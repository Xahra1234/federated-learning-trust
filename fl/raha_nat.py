"""
RAHA-NAT: Non-IID-Aware Adaptive Trust
A real-time client filtering and aggregation defense that:
- Keeps benign non-IID clients
- Softly down-weights uncertain clients
- Discards only persistent malicious clients
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering

from .utils import robust_center_scale


@dataclass
class RAHANATConfig:
    """RAHA-NAT configuration parameters"""
    # Latent representation
    latent_dim: int = 64  # Increased from 32 for richer features
    
    # Clustering
    cluster_method: str = "spectral"  # Use spectral clustering
    n_clusters: int = 2  # Binary: benign vs malicious
    use_krum_features: bool = True  # Combine with Krum distances
    
    # Trust thresholds
    trust_high_threshold: float = 0.6  # Lowered from 0.7 for more trusted clients
    trust_low_threshold: float = 0.35  # Raised from 0.3 to be more selective
    
    # Discard policy
    discard_rounds: int = 2  # Faster discard (was 3)
    hard_discard_threshold: float = 0.2  # Hard exclude if trust < 0.2
    
    # History tracking
    history_window: int = 5
    
    # Risk weights - Prioritize cluster and direction over validation
    w_validation_damage: float = 0.2  # Reduced from 0.3 (noisy signal)
    w_cluster_anomaly: float = 0.35  # Increased from 0.25
    w_direction_deviation: float = 0.30  # Increased from 0.25
    w_temporal_instability: float = 0.15  # Reduced from 0.2
    
    # Trust update
    trust_smoothing: float = 0.5  # Reduced from 0.7 for faster adaptation
    
    # Aggregation weights - More aggressive
    trusted_weight: float = 1.0
    uncertain_weight: float = 0.7  # Increased from 0.5
    fallback_weight: float = 0.2  # Increased from 0.1
    
    # Krum-style filtering
    use_krum_filter: bool = True  # Enable Krum pre-filtering
    krum_filter_ratio: float = 0.3  # Drop bottom 30% by Krum score
    
    # Adaptive weight scheduling
    adaptive_weights: bool = True  # Enable adaptive weight scheduling
    weight_decay_rate: float = 0.95  # Decay fallback weights over time


class RAHANATState:
    """Maintains temporal state for RAHA-NAT across rounds"""
    
    def __init__(self, n_clients: int, history_window: int = 5):
        self.n_clients = n_clients
        self.history_window = history_window
        
        # Trust scores
        self.trust_scores = {i: 1.0 for i in range(n_clients)}
        
        # Risk history
        self.risk_history = {i: deque(maxlen=history_window) for i in range(n_clients)}
        
        # Trust history
        self.trust_history = {i: deque(maxlen=history_window) for i in range(n_clients)}
        
        # Low trust counter (for discard policy)
        self.low_trust_counter = {i: 0 for i in range(n_clients)}
        
        # Discard flags
        self.discarded = {i: False for i in range(n_clients)}
        
        # Cluster assignments
        self.cluster_assignments = {i: 0 for i in range(n_clients)}
        
        # Route labels
        self.route_labels = {i: "trusted" for i in range(n_clients)}
    
    def update_trust(self, client_id: int, new_trust: float, low_threshold: float):
        """Update trust and track low trust periods"""
        self.trust_scores[client_id] = new_trust
        self.trust_history[client_id].append(new_trust)
        
        # Update low trust counter
        if new_trust < low_threshold:
            self.low_trust_counter[client_id] += 1
        else:
            self.low_trust_counter[client_id] = 0
    
    def update_risk(self, client_id: int, risk: float):
        """Update risk history"""
        self.risk_history[client_id].append(risk)
    
    def mark_discarded(self, client_id: int, discard_rounds: int):
        """Mark client as discarded if persistently low trust"""
        if self.low_trust_counter[client_id] >= discard_rounds:
            self.discarded[client_id] = True
    
    def is_discarded(self, client_id: int) -> bool:
        """Check if client is marked as discarded"""
        return self.discarded.get(client_id, False)


def build_enhanced_features(deltas: List[torch.Tensor], 
                           loss_improvements: List[float],
                           latent_dim: int) -> np.ndarray:
    """
    Build enhanced feature matrix combining:
    - Latent representation (PCA-like projection)
    - Norm features
    - Direction features (cosine with mean)
    - Loss improvement
    - Krum distances
    """
    n_clients = len(deltas)
    
    # 1. Latent representations
    latent_reps = []
    for delta in deltas:
        z = build_latent_representation(delta, latent_dim)
        latent_reps.append(z.cpu().numpy())
    latent_matrix = np.vstack(latent_reps)
    
    # 2. Norm features (normalized)
    norms = np.array([float(torch.norm(d).item()) for d in deltas])
    norms_normalized = (norms - norms.mean()) / (norms.std() + 1e-6)
    
    # 3. Direction features (cosine similarity with mean)
    mean_delta = torch.mean(torch.stack(deltas), dim=0)
    cos_sims = np.array([float(F.cosine_similarity(d, mean_delta, dim=0).item()) for d in deltas])
    
    # 4. Loss improvement features
    loss_feats = np.array(loss_improvements)
    loss_feats = (loss_feats - loss_feats.mean()) / (loss_feats.std() + 1e-6)
    
    # 5. Krum distance features
    krum_dists = compute_krum_distance_features(deltas)
    
    # Combine all features
    features = np.column_stack([
        latent_matrix,
        norms_normalized.reshape(-1, 1),
        cos_sims.reshape(-1, 1),
        loss_feats.reshape(-1, 1),
        krum_dists.reshape(-1, 1)
    ])
    
    return features


def compute_krum_distance_features(deltas: List[torch.Tensor]) -> np.ndarray:
    """
    Compute Krum-style distance features for each client
    Returns normalized distance scores (higher = more suspicious)
    """
    n = len(deltas)
    if n <= 2:
        return np.zeros(n)
    
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = float(torch.norm(deltas[i] - deltas[j]).item())
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, compute average distance to k nearest neighbors
    k = max(1, n // 2)
    krum_scores = []
    for i in range(n):
        dists_i = distances[i]
        # Get k nearest neighbors (excluding self)
        nearest_dists = np.partition(dists_i, k)[:k]
        avg_dist = np.mean(nearest_dists)
        krum_scores.append(avg_dist)
    
    krum_scores = np.array(krum_scores)
    # Normalize
    if krum_scores.std() > 1e-6:
        krum_scores = (krum_scores - krum_scores.mean()) / krum_scores.std()
    
    return krum_scores


def build_latent_representation(delta: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """
    Build latent representation z_i from client update
    Uses PCA-like dimensionality reduction
    """
    # Simple approach: random projection for efficiency
    # In practice, could use PCA or autoencoder
    delta_flat = delta.view(-1)
    
    # Use hash-based random projection for consistency
    seed = hash(delta_flat.shape[0]) % (2**32)
    torch.manual_seed(seed)
    projection = torch.randn(delta_flat.shape[0], latent_dim, device=delta.device)
    
    # Normalize projection
    projection = F.normalize(projection, dim=0)
    
    # Project
    z = torch.matmul(delta_flat, projection)
    return F.normalize(z, dim=0)


def cluster_clients(deltas: List[torch.Tensor],
                   loss_improvements: List[float],
                   latent_dim: int, 
                   n_clusters: int, 
                   method: str = "spectral",
                   use_krum_features: bool = True) -> np.ndarray:
    """
    Cluster clients using spectral clustering with enhanced features
    Returns cluster assignments
    """
    if len(deltas) < n_clusters:
        return np.zeros(len(deltas), dtype=int)
    
    # Build enhanced feature matrix
    if use_krum_features:
        X = build_enhanced_features(deltas, loss_improvements, latent_dim)
    else:
        # Fallback to simple latent representation
        latent_reps = []
        for delta in deltas:
            z = build_latent_representation(delta, latent_dim)
            latent_reps.append(z.cpu().numpy())
        X = np.vstack(latent_reps)
    
    # Cluster using spectral clustering
    if method == "spectral":
        try:
            # Use spectral clustering with affinity matrix
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='rbf',
                random_state=42,
                n_init=10
            )
            labels = clustering.fit_predict(X)
        except:
            # Fallback to kmeans if spectral fails
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
    else:
        # Use kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
    
    return labels


def compute_within_cluster_anomaly(deltas: List[torch.Tensor], 
                                   cluster_labels: np.ndarray,
                                   client_idx: int) -> float:
    """
    Compute within-cluster anomaly score for a client
    Measures how anomalous the client is within its cluster
    Higher score = more anomalous
    """
    client_cluster = cluster_labels[client_idx]
    
    # Find all clients in same cluster
    cluster_indices = np.where(cluster_labels == client_cluster)[0]
    
    if len(cluster_indices) <= 1:
        return 0.0
    
    # Compute distances to all cluster members
    client_delta = deltas[client_idx]
    distances = []
    
    for idx in cluster_indices:
        if idx != client_idx:
            dist = torch.norm(client_delta - deltas[idx]).item()
            distances.append(dist)
    
    distances = np.array(distances)
    
    # Use mean distance as anomaly indicator
    # Normalize by median distance across all pairs in cluster
    all_cluster_dists = []
    for i in cluster_indices:
        for j in cluster_indices:
            if i < j:
                dist = torch.norm(deltas[i] - deltas[j]).item()
                all_cluster_dists.append(dist)
    
    if len(all_cluster_dists) == 0:
        return 0.0
    
    median_cluster_dist = np.median(all_cluster_dists)
    
    if median_cluster_dist < 1e-10:
        # All updates are identical
        return 0.0
    
    # Anomaly = how much farther this client is compared to typical cluster distance
    mean_client_dist = np.mean(distances)
    anomaly_ratio = mean_client_dist / (median_cluster_dist + 1e-10)
    
    # Map to [0, 1] - ratio > 1 means more distant than typical
    anomaly = np.clip((anomaly_ratio - 1.0) / 2.0, 0.0, 1.0)
    
    return float(anomaly)


def compute_validation_damage(delta: torch.Tensor, global_model, 
                              val_loader, device) -> float:
    """
    Compute validation damage: how much does this update hurt validation accuracy?
    Uses a small clean validation set
    """
    # Save original state
    original_params = [p.clone() for p in global_model.parameters()]
    
    # Apply update temporarily
    idx = 0
    for p in global_model.parameters():
        n = p.numel()
        p.data.add_(delta[idx:idx+n].view_as(p))
        idx += n
    
    # Evaluate on validation set (small sample for efficiency)
    global_model.eval()
    total, correct = 0, 0
    max_batches = 5  # Increased from 2 for more stable signal
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = global_model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    # Restore original state
    for p, orig in zip(global_model.parameters(), original_params):
        p.data.copy_(orig)
    
    # Damage = 1 - accuracy (higher = worse)
    accuracy = correct / max(1, total)
    damage = 1.0 - accuracy
    
    return float(np.clip(damage, 0.0, 1.0))


def compute_direction_deviation(deltas: List[torch.Tensor], 
                                cluster_labels: np.ndarray,
                                client_idx: int) -> float:
    """
    Compute direction deviation within cluster
    Measures cosine distance from cluster mean direction
    """
    client_cluster = cluster_labels[client_idx]
    cluster_indices = np.where(cluster_labels == client_cluster)[0]
    
    if len(cluster_indices) <= 1:
        return 0.0
    
    # Compute cluster mean direction
    cluster_deltas = [deltas[i] for i in cluster_indices]
    cluster_mean = torch.mean(torch.stack(cluster_deltas), dim=0)
    
    # Cosine similarity
    client_delta = deltas[client_idx]
    cos_sim = F.cosine_similarity(client_delta, cluster_mean, dim=0).item()
    
    # Convert to deviation (0 = aligned, 1 = opposite)
    deviation = (1.0 - cos_sim) / 2.0
    
    return float(np.clip(deviation, 0.0, 1.0))


def compute_temporal_instability(state: RAHANATState, client_id: int) -> float:
    """
    Compute temporal instability from risk history
    High variance in risk = unstable/suspicious
    """
    risk_hist = list(state.risk_history[client_id])
    
    if len(risk_hist) < 2:
        return 0.0
    
    # Variance of risk scores
    variance = float(np.var(risk_hist))
    
    # Normalize to [0, 1]
    instability = np.clip(variance * 4.0, 0.0, 1.0)
    
    return float(instability)


def compute_risk_score(validation_damage: float, cluster_anomaly: float,
                      direction_deviation: float, temporal_instability: float,
                      cfg: RAHANATConfig) -> float:
    """
    Combine risk signals into normalized risk score
    """
    risk = (cfg.w_validation_damage * validation_damage +
            cfg.w_cluster_anomaly * cluster_anomaly +
            cfg.w_direction_deviation * direction_deviation +
            cfg.w_temporal_instability * temporal_instability)
    
    return float(np.clip(risk, 0.0, 1.0))


def update_trust_score(prev_trust: float, risk: float, 
                       smoothing: float) -> float:
    """
    Update trust using exponential moving average
    T_i^t = α * T_i^(t-1) + (1-α) * (1 - Risk_i^t)
    """
    reliability = 1.0 - risk
    new_trust = smoothing * prev_trust + (1 - smoothing) * reliability
    return float(np.clip(new_trust, 0.0, 1.0))


def route_clients(client_ids: List[int], trust_scores: Dict[int, float],
                 state: RAHANATState, cfg: RAHANATConfig) -> Tuple[List[int], List[int], List[int]]:
    """
    Route clients based on trust and discard status
    Returns: (trusted, uncertain, fallback)
    """
    trusted = []
    uncertain = []
    fallback = []
    
    for cid in client_ids:
        # Check if discarded
        if state.is_discarded(cid):
            fallback.append(cid)
            state.route_labels[cid] = "fallback"
            continue
        
        trust = trust_scores.get(cid, 1.0)
        
        if trust >= cfg.trust_high_threshold:
            trusted.append(cid)
            state.route_labels[cid] = "trusted"
        elif trust >= cfg.trust_low_threshold:
            uncertain.append(cid)
            state.route_labels[cid] = "uncertain"
        else:
            fallback.append(cid)
            state.route_labels[cid] = "fallback"
    
    return trusted, uncertain, fallback


def raha_nat_select_clients(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    loss_improvements: List[float],
    state: RAHANATState,
    cfg: RAHANATConfig,
    global_model=None,
    val_loader=None,
    device="cpu"
) -> Tuple[List[int], List[float], Dict[str, float], Dict[int, float]]:
    """
    RAHA-NAT: Non-IID-Aware Adaptive Trust
    
    Main entry point for client selection and weighting
    
    Returns: (kept_ids, kept_weights, stats, new_trust)
    """
    n_clients = len(client_ids)
    
    # Step 0: Optional Krum pre-filtering
    active_indices = list(range(n_clients))
    if cfg.use_krum_filter and n_clients > 3:
        krum_scores = compute_krum_distance_features(deltas)
        # Filter out bottom krum_filter_ratio by Krum score (higher score = more suspicious)
        n_filter = int(n_clients * cfg.krum_filter_ratio)
        if n_filter > 0:
            # Get indices of clients with lowest Krum scores (most trustworthy)
            sorted_indices = np.argsort(krum_scores)
            active_indices = sorted_indices[:n_clients - n_filter].tolist()
    
    # Work with filtered clients
    active_client_ids = [client_ids[i] for i in active_indices]
    active_deltas = [deltas[i] for i in active_indices]
    active_weights = [base_weights[i] for i in active_indices]
    active_loss_imps = [loss_improvements[i] for i in active_indices]
    n_active = len(active_client_ids)
    
    # Step 1: Cluster clients with enhanced features
    cluster_labels = cluster_clients(
        active_deltas, active_loss_imps, cfg.latent_dim, 
        cfg.n_clusters, cfg.cluster_method, cfg.use_krum_features
    )
    
    # Update cluster assignments
    for i, cid in enumerate(active_client_ids):
        state.cluster_assignments[cid] = int(cluster_labels[i])
    
    # Step 2: Compute risk components for each active client
    risk_scores = {}
    validation_damages = []
    cluster_anomalies = []
    direction_deviations = []
    temporal_instabilities = []
    
    for i, cid in enumerate(active_client_ids):
        # Validation damage (if validation loader provided)
        if val_loader is not None and global_model is not None:
            val_damage = compute_validation_damage(active_deltas[i], global_model, val_loader, device)
        else:
            # Fallback: use loss improvement as proxy
            val_damage = max(0.0, -active_loss_imps[i])
        validation_damages.append(val_damage)
        
        # Within-cluster anomaly
        cluster_anomaly = compute_within_cluster_anomaly(active_deltas, cluster_labels, i)
        cluster_anomalies.append(cluster_anomaly)
        
        # Direction deviation
        direction_dev = compute_direction_deviation(active_deltas, cluster_labels, i)
        direction_deviations.append(direction_dev)
        
        # Temporal instability
        temporal_inst = compute_temporal_instability(state, cid)
        temporal_instabilities.append(temporal_inst)
        
        # Combined risk score
        risk = compute_risk_score(val_damage, cluster_anomaly, direction_dev, temporal_inst, cfg)
        risk_scores[cid] = risk
        
        # Update risk history
        state.update_risk(cid, risk)
    
    # Mark filtered-out clients as high risk
    for i in range(n_clients):
        if i not in active_indices:
            cid = client_ids[i]
            risk_scores[cid] = 1.0  # Maximum risk
            state.update_risk(cid, 1.0)
    
    # Step 3: Update trust scores for all clients
    new_trust = {}
    for cid in client_ids:
        prev_trust = state.trust_scores.get(cid, 1.0)
        risk = risk_scores[cid]
        new_trust[cid] = update_trust_score(prev_trust, risk, cfg.trust_smoothing)
        
        # Update state
        state.update_trust(cid, new_trust[cid], cfg.trust_low_threshold)
        
        # Check for discard
        state.mark_discarded(cid, cfg.discard_rounds)
    
    # Step 4: Route clients
    trusted, uncertain, fallback = route_clients(active_client_ids, new_trust, state, cfg)
    
    # Step 5: Compute weights with hard discard and adaptive scheduling
    kept_ids = []
    kept_weights = []
    discarded_count = 0
    krum_filtered_count = n_clients - n_active
    
    # Adaptive weight decay based on round (if state tracks rounds)
    round_num = len(list(state.trust_history.values())[0]) if state.trust_history else 0
    if cfg.adaptive_weights and round_num > 0:
        # Decay fallback weight over time to be more aggressive
        adaptive_fallback_weight = cfg.fallback_weight * (cfg.weight_decay_rate ** round_num)
        adaptive_fallback_weight = max(0.05, adaptive_fallback_weight)  # Minimum 5%
    else:
        adaptive_fallback_weight = cfg.fallback_weight
    
    for i, cid in enumerate(client_ids):
        # Skip Krum-filtered clients
        if i not in active_indices:
            discarded_count += 1
            continue
            
        # Hard discard if trust is extremely low
        if new_trust[cid] < cfg.hard_discard_threshold:
            discarded_count += 1
            continue
        
        # Find index in active lists
        active_idx = active_indices.index(i)
        kept_ids.append(cid)
        
        if cid in trusted:
            # Trusted: full weight modulated by trust
            weight = base_weights[i] * cfg.trusted_weight * new_trust[cid]
        elif cid in uncertain:
            # Uncertain: reduced weight
            weight = base_weights[i] * cfg.uncertain_weight
        else:  # fallback
            # Fallback: minimal weight with adaptive decay
            weight = base_weights[i] * adaptive_fallback_weight
        
        kept_weights.append(weight)
    
    # Step 6: Compute statistics
    stats = {
        "kept": float(len(kept_ids)),
        "dropped": float(discarded_count),
        "krum_filtered": float(krum_filtered_count),
        "softened": float(len(uncertain) + len(fallback)),
        "routed_trusted": float(len(trusted)),
        "routed_uncertain": float(len(uncertain)),
        "routed_fallback": float(len(fallback)),
        "discarded_count": float(sum(1 for cid in client_ids if state.is_discarded(cid))),
        "mean_risk": float(np.mean(list(risk_scores.values()))),
        "mean_trust": float(np.mean(list(new_trust.values()))),
        "mean_validation_damage": float(np.mean(validation_damages)) if validation_damages else 0.0,
        "mean_cluster_anomaly": float(np.mean(cluster_anomalies)) if cluster_anomalies else 0.0,
        "mean_direction_deviation": float(np.mean(direction_deviations)) if direction_deviations else 0.0,
        "mean_temporal_instability": float(np.mean(temporal_instabilities)) if temporal_instabilities else 0.0,
        "n_clusters": float(len(np.unique(cluster_labels))),
        "adaptive_fallback_weight": float(adaptive_fallback_weight),
    }
    
    return kept_ids, kept_weights, stats, new_trust
