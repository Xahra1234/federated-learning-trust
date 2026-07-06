"""
RAHA: Risk-Adaptive Hybrid Aggregation
A novel federated learning defense that uses dynamic risk scoring,
temporal trust updates, and adaptive routing for robust aggregation.
"""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from .utils import robust_center_scale
from .config import RAHAConfig


class RAHAState:
    """Maintains temporal state for RAHA across rounds"""
    def __init__(self, n_clients: int, history_window: int = 5):
        self.trust_scores = {i: 1.0 for i in range(n_clients)}
        self.update_history = {i: deque(maxlen=history_window) for i in range(n_clients)}
        self.reliability_history = {i: deque(maxlen=history_window) for i in range(n_clients)}
        
    def update_history(self, client_id: int, update: torch.Tensor, reliability: float):
        """Store update and reliability for temporal variance computation"""
        self.update_history[client_id].append(update.clone().detach())
        self.reliability_history[client_id].append(reliability)


def compute_raha_signals(deltas: List[torch.Tensor], 
                         loss_improvements: List[float],
                         cfg: RAHAConfig) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    RAHA Step 2: Multi-signal feature extraction
    Returns: (cosine_scores, norm_scores, val_scores, geo_scores)
    """
    U = torch.stack(deltas)
    mean_u = torch.mean(U, dim=0)
    
    # Signal 1: Cosine similarity (alignment with mean)
    cos_scores = [float(F.cosine_similarity(d, mean_u, dim=0).item()) for d in deltas]
    
    # Signal 2: Norm deviation (robust normalization)
    norms = np.array([float(torch.norm(d).item()) for d in deltas], dtype=np.float64)
    med, mad = robust_center_scale(norms)
    z = np.abs((norms - med) / (mad + 1e-10))
    norm_scores = (1.0 / (1.0 + z)).tolist()
    
    # Signal 3: Validation behavior (loss improvement)
    li = np.clip(np.array(loss_improvements, dtype=np.float64), -1.0, 1.0)
    val_scores = ((li + 1.0) / 2.0).tolist()
    
    # Signal 4: Geometric consistency (Krum-like distance)
    geo_scores = compute_geometric_consistency(deltas)
    
    return cos_scores, norm_scores, val_scores, geo_scores


def compute_geometric_consistency(deltas: List[torch.Tensor], n_attackers: int = 2) -> List[float]:
    """Compute geometric consistency using Krum-like distance"""
    n = len(deltas)
    if n <= 2 * n_attackers + 2:
        return [1.0] * n
    
    # Pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(deltas[i] - deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Sum of distances to n-f-2 closest neighbors
    n_closest = n - n_attackers - 2
    raw_scores = []
    for i in range(n):
        dists_i = distances[i].numpy()
        closest_dists = np.partition(dists_i, n_closest)[:n_closest]
        raw_scores.append(float(np.sum(closest_dists)))
    
    # Normalize: lower distance = higher score
    raw_scores = np.array(raw_scores)
    if raw_scores.max() > raw_scores.min():
        normalized = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        normalized = np.ones_like(raw_scores)
    
    return normalized.tolist()


def compute_reliability(cos_scores: List[float], norm_scores: List[float],
                       val_scores: List[float], geo_scores: List[float],
                       cfg: RAHAConfig) -> List[float]:
    """
    RAHA Step 3: Reliability estimation
    R_i^t = α_1*S_cos + α_2*S_norm + α_3*S_val + α_4*S_geo
    """
    reliability = []
    for c, n, v, g in zip(cos_scores, norm_scores, val_scores, geo_scores):
        r = cfg.alpha_cos * c + cfg.alpha_norm * n + cfg.alpha_val * v + cfg.alpha_geo * g
        reliability.append(max(0.0, min(1.0, r)))  # Clip to [0,1]
    return reliability


def update_temporal_trust(client_ids: List[int], reliability: List[float],
                         state: RAHAState, cfg: RAHAConfig) -> Dict[int, float]:
    """
    RAHA Step 4: Temporal trust update with exponential smoothing
    T_i^t = λ*T_i^(t-1) + (1-λ)*R_i^t
    """
    new_trust = {}
    for cid, r in zip(client_ids, reliability):
        prev_trust = state.trust_scores.get(cid, 1.0)
        new_t = cfg.lambda_smooth * prev_trust + (1 - cfg.lambda_smooth) * r
        new_trust[cid] = max(0.0, min(1.0, new_t))  # Clip to [0,1]
    return new_trust


def compute_temporal_variance(client_id: int, state: RAHAState) -> float:
    """Compute variance of client's reliability over recent rounds"""
    history = list(state.reliability_history[client_id])
    if len(history) < 2:
        return 0.0
    return float(np.var(history))


def compute_geometric_deviation(deltas: List[torch.Tensor], client_idx: int) -> float:
    """Compute normalized geometric deviation for a client"""
    if len(deltas) < 2:
        return 0.0
    
    # Distance to mean
    mean_delta = torch.mean(torch.stack(deltas), dim=0)
    dist_to_mean = torch.norm(deltas[client_idx] - mean_delta).item()
    
    # Normalize by median distance
    all_dists = [torch.norm(d - mean_delta).item() for d in deltas]
    median_dist = float(np.median(all_dists))
    
    if median_dist < 1e-10:
        return 0.0
    
    normalized_dev = dist_to_mean / median_dist
    return max(0.0, min(1.0, normalized_dev / 3.0))  # Scale and clip


def compute_risk_scores(client_ids: List[int], deltas: List[torch.Tensor],
                       trust_scores: Dict[int, float], state: RAHAState,
                       cfg: RAHAConfig) -> List[float]:
    """
    RAHA Step 5: Dynamic risk computation
    Risk_i^t = β_1*(1-T_i^t) + β_2*D_i^t + β_3*V_i^t
    """
    risk_scores = []
    for i, cid in enumerate(client_ids):
        trust = trust_scores.get(cid, 1.0)
        deviation = compute_geometric_deviation(deltas, i)
        variance = compute_temporal_variance(cid, state)
        
        risk = (cfg.beta_trust * (1 - trust) + 
                cfg.beta_deviation * deviation + 
                cfg.beta_variance * variance)
        risk_scores.append(max(0.0, min(1.0, risk)))
    
    return risk_scores


def adaptive_thresholding(risk_scores: List[float], cfg: RAHAConfig) -> Tuple[float, float]:
    """
    RAHA Step 6: Adaptive thresholding
    τ_L = mean(risk), τ_H = mean(risk) + std(risk)
    """
    risks = np.array(risk_scores)
    mean_risk = float(np.mean(risks))
    std_risk = float(np.std(risks))
    
    tau_low = mean_risk * cfg.tau_low_factor
    tau_high = mean_risk + std_risk * cfg.tau_high_factor
    
    return tau_low, tau_high


def route_clients_by_risk(client_ids: List[int], risk_scores: List[float],
                         tau_low: float, tau_high: float) -> Tuple[List[int], List[int], List[int]]:
    """
    RAHA Step 7: Adaptive routing
    Returns: (trusted_pool, downweighted_pool, fallback_pool)
    """
    trusted = []
    downweighted = []
    fallback = []
    
    for cid, risk in zip(client_ids, risk_scores):
        if risk < tau_low:
            trusted.append(cid)
        elif risk < tau_high:
            downweighted.append(cid)
        else:
            fallback.append(cid)
    
    return trusted, downweighted, fallback


def aggregate_trusted(deltas: List[torch.Tensor], client_ids: List[int],
                     trust_scores: Dict[int, float], trusted_pool: List[int]) -> torch.Tensor:
    """
    RAHA Step 8: Trusted aggregation with trust weights
    β_i = T_i^t / Σ T_j^t
    """
    if not trusted_pool:
        return torch.zeros_like(deltas[0])
    
    # Map client IDs to indices
    id_to_idx = {cid: i for i, cid in enumerate(client_ids)}
    
    # Compute trust weights
    trust_sum = sum(trust_scores.get(cid, 1.0) for cid in trusted_pool)
    if trust_sum < 1e-10:
        trust_sum = 1.0
    
    # Weighted aggregation
    agg = torch.zeros_like(deltas[0])
    for cid in trusted_pool:
        idx = id_to_idx[cid]
        weight = trust_scores.get(cid, 1.0) / trust_sum
        agg += weight * deltas[idx]
    
    return agg


def robust_fallback_aggregate(deltas: List[torch.Tensor], client_ids: List[int],
                              fallback_pool: List[int], method: str = "krum") -> torch.Tensor:
    """
    RAHA Step 9: Robust fallback aggregation
    Apply Krum or Trimmed Mean on suspicious subset
    """
    if not fallback_pool:
        return torch.zeros_like(deltas[0])
    
    # Extract fallback deltas
    id_to_idx = {cid: i for i, cid in enumerate(client_ids)}
    fallback_deltas = [deltas[id_to_idx[cid]] for cid in fallback_pool]
    
    if method == "krum":
        return krum_aggregate(fallback_deltas)
    elif method == "trimmed_mean":
        return trimmed_mean_aggregate(fallback_deltas)
    else:
        # Default: simple mean
        return torch.mean(torch.stack(fallback_deltas), dim=0)


def krum_aggregate(deltas: List[torch.Tensor], n_attackers: int = 1) -> torch.Tensor:
    """Krum aggregation: select update with smallest distance sum"""
    n = len(deltas)
    if n == 0:
        raise ValueError("Empty delta list")
    if n == 1:
        return deltas[0]
    
    # Compute pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(deltas[i] - deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Select client with smallest distance sum to n-f-2 neighbors
    n_closest = max(1, n - n_attackers - 2)
    scores = []
    for i in range(n):
        dists_i = distances[i].numpy()
        closest_dists = np.partition(dists_i, min(n_closest, len(dists_i)-1))[:n_closest]
        scores.append(float(np.sum(closest_dists)))
    
    best_idx = int(np.argmin(scores))
    return deltas[best_idx]


def trimmed_mean_aggregate(deltas: List[torch.Tensor], trim_ratio: float = 0.2) -> torch.Tensor:
    """Trimmed mean: remove top and bottom trim_ratio and average"""
    if not deltas:
        raise ValueError("Empty delta list")
    if len(deltas) == 1:
        return deltas[0]
    
    stacked = torch.stack(deltas)
    n_trim = max(1, int(len(deltas) * trim_ratio))
    
    # Sort along client dimension and trim
    sorted_deltas, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_deltas[n_trim:-n_trim] if n_trim < len(deltas)//2 else sorted_deltas
    
    return torch.mean(trimmed, dim=0)


def raha_aggregate(client_ids: List[int], deltas: List[torch.Tensor],
                  base_weights: List[float], loss_improvements: List[float],
                  state: RAHAState, cfg: RAHAConfig) -> Tuple[List[int], List[float], Dict, Dict[int, float]]:
    """
    RAHA: Risk-Adaptive Hybrid Aggregation
    Main entry point that orchestrates all steps
    
    Returns: (kept_ids, kept_weights, stats, new_trust)
    """
    # Step 2: Multi-signal feature extraction
    cos_scores, norm_scores, val_scores, geo_scores = compute_raha_signals(
        deltas, loss_improvements, cfg
    )
    
    # Step 3: Reliability estimation
    reliability = compute_reliability(cos_scores, norm_scores, val_scores, geo_scores, cfg)
    
    # Step 4: Temporal trust update
    new_trust = update_temporal_trust(client_ids, reliability, state, cfg)
    
    # Update state
    for cid, r in zip(client_ids, reliability):
        state.trust_scores[cid] = new_trust[cid]
        state.reliability_history[cid].append(r)
    
    # Step 5: Dynamic risk computation
    risk_scores = compute_risk_scores(client_ids, deltas, new_trust, state, cfg)
    
    # Step 6: Adaptive thresholding
    tau_low, tau_high = adaptive_thresholding(risk_scores, cfg)
    
    # Step 7: Adaptive routing
    trusted_pool, downweighted_pool, fallback_pool = route_clients_by_risk(
        client_ids, risk_scores, tau_low, tau_high
    )
    
    # Compute weights for each pool
    kept_ids = client_ids.copy()
    kept_weights = []
    
    for i, cid in enumerate(client_ids):
        if cid in trusted_pool:
            # Full weight based on trust
            kept_weights.append(base_weights[i] * new_trust[cid])
        elif cid in downweighted_pool:
            # Downweighted based on risk
            risk = risk_scores[i]
            downweight_factor = 1.0 - risk  # Higher risk = lower weight
            kept_weights.append(base_weights[i] * downweight_factor)
        else:  # fallback_pool
            # Minimal weight (will use robust aggregation)
            kept_weights.append(base_weights[i] * 0.1)
    
    # Compute RAHA-specific metrics
    stats = {
        "kept": float(len(kept_ids)),
        "dropped": 0.0,  # RAHA doesn't drop clients
        "softened": float(len(downweighted_pool) + len(fallback_pool)),
        "routed_trusted": float(len(trusted_pool)),
        "routed_downweighted": float(len(downweighted_pool)),
        "routed_fallback": float(len(fallback_pool)),
        "avg_risk": float(np.mean(risk_scores)),
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "avg_trust": float(np.mean(list(new_trust.values()))),
        "trust_std": float(np.std(list(new_trust.values()))),
    }
    
    return kept_ids, kept_weights, stats, new_trust
