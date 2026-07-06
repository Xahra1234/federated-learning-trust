"""
Enhanced Defense Mechanisms for FedTrust

Implements 5 key improvements:
1. Per-layer scoring and clipping
2. Temporal drift/change-point detection in trust
3. Cluster-based suspicious-group filtering
4. Adaptive rule selection among baselines
5. Layer-wise temporal trust with adaptive aggregation (BIGGEST IMPROVEMENT)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from scipy.stats import chi2

from .utils import robust_center_scale

@dataclass
class EnhancedConfig:
    """Enhanced defense configuration"""
    # Layer-wise parameters
    layer_clip_threshold: float = 3.0  # Clip layers beyond 3 MAD
    layer_weight_decay: float = 0.9    # Exponential decay for layer importance
    
    # Temporal drift detection
    drift_window: int = 5              # Rounds to track for drift
    drift_threshold: float = 2.0       # Chi-square threshold for change-point
    
    # Clustering parameters
    cluster_eps: float = 0.5           # DBSCAN epsilon
    cluster_min_samples: int = 2       # DBSCAN min samples
    
    # Adaptive aggregation
    adaptive_mode: bool = True         # Enable adaptive rule selection
    performance_window: int = 3        # Rounds to evaluate performance
    
    # Trust parameters
    trust_init: float = 1.0
    trust_inc: float = 0.05
    trust_dec: float = 0.10
    trust_min: float = 0.0
    trust_max: float = 1.0
    high_trust: float = 0.70
    high_trust_downweight: float = 0.50

class LayerWiseAnalyzer:
    """Per-layer gradient analysis and clipping"""
    
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
        self.layer_history: Dict[int, List[List[float]]] = {}  # {client_id: [round_norms]}
    
    def extract_layer_norms(self, delta: torch.Tensor, model_shapes: List) -> List[float]:
        """Extract per-layer gradient norms"""
        norms = []
        offset = 0
        for shape in model_shapes:
            numel = np.prod(shape)
            layer_params = delta[offset:offset+numel]
            norms.append(float(torch.norm(layer_params).item()))
            offset += numel
        return norms
    
    def clip_layer_outliers(self, deltas: List[torch.Tensor], 
                           model_shapes: List) -> List[torch.Tensor]:
        """Clip per-layer outliers using robust statistics"""
        n_clients = len(deltas)
        n_layers = len(model_shapes)
        
        # Extract layer norms for all clients
        layer_norms = np.zeros((n_clients, n_layers))
        for i, delta in enumerate(deltas):
            layer_norms[i] = self.extract_layer_norms(delta, model_shapes)
        
        # Compute robust statistics per layer
        clipped_deltas = []
        for i, delta in enumerate(deltas):
            clipped = delta.clone()
            offset = 0
            
            for j, shape in enumerate(model_shapes):
                numel = np.prod(shape)
                layer_params = clipped[offset:offset+numel]
                
                # Robust clipping for this layer
                layer_vals = layer_norms[:, j]
                med, mad = robust_center_scale(layer_vals)
                z = abs((layer_norms[i, j] - med) / (mad + 1e-8))
                
                if z > self.cfg.layer_clip_threshold:
                    # Clip to threshold
                    scale = (med + self.cfg.layer_clip_threshold * mad) / (layer_norms[i, j] + 1e-8)
                    clipped[offset:offset+numel] = layer_params * scale
                
                offset += numel
            
            clipped_deltas.append(clipped)
        
        return clipped_deltas
    
    def compute_layer_scores(self, deltas: List[torch.Tensor], 
                            model_shapes: List) -> np.ndarray:
        """Compute per-client scores based on layer-wise analysis"""
        n_clients = len(deltas)
        n_layers = len(model_shapes)
        
        # Extract layer norms
        layer_norms = np.zeros((n_clients, n_layers))
        for i, delta in enumerate(deltas):
            layer_norms[i] = self.extract_layer_norms(delta, model_shapes)
        
        # Compute weighted anomaly scores (later layers more important)
        scores = np.zeros(n_clients)
        weights = np.array([self.cfg.layer_weight_decay ** (n_layers - j - 1) 
                           for j in range(n_layers)])
        weights /= weights.sum()
        
        for j in range(n_layers):
            layer_vals = layer_norms[:, j]
            med, mad = robust_center_scale(layer_vals)
            z = np.abs((layer_vals - med) / (mad + 1e-8))
            scores += weights[j] * z
        
        return scores

class TemporalDriftDetector:
    """Detect change-points in client behavior using temporal analysis"""
    
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
        self.history: Dict[int, List[np.ndarray]] = {}  # {client_id: [feature_vectors]}
    
    def update_history(self, client_id: int, features: np.ndarray):
        """Add current round features"""
        if client_id not in self.history:
            self.history[client_id] = []
        self.history[client_id].append(features)
        
        # Keep only recent window
        if len(self.history[client_id]) > self.cfg.drift_window:
            self.history[client_id] = self.history[client_id][-self.cfg.drift_window:]
    
    def detect_drift(self, client_id: int) -> Tuple[bool, float]:
        """
        Detect if client behavior has drifted significantly.
        Uses chi-square test on recent vs historical features.
        """
        if client_id not in self.history or len(self.history[client_id]) < 3:
            return False, 0.0
        
        hist = np.array(self.history[client_id])
        
        # Split into recent vs historical
        split = len(hist) // 2
        historical = hist[:split]
        recent = hist[split:]
        
        # Compute means
        hist_mean = np.mean(historical, axis=0)
        recent_mean = np.mean(recent, axis=0)
        
        # Chi-square statistic for distribution change
        hist_cov = np.cov(historical.T) + np.eye(historical.shape[1]) * 1e-6
        try:
            inv_cov = np.linalg.inv(hist_cov)
            diff = recent_mean - hist_mean
            chi2_stat = float(diff @ inv_cov @ diff)
        except:
            chi2_stat = 0.0
        
        # Test against threshold
        drift_detected = chi2_stat > self.cfg.drift_threshold
        return drift_detected, chi2_stat

class ClusterBasedFilter:
    """Cluster-based suspicious group detection"""
    
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
    
    def detect_suspicious_clusters(self, features: np.ndarray) -> List[bool]:
        """
        Use DBSCAN to identify outlier clusters.
        Clients in small/outlier clusters are suspicious.
        """
        if len(features) < 3:
            return [False] * len(features)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.cfg.cluster_eps, 
                           min_samples=self.cfg.cluster_min_samples).fit(features)
        labels = clustering.labels_
        
        # Outliers (label=-1) and small clusters are suspicious
        cluster_sizes = {label: np.sum(labels == label) for label in set(labels)}
        median_size = np.median(list(cluster_sizes.values()))
        
        suspicious = []
        for label in labels:
            if label == -1:  # Outlier
                suspicious.append(True)
            elif cluster_sizes[label] < median_size / 2:  # Small cluster
                suspicious.append(True)
            else:
                suspicious.append(False)
        
        return suspicious

class AdaptiveAggregator:
    """Adaptive rule selection based on recent performance"""
    
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
        self.performance_history: Dict[str, List[float]] = {
            'fedavg': [],
            'cosine': [],
            'multisignal': [],
            'trust': [],
        }
        self.current_mode = 'multisignal'
    
    def update_performance(self, mode: str, accuracy: float):
        """Track performance of each aggregation mode"""
        if mode in self.performance_history:
            self.performance_history[mode].append(accuracy)
            if len(self.performance_history[mode]) > self.cfg.performance_window:
                self.performance_history[mode] = \
                    self.performance_history[mode][-self.cfg.performance_window:]
    
    def select_best_mode(self) -> str:
        """Select aggregation mode with best recent performance"""
        if not self.cfg.adaptive_mode:
            return self.current_mode
        
        # Compute average performance
        avg_perf = {}
        for mode, hist in self.performance_history.items():
            if len(hist) > 0:
                avg_perf[mode] = np.mean(hist)
        
        if len(avg_perf) == 0:
            return self.current_mode
        
        # Select best
        best_mode = max(avg_perf.items(), key=lambda x: x[1])[0]
        self.current_mode = best_mode
        return best_mode

class LayerWiseTemporalTrust:
    """
    BIGGEST IMPROVEMENT: Layer-wise temporal trust with adaptive aggregation.
    
    Key innovation: Track trust per layer, allowing fine-grained detection
    of attacks that target specific layers (e.g., backdoors in final layer).
    """
    
    def __init__(self, cfg: EnhancedConfig, n_layers: int):
        self.cfg = cfg
        self.n_layers = n_layers
        
        # Per-client, per-layer trust: {client_id: [layer_trusts]}
        self.layer_trust: Dict[int, np.ndarray] = {}
        
        # Layer importance weights (learned adaptively)
        self.layer_importance = np.ones(n_layers) / n_layers
    
    def initialize_client(self, client_id: int):
        """Initialize trust for new client"""
        if client_id not in self.layer_trust:
            self.layer_trust[client_id] = np.ones(self.n_layers) * self.cfg.trust_init
    
    def update_layer_trust(self, client_id: int, layer_scores: np.ndarray):
        """
        Update per-layer trust based on layer-wise anomaly scores.
        
        Args:
            layer_scores: (n_layers,) anomaly scores per layer
        """
        self.initialize_client(client_id)
        
        for j in range(self.n_layers):
            if layer_scores[j] > 1.0:  # Anomalous
                self.layer_trust[client_id][j] -= self.cfg.trust_dec
            else:  # Normal
                self.layer_trust[client_id][j] += self.cfg.trust_inc
            
            # Clamp
            self.layer_trust[client_id][j] = np.clip(
                self.layer_trust[client_id][j],
                self.cfg.trust_min,
                self.cfg.trust_max
            )
    
    def get_aggregate_trust(self, client_id: int) -> float:
        """Compute weighted aggregate trust across layers"""
        self.initialize_client(client_id)
        return float(np.dot(self.layer_trust[client_id], self.layer_importance))
    
    def get_layer_weights(self, client_id: int) -> np.ndarray:
        """Get per-layer aggregation weights based on trust"""
        self.initialize_client(client_id)
        # Low trust layers get downweighted
        weights = self.layer_trust[client_id] / self.cfg.trust_max
        return weights
    
    def update_layer_importance(self, layer_contributions: np.ndarray):
        """
        Adaptively update layer importance based on contribution to accuracy.
        Layers that contribute more to model performance get higher weight.
        """
        # Exponential moving average
        alpha = 0.1
        self.layer_importance = (1 - alpha) * self.layer_importance + \
                               alpha * layer_contributions
        self.layer_importance /= self.layer_importance.sum()

def enhanced_select_clients(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    loss_improvements: List[float],
    model_shapes: List,
    cfg: EnhancedConfig,
    layer_analyzer: LayerWiseAnalyzer,
    drift_detector: TemporalDriftDetector,
    cluster_filter: ClusterBasedFilter,
    adaptive_agg: AdaptiveAggregator,
    layer_trust: LayerWiseTemporalTrust,
    current_accuracy: Optional[float] = None,
) -> Tuple[List[int], List[float], List[torch.Tensor], Dict]:
    """
    Enhanced client selection with all 5 improvements.
    
    Returns:
        kept_ids: Selected client IDs
        kept_weights: Aggregation weights
        kept_deltas: Processed gradients (clipped)
        stats: Diagnostic statistics
    """
    
    # 1. PER-LAYER SCORING AND CLIPPING
    layer_scores = layer_analyzer.compute_layer_scores(deltas, model_shapes)
    clipped_deltas = layer_analyzer.clip_layer_outliers(deltas, model_shapes)
    
    # 2. TEMPORAL DRIFT DETECTION
    drift_flags = []
    drift_stats = []
    for i, cid in enumerate(client_ids):
        # Extract features for drift detection
        features = np.array([
            float(torch.norm(deltas[i]).item()),
            loss_improvements[i],
            layer_scores[i],
        ])
        drift_detector.update_history(cid, features)
        drift, drift_score = drift_detector.detect_drift(cid)
        drift_flags.append(drift)
        drift_stats.append(drift_score)
    
    # 3. CLUSTER-BASED FILTERING
    # Combine multiple features for clustering
    cluster_features = np.column_stack([
        [float(torch.norm(d).item()) for d in deltas],
        loss_improvements,
        layer_scores,
    ])
    cluster_suspicious = cluster_filter.detect_suspicious_clusters(cluster_features)
    
    # 4. ADAPTIVE RULE SELECTION
    if current_accuracy is not None:
        current_mode = adaptive_agg.current_mode
        adaptive_agg.update_performance(current_mode, current_accuracy)
    best_mode = adaptive_agg.select_best_mode()
    
    # 5. LAYER-WISE TEMPORAL TRUST (BIGGEST IMPROVEMENT)
    # Update per-layer trust
    for i, cid in enumerate(client_ids):
        layer_anomaly = layer_analyzer.extract_layer_norms(deltas[i], model_shapes)
        layer_anomaly = np.array(layer_anomaly)
        # Normalize to anomaly scores
        med, mad = robust_center_scale(layer_anomaly)
        layer_anomaly = np.abs((layer_anomaly - med) / (mad + 1e-8))
        layer_trust.update_layer_trust(cid, layer_anomaly)
    
    # COMBINED DECISION
    keep = [True] * len(client_ids)
    eff_weights = list(base_weights)
    layer_weights_list = []
    
    for i, cid in enumerate(client_ids):
        # Aggregate suspicion signals
        suspicious = (
            layer_scores[i] > 2.0 or      # Layer-wise anomaly
            drift_flags[i] or              # Temporal drift
            cluster_suspicious[i]          # Cluster outlier
        )
        
        if suspicious:
            # Get layer-wise trust
            agg_trust = layer_trust.get_aggregate_trust(cid)
            
            if agg_trust >= cfg.high_trust:
                # High trust: apply layer-wise downweighting
                layer_weights = layer_trust.get_layer_weights(cid)
                layer_weights_list.append(layer_weights)
                eff_weights[i] *= cfg.high_trust_downweight
            else:
                # Low trust: drop
                keep[i] = False
                layer_weights_list.append(None)  # Placeholder for dropped client
        else:
            layer_weights_list.append(np.ones(layer_trust.n_layers))
    
    # Apply layer-wise weights to deltas
    final_deltas = []
    for i, (delta, keep_flag) in enumerate(zip(clipped_deltas, keep)):
        if keep_flag:
            # Apply layer-wise trust weights
            weighted_delta = delta.clone()
            offset = 0
            layer_weights = layer_weights_list[i]
            if layer_weights is not None:
                for j, shape in enumerate(model_shapes):
                    numel = np.prod(shape)
                    layer_weight = layer_weights[j]
                    weighted_delta[offset:offset+numel] *= layer_weight
                    offset += numel
            final_deltas.append(weighted_delta)
    
    kept_ids = [cid for cid, k in zip(client_ids, keep) if k]
    kept_weights = [w for w, k in zip(eff_weights, keep) if k]
    
    stats = {
        'kept': len(kept_ids),
        'dropped': sum(1 for k in keep if not k),
        'layer_score_mean': float(np.mean(layer_scores)),
        'drift_detected': sum(drift_flags),
        'cluster_suspicious': sum(cluster_suspicious),
        'adaptive_mode': best_mode,
        'avg_layer_trust': float(np.mean([layer_trust.get_aggregate_trust(cid) 
                                          for cid in client_ids])),
    }
    
    return kept_ids, kept_weights, final_deltas, stats
