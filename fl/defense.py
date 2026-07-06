from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F

from .utils import robust_center_scale
from .raha import RAHAState, raha_aggregate
from .config import RAHAConfig
from .raha_nat import RAHANATState, RAHANATConfig, raha_nat_select_clients
from .pebt import PEBTConfig, PretrainedEncoder, TemporalTracker, pebt_select_clients

@dataclass
class TrustConfig:
    """Trust system parameters"""
    trust_init: float = 0.5  # Start neutral
    trust_inc: float = 0.10
    trust_dec: float = 0.20
    trust_min: float = 0.0
    trust_max: float = 1.0
    high_trust: float = 0.70
    high_trust_downweight: float = 0.50
    lambda_smooth: float = 0.3
    beta1_trust: float = 0.75
    beta2_deviation: float = 0.15
    beta3_variance: float = 0.10
    history_window: int = 3

@dataclass
class ScoreConfig:
    """Multi-signal scoring weights"""
    w_krum: float = 0.25
    w_cos: float = 0.20
    w_norm: float = 0.15
    w_val: float = 0.25
    w_trust: float = 0.15
    base_thresh: float = 0.0
    drop_quantile: float = 0.35
    adaptive_weights: bool = False

def compute_krum_scores(deltas: List[torch.Tensor], attack_ratio_estimate: float = 0.2) -> List[float]:
    """Compute Krum distance scores (lower = more trustworthy)
    
    At low attack ratios, Krum's geometric distance is highly effective.
    """
    n = len(deltas)
    n_attackers = max(1, int(n * attack_ratio_estimate))
    if n <= 2 * n_attackers + 2:
        return [1.0] * n  # Not enough clients, return neutral scores
    
    # Compute pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(deltas[i] - deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, compute score = sum of distances to n-f-2 closest neighbors
    n_closest = n - n_attackers - 2
    raw_scores = []
    for i in range(n):
        dists_i = distances[i].numpy()
        closest_dists = np.partition(dists_i, n_closest)[:n_closest]
        raw_scores.append(float(np.sum(closest_dists)))
    
    # Normalize: lower distance = higher score (invert and normalize to [0,1])
    raw_scores = np.array(raw_scores)
    if raw_scores.max() > raw_scores.min():
        # Use exponential scaling to amplify differences at low ratios
        normalized = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        # Apply power transform to increase separation
        normalized = np.power(normalized, 0.7)  # Compress high scores, expand low scores
    else:
        normalized = np.ones_like(raw_scores)
    
    return normalized.tolist()

def compute_features(deltas: List[torch.Tensor], loss_improvements: List[float], 
                     attack_ratio: float = 0.2, validation_scores: List[float] = None):
    """Compute detection signals"""
    center_u = torch.stack(deltas).median(dim=0).values
    
    # Signal 1: Krum distance score
    krum_scores = compute_krum_scores(deltas, attack_ratio)
    
    # Signal 2: Cosine similarity
    cos_vals = [float(F.cosine_similarity(d, center_u, dim=0).item()) for d in deltas]

    # Signal 3: Norm score
    norms = np.array([float(torch.norm(d).item()) for d in deltas], dtype=np.float64)
    med, mad = robust_center_scale(norms)
    z = np.abs((norms - med) / (mad + 1e-6))
    norm_scores = (1.0 / (1.0 + 1.5 * z)).tolist()
    
    # Signal 4: Validation score
    if validation_scores is None:
        validation_scores = [0.5] * len(deltas)
    
    return krum_scores, cos_vals, norm_scores, validation_scores

def multisignal_scores(krum_scores, cos_vals, norm_scores, val_scores, trust_scores, cfg: ScoreConfig):
    """Weighted combination of 5 signals"""
    scores = [
        cfg.w_krum * k + cfg.w_cos * c + cfg.w_norm * n + cfg.w_val * v + cfg.w_trust * t
        for k, c, n, v, t in zip(krum_scores, cos_vals, norm_scores, val_scores, trust_scores)
    ]
    return scores

def compute_risk_score(trust: float, deviation: float, variance: float, 
                       beta1: float, beta2: float, beta3: float) -> float:
    """RAHA Step 5: Compute dynamic risk score
    
    Risk_i^t = β_1(1-T_i^t) + β_2*D_i^t + β_3*V_i^t
    """
    risk = beta1 * (1.0 - trust) + beta2 * deviation + beta3 * variance
    return max(0.0, min(1.0, risk))  # Clip to [0,1]

def compute_adaptive_thresholds(risk_scores: List[float], attack_ratio_estimate: float = 0.2) -> Tuple[float, float]:
    """RAHA Step 6: Compute adaptive thresholds from risk distribution
    
    CRITICAL: Very aggressive thresholds to catch attackers
    """
    risks = np.array(risk_scores)
    
    # Adaptive percentiles based on attack ratio
    if attack_ratio_estimate > 0.6:
        # High ratio: aggressive thresholds
        tau_low = float(np.percentile(risks, 10))  # Bottom 10%
        tau_high = float(np.percentile(risks, 35))  # 35th percentile
    elif attack_ratio_estimate < 0.2:
        # Low ratio: conservative thresholds to reduce FP
        tau_low = float(np.percentile(risks, 25))   # Bottom 25%
        tau_high = float(np.percentile(risks, 60))  # 60th percentile
    else:
        # Medium ratio: balanced
        tau_low = float(np.percentile(risks, 20))   # Bottom 20%
        tau_high = float(np.percentile(risks, 50))  # 50th percentile
    
    # Ensure minimum separation
    min_sep = 0.15 if attack_ratio_estimate < 0.6 else 0.10
    if tau_high - tau_low < min_sep:
        tau_high = tau_low + min_sep
    
    return tau_low, tau_high

def route_clients_raha(client_ids: List[int], risk_scores: Dict[int, float],
                       tau_low: float, tau_high: float) -> Tuple[List[int], List[int], List[int]]:
    """RAHA Step 7: Adaptive routing based on risk
    
    - Risk < τ_L: Trusted pool
    - τ_L ≤ Risk < τ_H: Down-weighted pool  
    - Risk ≥ τ_H: Robust fallback pool
    """
    trusted = []        # Low risk
    downweighted = []   # Medium risk
    fallback = []       # High risk
    
    for cid in client_ids:
        risk = risk_scores.get(cid, 0.5)
        if risk < tau_low:
            trusted.append(cid)
        elif risk < tau_high:
            downweighted.append(cid)
        else:
            fallback.append(cid)
    
    return trusted, downweighted, fallback

def compute_geometric_deviation(deltas: List[torch.Tensor], idx: int) -> float:
    """RAHA: Compute normalized geometric deviation D_i^t
    
    Measures how far client i's update is from neighbors
    """
    n = len(deltas)
    if n <= 1:
        return 0.0
    
    # Compute distances to all other clients
    distances = []
    for j in range(n):
        if j != idx:
            dist = float(torch.norm(deltas[idx] - deltas[j]).item())
            distances.append(dist)
    
    # Normalize using robust statistics
    distances = np.array(distances)
    median_dist = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_dist)))
    
    if mad < 1e-6:
        # If all distances similar, check if far from median
        if median_dist > np.percentile(distances, 75):
            return 0.8  # High deviation
        return 0.0
    
    # Z-score based on median distance - more aggressive
    z_score = (np.mean(distances) - median_dist) / (mad + 1e-6)
    # Amplified sigmoid mapping
    deviation = 1.0 / (1.0 + np.exp(-2.0 * z_score))  # Steeper curve
    
    return float(deviation)

def select_clients(
    mode: str,
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    loss_improvements: List[float],
    trust: Dict[int, float],
    trust_cfg: TrustConfig,
    score_cfg: ScoreConfig,
    client_history: Dict[int, List[float]] = None,
    raha_nat_state: RAHANATState = None,
    raha_nat_cfg: RAHANATConfig = None,
    global_model = None,
    val_loader = None,
    device: str = "cpu",
    pebt_encoder: PretrainedEncoder = None,
    pebt_tracker: TemporalTracker = None,
    pebt_cfg: PEBTConfig = None,
    known_attack_ratio: float = 0.2,  # NEW: Known attack ratio from experiment
) -> Tuple[List[int], List[float], Dict[str, float], Dict[int, float]]:
    """
    RAHA: Risk-Adaptive Hybrid Aggregation
    
    Modes:
    - fedavg: No defense (baseline)
    - cosine_only: Cosine similarity filtering (baseline)
    - krum: Krum robust aggregation (baseline)
    - multi_krum: Multi-Krum robust aggregation (baseline)
    - trimmed_mean: Trimmed mean aggregation (baseline)
    - fltrust: FLTrust aggregation (baseline)
    - scaffold: SCAFFOLD with control variates (baseline)
    - multisignal: Hybrid 4-signal WITHOUT temporal trust (baseline)
    - multisignal_trust: RAHA - Risk-Adaptive Hybrid Aggregation (PROPOSED)
    - raha_nat: RAHA-NAT - Non-IID-Aware Adaptive Trust (PROPOSED)
    - pebt: PEBT - Pre-trained Embedding-Based Trust (PROPOSED)
    
    RAHA Steps:
    Step 2: Multi-signal feature extraction (S_cos, S_norm, S_val, S_geo)
    Step 3: Reliability estimation R_i^t = Σ α_k S_i^k
    Step 4: Temporal trust update T_i^t = λT_i^(t-1) + (1-λ)R_i^t
    Step 5: Dynamic risk computation Risk_i^t = β_1(1-T) + β_2*D + β_3*V
    Step 6: Adaptive thresholding (τ_L, τ_H from risk distribution)
    Step 7: Adaptive routing (trusted/downweighted/fallback)
    Step 8-10: Hybrid aggregation with trust weights
    
    RAHA-NAT:
    Non-IID-aware variant with cluster-based anomaly detection,
    validation damage computation, and persistent attacker tracking.
    """
    start = time.time()
    
    # Initialize client history if not provided
    if client_history is None:
        client_history = {}
    
    # Use known attack ratio directly
    attack_ratio_estimate = known_attack_ratio
    
    # RAHA Step 2: Multi-signal feature extraction
    krum_scores, cos_vals, norm_scores, loss_scores = compute_features(deltas, loss_improvements, attack_ratio_estimate)
    
    # Initialize trust scores
    trust_scores = [trust.get(cid, trust_cfg.trust_init) for cid in client_ids]
    
    # RAHA Step 3: Reliability estimation R_i^t = Σ α_k S_i^k (with adaptive weights)
    reliability_scores = multisignal_scores(krum_scores, cos_vals, norm_scores, loss_scores, trust_scores, score_cfg)
    scores = reliability_scores  # Alias for compatibility

    # PHASE I: Multi-signal suspicion detection
    suspicious = [False]*len(client_ids)
    keep = [True]*len(client_ids)

    if mode == "fedavg":
        pass
    elif mode == "cosine_only":
        for i,c in enumerate(cos_vals):
            suspicious[i] = (c < score_cfg.base_thresh)
        keep = [not s for s in suspicious]
    elif mode == "krum":
        # Krum uses distance-based selection (handled in aggregation)
        pass
    elif mode == "multi_krum":
        # Multi-Krum uses distance-based selection (handled in aggregation)
        pass
    elif mode == "trimmed_mean":
        # Trimmed mean: handled in aggregation, keep all clients
        pass
    elif mode == "fltrust":
        # FLTrust: handled in aggregation with server reference, keep all clients
        pass
    elif mode == "scaffold":
        # SCAFFOLD uses control variates (handled in aggregation)
        pass
    elif mode == "multisignal":
        # PROPOSED: Multi-signal scoring WITHOUT temporal trust
        # Use multiple aggressive thresholds
        
        # Method 1: Quantile-based filtering (drop bottom performers)
        thr_quantile = float(np.quantile(np.array(scores), score_cfg.drop_quantile))
        
        # Method 2: Absolute threshold for very low scores
        thr_absolute = 0.25  # Flag scores below 0.25 as suspicious
        
        # Method 3: Statistical outlier detection
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        thr_outlier = mean_score - 1.5 * std_score  # 1.5 std below mean
        
        # Method 4: At low attack ratios, use stricter geometric filtering
        if attack_ratio_estimate < 0.3:
            # Boost Krum signal importance - use more aggressive threshold
            krum_threshold = float(np.percentile(krum_scores, 20))  # Bottom 20% by Krum
            # Require BOTH low combined score AND low Krum score for flagging
            for i, (s, k) in enumerate(zip(scores, krum_scores)):
                # More conservative: flag only if score is low AND Krum is very low
                suspicious[i] = ((s < thr_quantile) and (k < krum_threshold)) or \
                               (s < thr_absolute) or (k < 0.5)  # Very low Krum = definite attacker
        else:
            # Standard multi-threshold
            for i, s in enumerate(scores):
                suspicious[i] = (s < thr_quantile) or (s < thr_absolute) or (s < thr_outlier)
        
        keep = [not s for s in suspicious]
    elif mode == "multisignal_trust":
        # PROPOSED: Adaptive Hybrid - Multi-Krum OR Trust-based routing
        # Low ratio (<0.3): Use Multi-Krum only (it's already optimal)
        # High ratio (>=0.3): Use full trust-based routing
        
        # Initialize variables
        tau_low = 0.0
        tau_high = 1.0
        trusted = []
        downweighted = []
        fallback = []
        
        if attack_ratio_estimate < 0.3:
            # Low attack ratio: Multi-Krum is sufficient
            n = len(client_ids)
            n_attackers = max(1, int(n * attack_ratio_estimate))
            n_select = n - n_attackers
            krum_indices = np.argsort(krum_scores)[::-1][:n_select]
            
            # Keep only Multi-Krum selected clients
            for i, cid in enumerate(client_ids):
                if i in krum_indices:
                    keep[i] = True
                    suspicious[i] = False
                else:
                    keep[i] = False
                    suspicious[i] = True
            
            # Simple trust update
            new_trust = dict(trust)
            risk_scores = {}
            for i, cid in enumerate(client_ids):
                if keep[i]:
                    new_trust[cid] = min(1.0, new_trust.get(cid, trust_cfg.trust_init) + 0.1)
                    risk_scores[cid] = 0.0
                else:
                    new_trust[cid] = max(0.0, new_trust.get(cid, trust_cfg.trust_init) - 0.2)
                    risk_scores[cid] = 1.0
        else:
            # High attack ratio: Use full trust-based routing
            # RAHA Step 4: Temporal trust update
            new_trust = {}
            for i, cid in enumerate(client_ids):
                prev_trust = trust.get(cid, trust_cfg.trust_init)
                current_reliability = reliability_scores[i]
                
                if current_reliability < 0.30:
                    decay_boost = 0.20
                elif current_reliability < 0.45:
                    decay_boost = 0.10
                else:
                    decay_boost = 0.0
                
                new_trust[cid] = trust_cfg.lambda_smooth * prev_trust + \
                               (1 - trust_cfg.lambda_smooth) * current_reliability - decay_boost
                new_trust[cid] = float(max(0.0, min(1.0, new_trust[cid])))
                
                if cid not in client_history:
                    client_history[cid] = []
                client_history[cid].append(current_reliability)
                if len(client_history[cid]) > trust_cfg.history_window:
                    client_history[cid] = client_history[cid][-trust_cfg.history_window:]
            
            # RAHA Step 5: Dynamic risk computation
            risk_scores = {}
            for i, cid in enumerate(client_ids):
                deviation = compute_geometric_deviation(deltas, i)
                history = client_history.get(cid, [reliability_scores[i]])
                variance = float(np.std(history)) if len(history) > 1 else 0.0
                
                risk_scores[cid] = compute_risk_score(
                    new_trust[cid], deviation, variance,
                    trust_cfg.beta1_trust, trust_cfg.beta2_deviation, trust_cfg.beta3_variance
                )
            
            # RAHA Step 6: Adaptive thresholding
            tau_low, tau_high = compute_adaptive_thresholds(list(risk_scores.values()), attack_ratio_estimate)
            
            # RAHA Step 7: Adaptive routing
            trusted, downweighted, fallback = route_clients_raha(
                client_ids, risk_scores, tau_low, tau_high
            )
            
            reliability_threshold = 0.35
            for i, cid in enumerate(client_ids):
                if reliability_scores[i] < reliability_threshold:
                    if cid in trusted:
                        trusted.remove(cid)
                        fallback.append(cid)
                    elif cid in downweighted:
                        downweighted.remove(cid)
                        fallback.append(cid)
            
            # Mark clients
            for i, cid in enumerate(client_ids):
                if cid in fallback:
                    suspicious[i] = True
                    keep[i] = False
                else:
                    suspicious[i] = False
                    keep[i] = True
    elif mode == "raha_nat":
        # PROPOSED: RAHA-NAT - Non-IID-Aware Adaptive Trust
        if raha_nat_state is None or raha_nat_cfg is None:
            # Fallback to multisignal if not properly initialized
            thr = float(np.quantile(np.array(scores), score_cfg.drop_quantile))
            for i,s in enumerate(scores):
                suspicious[i] = (s < thr)
            keep = [not s for s in suspicious]
        else:
            # Use RAHA-NAT selection
            kept_ids, kept_weights, stats, new_trust = raha_nat_select_clients(
                client_ids, deltas, base_weights, loss_improvements,
                raha_nat_state, raha_nat_cfg, global_model, val_loader, device
            )
            
            # Return early with RAHA-NAT results
            return kept_ids, kept_weights, stats, new_trust
    elif mode == "pebt":
        # PROPOSED: PEBT - Pre-trained Embedding-Based Trust
        if pebt_encoder is None or pebt_tracker is None or pebt_cfg is None:
            # Fallback to multisignal if not properly initialized
            thr = float(np.quantile(np.array(scores), score_cfg.drop_quantile))
            for i,s in enumerate(scores):
                suspicious[i] = (s < thr)
            keep = [not s for s in suspicious]
        else:
            # Use PEBT selection
            kept_ids, kept_weights, stats, detected_mal = pebt_select_clients(
                client_ids, deltas, base_weights, global_model, val_loader, device,
                pebt_encoder, pebt_tracker, pebt_cfg
            )
            
            # Update trust based on detection
            new_trust = dict(trust)
            for cid in client_ids:
                if cid in detected_mal:
                    new_trust[cid] = max(0.0, new_trust.get(cid, 1.0) - 0.2)
                else:
                    new_trust[cid] = min(1.0, new_trust.get(cid, 1.0) + 0.1)
            
            # Return early with PEBT results
            return kept_ids, kept_weights, stats, new_trust
    elif mode == "trimmed_mean":
        # Trimmed mean: drop top and bottom quantile based on scores
        thr = float(np.quantile(np.array(scores), score_cfg.drop_quantile))
        for i,s in enumerate(scores):
            suspicious[i] = (s < thr)
        keep = [not s for s in suspicious]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # PHASE II: Risk-adaptive weighting
    eff_weights = list(base_weights)
    dropped = 0
    softened = 0
    routed_trusted = 0
    routed_downweighted = 0
    routed_fallback = 0
    
    if mode == "multisignal_trust":
        if attack_ratio_estimate < 0.3:
            # Low ratio: Simple weighting (Multi-Krum already selected)
            for i, cid in enumerate(client_ids):
                if not keep[i]:
                    eff_weights[i] = 0.0
                    dropped += 1
        else:
            # High ratio: Score-based downweighting
            for i, cid in enumerate(client_ids):
                score = reliability_scores[i]
                
                # Score-based weighting
                if score >= 0.70:
                    eff_weights[i] = base_weights[i]
                elif score >= 0.45:
                    eff_weights[i] = base_weights[i] * 0.4
                    softened += 1
                else:
                    eff_weights[i] = 0.0
                    dropped += 1
    elif mode == "multisignal_trust_old":
        # OLD VERSION: Simple trust modulation (for comparison)
        for i,cid in enumerate(client_ids):
            if suspicious[i]:
                t = trust.get(cid, trust_cfg.trust_init)
                if t >= trust_cfg.high_trust:
                    keep[i] = True
                    eff_weights[i] *= trust_cfg.high_trust_downweight
                    softened += 1
                else:
                    keep[i] = False
                    dropped += 1
    else:
        dropped = int(sum(1 for k in keep if not k))

    kept_ids = [cid for cid,k in zip(client_ids, keep) if k]
    kept_weights = [w for w,k in zip(eff_weights, keep) if k]

    # Trust update (temporal component)
    if mode == "multisignal_trust":
        # Already updated in Algorithm 1 Step 3 above
        pass
    elif mode == "multisignal_trust_old":
        # OLD VERSION: Simple increment/decrement
        new_trust = dict(trust)
        for i,cid in enumerate(client_ids):
            t = new_trust.get(cid, trust_cfg.trust_init)
            t = t - trust_cfg.trust_dec if suspicious[i] else t + trust_cfg.trust_inc
            new_trust[cid] = float(min(trust_cfg.trust_max, max(trust_cfg.trust_min, t)))
    else:
        new_trust = dict(trust)

    stats = {
        "kept": float(len(kept_ids)),
        "dropped": float(dropped),
        "softened": float(softened),  # Trust interventions
        "routed_trusted": float(routed_trusted),  # RAHA: Trusted pool
        "routed_downweighted": float(routed_downweighted),  # RAHA: Down-weighted pool
        "routed_fallback": float(routed_fallback),  # RAHA: Robust fallback pool
        "score_min": float(np.min(scores)),
        "score_med": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "time_select_s": float(time.time() - start),
    }
    
    # Add RAHA-specific stats if applicable
    if mode == "multisignal_trust":
        stats["tau_low"] = float(tau_low)
        stats["tau_high"] = float(tau_high)
        stats["mean_risk"] = float(np.mean(list(risk_scores.values())))
        stats["mean_trust"] = float(np.mean(list(new_trust.values())))
    
    return kept_ids, kept_weights, stats, new_trust
