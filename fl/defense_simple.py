"""
Simplified 5-Signal Defense
- Uses known attack ratio (no estimation)
- 5 signals: cosine, norm, loss, krum, validation
- Adaptive: Multi-Krum at low ratios, trust routing at high ratios
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

def compute_krum_scores(deltas: List[torch.Tensor], attack_ratio: float) -> List[float]:
    n = len(deltas)
    n_attackers = max(1, int(n * attack_ratio))
    if n <= 2 * n_attackers + 2:
        return [1.0] * n
    
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(deltas[i] - deltas[j]).item()
            distances[i, j] = distances[j, i] = dist
    
    n_closest = n - n_attackers - 2
    scores = []
    for i in range(n):
        closest = np.partition(distances[i].numpy(), n_closest)[:n_closest]
        scores.append(float(np.sum(closest)))
    
    scores = np.array(scores)
    if scores.max() > scores.min():
        normalized = 1.0 - (scores - scores.min()) / (scores.max() - scores.min())
        normalized = np.power(normalized, 0.7)
    else:
        normalized = np.ones_like(scores)
    
    return normalized.tolist()

def compute_5_signals(deltas: List[torch.Tensor], loss_imps: List[float], 
                      val_scores: List[float], attack_ratio: float):
    """Compute 5 detection signals"""
    center = torch.stack(deltas).median(dim=0).values
    
    # 1. Cosine similarity
    cos_vals = [float(F.cosine_similarity(d, center, dim=0).item()) for d in deltas]
    
    # 2. Norm score
    norms = np.array([float(torch.norm(d).item()) for d in deltas])
    med, mad = np.median(norms), np.median(np.abs(norms - np.median(norms)))
    z = np.abs((norms - med) / (mad + 1e-6))
    norm_scores = (1.0 / (1.0 + 1.5 * z)).tolist()
    
    # 3. Loss improvement
    li = np.clip(np.array(loss_imps), -2.0, 2.0)
    loss_scores = ((li + 2.0) / 4.0).tolist()
    
    # 4. Krum distance
    krum_scores = compute_krum_scores(deltas, attack_ratio)
    
    # 5. Validation score (already computed)
    
    return cos_vals, norm_scores, loss_scores, krum_scores, val_scores

def multisignal_trust_select(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    loss_imps: List[float],
    val_scores: List[float],
    attack_ratio: float,
    trust: Dict[int, float]
) -> Tuple[List[int], List[float], Dict[int, float]]:
    """
    Adaptive 5-signal selection:
    - Low ratio (<0.3): Use Multi-Krum only
    - High ratio (>=0.3): Use full trust-based routing
    """
    
    # Compute 5 signals
    cos_vals, norm_scores, loss_scores, krum_scores, val_scores = compute_5_signals(
        deltas, loss_imps, val_scores, attack_ratio
    )
    
    # Weighted combination
    w_cos, w_norm, w_loss, w_krum, w_val = 0.20, 0.15, 0.10, 0.25, 0.30
    scores = [
        w_cos*c + w_norm*n + w_loss*l + w_krum*k + w_val*v
        for c,n,l,k,v in zip(cos_vals, norm_scores, loss_scores, krum_scores, val_scores)
    ]
    
    if attack_ratio < 0.3:
        # Low ratio: Multi-Krum selection
        n = len(client_ids)
        n_select = n - max(1, int(n * attack_ratio))
        selected = np.argsort(krum_scores)[::-1][:n_select]
        
        kept_ids = [client_ids[i] for i in selected]
        kept_weights = [base_weights[i] for i in selected]
        
        # Simple trust update
        new_trust = dict(trust)
        for i, cid in enumerate(client_ids):
            if i in selected:
                new_trust[cid] = min(1.0, new_trust.get(cid, 0.0) + 0.1)
            else:
                new_trust[cid] = max(0.0, new_trust.get(cid, 0.0) - 0.2)
    else:
        # High ratio: Trust-based routing
        # Compute risk scores
        risks = [1.0 - s for s in scores]  # Invert: low score = high risk
        
        # Adaptive thresholds
        tau_low = float(np.percentile(risks, 30))
        tau_high = float(np.percentile(risks, 60))
        
        # Route clients
        trusted, downweighted, fallback = [], [], []
        for i, cid in enumerate(client_ids):
            if risks[i] < tau_low:
                trusted.append(i)
            elif risks[i] < tau_high:
                downweighted.append(i)
            else:
                fallback.append(i)
        
        # Assign weights
        kept_ids = []
        kept_weights = []
        for i, cid in enumerate(client_ids):
            if i in trusted:
                kept_ids.append(cid)
                kept_weights.append(base_weights[i])
            elif i in downweighted:
                kept_ids.append(cid)
                kept_weights.append(base_weights[i] * 0.3)
        
        # Trust update
        new_trust = {}
        for i, cid in enumerate(client_ids):
            prev = trust.get(cid, 0.0)
            new_trust[cid] = 0.7 * prev + 0.3 * scores[i]
            new_trust[cid] = float(np.clip(new_trust[cid], 0.0, 1.0))
    
    return kept_ids, kept_weights, new_trust
