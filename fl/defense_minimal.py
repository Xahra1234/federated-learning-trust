"""
Minimal Defense: Multi-Krum + Penalty Counter + Score-based Weighting

Expected Performance:
- 20% attackers: Strong robustness (should match clean accuracy)
- 40% attackers: Acceptable robustness (small accuracy drop)
- 80% attackers: Stress-test/adversarial-majority limitation (expected failure)

Note: 80% attack ratio represents adversarial majority - no defense can work
when attackers outnumber honest clients. This is a fundamental limitation.
"""
from typing import List, Dict, Tuple
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
            distances[i, j] = dist
            distances[j, i] = dist
    
    n_closest = n - n_attackers - 2
    scores = []
    for i in range(n):
        dists_i = distances[i].numpy()
        closest_dists = np.partition(dists_i, n_closest)[:n_closest]
        scores.append(float(np.sum(closest_dists)))
    
    scores = np.array(scores)
    if scores.max() > scores.min():
        normalized = 1.0 - (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = np.ones_like(scores)
    
    return normalized.tolist()

def select_clients_minimal(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    bad_count: Dict[int, int],
    attack_ratio: float = 0.2
) -> Tuple[List[int], List[float], Dict[int, int]]:
    """
    Minimal defense with score-based weighting
    
    Args:
        client_ids: List of client IDs
        deltas: Client updates
        base_weights: Client weights
        bad_count: Penalty counter for each client
        attack_ratio: Known attack ratio
    
    Returns:
        kept_ids, kept_weights, updated_bad_count
    """
    # Compute Krum scores
    krum_scores = compute_krum_scores(deltas, attack_ratio)
    
    # Detect malicious: bottom 20% by Krum score
    threshold = np.percentile(krum_scores, 20)
    detected_mal = [i for i, score in enumerate(krum_scores) if score < threshold]
    
    # Update penalty counter
    new_bad_count = dict(bad_count)
    for i, cid in enumerate(client_ids):
        if i in detected_mal:
            new_bad_count[cid] = new_bad_count.get(cid, 0) + 1
        else:
            new_bad_count[cid] = max(0, new_bad_count.get(cid, 0) - 1)
    
    # Apply score-based weighting
    kept_ids = []
    kept_weights = []
    for i, cid in enumerate(client_ids):
        score = krum_scores[i]
        
        # Weight based on score
        if score >= 0.70:
            weight = base_weights[i]
        elif score >= 0.45:
            weight = base_weights[i] * 0.4
        else:
            weight = 0.0
        
        # Exclude if bad_count >= 3
        if new_bad_count[cid] >= 3:
            weight = 0.0
        
        if weight > 0:
            kept_ids.append(cid)
            kept_weights.append(weight)
    
    return kept_ids, kept_weights, new_bad_count
