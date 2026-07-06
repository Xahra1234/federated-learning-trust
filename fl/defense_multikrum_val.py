"""
Multi-Krum + Validation Damage Defense

Expected Performance:
- 20% attackers: 0.87-0.88 (strong robustness)
- 40% attackers: 0.85-0.87 (major improvement from 0.73)
- 80% attackers: 0.10-0.30 (expected failure - adversarial majority)

Combines:
1. Multi-Krum geometric filtering (catches obvious poisoning)
2. Validation damage score (catches subtle label-flip attacks)
"""
from typing import List, Dict, Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F

def compute_krum_scores(deltas: List[torch.Tensor], attack_ratio: float) -> List[float]:
    """Compute Krum distance scores (lower distance = higher score)"""
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

def compute_validation_scores(
    deltas: List[torch.Tensor],
    global_model: torch.nn.Module,
    val_loader,
    device: str,
    max_batches: int = 2
) -> List[float]:
    """Compute validation damage: how much each update hurts validation loss"""
    baseline_loss = evaluate_loss(global_model, val_loader, device, max_batches)
    
    val_scores = []
    for delta in deltas:
        test_model = copy.deepcopy(global_model)
        apply_delta(test_model, delta)
        updated_loss = evaluate_loss(test_model, val_loader, device, max_batches)
        
        # Score: 1.0 if loss improves, 0.0 if loss increases
        damage = updated_loss - baseline_loss
        score = 1.0 / (1.0 + max(0.0, damage))
        val_scores.append(score)
    
    return val_scores

def evaluate_loss(model: torch.nn.Module, data_loader, device: str, max_batches: int) -> float:
    """Compute average validation loss"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0.0

def apply_delta(model: torch.nn.Module, delta: torch.Tensor) -> None:
    """Apply flattened delta to model parameters"""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(delta[idx:idx + n].view_as(p))
            idx += n

def select_clients_multikrum_val(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    global_model: torch.nn.Module,
    val_loader,
    device: str,
    attack_ratio: float = 0.2
) -> Tuple[List[int], List[float]]:
    """
    Multi-Krum + Validation Damage Defense
    
    Step 1: Multi-Krum geometric filtering
    Step 2: Validation damage scoring
    Step 3: Combined score-based weighting
    """
    # Step 1: Multi-Krum scores
    krum_scores = compute_krum_scores(deltas, attack_ratio)
    
    # Step 2: Validation damage scores
    val_scores = compute_validation_scores(deltas, global_model, val_loader, device)
    
    # Step 3: Combined scoring (60% Krum + 40% validation)
    combined_scores = [0.6 * k + 0.4 * v for k, v in zip(krum_scores, val_scores)]
    
    # Step 4: Score-based weighting
    kept_ids = []
    kept_weights = []
    for i, cid in enumerate(client_ids):
        score = combined_scores[i]
        
        if score >= 0.70:
            weight = base_weights[i]
        elif score >= 0.45:
            weight = base_weights[i] * 0.4
        else:
            weight = 0.0
        
        if weight > 0:
            kept_ids.append(cid)
            kept_weights.append(weight)
    
    return kept_ids, kept_weights
