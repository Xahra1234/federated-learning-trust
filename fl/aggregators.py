from __future__ import annotations
import torch
import numpy as np
from typing import List, Tuple

def fedavg_apply(global_model, deltas: List[torch.Tensor], weights: List[float]):
    total = float(sum(weights))
    if total <= 0:
        return
    agg = torch.zeros_like(deltas[0])
    for d, w in zip(deltas, weights):
        agg.add_(d, alpha=float(w)/total)
    idx = 0
    for p in global_model.parameters():
        n = p.numel()
        p.data.add_(agg[idx:idx+n].view_as(p))
        idx += n

def krum_select(deltas: List[torch.Tensor], n_attackers: int = None, multi_krum: bool = False) -> Tuple[List[int], List[float]]:
    """
    Krum/Multi-Krum robust aggregation (Blanchard et al., 2017)
    Selects clients based on distance to neighbors.
    
    Args:
        deltas: Client updates
        n_attackers: Expected number of Byzantine clients (auto-adjusted if None)
        multi_krum: If True, select multiple clients; if False, select only 1
    
    Returns:
        Selected client indices and their weights
    """
    n = len(deltas)
    if n_attackers is None:
        n_attackers = max(1, int(n * 0.25))  # Auto-adjust: 25% tolerance
    if n <= 2 * n_attackers + 2:
        # Not enough clients for Krum, fall back to all
        return list(range(n)), [1.0] * n
    
    # Compute pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(deltas[i] - deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, compute score = sum of distances to n-f-2 closest neighbors
    n_closest = n - n_attackers - 2
    scores = []
    for i in range(n):
        dists_i = distances[i].numpy()
        closest_dists = np.partition(dists_i, n_closest)[:n_closest]
        scores.append(float(np.sum(closest_dists)))
    
    if multi_krum:
        # Multi-Krum: select n-f clients with lowest scores
        n_select = n - n_attackers
        selected_indices = np.argsort(scores)[:n_select].tolist()
        return selected_indices, [1.0] * len(selected_indices)
    else:
        # Krum: select single client with lowest score
        best_idx = int(np.argmin(scores))
        return [best_idx], [1.0]

def trimmed_mean_apply(global_model, deltas: List[torch.Tensor], weights: List[float], trim_ratio: float = 0.1):
    """
    Trimmed Mean aggregation (Yin et al., 2018)
    Removes top and bottom trim_ratio of values for each parameter.
    
    Args:
        global_model: Global model to update
        deltas: Client updates
        weights: Client weights (not used in trimmed mean)
        trim_ratio: Fraction to trim from each end (default 0.1 = 10%)
    """
    n = len(deltas)
    if n == 0:
        return
    
    # Stack all deltas
    stacked = torch.stack(deltas)  # Shape: (n_clients, n_params)
    
    # Compute trimmed mean along client dimension
    n_trim = int(n * trim_ratio)
    if n_trim > 0:
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[n_trim:n-n_trim]
        agg = torch.mean(trimmed, dim=0)
    else:
        agg = torch.mean(stacked, dim=0)
    
    # Apply to global model
    idx = 0
    for p in global_model.parameters():
        n_params = p.numel()
        p.data.add_(agg[idx:idx+n_params].view_as(p))
        idx += n_params

def fltrust_apply(global_model, deltas: List[torch.Tensor], weights: List[float], 
                  server_delta: torch.Tensor, clip_threshold: float = 1.0):
    """
    FLTrust aggregation (Cao et al., 2021)
    Uses server's update on clean data as reference to weight client updates.
    
    Args:
        global_model: Global model to update
        deltas: Client updates
        weights: Client weights
        server_delta: Server's update on clean validation data
        clip_threshold: ReLU clipping threshold for trust scores
    """
    if len(deltas) == 0:
        return
    
    # Compute trust scores based on cosine similarity with server update
    trust_scores = []
    server_norm = torch.norm(server_delta)
    
    for delta in deltas:
        # Cosine similarity
        cos_sim = torch.dot(delta, server_delta) / (torch.norm(delta) * server_norm + 1e-10)
        # ReLU clipping
        trust_score = max(0.0, float(cos_sim.item()))
        trust_scores.append(trust_score)
    
    # Normalize trust scores
    total_trust = sum(trust_scores)
    if total_trust <= 0:
        # Fallback to uniform weights
        trust_scores = [1.0] * len(deltas)
        total_trust = float(len(deltas))
    
    # Weighted aggregation using trust scores
    agg = torch.zeros_like(deltas[0])
    for delta, trust in zip(deltas, trust_scores):
        agg.add_(delta, alpha=trust / total_trust)
    
    # Apply to global model
    idx = 0
    for p in global_model.parameters():
        n_params = p.numel()
        p.data.add_(agg[idx:idx+n_params].view_as(p))
        idx += n_params

def scaffold_apply(global_model, deltas: List[torch.Tensor], weights: List[float], 
                   client_controls: List[torch.Tensor], server_control: torch.Tensor,
                   lr: float = 1.0):
    """
    SCAFFOLD aggregation (Karimireddy et al., 2020)
    Uses control variates to reduce client drift.
    
    Note: This is a simplified version. Full SCAFFOLD requires maintaining
    control variates across rounds.
    """
    total = float(sum(weights))
    if total <= 0:
        return
    
    # Aggregate deltas with control variate correction
    agg = torch.zeros_like(deltas[0])
    for d, w, c in zip(deltas, weights, client_controls):
        corrected = d + c - server_control
        agg.add_(corrected, alpha=float(w)/total)
    
    # Apply to global model
    idx = 0
    for p in global_model.parameters():
        n = p.numel()
        p.data.add_(agg[idx:idx+n].view_as(p), alpha=lr)
        idx += n
