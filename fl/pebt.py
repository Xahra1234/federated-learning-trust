"""
PEBT: Pre-trained Embedding-Based Trust
Algorithm for federated learning with validation-based trust scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PEBTConfig:
    """PEBT configuration parameters"""
    # Trust score weights (tuned for better detection)
    w_update: float = 0.20      # Weight for update similarity
    w_anchor: float = 0.30      # Weight for pretrained anchor similarity (reduced)
    w_val: float = 0.40         # Weight for validation score (increased - most reliable)
    w_temp: float = 0.10        # Weight for temporal consistency
    
    # Soft aggregation
    temperature: float = 0.5    # Lower temp = stronger down-weighting of low-trust clients
    
    # Temporal tracking
    history_window: int = 5     # Track last K rounds
    
    # Thresholds - only for detection reporting, not rejection
    detection_threshold: float = 0.25  # Below this = flag as malicious for metrics

class PretrainedEncoder:
    """Pretrained encoder for extracting activation embeddings"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.anchor_embedding = None
    
    @torch.no_grad()
    def extract_features(self, model, data_loader):
        """Extract penultimate layer features from model on data"""
        model.eval()
        features = []
        
        # Get a batch from data loader
        for x, y in data_loader:
            x = x.to(self.device)
            
            # Forward pass and extract penultimate layer
            if hasattr(model, 'fc2'):  # FMNIST_CNN
                x = model.pool(F.relu(model.conv1(x)))
                x = model.pool(F.relu(model.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(model.fc1(x))  # Penultimate layer
            elif hasattr(model, 'linear'):  # ResNet
                x = F.relu(model.bn1(model.conv1(x)))
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)  # Penultimate layer
            elif hasattr(model, 'fc4'):  # NBIoT_MLP
                x = F.relu(model.fc1(x))
                x = model.dropout(x)
                x = F.relu(model.fc2(x))
                x = model.dropout(x)
                x = F.relu(model.fc3(x))  # Penultimate layer
            else:
                raise ValueError("Unknown model architecture")
            
            features.append(x.mean(dim=0))  # Average over batch
            break  # Only use first batch
        
        return torch.stack(features).mean(dim=0)  # Return mean feature vector
    
    def set_anchor(self, model, val_loader):
        """Set anchor embedding from pretrained model"""
        self.anchor_embedding = self.extract_features(model, val_loader)
    
    def get_anchor(self):
        """Get anchor embedding"""
        return self.anchor_embedding

class TemporalTracker:
    """Track client trust history over rounds"""
    def __init__(self, history_window: int = 5):
        self.history_window = history_window
        self.client_history: Dict[int, List[float]] = {}
    
    def update(self, client_id: int, trust_score: float):
        """Update client history"""
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        
        self.client_history[client_id].append(trust_score)
        
        # Keep only recent history
        if len(self.client_history[client_id]) > self.history_window:
            self.client_history[client_id] = self.client_history[client_id][-self.history_window:]
    
    def get_consistency_score(self, client_id: int) -> float:
        """Compute temporal consistency score (1 - std)"""
        if client_id not in self.client_history or len(self.client_history[client_id]) < 2:
            return 0.5  # Neutral for new clients
        
        history = self.client_history[client_id]
        std = float(np.std(history))
        
        # Low variance = high consistency
        consistency = 1.0 - min(1.0, std)
        return consistency

def compute_validation_score(global_model, delta, val_loader, device):
    """
    Step 1-2: Apply update temporarily and compute validation loss
    Returns normalized score in [0, 1] where higher = better
    """
    # Create temporary model
    temp_model = type(global_model)().to(device)
    temp_model.load_state_dict(global_model.state_dict())
    
    # Apply delta
    idx = 0
    for p in temp_model.parameters():
        n = p.numel()
        p.data.add_(delta[idx:idx+n].view_as(p))
        idx += n
    
    # Compute validation loss
    temp_model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = temp_model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            if total_samples >= 3 * y.size(0):  # Use 3 batches for stability
                break
    
    avg_loss = total_loss / max(1, total_samples)
    
    # Normalize to [0, 1]: lower loss = higher score
    score = 1.0 / (1.0 + avg_loss)
    return float(score)

def compute_update_similarity(delta, deltas):
    """
    Step 4: Compute cosine similarity with mean update
    Returns score in [0, 1]
    """
    # Compute mean update
    delta_mean = torch.stack(deltas).mean(dim=0)
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(delta.unsqueeze(0), delta_mean.unsqueeze(0), dim=1)
    
    # Clip negative values
    score = max(0.0, float(cos_sim.item()))
    return score

def compute_anchor_similarity(global_model, delta, encoder, val_loader, device):
    """
    Step 3-4: Extract activation embedding and compute similarity with anchor
    Returns score in [0, 1]
    """
    # Create temporary model with update applied
    temp_model = type(global_model)().to(device)
    temp_model.load_state_dict(global_model.state_dict())
    
    # Apply delta
    idx = 0
    for p in temp_model.parameters():
        n = p.numel()
        p.data.add_(delta[idx:idx+n].view_as(p))
        idx += n
    
    # Extract embedding
    z_i = encoder.extract_features(temp_model, val_loader)
    z_anchor = encoder.get_anchor()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(z_i.unsqueeze(0), z_anchor.unsqueeze(0), dim=1)
    
    # Clip negative values
    score = max(0.0, float(cos_sim.item()))
    return score

def pebt_select_clients(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    global_model,
    val_loader,
    device: str,
    encoder: PretrainedEncoder,
    temporal_tracker: TemporalTracker,
    cfg: PEBTConfig,
    malicious_ids: set = None  # For FP/FN tracking
) -> Tuple[List[int], List[float], Dict[str, float], List[int]]:
    """
    PEBT Aggregation Algorithm
    
    Returns:
        kept_ids: List of client IDs to keep
        kept_weights: Soft aggregation weights
        stats: Statistics dictionary
        detected_malicious: List of detected malicious client IDs
    """
    n_clients = len(client_ids)
    
    # Step 1-2: Compute validation scores
    val_scores = []
    for delta in deltas:
        s_val = compute_validation_score(global_model, delta, val_loader, device)
        val_scores.append(s_val)
    
    # Step 2: Compute update similarity scores
    update_scores = []
    for delta in deltas:
        s_update = compute_update_similarity(delta, deltas)
        update_scores.append(s_update)
    
    # Step 3: Compute anchor similarity scores
    anchor_scores = []
    for delta in deltas:
        s_anchor = compute_anchor_similarity(global_model, delta, encoder, val_loader, device)
        anchor_scores.append(s_anchor)
    
    # Step 4: Compute temporal consistency scores
    temp_scores = []
    for cid in client_ids:
        s_temp = temporal_tracker.get_consistency_score(cid)
        temp_scores.append(s_temp)
    
    # Step 5: Compute composite trust scores
    trust_scores = []
    for i in range(n_clients):
        trust = (cfg.w_update * update_scores[i] +
                cfg.w_anchor * anchor_scores[i] +
                cfg.w_val * val_scores[i] +
                cfg.w_temp * temp_scores[i])
        trust_scores.append(trust)
    
    # Robust normalization using median and MAD
    trust_scores = np.array(trust_scores)
    median = np.median(trust_scores)
    mad = np.median(np.abs(trust_scores - median))
    if mad > 1e-6:
        trust_scores = (trust_scores - median) / (mad * 1.4826)
        trust_scores = np.clip(trust_scores, -3, 3)  # Clip outliers
        trust_scores = (trust_scores + 3) / 6  # Scale to [0,1]
    else:
        trust_scores = np.ones_like(trust_scores) * 0.5
    
    # Step 6: Convert to soft aggregation weights using softmax
    trust_scores_tensor = torch.tensor(trust_scores, dtype=torch.float32)
    soft_weights = F.softmax(trust_scores_tensor / cfg.temperature, dim=0).numpy()
    
    # Step 7: Apply weights to base weights
    aggregation_weights = []
    for i in range(n_clients):
        aggregation_weights.append(base_weights[i] * soft_weights[i])
    
    # Step 8: Update temporal history and flag malicious (but keep all clients)
    detected_malicious = []
    for i, cid in enumerate(client_ids):
        temporal_tracker.update(cid, trust_scores[i])
        
        # Flag as malicious if trust score is low (for metrics only)
        if trust_scores[i] < cfg.detection_threshold:
            detected_malicious.append(cid)
    
    # Keep ALL clients but use soft weights (no hard rejection)
    kept_ids = client_ids
    kept_weights = aggregation_weights
    
    # Compute statistics
    stats = {
        "kept": len(kept_ids),
        "dropped": n_clients - len(kept_ids),
        "softened": n_clients,  # All clients are soft-weighted
        "mean_trust": float(np.mean(trust_scores)),
        "std_trust": float(np.std(trust_scores)),
        "min_trust": float(np.min(trust_scores)),
        "max_trust": float(np.max(trust_scores)),
        "mean_val_score": float(np.mean(val_scores)),
        "mean_anchor_score": float(np.mean(anchor_scores)),
        "mean_update_score": float(np.mean(update_scores)),
    }
    
    return kept_ids, kept_weights, stats, detected_malicious
