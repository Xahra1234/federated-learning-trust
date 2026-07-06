"""
Temporal Encoder Defense for Non-IID Federated Learning

Key Innovation: Uses LSTM/Transformer to learn temporal patterns of client behavior,
distinguishing between:
- Non-IID heterogeneity (legitimate data distribution differences)
- Malicious behavior (adversarial poisoning patterns)

Addresses current defense failure: Static features can't distinguish non-IID from attacks.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TemporalConfig:
    """Temporal encoder hyperparameters"""
    history_window: int = 10  # Rounds of history to track
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    anomaly_threshold: float = 0.7  # Anomaly score threshold

class TemporalEncoder(nn.Module):
    """LSTM-based encoder to learn normal client behavior patterns"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)  # Anomaly score
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - temporal sequence of features
        Returns:
            anomaly_score: (batch,) - higher = more anomalous
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Use last timestep
        score = torch.sigmoid(self.fc(last_hidden)).squeeze(-1)
        return score

class TemporalDefense:
    """
    Temporal defense system that learns to distinguish:
    1. Non-IID patterns (consistent over time, legitimate)
    2. Attack patterns (sudden deviations, malicious)
    """
    
    def __init__(self, cfg: TemporalConfig, device='cpu'):
        self.cfg = cfg
        self.device = device
        
        # Client behavior history: {client_id: [(round, features), ...]}
        self.history: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        
        # Temporal encoder (initialized on first use)
        self.encoder = None
        self.optimizer = None
        
        # Current round
        self.current_round = 0
        
    def extract_features(self, delta: torch.Tensor, loss_imp: float, 
                        base_weight: float) -> np.ndarray:
        """Extract per-round features for temporal encoding"""
        with torch.no_grad():
            norm = float(torch.norm(delta).item())
            mean = float(torch.mean(delta).item())
            std = float(torch.std(delta).item())
            
        return np.array([norm, mean, std, loss_imp, base_weight], dtype=np.float32)
    
    def update_history(self, client_id: int, features: np.ndarray):
        """Add current round features to client history"""
        if client_id not in self.history:
            self.history[client_id] = []
        
        self.history[client_id].append((self.current_round, features))
        
        # Keep only recent history
        if len(self.history[client_id]) > self.cfg.history_window:
            self.history[client_id] = self.history[client_id][-self.cfg.history_window:]
    
    def get_temporal_sequence(self, client_id: int) -> torch.Tensor:
        """Get temporal sequence for client (padded if needed)"""
        if client_id not in self.history or len(self.history[client_id]) == 0:
            # No history - return zeros
            return torch.zeros(1, self.cfg.history_window, 5, device=self.device)
        
        hist = self.history[client_id]
        features = [f for _, f in hist]
        
        # Pad if insufficient history
        while len(features) < self.cfg.history_window:
            features.insert(0, np.zeros(5, dtype=np.float32))
        
        # Take last window_size entries
        features = features[-self.cfg.history_window:]
        
        seq = torch.tensor(np.stack(features), dtype=torch.float32, device=self.device)
        return seq.unsqueeze(0)  # (1, seq_len, features)
    
    def initialize_encoder(self):
        """Initialize encoder on first use"""
        if self.encoder is None:
            self.encoder = TemporalEncoder(
                input_dim=5,  # [norm, mean, std, loss_imp, weight]
                hidden_dim=self.cfg.hidden_dim,
                num_layers=self.cfg.num_layers,
                dropout=self.cfg.dropout
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.encoder.parameters(), 
                                             lr=self.cfg.learning_rate)
    
    def train_on_benign(self, benign_sequences: List[torch.Tensor]):
        """
        Train encoder to recognize normal (benign) patterns.
        Called periodically with confirmed benign client sequences.
        """
        if len(benign_sequences) == 0:
            return
        
        self.initialize_encoder()
        self.encoder.train()
        
        # Benign clients should have low anomaly scores
        X = torch.cat(benign_sequences, dim=0)  # (batch, seq_len, features)
        target = torch.zeros(X.size(0), device=self.device)  # Benign = 0
        
        self.optimizer.zero_grad()
        pred = self.encoder(X)
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def detect_anomalies(self, client_ids: List[int], 
                        deltas: List[torch.Tensor],
                        loss_improvements: List[float],
                        base_weights: List[float]) -> Tuple[List[bool], List[float]]:
        """
        Detect anomalous clients using temporal patterns.
        
        Returns:
            suspicious: List of boolean flags
            anomaly_scores: List of anomaly scores [0,1]
        """
        self.current_round += 1
        
        # Extract and store features
        for i, cid in enumerate(client_ids):
            features = self.extract_features(deltas[i], loss_improvements[i], base_weights[i])
            self.update_history(cid, features)
        
        # If encoder not trained yet, use heuristic (first few rounds)
        if self.encoder is None or self.current_round < self.cfg.history_window:
            # Fallback: use simple norm-based detection
            norms = [float(torch.norm(d).item()) for d in deltas]
            median = float(np.median(norms))
            mad = float(np.median(np.abs(norms - median)))
            z_scores = [abs(n - median) / (mad + 1e-8) for n in norms]
            anomaly_scores = [min(1.0, z / 3.0) for z in z_scores]  # Normalize to [0,1]
            suspicious = [s > self.cfg.anomaly_threshold for s in anomaly_scores]
            return suspicious, anomaly_scores
        
        # Use trained encoder
        self.encoder.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for cid in client_ids:
                seq = self.get_temporal_sequence(cid)
                score = float(self.encoder(seq).item())
                anomaly_scores.append(score)
        
        suspicious = [s > self.cfg.anomaly_threshold for s in anomaly_scores]
        return suspicious, anomaly_scores
    
    def adaptive_train(self, client_ids: List[int], suspicious: List[bool]):
        """
        Adaptive training: Use non-suspicious clients as benign examples.
        This allows the encoder to learn non-IID patterns as normal.
        """
        benign_ids = [cid for cid, sus in zip(client_ids, suspicious) if not sus]
        
        if len(benign_ids) < 3:  # Need minimum benign samples
            return None
        
        benign_sequences = [self.get_temporal_sequence(cid) for cid in benign_ids]
        return self.train_on_benign(benign_sequences)

def select_clients_temporal(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    loss_improvements: List[float],
    temporal_defense: TemporalDefense,
    trust: Dict[int, float],
    trust_cfg,
) -> Tuple[List[int], List[float], Dict[str, float], Dict[int, float]]:
    """
    Enhanced client selection with temporal encoding.
    
    Key advantage: Learns to distinguish non-IID (consistent temporal pattern)
    from attacks (anomalous temporal pattern).
    """
    # Detect anomalies using temporal patterns
    suspicious, anomaly_scores = temporal_defense.detect_anomalies(
        client_ids, deltas, loss_improvements, base_weights
    )
    
    # Adaptive training on benign clients
    train_loss = temporal_defense.adaptive_train(client_ids, suspicious)
    
    # Trust-modulated decision
    keep = [True] * len(client_ids)
    eff_weights = list(base_weights)
    softened = 0
    dropped = 0
    
    for i, cid in enumerate(client_ids):
        if suspicious[i]:
            t = trust.get(cid, 1.0)
            if t >= trust_cfg.high_trust:
                # High trust: downweight instead of drop
                eff_weights[i] *= trust_cfg.high_trust_downweight
                softened += 1
            else:
                # Low trust: drop
                keep[i] = False
                dropped += 1
    
    # Update trust
    new_trust = dict(trust)
    for i, cid in enumerate(client_ids):
        t = new_trust.get(cid, 1.0)
        # Use anomaly score for fine-grained trust update
        t = t - (trust_cfg.trust_dec * anomaly_scores[i])
        if not suspicious[i]:
            t = t + trust_cfg.trust_inc
        new_trust[cid] = float(min(trust_cfg.trust_max, max(trust_cfg.trust_min, t)))
    
    kept_ids = [cid for cid, k in zip(client_ids, keep) if k]
    kept_weights = [w for w, k in zip(eff_weights, keep) if k]
    
    stats = {
        "kept": float(len(kept_ids)),
        "dropped": float(dropped),
        "softened": float(softened),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_max": float(np.max(anomaly_scores)),
        "train_loss": float(train_loss) if train_loss is not None else 0.0,
    }
    
    return kept_ids, kept_weights, stats, new_trust
