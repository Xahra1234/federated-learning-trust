"""
PEBT-LF: Pre-trained Embedding-Based Trust for Label-Flipping Attacks

Expected Performance:
- 20% attackers: Strong robustness
- 40% attackers: Acceptable robustness
- 80% attackers: Stress-test/adversarial-majority limitation (expected failure)

Note: No defense can work when attackers are the majority.

Optimized for label-flip poisoning with:
1. Validation-damage score (primary signal)
2. Median-centered update similarity
3. Temporal trust with persistent attacker memory
4. Known attack ratio for adaptive thresholding
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PEBTConfig:
    """
    PEBT configuration optimized for label-flip attacks.
    Simplified to 3 core signals.
    """
    # Composite trust weights (must sum to 1.0)
    w_val: float = 0.50      # Validation damage (primary)
    w_update: float = 0.30   # Median-centered similarity
    w_temp: float = 0.20     # Temporal trust

    # Score-based weighting thresholds
    high_trust: float = 0.70
    medium_trust: float = 0.45
    medium_weight: float = 0.40

    # Temporal tracking
    history_window: int = 5

    # Rejection threshold settings
    reject_threshold: float = 0.35
    min_threshold: float = 0.30
    max_threshold: float = 0.55

    # Validation batches (reduced for speed)
    val_batches: int = 2

    # Safe fallback settings
    fallback_top_frac: float = 0.30
    min_keep_frac: float = 0.20

    eps: float = 1e-8

    def __post_init__(self) -> None:
        total = self.w_val + self.w_update + self.w_temp
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"PEBT weights must sum to 1.0, but got {total:.6f}."
            )


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _apply_flat_delta_to_model(model: torch.nn.Module, delta: torch.Tensor) -> None:
    """
    Apply flattened client delta to model parameters in-place.
    """
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(delta[idx:idx + n].view_as(p))
            idx += n

    if idx != delta.numel():
        raise ValueError(
            f"Delta size mismatch: consumed {idx} values, but delta has {delta.numel()} values."
        )


def _evaluate_loss(
    model: torch.nn.Module,
    data_loader,
    device: str,
    max_batches: int = 5,
) -> float:
    """
    Compute average validation loss over a limited number of batches.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

    if total_samples == 0:
        raise ValueError("Validation loader produced no samples.")

    return total_loss / total_samples


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Safe cosine similarity clipped into [0, 1].
    Negative similarity is treated as zero trust.
    """
    if torch.norm(a) < 1e-12 or torch.norm(b) < 1e-12:
        return 0.0

    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
    return max(0.0, float(cos.item()))


def _robust_rescale_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Robustly calibrate scores using median/MAD while preserving raw signal.

    Pure robust z-score normalization can destroy absolute meaning.
    This combines:
        - 60% raw score
        - 40% robust relative score
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    raw_scores = np.clip(raw_scores, 0.0, 1.0)

    median = np.median(raw_scores)
    mad = np.median(np.abs(raw_scores - median))

    if mad < 1e-8:
        return raw_scores

    z = (raw_scores - median) / (1.4826 * mad + 1e-8)
    z = np.clip(z, -3.0, 3.0)

    robust_scores = (z + 3.0) / 6.0
    final_scores = 0.60 * raw_scores + 0.40 * robust_scores

    return np.clip(final_scores, 0.0, 1.0)


def compute_adaptive_threshold(
    trust_scores: np.ndarray,
    base_threshold: float = 0.35,
    min_threshold: float = 0.30,
    max_threshold: float = 0.55,
) -> float:
    """
    Compute rejection threshold.

    This avoids very weak thresholds such as 0.15, which allow too many
    poisoned label-flip clients to pass.
    """
    trust_scores = np.asarray(trust_scores, dtype=np.float64)

    q1 = np.percentile(trust_scores, 25)
    q3 = np.percentile(trust_scores, 75)
    iqr = q3 - q1

    adaptive_threshold = max(base_threshold, q1 - 0.5 * iqr)

    return float(np.clip(adaptive_threshold, min_threshold, max_threshold))


# ---------------------------------------------------------------------
# Pretrained encoder
# ---------------------------------------------------------------------

class PretrainedEncoder:
    """
    Extracts anchor embeddings from the current/global/pretrained model.

    This version uses multiple validation batches instead of one batch.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.anchor_embedding: Optional[torch.Tensor] = None

    @torch.no_grad()
    def extract_features(
        self,
        model: torch.nn.Module,
        data_loader,
        max_batches: int = 5,
    ) -> torch.Tensor:
        """
        Extract average penultimate-layer embedding over multiple batches.

        Supports common model structures used in this project:
        - FMNIST_CNN with conv1, conv2, pool, fc1, fc2
        - ResNet-like model with layer1-layer4 and linear
        - MLP/NBIoT-like model with fc1-fc4
        """
        model.eval()
        features = []

        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(self.device)

            # FMNIST_CNN-style architecture.
            if hasattr(model, "fc2") and hasattr(model, "conv1") and hasattr(model, "conv2"):
                x = model.pool(F.relu(model.conv1(x)))
                x = model.pool(F.relu(model.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(model.fc1(x))

            # ResNet-like architecture.
            elif hasattr(model, "linear") and hasattr(model, "layer1"):
                x = F.relu(model.bn1(model.conv1(x)))
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)

            # MLP / NBIoT-like architecture.
            elif hasattr(model, "fc4"):
                x = F.relu(model.fc1(x))
                if hasattr(model, "dropout"):
                    x = model.dropout(x)
                x = F.relu(model.fc2(x))
                if hasattr(model, "dropout"):
                    x = model.dropout(x)
                x = F.relu(model.fc3(x))

            else:
                raise ValueError(
                    "Unknown model architecture. Add penultimate-layer extraction logic "
                    "inside PretrainedEncoder.extract_features()."
                )

            features.append(x.mean(dim=0))

        if len(features) == 0:
            raise ValueError("Validation loader produced no batches for feature extraction.")

        return torch.stack(features).mean(dim=0)

    def set_anchor(
        self,
        model: torch.nn.Module,
        val_loader,
        max_batches: int = 5,
    ) -> None:
        """
        Set anchor embedding from the current clean/global model.
        """
        self.anchor_embedding = self.extract_features(
            model=model,
            data_loader=val_loader,
            max_batches=max_batches,
        )

    def get_anchor(self) -> torch.Tensor:
        """
        Return anchor embedding.
        """
        if self.anchor_embedding is None:
            raise ValueError("Anchor embedding is not set. Call encoder.set_anchor(...) first.")

        return self.anchor_embedding


# ---------------------------------------------------------------------
# Temporal tracker
# ---------------------------------------------------------------------

class TemporalTracker:
    """
    Tracks historical trust per client.

    Important fix:
    Pure consistency can reward clients that are consistently bad.
    This version combines:
        - historical average trust
        - consistency
        - persistent bad-streak penalty
    """

    def __init__(self, history_window: int = 5):
        self.history_window = history_window
        self.client_history: Dict[int, List[float]] = {}
        self.bad_streak: Dict[int, int] = {}

    def update(
        self,
        client_id: int,
        trust_score: float,
        flagged: bool = False,
    ) -> None:
        """
        Update client trust history and bad-streak counter.
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = []

        self.client_history[client_id].append(float(trust_score))

        if len(self.client_history[client_id]) > self.history_window:
            self.client_history[client_id] = self.client_history[client_id][-self.history_window:]

        if flagged:
            self.bad_streak[client_id] = self.bad_streak.get(client_id, 0) + 1
        else:
            self.bad_streak[client_id] = max(0, self.bad_streak.get(client_id, 0) - 1)

    def get_temporal_score(self, client_id: int) -> float:
        """
        Compute temporal trust score.

        New clients get neutral score 0.5.
        Existing clients are scored by average trust plus consistency.
        """
        if client_id not in self.client_history or len(self.client_history[client_id]) < 2:
            return 0.5

        history = np.asarray(self.client_history[client_id], dtype=np.float64)

        avg_trust = float(np.mean(history))
        std_trust = float(np.std(history))
        consistency = 1.0 - min(1.0, std_trust)

        score = 0.70 * avg_trust + 0.30 * consistency
        return float(np.clip(score, 0.0, 1.0))

    # Backward-compatible alias for older calling code.
    def get_consistency_score(self, client_id: int) -> float:
        return self.get_temporal_score(client_id)

    def get_persistent_penalty(self, client_id: int) -> float:
        """
        Stricter penalty for clients repeatedly flagged.
        Permanent ban after 3 strikes.
        """
        streak = self.bad_streak.get(client_id, 0)
        if streak >= 3:
            return 0.0  # Permanent ban
        if streak == 2:
            return 0.40
        if streak == 1:
            return 0.70
        return 1.0


# ---------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------

def compute_validation_score(
    global_model: torch.nn.Module,
    delta: torch.Tensor,
    val_loader,
    device: str,
    baseline_loss: Optional[float] = None,
    max_batches: int = 5,
) -> Tuple[float, float]:
    """
    Compute validation-damage score.

    Higher score = safer update.

    This compares the updated model loss against the baseline global model loss.
    That is stronger for label-flip than scoring absolute updated loss only.

    Returns:
        score:
            Normalized score in [0, 1].
        damage:
            updated_loss - baseline_loss. Positive means harmful update.
    """
    if baseline_loss is None:
        baseline_loss = _evaluate_loss(
            model=global_model,
            data_loader=val_loader,
            device=device,
            max_batches=max_batches,
        )

    temp_model = copy.deepcopy(global_model).to(device)
    temp_model.load_state_dict(global_model.state_dict())

    _apply_flat_delta_to_model(temp_model, delta)

    updated_loss = _evaluate_loss(
        model=temp_model,
        data_loader=val_loader,
        device=device,
        max_batches=max_batches,
    )

    damage = updated_loss - baseline_loss

    if damage <= 0:
        score = 1.0
    else:
        # Stronger penalty because label-flip harms decision boundaries.
        score = 1.0 / (1.0 + 3.0 * damage)

    return float(np.clip(score, 0.0, 1.0)), float(damage)


def compute_update_similarity(
    delta: torch.Tensor,
    deltas: List[torch.Tensor],
) -> float:
    """
    Median-centred update similarity.

    Mean-centred similarity is unsafe because attackers can poison the mean.
    """
    stacked = torch.stack(deltas)
    center_delta = torch.median(stacked, dim=0).values

    return _safe_cosine(delta, center_delta)


def compute_anchor_similarity(
    global_model: torch.nn.Module,
    delta: torch.Tensor,
    encoder: PretrainedEncoder,
    val_loader,
    device: str,
    max_batches: int = 5,
) -> float:
    """
    Compute embedding similarity between updated model and anchor model.

    For label-flip, this checks whether the client's update shifts the
    representation away from the trusted anchor.
    """
    if getattr(encoder, "anchor_embedding", None) is None:
        encoder.set_anchor(global_model, val_loader, max_batches=max_batches)

    temp_model = copy.deepcopy(global_model).to(device)
    temp_model.load_state_dict(global_model.state_dict())

    _apply_flat_delta_to_model(temp_model, delta)

    z_i = encoder.extract_features(
        model=temp_model,
        data_loader=val_loader,
        max_batches=max_batches,
    )
    z_anchor = encoder.get_anchor()

    score = _safe_cosine(z_i, z_anchor)

    # Representation-distance penalty. Kept mild to avoid punishing non-IID benign clients too hard.
    embedding_dist = torch.norm(z_i - z_anchor).item()

    if embedding_dist > 10.0:
        score *= 0.50
    elif embedding_dist > 5.0:
        score *= 0.75

    return float(np.clip(score, 0.0, 1.0))


def compute_anomaly_score(
    delta: torch.Tensor,
    deltas: List[torch.Tensor],
    validation_score: float,
) -> float:
    """
    Compute normality/anomaly score.

    Higher = more normal/safe.

    Uses:
        - magnitude consistency
        - median-centred direction consistency
        - validation behaviour
    """
    stacked = torch.stack(deltas)

    # Magnitude score using median/MAD.
    norms = torch.stack([torch.norm(d) for d in deltas])
    delta_norm = torch.norm(delta)

    median_norm = torch.median(norms)
    mad_norm = torch.median(torch.abs(norms - median_norm))

    z_score = torch.abs((delta_norm - median_norm) / (mad_norm + 1e-6))
    magnitude_score = float(1.0 / (1.0 + z_score.item()))

    # Direction score using median update.
    center_delta = torch.median(stacked, dim=0).values
    direction_score = _safe_cosine(delta, center_delta)

    anomaly_score = (
        0.35 * magnitude_score +
        0.30 * direction_score +
        0.35 * validation_score
    )

    return float(np.clip(anomaly_score, 0.0, 1.0))


# ---------------------------------------------------------------------
# Main PEBT selection function
# ---------------------------------------------------------------------

def pebt_select_clients(
    client_ids: List[int],
    deltas: List[torch.Tensor],
    base_weights: List[float],
    global_model: torch.nn.Module,
    val_loader,
    device: str,
    encoder: PretrainedEncoder,
    temporal_tracker: TemporalTracker,
    cfg: PEBTConfig,
    known_attack_ratio: float = 0.2,  # NEW: Known attack ratio
    malicious_ids: Optional[Set[int]] = None,
) -> Tuple[List[int], List[float], Dict[str, float], List[int]]:
    """
    PEBT-LF client selection and weighting.

    Args:
        client_ids:
            IDs of clients participating in this round.
        deltas:
            Flattened model updates from clients.
        base_weights:
            Original client weights, usually proportional to local data size.
        global_model:
            Current global model before aggregation.
        val_loader:
            Clean validation loader available to the server.
        device:
            'cpu' or 'cuda'.
        encoder:
            PretrainedEncoder instance.
        temporal_tracker:
            TemporalTracker instance.
        cfg:
            PEBTConfig instance.
        malicious_ids:
            Optional set of known malicious client IDs for FP/FN logging only.

    Returns:
        kept_ids:
            Selected client IDs.
        kept_weights:
            Normalized aggregation weights for selected clients.
        stats:
            Logging statistics.
        detected_malicious:
            Client IDs flagged as malicious by the detector.
    """

    if len(client_ids) == 0:
        return [], [], {}, []

    if not (len(client_ids) == len(deltas) == len(base_weights)):
        raise ValueError("client_ids, deltas, and base_weights must have the same length.")

    n_clients = len(client_ids)

    # Baseline validation loss once per round
    baseline_loss = _evaluate_loss(
        model=global_model,
        data_loader=val_loader,
        device=device,
        max_batches=cfg.val_batches,
    )

    # -------------------------------------------------------------
    # Step 1: Validation-damage scores
    # -------------------------------------------------------------
    val_scores: List[float] = []
    val_damages: List[float] = []

    for delta in deltas:
        score, damage = compute_validation_score(
            global_model=global_model,
            delta=delta,
            val_loader=val_loader,
            device=device,
            baseline_loss=baseline_loss,
            max_batches=cfg.val_batches,
        )
        val_scores.append(score)
        val_damages.append(damage)

    # -------------------------------------------------------------
    # Step 2: Median-centered update similarity
    # -------------------------------------------------------------
    update_scores: List[float] = []
    for delta in deltas:
        score = compute_update_similarity(delta, deltas)
        update_scores.append(score)

    # -------------------------------------------------------------
    # Step 3: Temporal trust scores
    # -------------------------------------------------------------
    temp_scores: List[float] = []
    for cid in client_ids:
        score = temporal_tracker.get_temporal_score(cid)
        temp_scores.append(score)

    # -------------------------------------------------------------
    # Step 4: Composite trust score (3 signals only)
    # -------------------------------------------------------------
    raw_trust_scores: List[float] = []
    for i, cid in enumerate(client_ids):
        raw_score = (
            cfg.w_val * val_scores[i] +
            cfg.w_update * update_scores[i] +
            cfg.w_temp * temp_scores[i]
        )
        # Persistent penalty
        raw_score *= temporal_tracker.get_persistent_penalty(cid)
        raw_trust_scores.append(float(np.clip(raw_score, 0.0, 1.0)))

    trust_scores = np.clip(raw_trust_scores, 0.0, 1.0)  # No rescaling

    # -------------------------------------------------------------
    # Step 5: Attack-ratio-aware adaptive threshold
    # -------------------------------------------------------------
    if known_attack_ratio < 0.3:
        # Low ratio: conservative threshold
        adaptive_threshold = max(0.40, cfg.reject_threshold)
    elif known_attack_ratio > 0.6:
        # High ratio: aggressive threshold
        adaptive_threshold = min(0.30, cfg.reject_threshold)
    else:
        # Medium ratio: adaptive
        adaptive_threshold = compute_adaptive_threshold(
            trust_scores=trust_scores,
            base_threshold=cfg.reject_threshold,
            min_threshold=cfg.min_threshold,
            max_threshold=cfg.max_threshold,
        )

    detected_malicious: List[int] = []

    for i, cid in enumerate(client_ids):
        if trust_scores[i] < adaptive_threshold:
            detected_malicious.append(cid)

    # -------------------------------------------------------------
    # Step 6: Simple score-based weighting
    # -------------------------------------------------------------
    aggregation_weights = [0.0 for _ in range(n_clients)]
    for i in range(n_clients):
        score = float(trust_scores[i])
        if score >= cfg.high_trust:
            aggregation_weights[i] = base_weights[i]
        elif score >= cfg.medium_trust:
            aggregation_weights[i] = base_weights[i] * cfg.medium_weight
        else:
            aggregation_weights[i] = 0.0

    # -------------------------------------------------------------
    # Step 7: Safe fallback
    # -------------------------------------------------------------
    kept_indices = [i for i, w in enumerate(aggregation_weights) if w > 0.0]

    min_keep = max(1, int(cfg.min_keep_frac * n_clients))

    # If too few clients remain, keep only the top-scoring clients.
    if len(kept_indices) < min_keep:
        top_k = max(min_keep, int(cfg.fallback_top_frac * n_clients))
        top_k = min(top_k, n_clients)

        kept_indices = list(np.argsort(trust_scores)[-top_k:])

        aggregation_weights = [0.0 for _ in range(n_clients)]
        for i in kept_indices:
            aggregation_weights[i] = base_weights[i]

    # Normalize selected weights.
    total_weight = sum(aggregation_weights)

    if total_weight <= cfg.eps:
        # Last-resort fallback: top 1 client only.
        best_idx = int(np.argmax(trust_scores))
        kept_indices = [best_idx]
        aggregation_weights = [0.0 for _ in range(n_clients)]
        aggregation_weights[best_idx] = 1.0
    else:
        aggregation_weights = [w / total_weight for w in aggregation_weights]

    kept_ids = [client_ids[i] for i in range(n_clients) if aggregation_weights[i] > 0.0]
    kept_weights = [aggregation_weights[i] for i in range(n_clients) if aggregation_weights[i] > 0.0]

    # -------------------------------------------------------------
    # Step 8: Update temporal tracker
    # -------------------------------------------------------------
    detected_set = set(detected_malicious)

    for i, cid in enumerate(client_ids):
        flagged = cid in detected_set
        temporal_tracker.update(
            client_id=cid,
            trust_score=float(trust_scores[i]),
            flagged=flagged,
        )

    # -------------------------------------------------------------
    # Step 9: FP/FN stats
    # -------------------------------------------------------------
    fp = 0
    fn = 0

    if malicious_ids is not None:
        malicious_ids = set(malicious_ids)
        detected_set = set(detected_malicious)

        fp = len([cid for cid in detected_set if cid not in malicious_ids])
        fn = len([cid for cid in malicious_ids if cid not in detected_set])

    # -------------------------------------------------------------
    # Step 10: Stats
    # -------------------------------------------------------------
    stats = {
        "kept": float(len(kept_ids)),
        "dropped": float(n_clients - len(kept_ids)),
        "softened": float(sum(1 for w in kept_weights if 0 < w < base_weights[0])),
        "baseline_loss": float(baseline_loss),
        "mean_trust": float(np.mean(trust_scores)),
        "min_trust": float(np.min(trust_scores)),
        "max_trust": float(np.max(trust_scores)),
        "adaptive_threshold": float(adaptive_threshold),
        "mean_val_score": float(np.mean(val_scores)),
        "mean_val_damage": float(np.mean(val_damages)),
        "max_val_damage": float(np.max(val_damages)),
        "mean_update_score": float(np.mean(update_scores)),
        "mean_temp_score": float(np.mean(temp_scores)),
        "FP": float(fp),
        "FN": float(fn),
    }

    return kept_ids, kept_weights, stats, detected_malicious
