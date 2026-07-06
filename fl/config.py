"""
Centralized configuration for full reproducibility.
Addresses Reviewer Comment #5.
"""
from dataclasses import dataclass, asdict
from typing import List
import json

@dataclass
class ModelConfig:
    """CNN architecture hyperparameters"""
    conv1_out: int = 32
    conv2_out: int = 64
    fc1_hidden: int = 128
    num_classes: int = 10
    dropout: float = 0.0

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    
@dataclass
class TrustConfig:
    """Trust system parameters"""
    trust_init: float = 1.0
    trust_inc: float = 0.05
    trust_dec: float = 0.10
    trust_min: float = 0.0
    trust_max: float = 1.0
    high_trust_threshold: float = 0.70
    high_trust_downweight: float = 0.50

@dataclass
class ScoreConfig:
    """Multi-signal scoring weights"""
    w_cosine: float = 0.25
    w_norm: float = 0.20
    w_loss: float = 0.20
    w_krum: float = 0.35  # Krum distance (highest weight for geometric consistency)
    base_threshold: float = 0.0
    drop_quantile: float = 0.2

@dataclass
class RAHAConfig:
    """RAHA: Risk-Adaptive Hybrid Aggregation parameters"""
    # Signal weights (α_1, α_2, α_3, α_4) - must sum to 1
    alpha_cos: float = 0.25
    alpha_norm: float = 0.20
    alpha_val: float = 0.20
    alpha_geo: float = 0.35
    # Risk combination weights (β_1, β_2, β_3)
    beta_trust: float = 0.5  # Weight for (1 - trust)
    beta_deviation: float = 0.3  # Weight for geometric deviation
    beta_variance: float = 0.2  # Weight for temporal variance
    # Temporal smoothing
    lambda_smooth: float = 0.7  # EMA smoothing factor
    # Adaptive thresholding (multipliers for mean and std)
    tau_low_factor: float = 1.0  # τ_L = mean(risk)
    tau_high_factor: float = 1.0  # τ_H = mean(risk) + std(risk)
    # Fallback aggregation
    fallback_method: str = "krum"  # "krum" or "trimmed_mean"
    # History tracking
    history_window: int = 5  # Track last K rounds for variance

@dataclass
class RAHANATConfig:
    """RAHA-NAT: Non-IID-Aware Adaptive Trust parameters"""
    # Latent representation
    latent_dim: int = 32
    # Clustering
    cluster_method: str = "kmeans"
    n_clusters: int = 3
    # Trust thresholds
    trust_high_threshold: float = 0.7
    trust_low_threshold: float = 0.3
    # Discard policy
    discard_rounds: int = 3
    # History tracking
    history_window: int = 5
    # Risk weights
    w_validation_damage: float = 0.3
    w_cluster_anomaly: float = 0.25
    w_direction_deviation: float = 0.25
    w_temporal_instability: float = 0.2
    # Trust update
    trust_smoothing: float = 0.7
    # Aggregation weights
    trusted_weight: float = 1.0
    uncertain_weight: float = 0.5
    fallback_weight: float = 0.1

@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    trust: TrustConfig
    score: ScoreConfig
    raha: RAHAConfig
    raha_nat: RAHANATConfig
    seeds: List[int]
    dataset: str
    data_alpha: float
    malicious_ratio: float
    attack_type: str
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**data['training']),
            trust=TrustConfig(**data['trust']),
            score=ScoreConfig(**data['score']),
            raha=RAHAConfig(**data.get('raha', {})),
            raha_nat=RAHANATConfig(**data.get('raha_nat', {})),
            seeds=data['seeds'],
            dataset=data['dataset'],
            data_alpha=data['data_alpha'],
            malicious_ratio=data['malicious_ratio'],
            attack_type=data['attack_type']
        )

def get_default_config(dataset: str = "fmnist") -> ExperimentConfig:
    """Get default configuration with all hyperparameters explicitly defined"""
    return ExperimentConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        trust=TrustConfig(),
        score=ScoreConfig(),
        raha=RAHAConfig(),
        raha_nat=RAHANATConfig(),
        seeds=[0, 1, 2, 3, 4],  # 5 seeds for confidence intervals
        dataset=dataset,
        data_alpha=0.5,
        malicious_ratio=0.2,
        attack_type="label_flip"
    )
