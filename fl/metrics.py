"""
Proper attack success metrics aligned with threat model.
Addresses Reviewer Comments #1 and #2.
"""
import numpy as np
from typing import Dict

def compute_attack_metrics(
    test_acc_clean: float,
    test_acc_attacked: float,
    backdoor_acc: float = None,
    threat_model: str = "untargeted"
) -> Dict[str, float]:
    """
    Compute attack success rate based on threat model.
    
    Args:
        test_acc_clean: Accuracy on clean test set without attack
        test_acc_attacked: Accuracy on clean test set under attack
        backdoor_acc: Accuracy on backdoor test set (for targeted attacks)
        threat_model: 'untargeted' (sign_flip, label_flip) or 'targeted' (backdoor)
    
    Returns:
        Dictionary with attack success metrics
    """
    metrics = {}
    
    if threat_model == "untargeted":
        # For untargeted poisoning: ASR = accuracy degradation
        acc_drop = test_acc_clean - test_acc_attacked
        asr = max(0.0, acc_drop / max(1e-9, test_acc_clean))  # Normalized drop
        metrics["attack_success_rate"] = float(asr)
        metrics["accuracy_degradation"] = float(acc_drop)
        metrics["relative_accuracy"] = float(test_acc_attacked / max(1e-9, test_acc_clean))
        
    elif threat_model == "targeted":
        # For targeted backdoor: ASR = backdoor accuracy
        if backdoor_acc is not None:
            metrics["attack_success_rate"] = float(backdoor_acc)
            metrics["backdoor_accuracy"] = float(backdoor_acc)
        metrics["main_task_accuracy"] = float(test_acc_attacked)
    
    return metrics

def compute_defense_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Compute detection metrics for malicious client identification.
    Addresses Reviewer Comment #1 - separates detection from attack success.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False negative rate (previously misnamed as ASR)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "false_negative_rate": float(fnr),
        "specificity": float(specificity),
    }
