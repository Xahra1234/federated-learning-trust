"""
Enhanced MultiSignal Trust with Multi-Krum Pre-filtering

Combines the best of both approaches:
1. Multi-Krum geometric selection (pre-filter outliers)
2. MultiSignal Trust adaptive routing (fine-grained trust management)
"""

def multisignal_trust_with_krum_prefilter(
    client_ids, deltas, base_weights, loss_improvements,
    trust, trust_cfg, score_cfg, client_history, attack_ratio_estimate=0.2
):
    """
    Two-stage defense:
    
    Stage 1: Multi-Krum Pre-filter
    - Use geometric distance to exclude extreme outliers
    - Select n-f clients with lowest Krum scores (like Multi-Krum)
    - This catches obvious attackers at low ratios
    
    Stage 2: MultiSignal Trust Routing
    - Apply full 4-signal scoring on pre-filtered clients
    - Use adaptive routing (trusted/downweighted/fallback)
    - This handles subtle attacks and high ratios
    """
    
    # STAGE 1: Multi-Krum Pre-filter
    n = len(client_ids)
    n_attackers = max(1, int(n * attack_ratio_estimate))
    
    # Compute Krum scores for all clients
    krum_scores_raw = compute_krum_scores(deltas, n_attackers)
    
    # Select n-f clients with lowest Krum distances (most trustworthy)
    n_select = n - n_attackers
    krum_indices = np.argsort(krum_scores_raw)[::-1][:n_select]  # Highest scores = lowest distances
    
    # Pre-filtered clients
    prefiltered_ids = [client_ids[i] for i in krum_indices]
    prefiltered_deltas = [deltas[i] for i in krum_indices]
    prefiltered_weights = [base_weights[i] for i in krum_indices]
    prefiltered_loss = [loss_improvements[i] for i in krum_indices]
    
    # STAGE 2: MultiSignal Trust on pre-filtered clients
    # Compute all 4 signals
    cos_vals, norm_scores, loss_scores, krum_scores = compute_features(
        prefiltered_deltas, prefiltered_loss
    )
    
    # Reliability scores with adaptive weights
    reliability_scores = multisignal_scores(
        cos_vals, norm_scores, loss_scores, krum_scores, 
        score_cfg, attack_ratio_estimate
    )
    
    # Temporal trust update
    new_trust = {}
    for i, cid in enumerate(prefiltered_ids):
        prev_trust = trust.get(cid, trust_cfg.trust_init)
        current_reliability = reliability_scores[i]
        
        # Exponential smoothing
        new_trust[cid] = trust_cfg.lambda_smooth * prev_trust + \
                        (1 - trust_cfg.lambda_smooth) * current_reliability
        new_trust[cid] = float(max(0.0, min(1.0, new_trust[cid])))
    
    # Risk computation and adaptive routing
    risk_scores = {}
    for i, cid in enumerate(prefiltered_ids):
        deviation = compute_geometric_deviation(prefiltered_deltas, i)
        history = client_history.get(cid, [reliability_scores[i]])
        variance = float(np.std(history)) if len(history) > 1 else 0.0
        
        risk_scores[cid] = compute_risk_score(
            new_trust[cid], deviation, variance,
            trust_cfg.beta1_trust, trust_cfg.beta2_deviation, trust_cfg.beta3_variance
        )
    
    # Adaptive thresholds and routing
    tau_low, tau_high = compute_adaptive_thresholds(
        list(risk_scores.values()), attack_ratio_estimate
    )
    trusted, downweighted, fallback = route_clients_raha(
        prefiltered_ids, risk_scores, tau_low, tau_high
    )
    
    # Final weights
    final_weights = []
    for i, cid in enumerate(prefiltered_ids):
        if cid in trusted:
            final_weights.append(prefiltered_weights[i])
        elif cid in downweighted:
            final_weights.append(prefiltered_weights[i] * 0.3)
        else:  # fallback
            final_weights.append(0.0)
    
    return prefiltered_ids, final_weights, new_trust


# Key advantages:
# 1. At low ratios (0.2): Multi-Krum pre-filter catches geometric outliers
# 2. At high ratios (0.4): MultiSignal Trust handles remaining subtle attacks
# 3. Best of both worlds: geometric robustness + behavioral intelligence
