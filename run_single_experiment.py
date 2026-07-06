#!/usr/bin/env python3
"""Run a single experiment configuration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.model_v2 import build_model
from fl.utils import set_seed, get_delta
from fl.aggregators import fedavg_apply, krum_select
from fl.defense import TrustConfig, ScoreConfig, select_clients
from fl.metrics import compute_defense_metrics
from fl.defense_multikrum_val import select_clients_multikrum_val

def load_npz(dirpath, name):
    z = np.load(os.path.join(dirpath, f"{name}.npz"))
    X = torch.tensor(z["X"], dtype=torch.float32)
    y = torch.tensor(z["y"], dtype=torch.long)
    if X.ndim == 4 and X.shape[1] == 32 and X.shape[2] in [3, 100]:
        X = X.permute(0, 2, 1, 3)
    return X, y

def make_loader(X, y, batch, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

def label_flip(y, num_classes):
    return (y + 1) % num_classes

def train_client(global_model, loader, device, lr, epochs, malicious, attack, num_classes):
    model = build_model(dataset=args.dataset, num_classes=num_classes).to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if malicious and attack == "label_flip":
                y = label_flip(y, num_classes)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    delta = get_delta(global_model, model)
    if malicious and attack == "sign_flip":
        delta = -delta
    return delta.detach()

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    num_classes = 100 if args.dataset == "cifar100" else 10
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    set_seed(args.seed)
    
    # Load data
    data_dir = f"data/{args.dataset}_noniid/alpha_{args.alpha}/seed_{args.seed}"
    X_val, y_val = load_npz(data_dir, "val")
    X_test, y_test = load_npz(data_dir, "test")
    val_loader = make_loader(X_val, y_val, 512, False)
    test_loader = make_loader(X_test, y_test, 512, False)
    
    client_data = []
    for cid in range(args.clients):
        Xc, yc = load_npz(data_dir, f"train_{cid}")
        client_data.append((Xc, yc))
    
    # Determine malicious clients
    n_mal = int(round(args.clients * args.ratio))
    set_seed(args.seed + 12345)
    mal_ids = set(np.random.choice(args.clients, n_mal, replace=False).tolist())
    
    # Train clean baseline
    clean_model = build_model(dataset=args.dataset, num_classes=num_classes).to(device)
    for _ in range(args.rounds):
        deltas, weights = [], []
        for cid in range(args.clients):
            Xc, yc = client_data[cid]
            loader = make_loader(Xc, yc, args.batch, True)
            delta = train_client(clean_model, loader, device, args.lr, args.local_epochs, False, args.attack, num_classes)
            deltas.append(delta)
            weights.append(float(len(yc)))
        fedavg_apply(clean_model, deltas, weights)
    _, clean_acc = eval_model(clean_model, test_loader, device)
    
    # Train with attack
    set_seed(args.seed)
    global_model = build_model(dataset=args.dataset, num_classes=num_classes).to(device)
    trust = {cid: 1.0 for cid in range(args.clients)}
    trust_cfg = TrustConfig()
    score_cfg = ScoreConfig()
    
    TP = FP = FN = TN = 0
    t0 = time.time()
    
    for rnd in range(args.rounds):
        deltas, weights, loss_imps, is_mals = [], [], [], []
        for cid in range(args.clients):
            Xc, yc = client_data[cid]
            loader = make_loader(Xc, yc, args.batch, True)
            is_mal = cid in mal_ids
            delta = train_client(global_model, loader, device, args.lr, args.local_epochs, is_mal, args.attack, num_classes)
            deltas.append(delta)
            weights.append(float(len(yc)))
            loss_imps.append(0.0)
            is_mals.append(is_mal)
        
        client_ids = list(range(args.clients))
        
        if args.mode == "krum":
            n_attackers = n_mal if n_mal > 0 else 1
            kept_ids, _ = krum_select(deltas, n_attackers, False)
        elif args.mode == "multi_krum":
            n_attackers = n_mal if n_mal > 0 else 1
            kept_ids, _ = krum_select(deltas, n_attackers, True)
        elif args.mode == "multikrum_val":
            kept_ids, _ = select_clients_multikrum_val(client_ids, deltas, weights, global_model, val_loader, device, args.ratio)
        elif args.mode in ["multisignal_trust"]:
            kept_ids, _, _, trust = select_clients(args.mode, client_ids, deltas, weights, loss_imps, trust, trust_cfg, score_cfg, {}, None, None, global_model, val_loader, device, None, None, None)
        else:  # fedavg
            kept_ids = client_ids
        
        kept = set(kept_ids)
        for cid, is_mal in enumerate(is_mals):
            dropped = cid not in kept
            if dropped and is_mal: TP += 1
            elif dropped and not is_mal: FP += 1
            elif not dropped and is_mal: FN += 1
            else: TN += 1
        
        kept_deltas = [deltas[i] for i in kept_ids]
        kept_weights = [weights[i] for i in kept_ids]
        if not kept_deltas:
            kept_deltas, kept_weights = deltas, weights
        fedavg_apply(global_model, kept_deltas, kept_weights)
    
    _, test_acc = eval_model(global_model, test_loader, device)
    runtime = time.time() - t0
    
    # Save result
    result = {
        "mode": args.mode, "alpha": args.alpha, "ratio": args.ratio, "attack": args.attack,
        "rounds": args.rounds, "clean_acc": clean_acc, "attacked_acc": test_acc,
        "accuracy_drop": clean_acc - test_acc, "tp": TP, "fp": FP, "fn": FN, "tn": TN,
        "runtime_s": runtime, "device": args.device
    }
    
    output_file = f"{args.output_dir}/{args.mode}_a{args.alpha}_r{args.ratio}_{args.attack}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ {args.mode} alpha={args.alpha} ratio={args.ratio} attack={args.attack} acc={test_acc:.4f} drop={clean_acc-test_acc:.4f}")

if __name__ == "__main__":
    main()
