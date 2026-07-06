#!/usr/bin/env python3
"""
Parallel GPU execution for CIFAR-100 experiments.
Distributes experiments across multiple GPUs.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Experiment configurations
MODES = ["fedavg", "krum", "multi_krum", "multisignal_trust", "multikrum_val"]
ALPHAS = [0.5, 1.0]
RATIOS = [0.2, 0.4]
ATTACKS = ["label_flip", "sign_flip"]
ROUNDS = 30

def generate_experiments():
    """Generate all experiment configurations"""
    experiments = []
    for mode in MODES:
        for alpha in ALPHAS:
            for ratio in RATIOS:
                for attack in ATTACKS:
                    experiments.append({
                        "mode": mode,
                        "alpha": alpha,
                        "ratio": ratio,
                        "attack": attack
                    })
    return experiments

def run_single_experiment(exp, gpu_id, output_dir):
    """Run a single experiment on specified GPU"""
    cmd = [
        "python", "scripts/run_single_experiment.py",
        "--dataset", "cifar100",
        "--mode", exp["mode"],
        "--alpha", str(exp["alpha"]),
        "--ratio", str(exp["ratio"]),
        "--attack", exp["attack"],
        "--rounds", str(ROUNDS),
        "--device", f"cuda:{gpu_id}",
        "--output_dir", output_dir
    ]
    
    log_file = f"{output_dir}/logs/{exp['mode']}_a{exp['alpha']}_r{exp['ratio']}_{exp['attack']}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    return process, log_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--output_dir", default="results_v2/cifar100", help="Output directory")
    args = parser.parse_args()
    
    experiments = generate_experiments()
    total = len(experiments)
    
    print(f"🚀 Starting {total} experiments across {args.num_gpus} GPUs")
    print(f"📊 Configuration: {ROUNDS} rounds, {len(MODES)} modes, {len(ALPHAS)} alphas, {len(RATIOS)} ratios, {len(ATTACKS)} attacks")
    print(f"⏱️  Estimated time: {total * 2 / args.num_gpus:.0f}-{total * 4 / args.num_gpus:.0f} minutes\n")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Track running processes
    running = {}
    completed = 0
    
    # Start initial batch
    for gpu_id in range(min(args.num_gpus, len(experiments))):
        exp = experiments.pop(0)
        process, log_file = run_single_experiment(exp, gpu_id, args.output_dir)
        running[gpu_id] = (process, exp, log_file)
        print(f"[GPU {gpu_id}] Started: {exp['mode']} alpha={exp['alpha']} ratio={exp['ratio']} attack={exp['attack']}")
    
    # Monitor and launch new experiments as GPUs become free
    while running or experiments:
        for gpu_id in list(running.keys()):
            process, exp, log_file = running[gpu_id]
            
            if process.poll() is not None:  # Process finished
                completed += 1
                print(f"[GPU {gpu_id}] ✅ Completed ({completed}/{total}): {exp['mode']} alpha={exp['alpha']} ratio={exp['ratio']} attack={exp['attack']}")
                del running[gpu_id]
                
                # Start next experiment on this GPU
                if experiments:
                    exp = experiments.pop(0)
                    process, log_file = run_single_experiment(exp, gpu_id, args.output_dir)
                    running[gpu_id] = (process, exp, log_file)
                    print(f"[GPU {gpu_id}] Started: {exp['mode']} alpha={exp['alpha']} ratio={exp['ratio']} attack={exp['attack']}")
        
        import time
        time.sleep(5)  # Check every 5 seconds
    
    print(f"\n✅ All {total} experiments completed!")
    print(f"📁 Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
