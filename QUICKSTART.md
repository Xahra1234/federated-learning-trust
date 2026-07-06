# CIFAR-100 Parallel GPU - Quick Reference

## 📦 Package Contents
```
cifar100_parallel_gpu/
├── README.md                    # Full documentation
├── prepare_cifar100.py          # Dataset preparation
├── run_parallel_gpu.py          # Parallel execution manager
├── run_single_experiment.py     # Single experiment runner
├── combine_results.py           # Results aggregator
├── run_all.sh                   # Complete workflow launcher
├── requirements.txt             # Dependencies
├── fl/                          # FL defense implementations
└── models/                      # Model architectures
```

## ⚡ One-Command Execution

```bash
# Extract and run everything (using 4 GPUs)
tar -xzf cifar100_parallel_gpu.tar.gz
cd cifar100_parallel_gpu
bash run_all.sh 4
```

## 🎯 Manual Step-by-Step

```bash
# 1. Install dependencies
pip install torch torchvision numpy pandas

# 2. Prepare dataset
python prepare_cifar100.py

# 3. Run experiments (4 GPUs)
python run_parallel_gpu.py --num_gpus 4

# 4. Combine results
python combine_results.py
```

## 📊 Experiment Matrix

- **40 total experiments**
- 5 modes × 2 alphas × 2 ratios × 2 attacks
- 30 rounds per experiment
- Excludes ratio=0.8

## ⏱️ Time Estimates

| GPUs | Total Time |
|------|------------|
| 1    | 2-3.5 hrs  |
| 2    | 1-1.75 hrs |
| 4    | 30-50 min  |
| 8    | 15-25 min  |

## 📁 Results Location

```
results_v2/cifar100/
├── all_results.csv              # All experiments
├── summary.csv                  # Aggregated stats
├── *.json                       # Individual results
└── logs/*.log                   # Execution logs
```

## 🔍 Monitor Progress

```bash
# Count completed
ls results_v2/cifar100/*.json | wc -l

# Watch live
watch -n 5 "ls results_v2/cifar100/*.json | wc -l"
```

## 🎓 Defense Methods Tested

1. **fedavg** - Baseline (no defense)
2. **krum** - Single Krum
3. **multi_krum** - Multi-Krum
4. **multisignal_trust** - Multi-signal trust
5. **multikrum_val** - Multi-Krum + validation

## 💾 Package Size

- **Compressed**: 101 KB
- **Extracted**: ~500 KB
- **With dataset**: ~200 MB
- **With results**: ~210 MB
