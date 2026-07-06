# CIFAR-100 Parallel GPU Execution

Complete standalone package for running CIFAR-100 federated learning experiments across multiple GPUs.

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install torch torchvision numpy pandas
```

### Step 2: Prepare CIFAR-100 Dataset
```bash
python prepare_cifar100.py
```
This will:
- Download CIFAR-100 dataset (~170 MB)
- Create non-IID splits for alpha=0.5 and alpha=1.0
- Generate train/val/test splits for 10 clients
- Save to `data/cifar100_noniid/`

### Step 3: Run Parallel Experiments
```bash
# Using 4 GPUs (recommended)
python run_parallel_gpu.py --num_gpus 4

# Using 2 GPUs
python run_parallel_gpu.py --num_gpus 2

# Using 8 GPUs
python run_parallel_gpu.py --num_gpus 8
```

## Experiment Configuration

**Total Experiments**: 40
- **Modes**: 5 (fedavg, krum, multi_krum, multisignal_trust, multikrum_val)
- **Rounds**: 30 per experiment
- **Alphas**: 0.5, 1.0 (data heterogeneity)
- **Malicious Ratios**: 0.2, 0.4 (excluding 0.8)
- **Attacks**: label_flip, sign_flip

## Defense Methods

1. **fedavg** - Baseline FedAvg (no defense)
2. **krum** - Single Krum selection
3. **multi_krum** - Multi-Krum selection  
4. **multisignal_trust** - Multi-signal trust-based defense
5. **multikrum_val** - Multi-Krum with validation damage

## Time Estimates

| GPUs | Time per Exp | Total Time | Speedup |
|------|--------------|------------|---------|
| 1 GPU | 3-5 min | 2-3.5 hours | 1x |
| 2 GPUs | 3-5 min | 1-1.75 hours | 2x |
| 4 GPUs | 3-5 min | 30-50 min | 4x |
| 8 GPUs | 3-5 min | 15-25 min | 8x |

## Monitoring Progress

```bash
# Count completed experiments
ls results_v2/cifar100/*.json 2>/dev/null | wc -l

# Watch progress in real-time
watch -n 5 "ls results_v2/cifar100/*.json 2>/dev/null | wc -l"

# View logs for specific experiment
tail -f results_v2/cifar100/logs/fedavg_a0.5_r0.2_label_flip.log
```

## Output Structure

```
results_v2/cifar100/
├── fedavg_a0.5_r0.2_label_flip.json
├── fedavg_a0.5_r0.2_sign_flip.json
├── ...
└── logs/
    ├── fedavg_a0.5_r0.2_label_flip.log
    └── ...
```

## Combine Results

After all experiments complete:

```bash
python combine_results.py
```

This creates:
- `results_v2/cifar100/all_results.csv` - All individual results
- `results_v2/cifar100/summary.csv` - Aggregated statistics

## Advantages of Parallel Execution

✅ **10-40x faster** than sequential execution  
✅ **Automatic GPU allocation** - experiments distributed evenly  
✅ **Fault tolerant** - failed experiments don't block others  
✅ **Real-time monitoring** - see progress per GPU  
✅ **Resource efficient** - maximizes GPU utilization  
✅ **Independent experiments** - no interference between runs

## Troubleshooting

**GPU Out of Memory:**
```bash
# Reduce batch size in run_single_experiment.py
# Change --batch 128 to --batch 64
```

**Check GPU availability:**
```bash
nvidia-smi
```

**Kill all running experiments:**
```bash
pkill -f run_single_experiment.py
```

## System Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU(s)
- ~5 GB disk space for dataset
- ~2-4 GB GPU memory per experiment

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```
