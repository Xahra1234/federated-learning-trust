# 🚀 CIFAR-100 Parallel GPU - User Guide

## For First-Time Users

### Option 1: Interactive Setup (Recommended)
```bash
tar -xzf cifar100_parallel_gpu.tar.gz
cd cifar100_parallel_gpu
bash setup_and_run.sh
```
**The script will guide you through everything step-by-step!**

### Option 2: Automated (One Command)
```bash
tar -xzf cifar100_parallel_gpu.tar.gz
cd cifar100_parallel_gpu
bash run_all.sh 4  # Use 4 GPUs
```

### Option 3: Manual Control
```bash
# Extract
tar -xzf cifar100_parallel_gpu.tar.gz
cd cifar100_parallel_gpu

# Install dependencies
pip install torch torchvision numpy pandas

# Prepare dataset
python prepare_cifar100.py

# Run experiments (specify number of GPUs)
python run_parallel_gpu.py --num_gpus 4

# Combine results
python combine_results.py
```

---

## What This Does

**Runs 40 federated learning experiments:**
- 5 defense methods (fedavg, krum, multi_krum, multisignal_trust, multikrum_val)
- 2 data distributions (alpha=0.5, 1.0)
- 2 attack ratios (0.2, 0.4)
- 2 attack types (label_flip, sign_flip)
- 30 training rounds each

**Time estimates:**
- 8 GPUs: 15-25 minutes
- 4 GPUs: 30-50 minutes
- 2 GPUs: 1-1.75 hours
- 1 GPU: 2-3.5 hours

---

## Results Location

After completion, find results in:
```
results_v2/cifar100/
├── all_results.csv      # All 40 experiments
├── summary.csv          # Aggregated statistics
├── *.json              # Individual results
└── logs/*.log          # Execution logs
```

---

## Monitor Progress

```bash
# Count completed experiments (out of 40)
ls results_v2/cifar100/*.json 2>/dev/null | wc -l

# Watch in real-time
watch -n 5 "ls results_v2/cifar100/*.json 2>/dev/null | wc -l"

# View specific log
tail -f results_v2/cifar100/logs/fedavg_a0.5_r0.2_label_flip.log
```

---

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable (optional but recommended)
- **Disk**: 5 GB free space
- **Memory**: 2-4 GB per GPU

---

## Troubleshooting

**GPU out of memory?**
```bash
# Edit run_single_experiment.py, line ~15
# Change: --batch 128
# To:     --batch 64
```

**No GPUs detected?**
```bash
# Check GPU availability
nvidia-smi

# If no GPUs, experiments will run on CPU (very slow)
```

**Dependencies missing?**
```bash
pip install torch torchvision numpy pandas
```

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Interactive setup | `bash setup_and_run.sh` |
| Auto run (4 GPUs) | `bash run_all.sh 4` |
| Prepare dataset | `python prepare_cifar100.py` |
| Run experiments | `python run_parallel_gpu.py --num_gpus 4` |
| Combine results | `python combine_results.py` |
| Check progress | `ls results_v2/cifar100/*.json \| wc -l` |

---

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review logs in `results_v2/cifar100/logs/`
3. Ensure all dependencies are installed

---

**That's it! The interactive script (`setup_and_run.sh`) handles everything automatically.**
