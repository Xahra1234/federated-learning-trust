# Federated Learning with Multi-Signal Trust Defense

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Robust federated learning framework with multi-signal trust-based defense mechanisms against Byzantine attacks.

## 🎯 Features

- **5 Defense Methods**: FedAvg, Krum, Multi-Krum, Multi-Signal Trust, Multi-Krum+Val
- **Multiple Datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST
- **Parallel GPU Execution**: Distribute experiments across multiple GPUs
- **Byzantine Attacks**: Label-flip and sign-flip attacks
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, trust scores

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/federated-learning-trust.git
cd federated-learning-trust
pip install -r requirements.txt
```

### Run CIFAR-100 Experiments (Parallel GPU)

```bash
cd cifar100_parallel_gpu
bash setup_and_run.sh
```

The interactive script will guide you through:
1. System requirements check
2. Dataset preparation
3. Parallel GPU execution
4. Results aggregation

## 📊 Supported Datasets

| Dataset | Classes | Image Size | Download |
|---------|---------|------------|----------|
| CIFAR-10 | 10 | 32×32×3 | Auto |
| CIFAR-100 | 100 | 32×32×3 | Auto |
| Fashion-MNIST | 10 | 28×28×1 | Auto |

## 🛡️ Defense Methods

1. **FedAvg** - Baseline federated averaging
2. **Krum** - Single Krum selection
3. **Multi-Krum** - Multi-Krum aggregation
4. **Multi-Signal Trust** - Proposed multi-signal trust-based defense
5. **Multi-Krum+Val** - Multi-Krum with validation damage

## 📁 Repository Structure

```
.
├── cifar100_parallel_gpu/     # Parallel GPU execution package
│   ├── setup_and_run.sh       # Interactive setup script
│   ├── prepare_cifar100.py    # Dataset preparation
│   ├── run_parallel_gpu.py    # Parallel execution manager
│   └── fl/                    # Defense implementations
├── fl/                        # Federated learning modules
│   ├── defense.py             # Defense mechanisms
│   ├── aggregators.py         # Aggregation methods
│   └── metrics.py             # Evaluation metrics
├── models/                    # Neural network models
├── scripts/                   # Experiment scripts
└── requirements.txt           # Dependencies

```

## 🔬 Experiment Configuration

**Default Settings:**
- **Clients**: 10
- **Rounds**: 30
- **Local Epochs**: 1
- **Batch Size**: 128
- **Learning Rate**: 0.05
- **Data Heterogeneity (α)**: 0.5, 1.0
- **Malicious Ratios**: 0.2, 0.4
- **Attacks**: label_flip, sign_flip

## 📈 Results

Results are saved in CSV format:
- `results_v2/{dataset}/all_results.csv` - Individual experiment results
- `results_v2/{dataset}/summary.csv` - Aggregated statistics

## ⚡ Performance

**Time Estimates (CIFAR-100, 40 experiments):**

| GPUs | Time |
|------|------|
| 8 GPUs | 15-25 min |
| 4 GPUs | 30-50 min |
| 2 GPUs | 1-1.75 hrs |
| 1 GPU | 2-3.5 hrs |

## 🔧 Advanced Usage

### Manual Execution

```bash
# Prepare dataset
python prepare_cifar100.py

# Run experiments on 4 GPUs
python run_parallel_gpu.py --num_gpus 4

# Combine results
python combine_results.py
```

### Custom Configuration

Edit experiment parameters in `run_parallel_gpu.py`:
```python
MODES = ["fedavg", "krum", "multi_krum", "multisignal_trust", "multikrum_val"]
ALPHAS = [0.5, 1.0]
RATIOS = [0.2, 0.4]
ATTACKS = ["label_flip", "sign_flip"]
ROUNDS = 30
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Multi-Signal Trust-Based Defense for Federated Learning},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- CIFAR dataset creators
- Federated learning research community

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com)

---

**Note**: Dataset files are not included in the repository. They will be automatically downloaded when you run the preparation scripts.
