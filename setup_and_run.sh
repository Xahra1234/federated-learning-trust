#!/bin/bash
# CIFAR-100 Parallel GPU - Interactive Setup & Run Script
# This script guides users through the complete setup and execution process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    print_msg "$1" "$BLUE"
    echo "=========================================="
    echo ""
}

print_success() {
    print_msg "✅ $1" "$GREEN"
}

print_warning() {
    print_msg "⚠️  $1" "$YELLOW"
}

print_error() {
    print_msg "❌ $1" "$RED"
}

# Welcome message
clear
print_header "CIFAR-100 Federated Learning - Parallel GPU Execution"
echo "This script will guide you through:"
echo "  1. System requirements check"
echo "  2. Dependency installation"
echo "  3. Dataset preparation"
echo "  4. Parallel GPU execution"
echo "  5. Results aggregation"
echo ""
read -p "Press Enter to continue..."

# Step 1: Check system requirements
print_header "Step 1: System Requirements Check"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check CUDA availability
echo "Checking CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_success "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    DEFAULT_GPUS=$GPU_COUNT
else
    print_warning "No CUDA GPUs detected. Will use CPU (very slow)"
    DEFAULT_GPUS=1
fi

# Check disk space
echo "Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -ge 5 ]; then
    print_success "Available disk space: ${AVAILABLE_SPACE}GB"
else
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB (5GB+ recommended)"
fi

echo ""
read -p "Continue with setup? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "Setup cancelled." "$YELLOW"
    exit 0
fi

# Step 2: Install dependencies
print_header "Step 2: Installing Dependencies"

echo "Required packages:"
echo "  - PyTorch (with CUDA support)"
echo "  - NumPy"
echo "  - Pandas"
echo ""

if python3 -c "import torch; import numpy; import pandas" 2>/dev/null; then
    print_success "All dependencies already installed"
    python3 -c "import torch; print('PyTorch version:', torch.__version__)"
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
else
    print_warning "Installing dependencies..."
    read -p "Install now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch torchvision numpy pandas
        print_success "Dependencies installed"
    else
        print_error "Dependencies required. Please install manually."
        exit 1
    fi
fi

# Step 3: Prepare dataset
print_header "Step 3: Dataset Preparation"

if [ -d "data/cifar100_noniid" ]; then
    print_success "CIFAR-100 dataset already prepared"
    echo "Dataset location: data/cifar100_noniid/"
    echo ""
    read -p "Re-download and prepare dataset? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        SKIP_DATASET=true
    fi
fi

if [ "$SKIP_DATASET" != "true" ]; then
    echo "Preparing CIFAR-100 dataset..."
    echo "This will:"
    echo "  - Download CIFAR-100 (~170 MB)"
    echo "  - Create non-IID splits (alpha=0.5, 1.0)"
    echo "  - Generate 10 client partitions"
    echo ""
    read -p "Start dataset preparation? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 prepare_cifar100.py
        print_success "Dataset prepared successfully"
    else
        print_error "Dataset required. Exiting."
        exit 1
    fi
fi

# Step 4: Configure experiment
print_header "Step 4: Experiment Configuration"

echo "Experiment details:"
echo "  - Total experiments: 40"
echo "  - Modes: fedavg, krum, multi_krum, multisignal_trust, multikrum_val"
echo "  - Rounds per experiment: 30"
echo "  - Alphas: 0.5, 1.0"
echo "  - Malicious ratios: 0.2, 0.4 (excluding 0.8)"
echo "  - Attacks: label_flip, sign_flip"
echo ""

# Ask for number of GPUs
echo "How many GPUs to use for parallel execution?"
echo "  - More GPUs = faster completion"
echo "  - Each GPU runs one experiment at a time"
echo ""
read -p "Number of GPUs [$DEFAULT_GPUS]: " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-$DEFAULT_GPUS}

# Estimate time
if [ "$NUM_GPUS" -ge 8 ]; then
    TIME_EST="15-25 minutes"
elif [ "$NUM_GPUS" -ge 4 ]; then
    TIME_EST="30-50 minutes"
elif [ "$NUM_GPUS" -ge 2 ]; then
    TIME_EST="1-1.75 hours"
else
    TIME_EST="2-3.5 hours"
fi

echo ""
print_msg "Configuration:" "$BLUE"
echo "  GPUs: $NUM_GPUS"
echo "  Estimated time: $TIME_EST"
echo "  Output: results_v2/cifar100/"
echo ""

read -p "Start experiments? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_msg "Experiments cancelled." "$YELLOW"
    exit 0
fi

# Step 5: Run experiments
print_header "Step 5: Running Parallel Experiments"

print_msg "Starting $NUM_GPUS parallel workers..." "$GREEN"
echo "Monitor progress with:"
echo "  watch -n 5 'ls results_v2/cifar100/*.json 2>/dev/null | wc -l'"
echo ""

python3 run_parallel_gpu.py --num_gpus $NUM_GPUS

print_success "All experiments completed!"

# Step 6: Combine results
print_header "Step 6: Aggregating Results"

echo "Combining individual results into CSV files..."
python3 combine_results.py

print_success "Results aggregated successfully!"

# Final summary
print_header "Execution Complete!"

echo "Results saved to:"
echo "  📊 results_v2/cifar100/all_results.csv"
echo "  📈 results_v2/cifar100/summary.csv"
echo "  📁 results_v2/cifar100/*.json (individual results)"
echo "  📝 results_v2/cifar100/logs/*.log (execution logs)"
echo ""

echo "View results:"
echo "  cat results_v2/cifar100/summary.csv"
echo "  python3 -c 'import pandas as pd; df=pd.read_csv(\"results_v2/cifar100/summary.csv\"); print(df.head())'"
echo ""

print_success "All done! 🎉"
