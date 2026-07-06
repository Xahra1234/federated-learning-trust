#!/bin/bash
# CIFAR-100 Parallel GPU Launcher
# Complete workflow from dataset preparation to results

set -e

echo "🚀 CIFAR-100 Parallel GPU Execution"
echo "===================================="
echo ""

# Step 1: Check dependencies
echo "📦 Step 1: Checking dependencies..."
python -c "import torch; import numpy; import pandas; print('✅ All dependencies installed')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install torch torchvision numpy pandas
}
echo ""

# Step 2: Prepare dataset
if [ ! -d "data/cifar100_noniid" ]; then
    echo "📥 Step 2: Preparing CIFAR-100 dataset..."
    python prepare_cifar100.py
else
    echo "✅ Step 2: CIFAR-100 dataset already prepared"
fi
echo ""

# Step 3: Run parallel experiments
echo "🔥 Step 3: Running parallel experiments..."
NUM_GPUS=${1:-4}
echo "Using $NUM_GPUS GPUs"
python run_parallel_gpu.py --num_gpus $NUM_GPUS
echo ""

# Step 4: Combine results
echo "📊 Step 4: Combining results..."
python combine_results.py
echo ""

echo "✅ All done! Results saved to: results_v2/cifar100/"
echo ""
echo "View results:"
echo "  cat results_v2/cifar100/summary.csv"
