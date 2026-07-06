#!/bin/bash
# Prepare repository for GitHub upload
# This script creates a clean repository structure without datasets

echo "🚀 Preparing GitHub Repository"
echo "=============================="
echo ""

# Create clean directory
REPO_DIR="federated-learning-trust"
rm -rf $REPO_DIR
mkdir -p $REPO_DIR

echo "📁 Creating repository structure..."

# Copy core files
cp -r fl models $REPO_DIR/
cp requirements.txt $REPO_DIR/
cp .gitignore LICENSE $REPO_DIR/

# Copy scripts
mkdir -p $REPO_DIR/scripts
cp run_single_experiment.py $REPO_DIR/scripts/

# Copy CIFAR-100 parallel package
mkdir -p $REPO_DIR/cifar100_parallel_gpu
cp prepare_cifar100.py run_parallel_gpu.py combine_results.py $REPO_DIR/cifar100_parallel_gpu/
cp setup_and_run.sh run_all.sh $REPO_DIR/cifar100_parallel_gpu/
cp USER_GUIDE.md QUICKSTART.md $REPO_DIR/cifar100_parallel_gpu/

# Copy documentation
cp README_GITHUB.md $REPO_DIR/README.md

# Create empty directories with .gitkeep
mkdir -p $REPO_DIR/data
touch $REPO_DIR/data/.gitkeep
mkdir -p $REPO_DIR/results_v2
touch $REPO_DIR/results_v2/.gitkeep

# Clean Python cache
find $REPO_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find $REPO_DIR -type f -name "*.pyc" -delete 2>/dev/null

echo "✅ Repository structure created"
echo ""

# Show structure
echo "📂 Repository contents:"
tree -L 2 $REPO_DIR 2>/dev/null || find $REPO_DIR -maxdepth 2 -type f -o -type d | head -20

echo ""
echo "📊 Repository statistics:"
echo "  Files: $(find $REPO_DIR -type f | wc -l)"
echo "  Size: $(du -sh $REPO_DIR | cut -f1)"
echo ""

echo "✅ Repository ready for GitHub!"
echo ""
echo "Next steps:"
echo "  1. cd $REPO_DIR"
echo "  2. git init"
echo "  3. git add ."
echo "  4. git commit -m 'Initial commit'"
echo "  5. git remote add origin https://github.com/yourusername/federated-learning-trust.git"
echo "  6. git push -u origin main"
