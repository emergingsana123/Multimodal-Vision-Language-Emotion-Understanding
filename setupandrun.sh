#!/bin/bash
# Setup and Run Script for AFEW-VA Analysis
# University VM Version

echo "=========================================="
echo "AFEW-VA Analysis Pipeline Setup"
echo "=========================================="

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: Neither conda nor mamba found!"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

echo "Using: $CONDA_CMD"

# Create conda environment
echo ""
echo "Creating conda environment 'multimodal'..."
$CONDA_CMD create -n multimodal python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate multimodal

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the analysis:"
echo "  1. Make sure you're in the conda environment:"
echo "     conda activate multimodal"
echo ""
echo "  2. Edit the script to update paths if needed:"
echo "     nano afew_va_analysis.py"
echo "     (Update DATA_DIR and PROJECT_ROOT)"
echo ""
echo "  3. Run the script:"
echo "     python afew_va_analysis.py"
echo ""
echo "  4. Or run in background with nohup:"
echo "     nohup python afew_va_analysis.py > output.log 2>&1 &"
echo ""
echo "  5. Monitor progress:"
echo "     tail -f output.log"
echo ""
echo "=========================================="