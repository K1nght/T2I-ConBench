#!/bin/bash

set -e

ENV_NAME="t2i-conbench"
PYTHON_VERSION="3.11"

echo "Setting up T2I-Conbench environment..."

# Check if environment exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Removing..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create conda environment
echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Install diffusers
echo "Installing diffusers..."
pip install git+https://github.com/huggingface/diffusers.git