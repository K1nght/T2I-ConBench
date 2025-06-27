#!/bin/bash

conda create -n t2i-conbench python=3.11

conda activate t2i-conbench

echo "Installing PyTorch with CUDA 12.1 support..."

# Install PyTorch with CUDA support from PyTorch index
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

echo "PyTorch installation completed!"

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 

# Install requirements 
pip install -r requirements.txt