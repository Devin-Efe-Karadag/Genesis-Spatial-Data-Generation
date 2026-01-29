#!/bin/bash
# Setup script for sandbox environment

set -e  # Exit on error

echo "Installing CUDA toolkit..."
conda install -y nvidia::cuda-toolkit==12.8.0

echo "Installing PyTorch..."
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing Kaolin..."
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

echo "Installing Blender Python..."
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

echo "Installing Taichi..."
pip install taichi

echo "Installing Genesis..."
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git@v0.3.13

echo "✓ Setup complete!"
