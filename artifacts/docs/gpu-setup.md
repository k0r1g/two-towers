# GPU Environment Setup for Two-Tower Model

This document describes the setup process for creating a GPU-enabled environment for training and running the Two-Tower model.

## Overview

Setting up a proper GPU environment involves several steps:
1. Installing CUDA-compatible versions of Conda and PyTorch
2. Creating a dedicated conda environment
3. Installing required dependencies
4. Configuring Weights & Biases for experiment tracking
5. Verifying GPU accessibility

## Requirements

- NVIDIA GPU with CUDA support (tested with NVIDIA GeForce RTX 3090)
- NVIDIA drivers installed (tested with driver version 570.86.15)
- CUDA toolkit (compatible with version 12.8)
- Internet connection for downloading packages

## Quick Start

For a quick setup, we provide an automated script that handles the entire process:

```bash
# Download the setup script
wget https://raw.githubusercontent.com/k0r1g/two-towers/main/setup.sh

# Make it executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

## Manual Setup Steps

If you prefer to set up the environment manually, follow these steps:

### 1. Install Conda

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Make the installer executable
chmod +x ~/miniconda.sh

# Run the installer in batch mode
~/miniconda.sh -b -p $HOME/miniconda

# Initialize conda for your shell
$HOME/miniconda/bin/conda init bash

# Source conda for current session
source $HOME/miniconda/etc/profile.d/conda.sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/k0r1g/two-towers.git
cd two-towers
```

### 3. Create and Activate Conda Environment

```bash
# Create a new environment with Python 3.11
conda create -y -n twotower python=3.11

# Activate the environment
conda activate twotower
```

### 4. Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

### 5. Install Required Packages

```bash
# Install dependencies from the requirements file
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### 6. Configure Weights & Biases

```bash
# Set up Weights & Biandes with your API key
wandb login YOUR_WANDB_API_KEY
```

Replace `YOUR_WANDB_API_KEY` with your actual Weights & Biandes API key from https://wandb.ai/authorize.

## Verifying GPU Availability

To ensure your GPU is properly configured with PyTorch, run:

```bash
# Check NVIDIA GPU status
nvidia-smi

# Verify PyTorch can see the GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

You should see output confirming GPU availability and the device name.

### Sample Output

```
# nvidia-smi output
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15    CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On | 00000000:81:00.0 Off |                  N/A |
| 40%   35C    P8              33W / 350W |     2MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

# PyTorch output
CUDA available: True Device: NVIDIA GeForce RTX 3090
```

## Automated Setup Script

We've created a setup script that automates the entire process. The script handles:
- Installing Conda if not present
- Cloning the repository
- Creating the conda environment
- Installing PyTorch and dependencies
- Setting up Weights & Biandes

Here's the script content:

```bash
#!/bin/bash
set -e  # Exit on error

# Configuration variables
REPO_NAME="two-towers"
REPO_URL="https://github.com/k0r1g/two-towers.git"
CONDA_ENV_NAME="twotower"
PYTHON_VERSION="3.11"
WANDB_API_KEY="YOUR_WANDB_API_KEY"  # Replace with your key
PROJECT_DIR="$HOME/$REPO_NAME"

# Check if conda is installed, if not install it
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
    
    # Make installer executable and run in batch mode
    chmod +x $HOME/miniconda.sh
    $HOME/miniconda.sh -b -p $HOME/miniconda
    
    # Initialize conda for bash
    $HOME/miniconda/bin/conda init bash
    
    # Source conda for current session
    source $HOME/miniconda/etc/profile.d/conda.sh
else
    echo "Conda is already installed."
fi

# Ensure conda commands are available in current session
if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
else
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Check if the project exists, if not clone it
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning $REPO_NAME repository from $REPO_URL..."
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    echo "$REPO_NAME repository already exists at $PROJECT_DIR."
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating conda environment $CONDA_ENV_NAME with Python $PYTHON_VERSION..."
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
else
    echo "Conda environment $CONDA_ENV_NAME already exists."
fi

# Activate the environment
echo "Activating conda environment $CONDA_ENV_NAME..."
conda activate "$CONDA_ENV_NAME"

# Install core dependencies
echo "Installing PyTorch..."
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html

echo "Installing requirements from $PROJECT_DIR/requirements.txt..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Warning: requirements.txt not found!"
fi

# Change to project directory for editable install
echo "Installing project in development mode..."
cd "$PROJECT_DIR"
pip install -e .

# Login to wandb
echo "Logging in to Weights & Biandes..."
wandb login "$WANDB_API_KEY"

echo "Setup complete! Environment '$CONDA_ENV_NAME' is ready."
echo "To activate this environment in the future, run: conda activate $CONDA_ENV_NAME"
```

Save this as `setup.sh`, replace the API key with your own, make it executable with `chmod +x setup.sh`, and run it.

## Common Issues and Troubleshooting

### Conda Activation Issues

If you encounter `conda command not found` or activation issues:

```bash
# Source conda manually
source $HOME/miniconda/etc/profile.d/conda.sh

# Verify conda is available
which conda
```

### PyTorch CUDA Issues

If PyTorch cannot detect your GPU:

1. Check if NVIDIA drivers are installed: `nvidia-smi`
2. Verify PyTorch CUDA compatibility: `python -c "import torch; print(torch.version.cuda)"`
3. Make sure the CUDA versions are compatible with your driver

### Weights & Biandes Login Issues

If you encounter authentication issues with wandb:

1. Verify your API key at https://wandb.ai/authorize
2. Try logging in manually: `wandb login`

## Next Steps

After setting up the environment, you can:

1. Review the [Configuration Guide](config.md) to understand model parameters
2. Start training with the command: `python train.py --config configs/your_config.yaml`
3. Explore the [Inference Guide](inference.md) for using your trained model

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Weights & Biandes Documentation](https://docs.wandb.ai/) 