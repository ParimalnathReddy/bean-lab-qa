#!/bin/bash
###############################################################################
# MSU HPCC Module Setup for LLM Project
# Python 3.11 + PyTorch 2.1.2 + CUDA 12.1.1
###############################################################################

# Clear any existing modules (optional - use if you want a clean slate)
# module purge

# Load PyTorch with CUDA support
# This automatically loads all dependencies:
# - Python/3.11.3
# - CUDA/12.1.1
# - cuDNN/8.9.2.26
# - foss/2023a (GCC, OpenMPI, OpenBLAS, FFTW, ScaLAPACK)
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Verify loaded modules
echo "=== Loaded Modules ==="
module list

# Verify CUDA availability
echo ""
echo "=== CUDA Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Verify Python and PyTorch
echo ""
echo "=== Python & PyTorch Verification ==="
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"

###############################################################################
# After loading modules, install your project dependencies:
# pip install --user sentence-transformers chromadb langchain pypdf
###############################################################################
