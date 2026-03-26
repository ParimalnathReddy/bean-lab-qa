# MSU HPCC Module Setup Guide for LLM Project

## Current Environment Status
- **Node**: dev-amd20-v100 (Development Node with 4x Tesla V100S-PCIE-32GB)
- **CUDA Driver**: 560.35.05 (supports CUDA 12.6)
- **Conda Environment**: `bean_llm` (already has PyTorch 2.10.0+cu128)

---

## ⚠️ IMPORTANT: Two Approaches

You have **two options** for running your LLM project on HPCC:

### **Option 1: Use Your Existing Conda Environment** (RECOMMENDED)
✅ You already have PyTorch with CUDA support in your `bean_llm` conda environment
✅ Faster setup - just install additional packages
✅ Better version control and reproducibility

### **Option 2: Use HPCC Modules**
✅ Uses system-provided PyTorch
✅ Tested and optimized for HPCC hardware
⚠️ Warning: Some tests fail, use at your own risk

---

## Option 1: Using Conda Environment (RECOMMENDED)

### Current Setup
Your `bean_llm` conda environment already has:
- Python 3.10.12
- PyTorch 2.10.0 with CUDA 12.8 support
- CUDA available: ✓
- 4 GPUs detected

### Install Additional Packages

```bash
# Activate your conda environment
conda activate bean_llm

# Install LLM project dependencies
pip install sentence-transformers chromadb langchain pypdf python-dotenv

# Optional: Install additional useful packages
pip install transformers accelerate bitsandbytes langchain-community
```

### Verify Installation

```bash
python3 -c "import sentence_transformers; print(f'sentence-transformers: {sentence_transformers.__version__}')"
python3 -c "import chromadb; print(f'chromadb: {chromadb.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### SLURM Job Script (Conda Approach)

```bash
#!/bin/bash
#SBATCH --job-name=llm_qa
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load conda
module load Miniforge3/24.3.0-0

# Activate conda environment
source activate bean_llm

# Set environment variables
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/scratch/$USER/llm_cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $SENTENCE_TRANSFORMERS_HOME

# Print environment info
echo "=== Environment Info ==="
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi

# Run your script
cd /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa
python3 src/your_script.py
```

---

## Option 2: Using HPCC Modules

### Load Modules

```bash
# Single command to load PyTorch with all dependencies
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

This automatically loads:
- **Python**: 3.11.3
- **CUDA**: 12.1.1
- **cuDNN**: 8.9.2.26
- **PyTorch**: 2.1.2
- **foss toolchain**: GCC 12.3.0, OpenMPI, OpenBLAS, FFTW, ScaLAPACK

### Install Additional Packages

```bash
# Install to user directory
pip install --user sentence-transformers chromadb langchain pypdf
```

### SLURM Job Script (Module Approach)

```bash
#!/bin/bash
#SBATCH --job-name=llm_qa
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load PyTorch module with CUDA support
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Set cache directories to scratch space
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/scratch/$USER/llm_cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $SENTENCE_TRANSFORMERS_HOME

# Print environment info
echo "=== Loaded Modules ==="
module list

echo ""
echo "=== Environment Info ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi

# Run your script
cd /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa
python src/your_script.py
```

---

## Available HPCC Modules Reference

### CUDA Versions Available
```bash
CUDA/11.7.0
CUDA/12.1.1  ← Used by PyTorch module
CUDA/12.3.0
CUDA/12.4.0
CUDA/12.6.0
CUDA/12.9.1
```

### Python Versions Available
```bash
Python/3.10.8
Python/3.11.3  ← Used by PyTorch module
Python/3.11.5
Python/3.12.3
Python/3.13.1
```

### PyTorch Module
```bash
PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

### Other Useful Modules
```bash
torchvision/0.16.0-foss-2023a-CUDA-12.1.1
```

---

## Key Recommendations

### 1. **Use Option 1 (Conda)** for most cases
- Your conda environment already has newer PyTorch (2.10.0)
- Better compatibility with latest packages
- Easier dependency management

### 2. **Set Cache Directories**
Always export these variables to avoid filling up home directory:

```bash
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/scratch/$USER/llm_cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers
```

Add to `~/.bashrc` for persistence:
```bash
echo 'export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/scratch/$USER/llm_cache/huggingface' >> ~/.bashrc
echo 'export SENTENCE_TRANSFORMERS_HOME=/mnt/scratch/$USER/llm_cache/sentence_transformers' >> ~/.bashrc
```

### 3. **GPU Partitions on HPCC**
```bash
# Interactive development (what you're on now)
--partition=dev-gpu

# Production jobs
--partition=gpu

# V100 GPUs specifically
--partition=v100

# A100 GPUs (if available)
--partition=a100
```

### 4. **Check Module Details**
```bash
# Search for modules
module spider pytorch

# Show module details
module show PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# List currently loaded modules
module list

# Purge all modules
module purge
```

---

## Quick Start Commands

### Using Conda (Recommended)
```bash
# Install packages
conda activate bean_llm
pip install sentence-transformers chromadb langchain pypdf

# Test installation
python3 -c "import torch; import sentence_transformers; import chromadb; print('All packages loaded successfully!')"
```

### Using Modules
```bash
# Load module
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Install packages
pip install --user sentence-transformers chromadb langchain pypdf

# Test installation
python -c "import torch; import sentence_transformers; import chromadb; print('All packages loaded successfully!')"
```

---

## Troubleshooting

### Issue: "No CUDA devices found"
```bash
# Check GPU availability
nvidia-smi

# Check SLURM GPU allocation
echo $CUDA_VISIBLE_DEVICES

# In SLURM script, ensure you have:
#SBATCH --gpus=1
```

### Issue: "Out of memory on home directory"
```bash
# Check quota
quota -s

# Move cache to scratch
export HF_HOME=/mnt/scratch/$USER/llm_cache/huggingface
```

### Issue: "Module not found"
```bash
# Conda approach: activate environment first
conda activate bean_llm

# Module approach: load module first
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

---

## Next Steps

1. Choose your approach (Conda recommended)
2. Install required packages
3. Create a test script
4. Submit a test SLURM job
5. Monitor with `squeue -u $USER`
