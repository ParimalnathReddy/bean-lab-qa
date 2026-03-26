#!/bin/bash
################################################################################
# Ollama Setup Script for MSU HPCC
# Sets up Ollama for use in LLM QA system
#
# Architecture: zen2 (dev-amd20-v100)
# Ollama Version: 0.15.5 (latest on HPCC)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Ollama Setup for MSU HPCC${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Project directory
PROJECT_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
MODELS_DIR="${PROJECT_DIR}/models/ollama"

# Check if running on compute node (not login node)
if [[ $(hostname) == *"dev"* ]] || [[ $(hostname) == *"nod"* ]]; then
    echo -e "${GREEN}✓ Running on compute node: $(hostname)${NC}"
else
    echo -e "${YELLOW}⚠ Warning: You appear to be on a login node${NC}"
    echo -e "${YELLOW}  Ollama should only run on compute nodes${NC}"
    echo -e "${YELLOW}  Use: salloc or sbatch to get a compute node${NC}"
    echo ""
fi

# Step 1: Create models directory
echo -e "${YELLOW}Step 1: Creating Ollama models directory${NC}"
mkdir -p "${MODELS_DIR}"
echo -e "${GREEN}✓ Created: ${MODELS_DIR}${NC}"
echo ""

# Step 2: Load Ollama module
echo -e "${YELLOW}Step 2: Loading Ollama module${NC}"
module load Ollama/0.15.5
echo -e "${GREEN}✓ Ollama module loaded${NC}"

# Verify module loaded
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version)
    echo -e "${GREEN}✓ Ollama version: ${OLLAMA_VERSION}${NC}"
else
    echo -e "${RED}✗ Ollama command not found${NC}"
    echo -e "${RED}  Module may not have loaded correctly${NC}"
    exit 1
fi
echo ""

# Step 3: Set environment variables
echo -e "${YELLOW}Step 3: Setting environment variables${NC}"

# Set OLLAMA_MODELS to project directory
export OLLAMA_MODELS="${MODELS_DIR}"
echo -e "${GREEN}✓ OLLAMA_MODELS=${OLLAMA_MODELS}${NC}"

# Set OLLAMA_HOST for network access (needed for SLURM jobs)
export OLLAMA_HOST="0.0.0.0:11434"
echo -e "${GREEN}✓ OLLAMA_HOST=${OLLAMA_HOST}${NC}"

# Optional: Set number of parallel requests
export OLLAMA_NUM_PARALLEL=1
echo -e "${GREEN}✓ OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}${NC}"

echo ""

# Step 4: Add to bashrc (optional)
echo -e "${YELLOW}Step 4: Add to ~/.bashrc? (y/n)${NC}"
read -p "This will make Ollama settings permanent: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    BASHRC_FILE="$HOME/.bashrc"

    # Check if already added
    if grep -q "OLLAMA_MODELS=${MODELS_DIR}" "$BASHRC_FILE" 2>/dev/null; then
        echo -e "${YELLOW}⚠ Settings already in ~/.bashrc${NC}"
    else
        echo "" >> "$BASHRC_FILE"
        echo "# Ollama settings for Bean Lab LLM project" >> "$BASHRC_FILE"
        echo "export OLLAMA_MODELS=${MODELS_DIR}" >> "$BASHRC_FILE"
        echo "export OLLAMA_HOST=0.0.0.0:11434" >> "$BASHRC_FILE"
        echo "export OLLAMA_NUM_PARALLEL=1" >> "$BASHRC_FILE"
        echo -e "${GREEN}✓ Added to ~/.bashrc${NC}"
        echo -e "${YELLOW}  Run 'source ~/.bashrc' to apply${NC}"
    fi
fi
echo ""

# Step 5: Check available models
echo -e "${YELLOW}Step 5: Checking for downloaded models${NC}"
ollama list 2>/dev/null || echo "No models downloaded yet"
echo ""

# Step 6: Information about downloading models
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Download llama3.1:8b model:"
echo -e "   ${GREEN}ollama pull llama3.1:8b${NC}"
echo ""
echo "2. Or use the download script:"
echo -e "   ${GREEN}bash ${PROJECT_DIR}/scripts/download_llama_model.sh${NC}"
echo ""
echo "3. Test Ollama:"
echo -e "   ${GREEN}bash ${PROJECT_DIR}/scripts/test_ollama.sh${NC}"
echo ""
echo "4. Start Ollama server in SLURM job:"
echo -e "   ${GREEN}bash ${PROJECT_DIR}/scripts/start_ollama.sh${NC}"
echo ""

echo "Models directory: ${MODELS_DIR}"
echo "Project directory: ${PROJECT_DIR}"
echo ""
