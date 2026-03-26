#!/bin/bash
################################################################################
# Download LLaMA 3.1 8B Model for Ollama
#
# Model: llama3.1:8b
# Size: ~4.7GB
# Quantization: Q4_0 (4-bit quantization for efficiency)
#
# IMPORTANT: Run this on a compute node, not login node
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Download LLaMA 3.1 8B Model${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Project directory
PROJECT_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
MODELS_DIR="${PROJECT_DIR}/models/ollama"

# Check if running on compute node
if [[ $(hostname) == *"dev"* ]] || [[ $(hostname) == *"nod"* ]]; then
    echo -e "${GREEN}✓ Running on compute node: $(hostname)${NC}"
else
    echo -e "${RED}✗ ERROR: Must run on compute node!${NC}"
    echo -e "${YELLOW}  Use: salloc --mem=16GB --time=1:00:00${NC}"
    echo -e "${YELLOW}  Then run this script${NC}"
    exit 1
fi
echo ""

# Load Ollama module
echo "Loading Ollama module..."
module load Ollama/0.15.5

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama command not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ollama loaded: $(ollama --version)${NC}"
echo ""

# Set environment variables
export OLLAMA_MODELS="${MODELS_DIR}"
echo "Models directory: ${OLLAMA_MODELS}"
echo ""

# Create models directory
mkdir -p "${OLLAMA_MODELS}"

# Check disk space
echo "Checking disk space..."
AVAILABLE_SPACE=$(df -BG "${MODELS_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_SPACE=10  # Need at least 10GB for safety

echo "Available space: ${AVAILABLE_SPACE}GB"
echo "Required space: ~5GB (model) + buffer"

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo -e "${RED}✗ Not enough disk space!${NC}"
    echo "Free up space and try again"
    exit 1
fi

echo -e "${GREEN}✓ Sufficient disk space${NC}"
echo ""

# Check if model already exists
echo "Checking for existing models..."
if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    echo -e "${YELLOW}⚠ llama3.1:8b already downloaded${NC}"
    echo ""
    ollama list
    echo ""
    read -p "Re-download? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download"
        exit 0
    fi
fi

# Download model
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Downloading llama3.1:8b${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Model: llama3.1:8b (Meta's LLaMA 3.1 8B parameter model)"
echo "Size: ~4.7GB"
echo "Quantization: 4-bit (Q4_0)"
echo ""
echo -e "${YELLOW}This will take 5-15 minutes depending on network speed${NC}"
echo ""

# Pull model
ollama pull llama3.1:8b

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}✓ Download Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""

    # List models
    echo "Downloaded models:"
    ollama list
    echo ""

    # Show model info
    echo "Model information:"
    ollama show llama3.1:8b
    echo ""

    # Check disk usage
    echo "Disk usage:"
    du -sh "${OLLAMA_MODELS}"
    echo ""

    echo "Model ready to use!"
    echo ""
    echo "Test with:"
    echo -e "  ${GREEN}bash ${PROJECT_DIR}/scripts/test_ollama.sh${NC}"
    echo ""
    echo "Or run directly:"
    echo -e "  ${GREEN}ollama run llama3.1:8b${NC}"
    echo ""

else
    echo ""
    echo -e "${RED}✗ Download failed!${NC}"
    echo "Check your internet connection and try again"
    exit 1
fi
