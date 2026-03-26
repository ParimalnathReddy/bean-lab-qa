#!/bin/bash
################################################################################
# Quick Setup Script for PDF Processing
# Sets up environment and runs initial test
################################################################################

set -e  # Exit on error

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}PDF Processing Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Get project directory
PROJECT_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Check conda environment
echo -e "${YELLOW}Step 1: Checking conda environment...${NC}"
if conda env list | grep -q "bean_llm"; then
    echo -e "${GREEN}✓ Found bean_llm conda environment${NC}"
else
    echo -e "${RED}✗ bean_llm conda environment not found${NC}"
    echo "Please create it first with:"
    echo "  conda create -n bean_llm python=3.10"
    exit 1
fi
echo ""

# Step 2: Install dependencies
echo -e "${YELLOW}Step 2: Installing dependencies...${NC}"
echo "This will install: PyPDF2, tiktoken, tqdm, and other required packages"
read -p "Install dependencies now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda activate bean_llm
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo "Skipping dependency installation"
    echo "You can install later with: pip install -r requirements.txt"
fi
echo ""

# Step 3: Set up PDF directory
echo -e "${YELLOW}Step 3: Setting up PDF directory...${NC}"
SOURCE_DIR_1="/mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/1961-2006"
SOURCE_DIR_2="/mnt/research/BeanLab/Parimal/BEAN_LLM/Data/Data/2007-2026"
DEST_DIR="$PROJECT_DIR/data/pdfs"

if [ -d "$DEST_DIR/1961-2006" ] && [ -d "$DEST_DIR/2007-2026" ]; then
    echo -e "${GREEN}✓ PDF directories already set up${NC}"
    PDF_COUNT=$(find "$DEST_DIR" -name "*.pdf" | wc -l)
    echo "  Found $PDF_COUNT PDFs"
else
    echo "PDF directories not found. Choose setup method:"
    echo "  1) Create symbolic links (recommended - saves 347MB)"
    echo "  2) Copy PDFs (creates independent copy)"
    echo "  3) Skip for now"
    read -p "Enter choice (1/2/3): " -n 1 -r
    echo ""

    if [[ $REPLY == "1" ]]; then
        mkdir -p "$DEST_DIR"
        ln -sf "$SOURCE_DIR_1" "$DEST_DIR/1961-2006"
        ln -sf "$SOURCE_DIR_2" "$DEST_DIR/2007-2026"
        echo -e "${GREEN}✓ Symbolic links created${NC}"
        PDF_COUNT=$(find "$DEST_DIR" -name "*.pdf" | wc -l)
        echo "  Found $PDF_COUNT PDFs"
    elif [[ $REPLY == "2" ]]; then
        echo "Copying PDFs (this may take a minute)..."
        mkdir -p "$DEST_DIR/1961-2006"
        mkdir -p "$DEST_DIR/2007-2026"
        cp "$SOURCE_DIR_1"/*.pdf "$DEST_DIR/1961-2006/"
        cp "$SOURCE_DIR_2"/*.pdf "$DEST_DIR/2007-2026/"
        echo -e "${GREEN}✓ PDFs copied${NC}"
        PDF_COUNT=$(find "$DEST_DIR" -name "*.pdf" | wc -l)
        echo "  Copied $PDF_COUNT PDFs"
    else
        echo "Skipped PDF setup"
    fi
fi
echo ""

# Step 4: Create necessary directories
echo -e "${YELLOW}Step 4: Creating directories...${NC}"
mkdir -p logs
mkdir -p data/test_output
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 5: Run test
if [ -d "$DEST_DIR" ] && [ -n "$(ls -A $DEST_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}Step 5: Running test on sample PDFs...${NC}"
    read -p "Run test now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 src/test_processor.py
    else
        echo "Skipping test. You can run it later with:"
        echo "  python3 src/test_processor.py"
    fi
else
    echo -e "${YELLOW}Step 5: Test skipped (no PDFs found)${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Test the processor (if not done):"
echo "   ${GREEN}python3 src/test_processor.py${NC}"
echo ""
echo "2. Process all PDFs interactively:"
echo "   ${GREEN}python3 src/pdf_processor.py${NC}"
echo ""
echo "3. Or submit as SLURM job (recommended):"
echo "   ${GREEN}sbatch jobs/process_pdfs.sb${NC}"
echo ""
echo "4. Monitor job:"
echo "   ${GREEN}squeue -u \$USER${NC}"
echo "   ${GREEN}tail -f logs/pdf_processing_JOBID.out${NC}"
echo ""
echo "For more info, see: README_PDF_PROCESSOR.md"
echo ""
