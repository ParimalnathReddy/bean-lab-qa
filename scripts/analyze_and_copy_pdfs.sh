#!/bin/bash
################################################################################
# PDF Analysis and Organization Script
# Purpose: Analyze and copy PDFs from Data directories to project structure
# Location: /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}PDF Analysis and Organization${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Define paths
BASE_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM"
SOURCE_DIR_1="${BASE_DIR}/Data/Data/1961-2006"
SOURCE_DIR_2="${BASE_DIR}/Data/Data/2007-2026"
DEST_DIR="${BASE_DIR}/hpcc-llm-qa/data/pdfs"

################################################################################
# 1. COUNT TOTAL PDFs IN EACH DIRECTORY
################################################################################
echo -e "${GREEN}1. Counting PDFs in each directory...${NC}"
echo ""

COUNT_1=$(find "${SOURCE_DIR_1}" -type f -name "*.pdf" 2>/dev/null | wc -l)
echo -e "  📁 1961-2006: ${YELLOW}${COUNT_1}${NC} PDFs"

COUNT_2=$(find "${SOURCE_DIR_2}" -type f -name "*.pdf" 2>/dev/null | wc -l)
echo -e "  📁 2007-2026: ${YELLOW}${COUNT_2}${NC} PDFs"

TOTAL_COUNT=$((COUNT_1 + COUNT_2))
echo -e "  📊 ${GREEN}Total: ${TOTAL_COUNT} PDFs${NC}"
echo ""

################################################################################
# 2. CHECK TOTAL FILE SIZE
################################################################################
echo -e "${GREEN}2. Checking total file sizes...${NC}"
echo ""

SIZE_1=$(du -sh "${SOURCE_DIR_1}" | cut -f1)
echo -e "  💾 1961-2006: ${YELLOW}${SIZE_1}${NC}"

SIZE_2=$(du -sh "${SOURCE_DIR_2}" | cut -f1)
echo -e "  💾 2007-2026: ${YELLOW}${SIZE_2}${NC}"

TOTAL_SIZE=$(du -sh "${BASE_DIR}/Data/Data/" | cut -f1)
echo -e "  💾 ${GREEN}Total: ${TOTAL_SIZE}${NC}"
echo ""

################################################################################
# 3. SHOW SAMPLE FILENAMES (FIRST 10)
################################################################################
echo -e "${GREEN}3. Sample filenames (first 10 from each directory)...${NC}"
echo ""

echo -e "${YELLOW}From 1961-2006:${NC}"
find "${SOURCE_DIR_1}" -type f -name "*.pdf" 2>/dev/null | head -10 | while read pdf; do
    basename "$pdf"
done
echo ""

echo -e "${YELLOW}From 2007-2026:${NC}"
find "${SOURCE_DIR_2}" -type f -name "*.pdf" 2>/dev/null | head -10 | while read pdf; do
    basename "$pdf"
done
echo ""

################################################################################
# 4. CHECK IF PDFs ARE READABLE/VALID
################################################################################
echo -e "${GREEN}4. Validating PDF files (checking 5 random samples)...${NC}"
echo ""

echo -e "${YELLOW}Checking samples from 1961-2006:${NC}"
find "${SOURCE_DIR_1}" -type f -name "*.pdf" 2>/dev/null | shuf | head -3 | while read pdf; do
    filename=$(basename "$pdf")
    file_info=$(file "$pdf" 2>/dev/null)
    if echo "$file_info" | grep -q "PDF document"; then
        pages=$(echo "$file_info" | grep -oP '\d+(?= pages?)' || echo "unknown")
        echo -e "  ✅ ${filename}: ${pages} pages"
    else
        echo -e "  ❌ ${filename}: NOT a valid PDF"
    fi
done
echo ""

echo -e "${YELLOW}Checking samples from 2007-2026:${NC}"
find "${SOURCE_DIR_2}" -type f -name "*.pdf" 2>/dev/null | shuf | head -2 | while read pdf; do
    filename=$(basename "$pdf")
    file_info=$(file "$pdf" 2>/dev/null)
    if echo "$file_info" | grep -q "PDF document"; then
        pages=$(echo "$file_info" | grep -oP '\d+(?= pages?)' || echo "unknown")
        echo -e "  ✅ ${filename}: ${pages} pages"
    else
        echo -e "  ❌ ${filename}: NOT a valid PDF"
    fi
done
echo ""

################################################################################
# 5. COPY ALL PDFs TO DESTINATION (PRESERVING ORGANIZATION)
################################################################################
echo -e "${GREEN}5. Preparing to copy PDFs to project directory...${NC}"
echo ""
echo -e "Destination: ${BLUE}${DEST_DIR}${NC}"
echo ""
echo -e "${YELLOW}This will create the following structure:${NC}"
echo "  ${DEST_DIR}/"
echo "  ├── 1961-2006/  (${COUNT_1} files, ${SIZE_1})"
echo "  └── 2007-2026/  (${COUNT_2} files, ${SIZE_2})"
echo ""

# Ask for confirmation (comment out for non-interactive use)
read -p "Do you want to proceed with copying? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Copy operation cancelled.${NC}"
    exit 0
fi

# Create destination directories
echo -e "${YELLOW}Creating destination directories...${NC}"
mkdir -p "${DEST_DIR}/1961-2006"
mkdir -p "${DEST_DIR}/2007-2026"

# Copy files
echo -e "${YELLOW}Copying PDFs from 1961-2006...${NC}"
cp -v "${SOURCE_DIR_1}"/*.pdf "${DEST_DIR}/1961-2006/" 2>/dev/null
COPIED_1=$?

echo ""
echo -e "${YELLOW}Copying PDFs from 2007-2026...${NC}"
cp -v "${SOURCE_DIR_2}"/*.pdf "${DEST_DIR}/2007-2026/" 2>/dev/null
COPIED_2=$?

echo ""
################################################################################
# VERIFY COPY OPERATION
################################################################################
echo -e "${GREEN}6. Verifying copy operation...${NC}"
echo ""

DEST_COUNT_1=$(find "${DEST_DIR}/1961-2006" -type f -name "*.pdf" 2>/dev/null | wc -l)
echo -e "  Copied to 1961-2006: ${YELLOW}${DEST_COUNT_1}${NC} / ${COUNT_1} PDFs"

DEST_COUNT_2=$(find "${DEST_DIR}/2007-2026" -type f -name "*.pdf" 2>/dev/null | wc -l)
echo -e "  Copied to 2007-2026: ${YELLOW}${DEST_COUNT_2}${NC} / ${COUNT_2} PDFs"

DEST_TOTAL=$((DEST_COUNT_1 + DEST_COUNT_2))
echo -e "  ${GREEN}Total copied: ${DEST_TOTAL} / ${TOTAL_COUNT} PDFs${NC}"

# Check for errors
if [ $DEST_TOTAL -eq $TOTAL_COUNT ]; then
    echo -e "\n${GREEN}✅ All PDFs copied successfully!${NC}"
else
    echo -e "\n${RED}⚠️  Warning: Not all PDFs were copied. Please check for errors.${NC}"
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Analysis Complete!${NC}"
echo -e "${BLUE}================================${NC}"
