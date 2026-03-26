#!/bin/bash
################################################################################
# Test Ollama Installation and Model
#
# Tests:
# 1. Module loading
# 2. Environment variables
# 3. Server connectivity
# 4. Model availability
# 5. Simple inference test
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Ollama Test Suite${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Project directory
PROJECT_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
MODELS_DIR="${PROJECT_DIR}/models/ollama"

TESTS_PASSED=0
TESTS_FAILED=0

# Function to report test result
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Test 1: Module availability
echo "Test 1: Ollama module availability"
echo "-----------------------------------"
module load Ollama/0.15.5 2>&1
test_result $?

# Test 2: Ollama command exists
echo "Test 2: Ollama command"
echo "-----------------------------------"
if command -v ollama &> /dev/null; then
    VERSION=$(ollama --version)
    echo "Ollama version: ${VERSION}"
    test_result 0
else
    echo "Ollama command not found"
    test_result 1
fi

# Test 3: Environment variables
echo "Test 3: Environment variables"
echo "-----------------------------------"
export OLLAMA_MODELS="${MODELS_DIR}"
export OLLAMA_HOST="0.0.0.0:11434"

echo "OLLAMA_MODELS=${OLLAMA_MODELS}"
echo "OLLAMA_HOST=${OLLAMA_HOST}"

if [ -d "${OLLAMA_MODELS}" ]; then
    echo "Models directory exists"
    test_result 0
else
    echo "Models directory does not exist"
    test_result 1
fi

# Test 4: Check if Ollama server is running
echo "Test 4: Ollama server status"
echo "-----------------------------------"
if pgrep -x "ollama" > /dev/null; then
    echo "Ollama server is running"
    OLLAMA_PID=$(pgrep -x "ollama")
    echo "PID: ${OLLAMA_PID}"

    # Test API connectivity
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "API is responding"
        test_result 0
    else
        echo "API is not responding"
        test_result 1
    fi
else
    echo "Ollama server not running"
    echo "Starting server..."

    # Try to start server
    bash "${PROJECT_DIR}/scripts/start_ollama.sh"

    if [ $? -eq 0 ]; then
        echo "Server started successfully"
        test_result 0
    else
        echo "Failed to start server"
        test_result 1
    fi
fi

# Test 5: List models
echo "Test 5: Available models"
echo "-----------------------------------"
MODEL_LIST=$(ollama list 2>&1)
echo "$MODEL_LIST"

if echo "$MODEL_LIST" | grep -q "llama3.1:8b"; then
    echo "✓ llama3.1:8b model found"
    test_result 0
else
    echo "⚠ llama3.1:8b model not found"
    echo ""
    echo "Download with:"
    echo "  bash ${PROJECT_DIR}/scripts/download_llama_model.sh"
    test_result 1
fi

# Test 6: Simple inference (if model available)
if echo "$MODEL_LIST" | grep -q "llama3.1:8b"; then
    echo "Test 6: Simple inference"
    echo "-----------------------------------"
    echo "Prompt: 'What is 2+2? Answer in one word.'"
    echo ""

    # Run inference with timeout
    RESPONSE=$(timeout 30 ollama run llama3.1:8b "What is 2+2? Answer in one word." 2>&1)
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "Response: ${RESPONSE}"
        echo ""
        test_result 0
    elif [ $RESULT -eq 124 ]; then
        echo "Timeout (inference took >30s)"
        test_result 1
    else
        echo "Inference failed"
        echo "Error: ${RESPONSE}"
        test_result 1
    fi
fi

# Test 7: API test
echo "Test 7: API endpoint test"
echo "-----------------------------------"

# Create test request
TEST_REQUEST='{
  "model": "llama3.1:8b",
  "prompt": "Say hello",
  "stream": false
}'

echo "Testing API endpoint..."
API_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "$TEST_REQUEST" 2>&1)

if [ $? -eq 0 ] && echo "$API_RESPONSE" | grep -q "response"; then
    echo "✓ API endpoint working"
    echo "Response preview:"
    echo "$API_RESPONSE" | head -c 200
    echo "..."
    test_result 0
else
    echo "API test failed"
    test_result 1
fi

# Summary
echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Tests passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Ollama is ready to use"
    echo ""
    echo "Next steps:"
    echo "  - Integrate with QA system"
    echo "  - Test with ChromaDB vector store"
    echo "  - Build retrieval-augmented generation"
    exit 0
else
    echo -e "${YELLOW}⚠ Some tests failed${NC}"
    echo ""
    echo "Check the failures above and:"
    echo "  - Ensure you're on a compute node"
    echo "  - Download model if missing"
    echo "  - Check server logs in logs/ directory"
    exit 1
fi
