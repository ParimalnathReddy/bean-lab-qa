#!/bin/bash
################################################################################
# Start Ollama Server Script
# For use in SLURM jobs or interactive sessions
#
# This script:
# 1. Loads Ollama module
# 2. Sets environment variables
# 3. Starts Ollama server in background
# 4. Waits for server to be ready
# 5. Returns when ready for use
################################################################################

# Project directory
PROJECT_DIR="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa"
MODELS_DIR="${PROJECT_DIR}/models/ollama"
LOG_DIR="${PROJECT_DIR}/logs"

# Create log directory
mkdir -p "${LOG_DIR}"

# Log file for this session
LOG_FILE="${LOG_DIR}/ollama_server_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Starting Ollama Server"
echo "========================================"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Models directory: ${MODELS_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Load Ollama module
echo "Loading Ollama module..."
module load Ollama/0.15.5

if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama command not found after loading module"
    exit 1
fi

echo "✓ Ollama loaded: $(ollama --version)"
echo ""

# Set environment variables
echo "Setting environment variables..."
export OLLAMA_MODELS="${MODELS_DIR}"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_NUM_PARALLEL=1

echo "✓ OLLAMA_MODELS=${OLLAMA_MODELS}"
echo "✓ OLLAMA_HOST=${OLLAMA_HOST}"
echo "✓ OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo ""

# Check if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    echo "⚠ Ollama server already running"
    echo "PID: $(pgrep -x ollama)"
    echo ""
    echo "To kill existing server:"
    echo "  pkill ollama"
    echo ""
    exit 0
fi

# Start Ollama server in background
echo "Starting Ollama server..."
nohup ollama serve > "${LOG_FILE}" 2>&1 &
OLLAMA_PID=$!

echo "✓ Ollama server started (PID: ${OLLAMA_PID})"
echo "  Log: ${LOG_FILE}"
echo ""

# Wait for server to be ready
echo "Waiting for Ollama server to be ready..."
MAX_WAIT=30  # Maximum wait time in seconds
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    # Try to connect to Ollama API
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama server is ready!"
        echo ""
        break
    fi

    # Check if process is still running
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama server process died"
        echo "Check log file: ${LOG_FILE}"
        exit 1
    fi

    echo "  Waiting... (${WAIT_TIME}s)"
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: Ollama server did not start within ${MAX_WAIT} seconds"
    echo "Check log file: ${LOG_FILE}"
    pkill -P $OLLAMA_PID
    exit 1
fi

# List available models
echo "Available models:"
ollama list
echo ""

echo "========================================"
echo "Ollama Server Ready!"
echo "========================================"
echo "Server URL: http://$(hostname):11434"
echo "PID: ${OLLAMA_PID}"
echo "Log: ${LOG_FILE}"
echo ""
echo "To use Ollama:"
echo "  ollama run llama3.1:8b"
echo ""
echo "To stop server:"
echo "  pkill ollama"
echo ""
echo "API endpoint:"
echo "  http://localhost:11434/api/generate"
echo ""
