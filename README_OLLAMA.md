# Ollama Setup for MSU HPCC

Complete guide for running Ollama LLM on MSU HPCC for your QA system.

---

## Overview

**Ollama** is a tool for running large language models locally. On HPCC, it's available as a module.

### Your Setup
- **Architecture**: zen2 (dev-amd20-v100)
- **Ollama Version**: 0.15.5 (latest on HPCC)
- **Model**: llama3.1:8b (~4.7GB)
- **GPU**: Tesla V100S-PCIE-32GB

---

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Run setup script
bash /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/scripts/setup_ollama.sh
```

This creates:
- Models directory at `hpcc-llm-qa/models/ollama/`
- Environment variables
- Configuration files

### 2. Download Model (One-time, ~10 minutes)

**IMPORTANT**: Must run on compute node, not login node!

```bash
# Get compute node
salloc --mem=16GB --time=1:00:00

# Download model
bash /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/scripts/download_llama_model.sh

# Exit compute node
exit
```

### 3. Test Installation

```bash
# Get compute node
salloc --mem=16GB --gpus=1 --time=1:00:00

# Run tests
bash /mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/scripts/test_ollama.sh

# Exit
exit
```

---

## Available Ollama Modules

MSU HPCC provides these Ollama versions:

| Version | Status | Recommended |
|---------|--------|-------------|
| 0.3.12 | Old | No |
| 0.4.7 | Old | No |
| 0.6.2 | Old | No |
| 0.6.7 | Old | No |
| 0.12.6 | Stable | No |
| **0.15.5** | **Latest** | **Yes** ✓ |

**Load with**: `module load Ollama/0.15.5`

---

## Directory Structure

```
hpcc-llm-qa/
├── models/
│   └── ollama/                     # Ollama models stored here
│       └── manifests/
│           └── registry.ollama.ai/
│               └── library/
│                   └── llama3.1/   # Model files (~4.7GB)
│
├── scripts/
│   ├── setup_ollama.sh            # Initial setup
│   ├── start_ollama.sh            # Start server
│   ├── download_llama_model.sh    # Download model
│   └── test_ollama.sh             # Test suite
│
├── jobs/
│   └── run_ollama_qa.sb           # SLURM job example
│
└── logs/
    └── ollama_server_*.log        # Server logs
```

---

## Environment Variables

**Critical variables** (set automatically by scripts):

```bash
# Where models are stored (YOUR PROJECT DIRECTORY, NOT HOME!)
export OLLAMA_MODELS="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/models/ollama"

# Server host (needed for SLURM job access)
export OLLAMA_HOST="0.0.0.0:11434"

# Number of parallel requests
export OLLAMA_NUM_PARALLEL=1
```

### Why These Settings?

1. **OLLAMA_MODELS**:
   - Default is `~/.ollama` (home directory)
   - Home has only 30GB quota
   - Models need ~5GB+ each
   - **Solution**: Store in research space

2. **OLLAMA_HOST**:
   - Default is `127.0.0.1:11434` (localhost only)
   - SLURM jobs need network access
   - **Solution**: Bind to all interfaces

3. **OLLAMA_NUM_PARALLEL**:
   - Controls concurrent requests
   - Set to 1 for single-user SLURM jobs

---

## Usage

### Interactive Session

```bash
# 1. Get compute node with GPU
salloc --mem=16GB --gpus=1 --time=2:00:00

# 2. Load module
module load Ollama/0.15.5

# 3. Set environment
export OLLAMA_MODELS="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/models/ollama"
export OLLAMA_HOST="0.0.0.0:11434"

# 4. Start server
ollama serve &

# 5. Use Ollama
ollama run llama3.1:8b

# 6. When done, kill server
pkill ollama

# 7. Exit node
exit
```

### SLURM Job

```bash
# Submit job
sbatch jobs/run_ollama_qa.sb

# Monitor
squeue -u $USER
tail -f logs/ollama_qa_*.out
```

### Python API

```python
import requests
import json

def query_ollama(prompt, model="llama3.1:8b"):
    """Query Ollama API."""

    url = "http://localhost:11434/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    result = response.json()

    return result['response']

# Example usage
answer = query_ollama("What are the main bean breeding techniques?")
print(answer)
```

---

## Model Information

### LLaMA 3.1 8B

**Specifications**:
- **Parameters**: 8 billion
- **Context Length**: 128K tokens
- **Quantization**: Q4_0 (4-bit)
- **Size on Disk**: ~4.7GB
- **Memory Usage**: ~6-8GB during inference
- **Speed**: ~10-20 tokens/second on V100

**Capabilities**:
- Question answering
- Summarization
- Code generation
- Reasoning
- Multi-turn conversations

**Strengths**:
- Good balance of quality and speed
- Fits in V100 GPU memory
- Fast inference
- Strong reasoning abilities

---

## Common Commands

### Server Management

```bash
# Start server
ollama serve

# Start in background
nohup ollama serve > logs/ollama.log 2>&1 &

# Check if running
pgrep -a ollama

# Stop server
pkill ollama
```

### Model Management

```bash
# List downloaded models
ollama list

# Pull/download model
ollama pull llama3.1:8b

# Remove model
ollama rm llama3.1:8b

# Show model info
ollama show llama3.1:8b
```

### Running Models

```bash
# Interactive chat
ollama run llama3.1:8b

# Single prompt
ollama run llama3.1:8b "What is machine learning?"

# With parameters
ollama run llama3.1:8b "Explain photosynthesis" \
    --verbose \
    --temperature 0.7
```

---

## Integration with QA System

### Basic RAG Pipeline

```python
import chromadb
import requests
import json

# 1. Load vector store
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("bean_research_docs")

# 2. Search for relevant documents
def search_documents(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

# 3. Query Ollama with context
def qa_with_context(question):
    # Get relevant docs
    docs = search_documents(question)

    # Build prompt with context
    context = "\n\n".join(docs)
    prompt = f"""Based on the following research documents, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    # Query Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()['response']

# Use it
answer = qa_with_context("What are effective bean breeding techniques?")
print(answer)
```

---

## Troubleshooting

### Issue: Model Download Fails

**Symptoms**:
```
Error: failed to download model
```

**Solutions**:
1. Check internet connectivity:
   ```bash
   ping ollama.com
   ```
2. Try on different compute node
3. Check disk space:
   ```bash
   df -h /mnt/research/BeanLab/Parimal/BEAN_LLM
   ```

### Issue: Server Won't Start

**Symptoms**:
```
Error: listen tcp :11434: bind: address already in use
```

**Solutions**:
1. Kill existing server:
   ```bash
   pkill ollama
   ```
2. Check if port is in use:
   ```bash
   netstat -tuln | grep 11434
   ```

### Issue: Out of Memory

**Symptoms**:
```
Error: CUDA out of memory
```

**Solutions**:
1. Check GPU memory:
   ```bash
   nvidia-smi
   ```
2. Request more memory in SLURM:
   ```bash
   #SBATCH --mem=64GB
   ```
3. Use smaller model or reduce batch size

### Issue: Slow Inference

**Symptoms**: Responses take >30 seconds

**Solutions**:
1. Check GPU utilization:
   ```bash
   nvidia-smi dmon
   ```
2. Verify model loaded on GPU (not CPU)
3. Reduce context length in prompts

### Issue: Running on Login Node

**Symptoms**:
```
Warning: Do not run on login node
```

**Solutions**:
1. Get compute node:
   ```bash
   salloc --mem=16GB --gpus=1 --time=2:00:00
   ```
2. Or submit SLURM job:
   ```bash
   sbatch jobs/run_ollama_qa.sb
   ```

---

## Best Practices

### 1. Always Use Compute Nodes

❌ **Don't**: Run on login node
```bash
# Login node (WRONG!)
[kodumuru@dev-intel16]$ ollama serve
```

✓ **Do**: Use compute node
```bash
# Get compute node first
salloc --mem=16GB --gpus=1 --time=2:00:00

# Then run Ollama
[kodumuru@nod001]$ ollama serve
```

### 2. Store Models in Research Space

❌ **Don't**: Use home directory
```bash
export OLLAMA_MODELS=~/.ollama  # Only 30GB quota!
```

✓ **Do**: Use research space
```bash
export OLLAMA_MODELS="/mnt/research/BeanLab/Parimal/BEAN_LLM/hpcc-llm-qa/models/ollama"
```

### 3. Clean Up After Jobs

```bash
# At end of SLURM job
pkill ollama  # Stop server
```

### 4. Monitor Resource Usage

```bash
# GPU usage
watch -n 1 nvidia-smi

# Memory usage
htop

# Disk usage
du -sh models/ollama/
```

### 5. Use API for Production

✓ **Do**: Use API (can integrate with other tools)
```python
requests.post("http://localhost:11434/api/generate", ...)
```

❌ **Don't**: Only use CLI (harder to automate)
```bash
ollama run llama3.1:8b < input.txt
```

---

## Performance Optimization

### Prompt Engineering

**Good prompt**:
```
Based on the context below, answer concisely.

Context: [relevant excerpts]

Question: What are the main factors?

Answer:
```

**Bad prompt**:
```
tell me everything you know about beans including history, cultivation methods, genetics, breeding, diseases, pests, climate factors, soil requirements, and future prospects
```

### Context Management

- Limit context to ~2000-4000 tokens
- Use top 5-10 most relevant chunks
- Pre-filter by metadata when possible

### Batch Processing

```python
# Process multiple questions efficiently
questions = [q1, q2, q3, ...]

for question in questions:
    # Use same Ollama server instance
    answer = query_ollama(question)
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_ollama.sh` | Initial setup | Run once |
| `download_llama_model.sh` | Download model | Run on compute node |
| `start_ollama.sh` | Start server | Use in SLURM jobs |
| `test_ollama.sh` | Verify installation | Test after setup |

---

## Next Steps

1. **Test Ollama** (you are here)
2. **Integrate with vector store**
3. **Build QA pipeline**
4. **Create web interface**

---

## References

- **Ollama**: https://ollama.com
- **LLaMA 3.1**: https://ai.meta.com/blog/meta-llama-3-1/
- **MSU HPCC Docs**: https://docs.icer.msu.edu/
- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md

---

**Ready to use Ollama?**

```bash
# 1. Get compute node
salloc --mem=16GB --gpus=1 --time=2:00:00

# 2. Run setup
bash scripts/setup_ollama.sh

# 3. Download model
bash scripts/download_llama_model.sh

# 4. Test
bash scripts/test_ollama.sh
```
