# ⚡ Performance Optimization Guide - TinyLlama AI

## 🎯 Performance Overview

TinyLlama can run on various hardware configurations. Here's how to optimize for your system:

| Hardware Type | Expected Performance | Optimization Level |
|---------------|---------------------|-------------------|
| **High-end GPU** | <1s response | Excellent |
| **Mid-range GPU** | 1-3s response | Very Good |
| **Modern CPU** | 3-8s response | Good |
| **Older CPU** | 8-15s response | Acceptable |

## 🚀 GPU Acceleration (Recommended)

### NVIDIA GPU Setup
```bash
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Usage
```python
# Run this test:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### GPU Memory Management
```bash
# In .env file:
TINYLLAMA_DEVICE=cuda
# OR auto-detect:
TINYLLAMA_DEVICE=auto
```

## 💾 Memory Optimization

### For 4GB RAM Systems
```bash
# .env settings for limited RAM:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128
TINYLLAMA_TEMPERATURE=0.6
```

### For 8GB+ RAM Systems
```bash
# .env settings for better performance:
TINYLLAMA_MAX_LENGTH=512
TINYLLAMA_MAX_NEW_TOKENS=256
TINYLLAMA_TEMPERATURE=0.7
```

### For 16GB+ RAM Systems
```bash
# .env settings for maximum quality:
TINYLLAMA_MAX_LENGTH=1024
TINYLLAMA_MAX_NEW_TOKENS=512
TINYLLAMA_TEMPERATURE=0.8
```

## 🔧 CPU Optimization

### Multi-threading
```bash
# Set CPU threads (in .env):
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

### CPU-specific PyTorch
```bash
# For Intel CPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For better Intel performance:
pip install intel-extension-for-pytorch
```

## 💿 Storage Optimization

### SSD vs HDD Performance
- **SSD**: Model loads in 10-30 seconds
- **HDD**: Model loads in 1-3 minutes

### Model Cache Location
```bash
# Move model cache to fastest drive:
# Edit app.py:
MODEL_CACHE_DIR = '/path/to/fast/ssd/models'
```

### Pre-download Models
```python
# Pre-download script (save as download_models.py):
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("Downloading TinyLlama...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="./models")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="./models")

print("Downloading embedding model...")
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

print("All models downloaded! First run will be faster.")
```

## ⚙️ Advanced Configuration

### Precision Optimization

#### For GPU Systems
```python
# In app.py, TinyLlamaManager class:
torch_dtype=torch.float16  # Use half precision for speed
```

#### For CPU Systems
```python
# In app.py, TinyLlamaManager class:
torch_dtype=torch.float32  # Use full precision for stability
```

### Batch Processing
```bash
# For handling multiple requests (advanced):
TINYLLAMA_BATCH_SIZE=1  # Start with 1, increase if you have extra RAM
```

### Temperature Tuning
```bash
# Creativity vs Speed tradeoff:
TINYLLAMA_TEMPERATURE=0.1  # Fast, repetitive responses
TINYLLAMA_TEMPERATURE=0.7  # Balanced (recommended)
TINYLLAMA_TEMPERATURE=1.0  # Creative, slower responses
```

## 🌐 Web Research Optimization

### Crawling Performance
```bash
# Limit concurrent crawls:
MAX_CONCURRENT_CRAWLS=2

# Adjust timeouts:
CRAWL_TIMEOUT=30

# Cache duration:
CRAWL_CACHE_HOURS=1
```

### Embedding Optimization
```bash
# Use faster embedding model:
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, good quality

# Or smaller model for speed:
EMBEDDING_MODEL=all-MiniLM-L12-v2  # Slower, better quality
```

## 📊 Performance Monitoring

### Built-in Status Check
```bash
# Check system performance:
python health_check.py
```

### Real-time Monitoring
```python
# Add to your environment:
import psutil
import GPUtil  # pip install gputil

# Monitor in real-time:
def monitor_performance():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"RAM: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.load*100:.1f}%")
        print(f"VRAM: {gpu.memoryUtil*100:.1f}%")
```

## 🔥 Performance Profiles

### Speed Profile (Fast responses)
```bash
# .env settings:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=64
TINYLLAMA_TEMPERATURE=0.5
TINYLLAMA_DEVICE=cuda
```

### Balanced Profile (Recommended)
```bash
# .env settings:
TINYLLAMA_MAX_LENGTH=512
TINYLLAMA_MAX_NEW_TOKENS=256
TINYLLAMA_TEMPERATURE=0.7
TINYLLAMA_DEVICE=auto
```

### Quality Profile (Best responses)
```bash
# .env settings:
TINYLLAMA_MAX_LENGTH=1024
TINYLLAMA_MAX_NEW_TOKENS=512
TINYLLAMA_TEMPERATURE=0.9
TINYLLAMA_DEVICE=cuda
```

## 🐛 Performance Troubleshooting

### Slow First Response
**Cause**: Model loading
**Solution**: 
```bash
# Pre-load models:
python download_models.py
```

### High Memory Usage
**Cause**: Large context/response settings
**Solution**:
```bash
# Reduce settings in .env:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128
```

### GPU Not Being Used
**Check**:
```bash
nvidia-smi  # Should show Python process
```
**Fix**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Slow Web Research
**Cause**: Network or crawling issues
**Solution**:
```bash
# Test crawl4ai directly:
python -c "from crawl4ai import AsyncWebCrawler; print('Crawl4AI OK')"
```

## 📈 Benchmarking

### Response Time Goals
- **GPU**: <2 seconds per response
- **Modern CPU**: <5 seconds per response
- **Older CPU**: <10 seconds per response

### Memory Usage Targets
- **Minimum**: 2-3GB RAM usage
- **Recommended**: 4-6GB RAM usage
- **Maximum**: 8GB+ for best quality

### First Load Time
- **With SSD**: 30-60 seconds
- **With HDD**: 2-5 minutes
- **Pre-downloaded**: 10-30 seconds

## 🎯 Quick Performance Check

Run this to test your setup:

```python
# save as perf_test.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("🧪 TinyLlama Performance Test")
print("=" * 40)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model (timed)
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
load_time = time.time() - start_time

print(f"Model Load Time: {load_time:.1f}s")

# Test inference (timed)
test_prompt = "Hello, how are you today?"
start_time = time.time()
inputs = tokenizer.encode(test_prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=50)
response_time = time.time() - start_time

print(f"Response Time: {response_time:.1f}s")

# Performance rating
if response_time < 2:
    rating = "Excellent ⭐⭐⭐⭐⭐"
elif response_time < 5:
    rating = "Good ⭐⭐⭐⭐"
elif response_time < 10:
    rating = "Acceptable ⭐⭐⭐"
else:
    rating = "Needs optimization ⭐⭐"

print(f"Performance Rating: {rating}")
```

## 🎉 Optimization Complete!

Your TinyLlama should now be running at optimal performance for your hardware. Remember:

- 🚀 **GPU acceleration** provides the biggest speed boost
- 💾 **SSD storage** helps with model loading
- ⚙️ **Tuned parameters** balance speed vs quality
- 🌐 **Web research** adds slight delay but provides current info

Enjoy your blazing-fast local AI assistant! 🔥