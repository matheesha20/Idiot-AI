# 🚀 Quick Install Guide - TinyLlama AI Assistant

## ⚡ 5-Minute Setup

### Step 1: Download & Extract
```bash
# Download all files to a new folder
# Extract to: my-ai-assistant/
```

### Step 2: Auto Setup (Recommended)
```bash
cd my-ai-assistant
python setup.py
```
*Follow the prompts - no API keys needed!*

### Step 3: Launch
```bash
# Linux/Mac
./start.sh

# Windows
start.bat

# Manual
python app.py
```

### Step 4: Open Browser
```
http://localhost:5000
```

## 🧠 First Run (One-Time Setup)

**What happens on first run:**
1. TinyLlama downloads (~2GB) - be patient! ☕
2. Models load into memory (~1-2 minutes)
3. Web browser opens automatically
4. Start chatting with your local AI!

**After first run:**
- Everything loads in seconds
- Works completely offline (except web research)
- Zero API costs forever!

## 🔧 Manual Install (If Auto Setup Fails)

### Install Dependencies
```bash
# Core AI dependencies
pip install torch transformers sentence-transformers

# Web framework
pip install flask python-dotenv

# Vector database
pip install faiss-cpu numpy

# Web crawling (optional)
pip install crawl4ai
python -m playwright install
```

### Create .env File
```bash
# Create .env file
FLASK_SECRET_KEY=your-secret-key-change-this
TINYLLAMA_DEVICE=auto
```

### Run Application
```bash
python app.py
```

## 🎯 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4GB | 8GB+ |
| **Storage** | 3GB free | 5GB+ |
| **CPU** | Any modern CPU | Multi-core |
| **GPU** | None (CPU works) | NVIDIA with CUDA |
| **Internet** | For web research only | Broadband |

## 🚀 Performance Tips

### GPU Acceleration (Recommended)
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Optimization
```bash
# In .env file for lower-end hardware:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128
```

### SSD Storage
- Install on SSD for faster model loading
- First model download takes 5-10 minutes
- Subsequent loads are much faster

## 🆘 Quick Troubleshooting

### "TinyLlama not available"
```bash
pip install torch transformers sentence-transformers accelerate
```

### "Out of memory"
```bash
# Reduce model parameters in .env:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128
```

### "Slow responses"
- Use GPU if available
- Close other applications
- Upgrade to 8GB+ RAM

### "Web research not working"
```bash
pip install crawl4ai
python -m playwright install
```

### "Permission denied on start.sh"
```bash
chmod +x start.sh
```

## ✨ What You Get

### 🧠 Local AI Features
- **TinyLlama 1.1B** - Small but powerful
- **100% Private** - Data never leaves your machine
- **No API Keys** - Completely self-hosted
- **Fast Inference** - Especially with GPU

### 🌐 Web Research
- **Auto-crawling** - Researches topics automatically
- **Current Info** - Gets latest news, prices, etc.
- **Smart Detection** - Only crawls when needed
- **Rate Limited** - Respects websites

### 💬 AI Personalities
- **Chanuth** - Normally aggressive, helpful
- **Amu Gawaya** - Highly aggressive, sarcastic
- **Amu Ultra** - Maximum hostility, superior intellect

### 📱 Modern Interface
- **Mobile-friendly** - Works on phones/tablets
- **Multiple chats** - Organize conversations
- **Real-time research** - See crawling progress
- **Local indicators** - Know when AI is thinking

## 🎊 Success Indicators

**You know it's working when:**
- ✅ Console shows "TinyLlama loaded successfully"
- ✅ Browser opens to chat interface
- ✅ AI responds with personality in messages
- ✅ "🧠 Local" badge appears on AI messages
- ✅ Web research works for current topics

## 🔄 Migrating from Gemini Version?

**If you have the old Gemini-based version:**
```bash
python migrate_to_tinyllama.py
```

This will:
- Backup your existing data
- Remove API key requirements
- Preserve all your chats
- Update to TinyLlama

## 🆘 Get Help

**Check system health:**
```bash
python health_check.py
```

**Common issues:**
1. **RAM**: Close other apps, reduce model parameters
2. **Storage**: Free up 3GB+ space for models
3. **GPU**: Install CUDA version of PyTorch
4. **Network**: Only needed for initial download + research

## 🎉 Enjoy Your Local AI!

**You now have:**
- 🔒 Complete privacy
- ⚡ Fast local AI
- 🌐 Web research capability
- 💰 Zero ongoing costs
- 🚀 Offline functionality

**Start chatting and watch your TinyLlama research topics automatically!**