# 🤖 Sirimath AI Assistant with TinyLlama + Auto-Research

## 🌟 What's New - Completely LOCAL AI!

Your AI assistant now runs **100% locally with TinyLlama** - no API keys needed! Plus **automatic web research**:

- 🧠 **TinyLlama running locally** - completely private and self-hosted
- 🌐 **Automatically crawls relevant websites**
- 📊 **Gathers real-time information**
- 🔒 **No cloud dependencies** - your data never leaves your machine
- 💬 **Three unique AI personalities** to choose from
- ⚡ **Fast inference** on CPU or GPU
- 🛡️ **Robust fallback system** for stability

No more API keys, no more cloud dependency - everything runs on YOUR machine! 🚀

## 🚀 Quick Setup (Automated)

### Option 1: Automated Setup (Recommended)

1. **Download all files** to a new folder
2. **Run the setup script**:
   ```bash
   python setup.py
   ```
3. **Follow the prompts** (no API keys needed!)
4. **Start the assistant**:
   ```bash
   ./start.sh    # Linux/Mac
   start.bat     # Windows
   ```

### Option 2: Manual Setup

If you prefer to set up manually:

#### Step 1: Install Python Requirements
```bash
pip install -r requirements.txt
```

#### Step 2: Install Playwright (for web crawling)
```bash
python -m playwright install
python -m playwright install-deps
```

#### Step 3: Create Environment File
Create a `.env` file with:
```
FLASK_SECRET_KEY=your-secret-key-change-this
# No API keys needed - TinyLlama runs locally!
```

#### Step 4: Run the Application
```bash
python app.py
```

### Option 3: Test Mode (If TinyLlama Fails)
```bash
# Run simple test version first
python test_app.py
```

## 🧠 TinyLlama - Your Local AI Brain

### What is TinyLlama?
- **1.1B parameter language model** - small but powerful
- **Runs on modest hardware** - even laptops work fine
- **No internet required** after initial download
- **Complete privacy** - your conversations stay local
- **Fast inference** - especially with GPU acceleration
- **Fallback mode** - graceful degradation if resources are limited

### System Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB RAM, dedicated GPU
- **Storage**: ~2GB for model download
- **Internet**: Only needed for initial model download + web research

## 💬 AI Models Available

### 🟢 F-1 (Standard)
- **Personality**: Normal humanized AI with friendly conversational style
- **Tone**: Warm, approachable, naturally helpful
- **Research**: Automatically looks up current info when needed
- **Best for**: General conversations and friendly assistance
- **Response Style**: Natural human speech, engaging and conversational

### 🔷 F-1.5 (Enhanced)
- **Personality**: Professional AI assistant focused on accuracy and efficiency
- **Tone**: Business-like but approachable, clear and precise
- **Research**: Methodical research with structured responses
- **Best for**: Professional tasks, detailed information, work-related queries
- **Response Style**: Well-structured, professional, thorough

### 🟣 F-o1 (Ultra High)
- **Personality**: Research-focused AI with analytical approach
- **Tone**: Analytical, thorough, academic-like
- **Research**: Comprehensive investigation with cross-referencing
- **Best for**: Complex research, analysis, in-depth exploration
- **Response Style**: Systematic, comprehensive, methodical

## 🌐 Auto-Research Examples

The AI will automatically research when you ask about:

### 💼 Companies
- "Tell me about Apple company"
- "What's Tesla doing lately?"
- "Microsoft recent developments"

### 📈 Financial Information
- "Apple stock price"
- "Tesla market performance" 
- "Tech stocks today"

### 📰 Current Events
- "Latest AI news"
- "Recent tech developments"
- "Current market trends"

### 🔍 General Information
- "What is [any company/topic]"
- "Latest updates on [anything]"
- "Current information about [topic]"

## 🛠️ System Requirements

### Minimum Requirements
- **Python 3.8+**
- **4GB+ RAM**
- **2GB free storage** (for model)
- **Internet connection** (for research only)

### Recommended for Best Performance
- **8GB+ RAM**
- **NVIDIA GPU** with CUDA support
- **SSD storage** for faster model loading
- **Modern CPU** (4+ cores)

### GPU Acceleration (Optional but Recommended)
```bash
# For NVIDIA GPUs with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# The app will automatically detect and use GPU if available
```

## 📁 Project Structure

```
sirimath-ai/
├── app.py                     # Main Sirimath application
├── test_app.py               # Simple test version
├── requirements.txt           # Python dependencies
├── setup.py                  # Automated setup script
├── health_check.py           # System health checker
├── templates/
│   └── multi_model_chat.html  # Enhanced UI
├── models/                   # TinyLlama model cache (auto-created)
├── .env                      # Your configuration (create this)
├── start.sh                 # Linux/Mac launch script
├── start.bat                # Windows launch script
└── chatbot_users.db         # SQLite database (auto-created)
```

## 🎯 Key Features

### 🧠 Local TinyLlama AI
- **Completely private** - no data sent to cloud
- **Fast inference** on CPU or GPU
- **No API keys** required
- **Works offline** (except for web research)
- **Fallback mode** for limited-resource systems

### 🌐 Intelligent Auto-Research
- **Smart Detection**: Automatically detects when queries need current info
- **Multi-Source**: Crawls multiple relevant websites
- **Rate Limited**: Respects website resources
- **Cached Results**: Avoids duplicate crawls

### 🤖 Multiple AI Personalities
- **Three distinct models** with different approaches
- **Context-aware responses** using fresh web data
- **Conversation memory** across chat sessions
- **Emotion detection** for better responses

### 💾 Advanced Features
- **Persistent chat history**
- **User preferences**
- **Mobile-responsive design**
- **Real-time research indicators**
- **Local embeddings** for semantic search
- **Robust error handling**

## 🚨 Troubleshooting

### Common Issues

#### Segmentation Fault on Startup
```bash
# Try the test version first:
python test_app.py

# If test works, the issue is TinyLlama loading
# Check available RAM and try:
export OMP_NUM_THREADS=1
python app.py
```

#### "TinyLlama dependencies not available"
```bash
pip install torch transformers sentence-transformers
# For GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### "Crawl4AI not available"
```bash
pip install crawl4ai
python -m playwright install
```

#### "Permission denied" on start.sh
```bash
chmod +x start.sh
```

#### First run is slow
- **Normal behavior** - TinyLlama downloads ~2GB model on first run
- **Subsequent runs** are much faster
- **Use GPU** for best performance

#### Out of memory errors
```bash
# Reduce model parameters in .env file:
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128
```

### Performance Tips

1. **First run downloads model** - be patient (~2GB download)
2. **Use GPU if available** - much faster inference
3. **Close other applications** to free RAM
4. **SSD storage** improves model loading speed
5. **Research takes 2-5 seconds** - be patient for fresh data
6. **Fallback mode available** if full model won't load

## 🔧 Advanced Configuration

### Environment Variables
```bash
# .env file options
FLASK_SECRET_KEY=your_secret_key

# TinyLlama settings
TINYLLAMA_DEVICE=auto                    # auto, cpu, cuda
TINYLLAMA_MAX_LENGTH=256                 # Input context length (conservative)
TINYLLAMA_MAX_NEW_TOKENS=128            # Response length (conservative)
TINYLLAMA_TEMPERATURE=0.7               # Creativity (0.1-1.0)

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2        # Local embedding model

# Web crawling
ENABLE_WEB_CRAWLING=true                # Enable/disable research
```

### Hardware Optimization
```python
# For systems with limited RAM (4GB)
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128

# For systems with good RAM (8GB+)
TINYLLAMA_MAX_LENGTH=512
TINYLLAMA_MAX_NEW_TOKENS=256

# For powerful systems (16GB+)
TINYLLAMA_MAX_LENGTH=1024
TINYLLAMA_MAX_NEW_TOKENS=512
```

## 🎉 Usage Examples

### Friendly Conversation (F-1)
```
You: "Tell me about Apple company"
F-1: "Hello! Apple Inc. is a major technology company known for designing and manufacturing consumer electronics like iPhones, iPads, and MacBooks. They're currently one of the world's most valuable companies and are headquartered in Cupertino, California. Would you like me to look up their latest news or stock information?"
```

### Professional Query (F-1.5)
```
You: "Tesla stock performance"
F-1.5: "I'll gather the current Tesla stock information for you. Tesla (TSLA) is currently trading around $240 per share. The stock has experienced volatility this quarter due to various market factors including EV competition and production updates. Would you like a more detailed analysis of their recent performance metrics?"
```

### Research Analysis (F-o1)
```
You: "Latest AI developments"
F-o1: "Conducting comprehensive analysis of current AI developments... Recent significant advances include improvements in large language models, multimodal AI systems, and AI safety research. Key developments include OpenAI's latest model iterations, Google's Gemini advances, and increased focus on AI alignment research. This represents a continued acceleration in the field with both technical and regulatory implications."
```

## 🔄 Updates & Maintenance

The system automatically:
- ✅ Caches web content to avoid spam
- ✅ Respects rate limits
- ✅ Updates knowledge base with fresh data
- ✅ Maintains conversation context
- ✅ Stores models locally for offline use
- ✅ Gracefully handles system limitations
- ✅ Provides fallback responses when needed

## 📞 Support

If you encounter issues:
1. **Try test mode first**: `python test_app.py`
2. **Run health check**: `python health_check.py`
3. Check the troubleshooting section above
4. Ensure all dependencies are installed
5. Verify sufficient RAM/storage available
6. Check that Playwright browsers are installed

### Health Check Commands
```bash
# Check system compatibility
python health_check.py

# Test basic functionality
python test_app.py

# Check model loading
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## 🎊 Enjoy Your LOCAL AI Assistant!

Sirimath now runs **completely on your machine** with TinyLlama and will automatically research topics to give you the most current information - all with three distinct personalities to match your needs!

Ask about companies, stocks, news, or anything that needs fresh data. The AI will handle the research automatically and respond according to their personality.

**Key Benefits:**
- 🔒 **Complete Privacy** - your data never leaves your machine
- ⚡ **Fast Response** - no network latency for AI inference
- 🌐 **Current Info** - automatic web research when needed
- 💰 **No Costs** - no API fees or subscriptions
- 🚀 **Always Available** - works offline after initial setup
- 🛡️ **Reliable** - fallback modes ensure it always works
- 🎭 **Personalized** - choose the AI personality that fits your needs

**Three Personalities for Every Need:**
- 🟢 **F-1**: Your friendly, conversational companion
- 🔷 **F-1.5**: Your professional, efficient assistant
- 🟣 **F-o1**: Your thorough, research-focused analyst

**No API keys needed - just ask and watch the magic happen!** ✨
