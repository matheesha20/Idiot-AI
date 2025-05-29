# 🤖 Multi-Model AI Assistant with TinyLlama + Auto-Research

## 🌟 What's New - Completely LOCAL AI!

Your AI assistant now runs **100% locally with TinyLlama** - no API keys needed! Plus **automatic web research**:

- 🧠 **TinyLlama running locally** - completely private and self-hosted
- 🌐 **Automatically crawls relevant websites**
- 📊 **Gathers real-time information**
- 🔒 **No cloud dependencies** - your data never leaves your machine
- 💬 **Responds in your chosen AI personality**
- ⚡ **Fast inference** on CPU or GPU

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

## 🧠 TinyLlama - Your Local AI Brain

### What is TinyLlama?
- **1.1B parameter language model** - small but powerful
- **Runs on modest hardware** - even laptops work fine
- **No internet required** after initial download
- **Complete privacy** - your conversations stay local
- **Fast inference** - especially with GPU acceleration

### System Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB RAM, dedicated GPU
- **Storage**: ~2GB for model download
- **Internet**: Only needed for initial model download + web research

## 💬 AI Models Available

### 🟢 Chanuth (Standard)
- **Personality**: Normally aggressive with human-like attitude
- **Research**: Automatically looks up current info when needed
- **Best for**: General questions with some attitude
- **Powered by**: TinyLlama running locally

### 🔴 Amu Gawaya (Enhanced)
- **Personality**: Highly aggressive, uses slang and sarcasm
- **Research**: Researches stuff online with attitude
- **Best for**: When you want brutal honesty
- **Powered by**: TinyLlama with enhanced prompting

### 🟣 Amu Gawaya Ultra Pro Max (Ultra)
- **Personality**: Maximum hostility with superior intelligence
- **Research**: Advanced web research with contemptuous responses
- **Best for**: Complex questions when you can handle the attitude
- **Powered by**: TinyLlama with sophisticated prompt engineering

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
multi-model-ai/
├── app.py                     # Main TinyLlama application
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

### 🌐 Intelligent Auto-Research
- **Smart Detection**: Automatically detects when queries need current info
- **Multi-Source**: Crawls multiple relevant websites
- **Rate Limited**: Respects website resources
- **Cached Results**: Avoids duplicate crawls

### 🤖 Multiple AI Personalities
- **Three distinct models** with different aggression levels
- **Context-aware responses** using fresh web data
- **Conversation memory** across chat sessions
- **Emotion detection** for better responses

### 💾 Advanced Features
- **Persistent chat history**
- **User preferences**
- **Mobile-responsive design**
- **Real-time research indicators**
- **Local embeddings** for semantic search

## 🚨 Troubleshooting

### Common Issues

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

## 🔧 Advanced Configuration

### Environment Variables
```bash
# .env file options
FLASK_SECRET_KEY=your_secret_key

# TinyLlama settings
TINYLLAMA_DEVICE=auto                    # auto, cpu, cuda
TINYLLAMA_MAX_LENGTH=512                 # Input context length
TINYLLAMA_MAX_NEW_TOKENS=256            # Response length
TINYLLAMA_TEMPERATURE=0.7               # Creativity (0.1-1.0)

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2        # Local embedding model

# Web crawling
ENABLE_WEB_CRAWLING=true                # Enable/disable research
```

### Hardware Optimization
```python
# For systems with limited RAM
TINYLLAMA_MAX_LENGTH=256
TINYLLAMA_MAX_NEW_TOKENS=128

# For powerful systems
TINYLLAMA_MAX_LENGTH=1024
TINYLLAMA_MAX_NEW_TOKENS=512
```

## 🎉 Usage Examples

### Basic Conversation
```
You: "Tell me about Apple company"
Chanuth: "Ugh, fine. Apple's doing their usual thing - they just announced some new iPhone nonsense and their stock is around $180. They're still making ridiculous amounts of money selling overpriced gadgets to people who think they need the latest shiny thing."
```

### Financial Query
```
You: "Tesla stock price"
Amu Gawaya: "tesla's at like $240 or whatever, down from yesterday because musk probably tweeted something stupid again 🙄 why don't you just check your own portfolio app?"
```

### Technical Information
```
You: "Latest AI developments"
Amu Ultra: "The current AI landscape is dominated by predictable corporate positioning and incremental improvements to large language models. Recent developments include OpenAI's latest model iterations and continued open-source advancement, though I doubt your limited comprehension can appreciate the technical complexities involved."
```

## 🔄 Updates & Maintenance

The system automatically:
- ✅ Caches web content to avoid spam
- ✅ Respects rate limits
- ✅ Updates knowledge base with fresh data
- ✅ Maintains conversation context
- ✅ Stores models locally for offline use

## 📞 Support

If you encounter issues:
1. Run the health check: `python health_check.py`
2. Check the troubleshooting section above
3. Ensure all dependencies are installed
4. Verify sufficient RAM/storage available
5. Check that Playwright browsers are installed

## 🎊 Enjoy Your LOCAL AI Assistant!

Your AI now runs **completely on your machine** with TinyLlama and will automatically research topics to give you the most current information - all while maintaining their wonderfully aggressive personalities! 

Ask about companies, stocks, news, or anything that needs fresh data. The AI will handle the research automatically and respond in character.

**Key Benefits:**
- 🔒 **Complete Privacy** - your data never leaves your machine
- ⚡ **Fast Response** - no network latency for AI inference
- 🌐 **Current Info** - automatic web research when needed
- 💰 **No Costs** - no API fees or subscriptions
- 🚀 **Always Available** - works offline after initial setup

**No API keys needed - just ask and watch the magic happen!** ✨