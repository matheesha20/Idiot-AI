# 🤖 Multi-Model AI Assistant with Auto-Research

## 🌟 What's New - Auto-Research Feature!

Your AI assistant now **automatically researches topics online** without needing any commands! Just ask about companies, current events, or any topic that needs fresh information, and the AI will:

- 🌐 **Automatically crawl relevant websites**
- 📊 **Gather real-time information**
- 🧠 **Integrate fresh data into responses**
- 💬 **Respond in your chosen AI personality**

No more `/crawl` commands needed - it's all automatic! 🚀

## 🚀 Quick Setup (Automated)

### Option 1: Automated Setup (Recommended)

1. **Download all files** to a new folder
2. **Run the setup script**:
   ```bash
   python setup.py
   ```
3. **Follow the prompts** (you'll need a Gemini API key)
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
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your-secret-key-change-this
```

#### Step 4: Run the Application
```bash
python app.py
```

## 🔑 Getting Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and paste it in your `.env` file

## 💬 AI Models Available

### 🟢 Chanuth (Standard)
- **Personality**: Normally aggressive with human-like attitude
- **Research**: Automatically looks up current info when needed
- **Best for**: General questions with some attitude

### 🔴 Amu Gawaya (Enhanced)
- **Personality**: Highly aggressive, uses slang and sarcasm
- **Research**: Researches stuff online with attitude
- **Best for**: When you want brutal honesty

### 🟣 Amu Gawaya Ultra Pro Max (Ultra)
- **Personality**: Maximum hostility with superior intelligence
- **Research**: Advanced web research with contemptuous responses
- **Best for**: Complex questions when you can handle the attitude

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

- **Python 3.8+**
- **4GB+ RAM** (for web crawling)
- **Internet connection** (for research)
- **Modern browser** (Chrome/Firefox/Safari)

## 📁 Project Structure

```
multi-model-ai/
├── app.py                 # Main application with auto-research
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
├── templates/
│   └── multi_model_chat.html  # Enhanced UI
├── .env                  # Your API keys (create this)
├── start.sh             # Linux/Mac launch script
├── start.bat            # Windows launch script
└── chatbot_users.db     # SQLite database (auto-created)
```

## 🎯 Key Features

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

## 🚨 Troubleshooting

### Common Issues

#### "Crawl4AI not available"
```bash
pip install crawl4ai
python -m playwright install
```

#### "Permission denied" on start.sh
```bash
chmod +x start.sh
```

#### Gemini API errors
- Check your API key in `.env`
- Ensure you have API quota remaining
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

#### Playwright installation issues
```bash
# Try these commands
python -m playwright install --with-deps
# Or on Ubuntu/Debian:
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libgtk-3-0
```

### Performance Tips

1. **First run may be slow** - Playwright needs to download browsers
2. **Research takes 2-5 seconds** - be patient for fresh data
3. **Rate limiting protects websites** - some queries may skip research
4. **Clear cache occasionally** by restarting the app

## 🔧 Advanced Configuration

### Environment Variables
```bash
# .env file options
GEMINI_API_KEY=your_key_here
FLASK_SECRET_KEY=your_secret_key

# Optional: Disable auto-research if needed
# (But why would you want to?)
```

### Customizing Research
Edit the `SmartWebCrawler` class in `app.py` to:
- Add more company URLs
- Modify crawl indicators
- Adjust rate limits
- Change content filtering

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
Amu Ultra: "The current AI landscape is dominated by predictable corporate positioning and incremental improvements to large language models. Recent developments include OpenAI's latest model iterations and Google's continued Gemini enhancements, though I doubt your limited comprehension can appreciate the technical complexities involved."
```

## 🔄 Updates & Maintenance

The system automatically:
- ✅ Caches web content to avoid spam
- ✅ Respects rate limits
- ✅ Updates knowledge base with fresh data
- ✅ Maintains conversation context

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify your Gemini API key is valid
4. Check that Playwright browsers are installed

## 🎊 Enjoy Your Enhanced AI Assistant!

Your AI now has access to the entire web and will automatically research topics to give you the most current information - all while maintaining their wonderfully aggressive personalities! 

Ask about companies, stocks, news, or anything that needs fresh data. The AI will handle the research automatically and respond in character.

**No commands needed - just ask and watch the magic happen!** ✨