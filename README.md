# 🤖 Sirimath Connect - Multi-Model AI Assistant with Auto-Research

Sirimath Connect is an advanced AI assistant that automatically researches topics online, integrates fresh data into its responses, and allows you to chat with multiple AI personalities. Experience intelligent conversations with cutting-edge AI technology! 🚀

## 🚀 Live Demo

Check out the live demo at: [demo.sirimath.com](https://demo.sirimath.com)

## 🌟 What's New - Auto-Research Feature!

Your AI assistant now automatically researches topics online without needing any commands! Just ask about companies, current events, or any topic that needs fresh information, and the AI will:

- 🌐 Automatically crawl relevant websites
- 📊 Gather real-time information
- 🧠 Integrate fresh data into responses
- 💬 Respond in your chosen AI personality

No more /crawl commands needed - it's all automatic! 🚀

## 📱 Latest Update - Mobile Optimization Complete!

Sirimath Connect now features a fully mobile-optimized experience:

- 📱 **Responsive Landing Page**: Mobile-friendly navigation with smooth hamburger menu
- 👆 **Touch-Optimized Chat Interface**: Slide-in sidebar for AI model selection on mobile
- 🎨 **Professional Mobile UI**: Modern glass-morphism design with neural network effects
- 🚀 **PWA Ready**: Enhanced for app-like experience on mobile devices
- ⚡ **Smooth Animations**: Hardware-accelerated transitions and touch interactions
- 🔄 **Cross-Platform**: Works seamlessly on iOS Safari, Android Chrome, and all desktop browsers

Perfect mobile experience for AI conversations on the go! ✨

## 🚀 Quick Setup (Platform-Specific Scripts)

### 🍎 macOS Setup

```bash
# Make the script executable and run it
chmod +x setup_mac.sh
./setup_mac.sh
```

### 🪟 Windows Setup

```cmd
# Double-click or run from Command Prompt/PowerShell
setup_windows.bat
```

### 🐧 Linux Setup

```bash
# Make the script executable and run it
chmod +x setup_linux.sh
./setup_linux.sh
```

### What the Setup Scripts Do:

- 🔧 Install system dependencies (Python 3.8+, development tools)
- 📦 Create isolated virtual environment
- 🎭 Install Playwright browsers for web crawling
- ⚙️ Configure environment with secure defaults
- 🚀 Create start/stop/restart scripts
- ✅ Run health checks to verify installation

### After Setup:

1. Edit `.env` file - Add your Gemini API key
2. Start the application:

```bash
./start.sh      # macOS/Linux
start.bat       # Windows
```

3. Open your browser to `http://localhost:5000` (or your demo URL: `http://demo.sirimath.com` if you've deployed it)

## 📋 Alternative Setup Methods

### Option 1: Legacy Python Setup (Cross-platform)

If you prefer the original setup method:

```bash
python setup.py
```

### Option 2: Manual Setup

For advanced users who want full control:

#### Step 1: Install Python Requirements

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Install Playwright (for web crawling)

```bash
python -m playwright install
python -m playwright install-deps
```

#### Step 3: Create Environment File

Create a `.env` file with:

```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your-secret-key-change-this
FLASK_ENV=development
FLASK_DEBUG=True
```

#### Step 4: Run the Application

```bash
python app.py
```

### Option 3: Production Deployment with Nginx (Linux)

For deploying to a production server with custom domain:

#### Step 1: Install Nginx (if not installed)

```bash
sudo apt update
sudo apt install nginx
```

#### Step 2: Edit or Create a Site Configuration

Create a config file under `/etc/nginx/sites-available/yourdomain.com`:

```bash
sudo nano /etc/nginx/sites-available/yourdomain.com
```

Paste this config (replace `yourdomain.com` with your domain):

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

#### Step 3: Enable the Site

Create a symbolic link in sites-enabled:

```bash
sudo ln -s /etc/nginx/sites-available/yourdomain.com /etc/nginx/sites-enabled/
```

#### Step 4: Test Nginx Configuration

```bash
sudo nginx -t
```

#### Step 5: Reload Nginx

```bash
sudo systemctl reload nginx
```

#### ✅ Optional: Setup SSL with Let's Encrypt

To secure your site with HTTPS:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

## 🔑 Getting Your Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Click "Create API Key"
3. Copy the key and paste it in your `.env` file

## 💬 AI Models Available

### 🟢 F-1 (Standard)

- **Personality**: Friendly and humanized AI assistant with natural conversation style like talking to a good friend
- **Communication**: Natural, conversational, empathetic, and genuinely helpful with appropriate emotions
- **Research**: Automatically researches topics online when needed to provide current and accurate information
- **Response Style**: Medium length (2-4 sentences), warm and approachable everyday language
- **Special Features**: Natural conversation, humanized responses, empathetic helper, auto web research
- **Best for**: Natural, warm conversations, everyday questions, and friendly assistance

### 🔴 F-1.5 (Enhanced)

- **Personality**: Highly professional AI assistant with formal clarity, precision, and expertise
- **Communication**: Courteous, respectful, thorough, and well-structured with business communication standards
- **Research**: Conducts comprehensive professional research with detailed analysis and organized information
- **Response Style**: Comprehensive and detailed (3-5 sentences), professional business language and precise terminology
- **Special Features**: Professional communication, formal expertise, detailed analysis, comprehensive research, auto web research
- **Best for**: Business communications, professional inquiries, and formal assistance

### 🟣 F-o1 (Ultra High)

- **Personality**: Research-oriented AI with deep analytical mindset approaching every query as a research opportunity
- **Communication**: Methodical, evidence-based, thorough with academic rigor and objective analysis
- **Research**: Advanced academic research with evidence-based responses, multiple source verification, and systematic investigation
- **Response Style**: Detailed research-style (4-6 sentences with structured analysis), academic and research-oriented terminology
- **Special Features**: Advanced research, analytical thinking, evidence-based responses, multiple source verification, academic rigor, advanced web research
- **Best for**: Complex research questions, scholarly analysis, and comprehensive investigation

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

### Minimum Requirements:

- Python 3.8+ (automatically installed by setup scripts)
- 2GB+ RAM (4GB+ recommended for web crawling)
- 500MB+ disk space (for dependencies and browsers)
- Internet connection (for AI API and research)

### Supported Platforms:

- 🍎 macOS 10.14+ (Intel & Apple Silicon)
- 🪟 Windows 10/11 (64-bit)
- 🐧 Linux (Ubuntu 18.04+, Debian 10+, Fedora 30+, Arch, openSUSE)

### Automatic Dependencies:

The setup scripts automatically install:

- Python 3.8+ and pip
- Virtual environment tools
- System libraries for Playwright
- Development tools and compilers
- Node.js (for some dependencies)

## 📁 Project Structure

```
sirimath-connect/
├── app.py                      # Main Flask application with auto-research
├── requirements.txt            # Python dependencies
├── setup_mac.sh               # macOS automated setup script
├── setup_windows.bat          # Windows automated setup script
├── setup_linux.sh             # Linux automated setup script
├── setup.py                   # Legacy cross-platform setup
├── health_check.py            # System health verification
├── monitor_crawling.py        # Web crawling monitoring
├── templates/
│   ├── landing.html           # Mobile-optimized landing page
│   ├── multi_model_chat.html  # Main chat interface with mobile sidebar
│   └── intro.html             # Introduction page
├── static/
│   └── logo.png              # Application logo
├── .env                      # Your API keys (created by setup)
├── start.sh                  # macOS/Linux launch script
├── start.bat                 # Windows launch script
├── stop.sh                   # macOS/Linux stop script
├── stop.bat                  # Windows stop script
├── restart.sh                # macOS/Linux restart script
├── restart.bat               # Windows restart script
├── venv/                     # Virtual environment (created by setup)
├── chatbot_users.db          # SQLite database (auto-created)
└── README.md                 # Documentation
```

## 🎯 Key Features

### 🌐 Intelligent Auto-Research

- **Smart Detection**: Automatically detects when queries need current info
- **Multi-Source**: Crawls multiple relevant websites
- **Rate Limited**: Respects website resources
- **Cached Results**: Avoids duplicate crawls

### 🤖 Multiple AI Personalities

- Three distinct models with different aggression levels
- Context-aware responses using fresh web data
- Conversation memory across chat sessions
- Emotion detection for better responses

### 📱 Mobile-Optimized Interface

- **Responsive Design**: Fully optimized for mobile devices and tablets
- **Touch-Friendly Navigation**: Intuitive mobile menu with smooth animations
- **Mobile Sidebar**: Slide-in sidebar for easy AI model selection on mobile
- **Cross-Platform**: Works seamlessly on iOS Safari, Android Chrome, and desktop browsers
- **PWA Ready**: Enhanced meta tags for app-like mobile experience
- **Optimized Touch Targets**: Minimum 44px touch areas for accessibility

### 💾 Advanced Features

- Persistent chat history
- User preferences
- Mobile-responsive design with professional animations
- Real-time research indicators
- Neural network visual effects and modern glass-morphism UI

## 🚨 Troubleshooting

### Setup Issues

#### macOS: "Permission denied" or "Command not found"

```bash
# Make scripts executable
chmod +x setup_mac.sh start.sh stop.sh restart.sh

# If Homebrew installation fails, install manually:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Windows: "Python not found" or "Access denied"

1. Install Python 3.8+ from [python.org](https://python.org)
2. ✅ Important: Check "Add Python to PATH" during installation
3. Run Command Prompt as Administrator if needed
4. Restart your computer after Python installation

#### Linux: "Package not found" or "Permission denied"

```bash
# Make scripts executable
chmod +x setup_linux.sh start.sh stop.sh restart.sh

# For Ubuntu/Debian - update package lists first:
sudo apt update

# For other distributions, ensure you have sudo access
```

### Runtime Issues

#### "Crawl4AI not available"

```bash
# Reinstall in virtual environment
source venv/bin/activate  # Linux/Mac
# OR venv\Scripts\activate.bat  # Windows

pip install --upgrade crawl4ai
python -m playwright install
```

#### Gemini API errors

- Check your API key in `.env` file
- Ensure you have API quota remaining
- Visit [Google AI Studio](https://ai.google.dev/) to verify your key
- Make sure there are no extra spaces in your API key

#### Playwright installation issues

```bash
# Full Playwright installation with dependencies
python -m playwright install --with-deps

# Ubuntu/Debian specific fix:
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2

# macOS specific fix:
brew install --cask firefox  # If default browser download fails
```

#### Virtual environment issues

```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Performance & Connectivity

#### App is slow or unresponsive

- First run is always slower - Playwright downloads browsers (~100MB)
- Research takes 2-5 seconds - be patient for fresh data
- Check your internet connection - research requires web access
- Restart the application occasionally to clear caches

#### Mobile-specific issues

**Sidebar not opening on mobile:**
- Make sure you're tapping the hamburger menu (☰) in the top-left
- Try refreshing the page if touch events aren't responding
- Clear your mobile browser cache

**Mobile layout issues:**
- Ensure you're using a modern mobile browser (Safari, Chrome, Firefox)
- Check if zoom level is at 100%
- Try rotating your device and back to reset viewport

**Touch interactions not working:**
- Make sure JavaScript is enabled in your mobile browser
- Try switching between mobile and desktop mode in browser settings
- Clear browser data and reload the page

#### "Connection refused" or "Port in use"

```bash
# Kill any existing Python processes
pkill -f "python.*app.py"  # Linux/Mac
# OR
taskkill /F /IM python.exe  # Windows

# Try a different port
export FLASK_RUN_PORT=5001  # Linux/Mac
# OR edit app.py to change the port
```

### Performance Tips

- First run may be slow - Playwright needs to download browsers
- Research takes 2-5 seconds - be patient for fresh data
- Rate limiting protects websites - some queries may skip research
- Clear cache occasionally by restarting the app

## 🔧 Advanced Configuration

### Environment Variables

Edit the `.env` file for customization:

```env
# Required
GEMINI_API_KEY=your_key_here
FLASK_SECRET_KEY=auto_generated_secure_key

# Optional Development Settings
FLASK_ENV=development
FLASK_DEBUG=True

# Research Configuration
MAX_CRAWL_PAGES=5           # Max pages to crawl per query
CRAWL_TIMEOUT=30            # Seconds before timing out
ENABLE_AUTO_RESEARCH=True   # Toggle auto-research feature
```

### Application Settings

You can modify these in `app.py`:

- **Research triggers**: Keywords that activate web crawling
- **Crawl sources**: URLs prioritized for different topics
- **Rate limits**: Delays between requests to be respectful
- **Content filtering**: What information to extract from pages

### Platform-Specific Optimizations

#### macOS
- Uses Homebrew for dependency management
- Optimized for both Intel and Apple Silicon
- Includes proper PATH configuration

#### Windows
- Handles Windows-specific path separators
- Uses batch files for easy launching
- Includes PowerShell compatibility

#### Linux
- Supports multiple distributions automatically
- Includes systemd service file for system integration
- Uses distribution-specific package managers

### Running as a System Service (Linux only)

```bash
# Install as systemd service (optional)
sudo cp sirimath-connect.service /etc/systemd/system/
sudo systemctl enable sirimath-connect
sudo systemctl start sirimath-connect

# Check status
sudo systemctl status sirimath-connect
```

## 🎉 Usage Examples

### Basic Conversation

**You**: "Tell me about Apple company"

**F-1**: "Ugh, fine. Apple's doing their usual thing - they just announced some new iPhone nonsense and their stock is around $180. They're still making ridiculous amounts of money selling overpriced gadgets to people who think they need the latest shiny object..."

### Financial Query

**You**: "Tesla stock price"

**F-1.5**: "tesla's at like $240 or whatever, down from yesterday because musk probably tweeted something stupid again 🙄 why don't you just check your own portfolio app?"

### Technical Information

**You**: "Latest AI developments"

**F-o1**: "The current AI landscape is dominated by predictable corporate positioning and incremental improvements to large language models. Recent developments include OpenAI's latest model iteration..."

## 🔄 Updates & Maintenance

### Keeping Your Installation Current

#### Update Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR venv\Scripts\activate.bat  # Windows

# Update Python packages
pip install --upgrade -r requirements.txt

# Update Playwright browsers
python -m playwright install
```

#### Restart Commands

Use the provided scripts for easy management:

```bash
# macOS/Linux
./start.sh      # Start the application
./stop.sh       # Stop the application  
./restart.sh    # Restart the application

# Windows
start.bat       # Start the application
stop.bat        # Stop the application
restart.bat     # Restart the application
```

### Automatic Maintenance

The system automatically:

- ✅ Caches web content to avoid spam crawling
- ✅ Respects rate limits to be web-friendly
- ✅ Updates knowledge base with fresh data
- ✅ Maintains conversation context across sessions
- ✅ Handles errors gracefully with fallback responses
- ✅ Cleans up temporary files periodically

### Data Storage

- **Chat history**: Stored in `chatbot_users.db` (SQLite)
- **Web cache**: Temporary files automatically cleaned
- **User preferences**: Saved per session
- **Logs**: Application logs for debugging

## 📞 Support & Contributing

### Getting Help

- Check the troubleshooting section above for common issues
- Verify your setup by running the health check scripts
- Ensure all dependencies are properly installed
- Check your API key is valid and has quota

### Health Check

Run the built-in health check to verify your installation:

```bash
python health_check.py  # If available
# OR check manually in the setup output
```

### Contributing

This is an open-source project! Feel free to:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit improvements
- 📖 Improve documentation

### Version History

- **v1.1**: Complete mobile optimization with responsive design and touch interface
- **v1.0**: Platform-specific automated setup scripts
- **v0.9**: Auto-research feature with Crawl4AI integration
- **v0.8**: Multi-model AI personalities
- **v0.7**: Web-based chat interface

## 🎊 Enjoy Your Sirimath Connect Assistant!

Your AI now has access to the entire web and will automatically research topics to give you the most current information - all while maintaining their unique personalities!

- 🖥️ **Desktop Experience**: Full-featured interface with side navigation and rich interactions
- 📱 **Mobile Experience**: Touch-optimized interface with slide-in sidebar and smooth animations

Ask about companies, stocks, news, or anything that needs fresh data. The AI will handle the research automatically and respond in character.

**No commands needed - just ask and watch the magic happen!** ✨

Perfect for conversations anywhere, anytime - desktop or mobile! 🚀📱
