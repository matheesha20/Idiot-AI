#!/bin/bash

# =============================================================================
# Sirimath Connect - macOS Setup Script
# =============================================================================
# This script will install all dependencies and set up the environment
# for the Multi-Model AI Assistant on macOS
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                    SIRIMATH CONNECT                           ║
║               macOS Setup Script v1.0                        ║
║                                                               ║
║  Multi-Model AI Assistant with Auto-Research Feature         ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "This script is designed for macOS only!"
    exit 1
fi

success "Running on macOS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check and install Homebrew
log "Checking for Homebrew..."
if ! command_exists brew; then
    warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    
    success "Homebrew installed successfully"
else
    success "Homebrew is already installed"
fi

# Update Homebrew
log "Updating Homebrew..."
brew update

# Check and install Python 3.8+
log "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]] || [[ $PYTHON_MAJOR -gt 3 ]]; then
        success "Python $PYTHON_VERSION is compatible"
    else
        warning "Python $PYTHON_VERSION is too old. Installing Python 3.11..."
        brew install python@3.11
        # Update symlinks
        brew link --force python@3.11
    fi
else
    warning "Python 3 not found. Installing Python 3.11..."
    brew install python@3.11
fi

# Verify Python installation
if ! command_exists python3; then
    error "Python 3 installation failed!"
    exit 1
fi

# Check and install pip
log "Checking pip installation..."
if ! command_exists pip3; then
    warning "pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install Node.js and npm (required for some dependencies)
log "Checking Node.js installation..."
if ! command_exists node; then
    warning "Node.js not found. Installing Node.js..."
    brew install node
else
    success "Node.js is already installed"
fi

# Create virtual environment
log "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Virtual environment created"
else
    success "Virtual environment already exists"
fi

# Activate virtual environment
log "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip in virtual environment
log "Upgrading pip in virtual environment..."
pip install --upgrade pip

# Install wheel and setuptools
log "Installing build tools..."
pip install wheel setuptools

# Install Python requirements
log "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    success "Python dependencies installed"
else
    error "requirements.txt not found!"
    exit 1
fi

# Install Playwright browsers
log "Installing Playwright browsers..."
python -m playwright install

# Install system dependencies for Playwright
log "Installing Playwright system dependencies..."
python -m playwright install-deps

# Create .env file if it doesn't exist
log "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Sirimath Connect Environment Configuration
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=$(openssl rand -hex 32)
FLASK_ENV=development
FLASK_DEBUG=True

# Optional: Customize these settings
MAX_CRAWL_PAGES=5
CRAWL_TIMEOUT=30
EOF
    
    warning "Created .env file. Please edit it and add your GEMINI_API_KEY!"
    success "Environment file created at .env"
else
    success "Environment file already exists"
fi

# Create start script
log "Creating start script..."
cat > start.sh << 'EOF'
#!/bin/bash

# Sirimath Connect - macOS Start Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Sirimath Connect...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_mac.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists and has API key
if [ ! -f ".env" ]; then
    echo "Environment file not found. Please run setup_mac.sh first."
    exit 1
fi

if grep -q "your_gemini_api_key_here" .env; then
    echo "⚠️  Please edit .env file and add your GEMINI_API_KEY"
    echo "You can get a free API key from: https://ai.google.dev/"
    exit 1
fi

# Start the application
echo -e "${GREEN}🚀 Starting Sirimath Connect on http://localhost:5000${NC}"
python app.py
EOF

chmod +x start.sh
success "Start script created (start.sh)"

# Create stop script
log "Creating stop script..."
cat > stop.sh << 'EOF'
#!/bin/bash

# Sirimath Connect - Stop Script

echo "🛑 Stopping Sirimath Connect..."

# Find and kill Python processes running app.py
pkill -f "python.*app.py"

echo "✅ Sirimath Connect stopped"
EOF

chmod +x stop.sh
success "Stop script created (stop.sh)"

# Create restart script
log "Creating restart script..."
cat > restart.sh << 'EOF'
#!/bin/bash

# Sirimath Connect - Restart Script

echo "🔄 Restarting Sirimath Connect..."

# Stop the application
./stop.sh

# Wait a moment
sleep 2

# Start the application
./start.sh
EOF

chmod +x restart.sh
success "Restart script created (restart.sh)"

# Run health check
log "Running health check..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import flask
    print(f'✅ Flask: {flask.__version__}')
except ImportError:
    print('❌ Flask not installed')

try:
    import google.generativeai
    print('✅ Google Generative AI imported successfully')
except ImportError:
    print('❌ Google Generative AI not installed')

try:
    import crawl4ai
    print('✅ Crawl4AI imported successfully')
except ImportError:
    print('❌ Crawl4AI not installed')

try:
    import playwright
    print('✅ Playwright imported successfully')
except ImportError:
    print('❌ Playwright not installed')

try:
    import faiss
    print('✅ FAISS imported successfully')
except ImportError:
    print('❌ FAISS not installed')
"

echo
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                    SETUP COMPLETED! 🎉                       ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Edit the .env file and add your GEMINI_API_KEY"
echo "   Get a free API key from: https://ai.google.dev/"
echo ""
echo "2. Start the application:"
echo "   ./start.sh"
echo ""
echo "3. Open your browser and go to:"
echo "   http://localhost:5000"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "• Start:   ./start.sh"
echo "• Stop:    ./stop.sh"
echo "• Restart: ./restart.sh"
echo ""
echo -e "${GREEN}Happy chatting with your AI assistant! 🤖${NC}"
