#!/bin/bash

# =============================================================================
# Sirimath Connect - Linux Setup Script
# =============================================================================
# This script will install all dependencies and set up the environment
# for the Multi-Model AI Assistant on Linux
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
║                Linux Setup Script v1.0                       ║
║                                                               ║
║  Multi-Model AI Assistant with Auto-Research Feature         ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
else
    DISTRO="unknown"
fi

success "Detected Linux distribution: $DISTRO"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages based on distribution
install_packages() {
    case $DISTRO in
        ubuntu|debian)
            log "Updating package lists..."
            sudo apt update
            log "Installing system dependencies..."
            sudo apt install -y python3 python3-pip python3-venv python3-dev \
                build-essential libssl-dev libffi-dev libnss3-dev \
                libgconf-2-4 libxrandr2 libasound2 libpangocairo-1.0-0 \
                libatk1.0-0 libcairo-gobject2 libgtk-3-0 libgdk-pixbuf2.0-0 \
                libxss1 libglib2.0-0 libxcomposite1 libxdamage1 libxext6 \
                libxfixes3 libxi6 libxrender1 libxtst6 libxrandr2 \
                libasound2 libpangocairo-1.0-0 libcups2 \
                curl wget git nodejs npm
            ;;
        fedora|rhel|centos)
            log "Installing system dependencies..."
            if command_exists dnf; then
                sudo dnf install -y python3 python3-pip python3-devel \
                    gcc gcc-c++ make openssl-devel libffi-devel \
                    nss libXrandr alsa-lib pango atk cairo-gobject3 \
                    gtk3 gdk-pixbuf2 libXScrnSaver glib2 libXcomposite \
                    libXdamage libXext libXfixes libXi libXrender libXtst \
                    cups-libs curl wget git nodejs npm
            else
                sudo yum install -y python3 python3-pip python3-devel \
                    gcc gcc-c++ make openssl-devel libffi-devel \
                    curl wget git
            fi
            ;;
        arch|manjaro)
            log "Installing system dependencies..."
            sudo pacman -Sy --noconfirm python python-pip base-devel \
                openssl libffi nss libxrandr alsa-lib pango atk \
                cairo gtk3 gdk-pixbuf2 libxss glib2 libxcomposite \
                libxdamage libxext libxfixes libxi libxrender libxtst \
                libxrandr alsa-lib cups curl wget git nodejs npm
            ;;
        opensuse*)
            log "Installing system dependencies..."
            sudo zypper install -y python3 python3-pip python3-devel \
                gcc gcc-c++ make libopenssl-devel libffi-devel \
                curl wget git nodejs npm
            ;;
        *)
            warning "Unsupported distribution: $DISTRO"
            warning "Please install Python 3.8+, pip, and development tools manually"
            ;;
    esac
}

# Install system dependencies
log "Installing system dependencies for $DISTRO..."
install_packages

# Check Python version
log "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]] || [[ $PYTHON_MAJOR -gt 3 ]]; then
        success "Python $PYTHON_VERSION is compatible"
    else
        error "Python $PYTHON_VERSION is too old. Please install Python 3.8+"
        exit 1
    fi
else
    error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
log "Checking pip installation..."
if ! command_exists pip3; then
    warning "pip3 not found. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install or upgrade pip
python3 -m pip install --user --upgrade pip

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
FLASK_SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
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

# Sirimath Connect - Linux Start Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Starting Sirimath Connect...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run setup_linux.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists and has API key
if [ ! -f ".env" ]; then
    echo -e "${RED}Environment file not found. Please run setup_linux.sh first.${NC}"
    exit 1
fi

if grep -q "your_gemini_api_key_here" .env; then
    echo -e "${YELLOW}⚠️  Please edit .env file and add your GEMINI_API_KEY${NC}"
    echo -e "${YELLOW}You can get a free API key from: https://ai.google.dev/${NC}"
    exit 1
fi

# Start the application
echo -e "${GREEN}🚀 Starting Sirimath Connect on http://localhost:5000${NC}"
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
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
pkill -f "python.*app.py" 2>/dev/null || true

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

# Create systemd service file (optional)
log "Creating systemd service file..."
cat > sirimath-connect.service << EOF
[Unit]
Description=Sirimath Connect - Multi-Model AI Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python $(pwd)/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

success "Systemd service file created (sirimath-connect.service)"
echo "To install as a system service, run:"
echo "  sudo cp sirimath-connect.service /etc/systemd/system/"
echo "  sudo systemctl enable sirimath-connect"
echo "  sudo systemctl start sirimath-connect"

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
echo -e "${YELLOW}Optional - Install as system service:${NC}"
echo "• sudo cp sirimath-connect.service /etc/systemd/system/"
echo "• sudo systemctl enable sirimath-connect"
echo "• sudo systemctl start sirimath-connect"
echo ""
echo -e "${GREEN}Happy chatting with your AI assistant! 🤖${NC}"
