@echo off
REM =============================================================================
REM Sirimath Connect - Windows Setup Script
REM =============================================================================
REM This script will install all dependencies and set up the environment
REM for the Multi-Model AI Assistant on Windows
REM =============================================================================

setlocal enabledelayedexpansion

REM Colors for output (Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Banner
echo %BLUE%
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                    SIRIMATH CONNECT                           ║
echo ║              Windows Setup Script v1.0                       ║
echo ║                                                               ║
echo ║  Multi-Model AI Assistant with Auto-Research Feature         ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo %NC%

echo %GREEN%✅ Starting Windows setup...%NC%

REM Check if running on Windows
ver >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %RED%❌ This script is designed for Windows only!%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ Running on Windows%NC%

REM Check for Python 3.8+
echo %BLUE%🔧 Checking Python installation...%NC%
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%⚠️  Python not found in PATH. Please install Python 3.8+ from https://python.org%NC%
    echo %YELLOW%Make sure to check "Add Python to PATH" during installation%NC%
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✅ Python %PYTHON_VERSION% found%NC%

REM Check Python version compatibility
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if %ERRORLEVEL% neq 0 (
    echo %RED%❌ Python 3.8 or higher is required. Current version: %PYTHON_VERSION%%NC%
    echo %YELLOW%Please install Python 3.8+ from https://python.org%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ Python version is compatible%NC%

REM Check for pip
echo %BLUE%🔧 Checking pip installation...%NC%
pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%⚠️  pip not found. Installing pip...%NC%
    python -m ensurepip --upgrade
)

echo %GREEN%✅ pip is available%NC%

REM Check for Git (optional but recommended)
echo %BLUE%🔧 Checking Git installation...%NC%
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%⚠️  Git not found. Git is recommended but not required.%NC%
    echo %YELLOW%You can download it from https://git-scm.com/download/win%NC%
) else (
    echo %GREEN%✅ Git is installed%NC%
)

REM Create virtual environment
echo %BLUE%🔧 Creating Python virtual environment...%NC%
if exist "venv" (
    echo %GREEN%✅ Virtual environment already exists%NC%
) else (
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo %RED%❌ Failed to create virtual environment%NC%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Virtual environment created%NC%
)

REM Activate virtual environment
echo %BLUE%🔧 Activating virtual environment...%NC%
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo %RED%❌ Failed to activate virtual environment%NC%
    pause
    exit /b 1
)

echo %GREEN%✅ Virtual environment activated%NC%

REM Upgrade pip in virtual environment
echo %BLUE%🔧 Upgrading pip in virtual environment...%NC%
python -m pip install --upgrade pip

REM Install wheel and setuptools
echo %BLUE%🔧 Installing build tools...%NC%
pip install wheel setuptools

REM Install Python requirements
echo %BLUE%🔧 Installing Python dependencies...%NC%
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo %RED%❌ Failed to install Python dependencies%NC%
        pause
        exit /b 1
    )
    echo %GREEN%✅ Python dependencies installed%NC%
) else (
    echo %RED%❌ requirements.txt not found!%NC%
    pause
    exit /b 1
)

REM Install Playwright browsers
echo %BLUE%🔧 Installing Playwright browsers...%NC%
python -m playwright install
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%⚠️  Playwright browser installation had issues, but continuing...%NC%
)

REM Install Playwright system dependencies (Windows)
echo %BLUE%🔧 Installing Playwright system dependencies...%NC%
python -m playwright install-deps
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%⚠️  Some Playwright dependencies may need manual installation%NC%
)

REM Create .env file if it doesn't exist
echo %BLUE%🔧 Setting up environment configuration...%NC%
if exist ".env" (
    echo %GREEN%✅ Environment file already exists%NC%
) else (
    (
    echo # Sirimath Connect Environment Configuration
    echo GEMINI_API_KEY=your_gemini_api_key_here
    echo FLASK_SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%
    echo FLASK_ENV=development
    echo FLASK_DEBUG=True
    echo.
    echo # Optional: Customize these settings
    echo MAX_CRAWL_PAGES=5
    echo CRAWL_TIMEOUT=30
    ) > .env
    
    echo %YELLOW%⚠️  Created .env file. Please edit it and add your GEMINI_API_KEY!%NC%
    echo %GREEN%✅ Environment file created at .env%NC%
)

REM Create start script
echo %BLUE%🔧 Creating start script...%NC%
(
echo @echo off
echo REM Sirimath Connect - Windows Start Script
echo.
echo echo 🚀 Starting Sirimath Connect...
echo.
echo REM Check if virtual environment exists
echo if not exist "venv" ^(
echo     echo Virtual environment not found. Please run setup_windows.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Check if .env file exists and has API key
echo if not exist ".env" ^(
echo     echo Environment file not found. Please run setup_windows.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo findstr /C:"your_gemini_api_key_here" .env ^>nul
echo if %%ERRORLEVEL%% equ 0 ^(
echo     echo ⚠️  Please edit .env file and add your GEMINI_API_KEY
echo     echo You can get a free API key from: https://ai.google.dev/
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Start the application
echo echo 🚀 Starting Sirimath Connect on http://localhost:5000
echo echo Press Ctrl+C to stop the server
echo python app.py
echo pause
) > start.bat

echo %GREEN%✅ Start script created (start.bat)%NC%

REM Create stop script
echo %BLUE%🔧 Creating stop script...%NC%
(
echo @echo off
echo REM Sirimath Connect - Stop Script
echo.
echo echo 🛑 Stopping Sirimath Connect...
echo.
echo REM Kill Python processes running app.py
echo taskkill /F /IM python.exe /FI "WINDOWTITLE eq Sirimath Connect*" 2^>nul
echo taskkill /F /IM python.exe /FI "COMMANDLINE eq *app.py*" 2^>nul
echo.
echo echo ✅ Sirimath Connect stopped
echo pause
) > stop.bat

echo %GREEN%✅ Stop script created (stop.bat)%NC%

REM Create restart script
echo %BLUE%🔧 Creating restart script...%NC%
(
echo @echo off
echo REM Sirimath Connect - Restart Script
echo.
echo echo 🔄 Restarting Sirimath Connect...
echo.
echo REM Stop the application
echo call stop.bat
echo.
echo REM Wait a moment
echo timeout /t 3 /nobreak ^>nul
echo.
echo REM Start the application
echo call start.bat
) > restart.bat

echo %GREEN%✅ Restart script created (restart.bat)%NC%

REM Run health check
echo %BLUE%🔧 Running health check...%NC%
python -c "
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

echo.
echo %GREEN%
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                    SETUP COMPLETED! 🎉                       ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo %NC%

echo %YELLOW%Next Steps:%NC%
echo 1. Edit the .env file and add your GEMINI_API_KEY
echo    Get a free API key from: https://ai.google.dev/
echo.
echo 2. Start the application:
echo    start.bat
echo.
echo 3. Open your browser and go to:
echo    http://localhost:5000
echo.
echo %YELLOW%Useful commands:%NC%
echo • Start:   start.bat
echo • Stop:    stop.bat
echo • Restart: restart.bat
echo.
echo %GREEN%Happy chatting with your AI assistant! 🤖%NC%

pause
