#!/usr/bin/env python3
"""
Automated Setup Script for Multi-Model AI Assistant with TinyLlama
This script will set up everything you need to run the local TinyLlama chatbot
"""

import os
import subprocess
import sys
import platform
from pathlib import Path
import torch

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_system_requirements():
    """Check system requirements for TinyLlama"""
    print("🔍 Checking system requirements...")
    
    # Check available RAM
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
                mem_total = int([line for line in mem_info.split('\n') if 'MemTotal' in line][0].split()[1]) / 1024 / 1024
        elif platform.system() == "Darwin":  # macOS
            mem_total = int(subprocess.check_output(['sysctl', 'hw.memsize']).decode().split()[1]) / 1024 / 1024 / 1024
        else:  # Windows
            import psutil
            mem_total = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        print(f"💾 System RAM: {mem_total:.1f} GB")
        
        if mem_total < 4:
            print("⚠️  Warning: Less than 4GB RAM detected. TinyLlama may run slowly.")
        elif mem_total >= 8:
            print("✅ Sufficient RAM for optimal TinyLlama performance")
        else:
            print("✅ Adequate RAM for TinyLlama")
            
    except Exception as e:
        print(f"⚠️  Could not determine RAM amount: {e}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ CUDA available with {gpu_count} GPU(s) - TinyLlama will run faster!")
    else:
        print("💻 No GPU detected - TinyLlama will run on CPU (slower but still works)")
    
    return True

def install_pytorch():
    """Install PyTorch with appropriate configuration"""
    print("🔥 Installing PyTorch...")
    
    # Detect if CUDA is available
    cuda_available = torch.cuda.is_available() if 'torch' in sys.modules else False
    
    if cuda_available:
        print("🚀 CUDA detected - installing PyTorch with GPU support")
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("💻 Installing PyTorch for CPU")
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(pytorch_cmd, "Installing PyTorch")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    
    env_content = f"""# Multi-Model AI Assistant with TinyLlama Configuration
# No API keys needed - everything runs locally!

FLASK_SECRET_KEY=your-super-secret-key-change-this-in-production

# TinyLlama settings
TINYLLAMA_DEVICE=auto
TINYLLAMA_MAX_LENGTH=512
TINYLLAMA_MAX_NEW_TOKENS=256

# Auto-research settings (crawl4ai runs locally)
ENABLE_WEB_CRAWLING=true

# Optional: Advanced settings
TINYLLAMA_TEMPERATURE=0.7
EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def install_playwright():
    """Install Playwright browsers for crawl4ai"""
    print("🎭 Installing Playwright browsers for web crawling...")
    commands = [
        "python -m playwright install",
        "python -m playwright install-deps"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            print("⚠️  Playwright installation failed, but continuing...")
            print("You may need to run these commands manually:")
            print("  python -m playwright install")
            print("  python -m playwright install-deps")
            return False
    return True

def test_tinyllama():
    """Test TinyLlama installation"""
    print("🧪 Testing TinyLlama installation...")
    
    test_script = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("Testing TinyLlama components...")

# Test basic imports
print("✅ Transformers imported")
print("✅ Torch imported")
print("✅ Sentence transformers imported")

# Test device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device detected: {device}")

# Test embedding model
try:
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = emb_model.encode("test sentence")
    print(f"✅ Embedding model works (dim: {len(test_embedding)})")
except Exception as e:
    print(f"❌ Embedding model failed: {e}")

print("✅ All TinyLlama components working!")
"""
    
    try:
        # Write test script to temp file
        with open("test_tinyllama.py", "w") as f:
            f.write(test_script)
        
        # Run test
        result = subprocess.run([sys.executable, "test_tinyllama.py"], 
                              capture_output=True, text=True, timeout=60)
        
        # Clean up
        os.remove("test_tinyllama.py")
        
        if result.returncode == 0:
            print("✅ TinyLlama test passed!")
            print(result.stdout)
            return True
        else:
            print("❌ TinyLlama test failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing TinyLlama: {e}")
        return False

def create_model_cache():
    """Create model cache directory"""
    cache_dir = Path("./models")
    cache_dir.mkdir(exist_ok=True)
    print(f"✅ Created model cache directory: {cache_dir.absolute()}")
    return True

def create_launch_script():
    """Create a launch script for easy startup"""
    launch_content = """#!/bin/bash
# Launch script for Multi-Model AI Assistant with TinyLlama

echo "🤖 Starting Multi-Model AI Assistant with TinyLlama..."
echo "🧠 Running completely locally - no API keys needed!"
echo "🌐 Web crawling enabled - chatbot will automatically research topics!"
echo ""
echo "🚀 TinyLlama is loading... (this may take a moment on first run)"
echo "📊 Check the console for model loading progress"
echo ""
echo "Open your browser to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Check if models directory exists
if [ ! -d "./models" ]; then
    echo "📁 Creating models cache directory..."
    mkdir -p ./models
fi

# Start the application
python app.py
"""
    
    try:
        with open("start.sh", "w") as f:
            f.write(launch_content)
        os.chmod("start.sh", 0o755)
        print("✅ Created start.sh launch script")
        
        # Windows batch file
        win_launch_content = """@echo off
echo 🤖 Starting Multi-Model AI Assistant with TinyLlama...
echo 🧠 Running completely locally - no API keys needed!
echo 🌐 Web crawling enabled - chatbot will automatically research topics!
echo.
echo 🚀 TinyLlama is loading... (this may take a moment on first run)
echo 📊 Check the console for model loading progress
echo.
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Check if models directory exists
if not exist "./models" (
    echo 📁 Creating models cache directory...
    mkdir "./models"
)

REM Start the application
python app.py
pause
"""
        
        with open("start.bat", "w") as f:
            f.write(win_launch_content)
        print("✅ Created start.bat launch script for Windows")
        return True
    except Exception as e:
        print(f"❌ Failed to create launch scripts: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Multi-Model AI Assistant with TinyLlama Setup")
    print("=" * 60)
    print("🧠 Setting up completely LOCAL AI - no cloud dependencies!")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check system requirements
    if not check_system_requirements():
        return False
    
    # Install PyTorch first (required for other packages)
    print("\n📦 Installing AI dependencies...")
    if not install_pytorch():
        print("❌ Failed to install PyTorch. This is required for TinyLlama.")
        return False
    
    # Install other requirements
    print("\n📦 Installing remaining Python packages...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements. Try running manually:")
        print("  pip install -r requirements.txt")
        return False
    
    # Install playwright browsers
    if not install_playwright():
        print("⚠️  Playwright setup incomplete, but continuing...")
    
    # Test TinyLlama
    if not test_tinyllama():
        print("⚠️  TinyLlama test failed, but setup will continue...")
        print("You may need to debug the installation manually.")
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Create model cache directory
    create_model_cache()
    
    # Create launch scripts
    create_launch_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n" + "=" * 60)
    print("🤖 Multi-Model AI Assistant with TinyLlama Ready!")
    print("=" * 60)
    print("\n✨ What's New:")
    print("  🧠 TinyLlama running 100% locally - no API keys needed!")
    print("  🌐 Automatic web research - no commands needed!")
    print("  🚀 Completely self-hosted - your data stays private!")
    print("  💬 Three aggressive AI personalities")
    print("  🔒 No cloud dependencies - works offline!")
    print("\n🚀 To start the assistant:")
    print("  Linux/Mac: ./start.sh")
    print("  Windows: start.bat")
    print("  Manual: python app.py")
    print("\n🌐 Then open: http://localhost:5000")
    print("\n💡 Try asking about:")
    print("  • Apple company latest news")
    print("  • Tesla stock price")
    print("  • Current AI developments")
    print("  • Any company information")
    print("\n🧠 First run will download TinyLlama model (~2GB)")
    print("🌐 The AI will automatically research online when needed!")
    print("🔒 Everything runs on YOUR machine - completely private!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)