#!/usr/bin/env python3
"""
Automated Setup Script for Multi-Model AI Assistant with Crawl4AI
This script will set up everything you need to run the enhanced chatbot
"""

import os
import subprocess
import sys
from pathlib import Path

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

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    gemini_key = input("Enter your Gemini API key (get it from https://makersuite.google.com/app/apikey): ")
    
    env_content = f"""# Multi-Model AI Assistant Configuration
GEMINI_API_KEY={gemini_key}
FLASK_SECRET_KEY=your-super-secret-key-change-this-in-production

# Auto-research settings (crawl4ai is now integrated directly)
# No external API needed - crawl4ai runs locally
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

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    modules_to_test = [
        'flask',
        'google.generativeai',
        'numpy',
        'faiss',
        'crawl4ai',
        'requests',
        'sqlite3'
    ]
    
    failed_imports = []
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All modules imported successfully")
    return True

def create_launch_script():
    """Create a launch script for easy startup"""
    launch_content = """#!/bin/bash
# Launch script for Multi-Model AI Assistant

echo "🤖 Starting Multi-Model AI Assistant with Auto-Research..."
echo "🌐 Web crawling enabled - chatbot will automatically research topics!"
echo ""
echo "Open your browser to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
"""
    
    try:
        with open("start.sh", "w") as f:
            f.write(launch_content)
        os.chmod("start.sh", 0o755)
        print("✅ Created start.sh launch script")
        
        # Windows batch file
        win_launch_content = """@echo off
echo 🤖 Starting Multi-Model AI Assistant with Auto-Research...
echo 🌐 Web crawling enabled - chatbot will automatically research topics!
echo.
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

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
    print("🚀 Multi-Model AI Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements. Try running manually:")
        print("  pip install -r requirements.txt")
        return False
    
    # Install playwright browsers
    if not install_playwright():
        print("⚠️  Playwright setup incomplete, but continuing...")
    
    # Test imports
    if not test_imports():
        print("❌ Some modules failed to import. Check the error messages above.")
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Create launch scripts
    create_launch_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n" + "=" * 50)
    print("🤖 Multi-Model AI Assistant Ready!")
    print("=" * 50)
    print("\n✨ What's New:")
    print("  🌐 Automatic web research - no commands needed!")
    print("  🧠 AI automatically crawls relevant websites")
    print("  📊 Real-time information gathering")
    print("  💬 Three aggressive AI personalities")
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
    print("\nThe AI will automatically research online! 🔍")
    
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