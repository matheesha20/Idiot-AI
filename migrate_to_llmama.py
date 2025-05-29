#!/usr/bin/env python3
"""
Migration Script: Gemini to TinyLlama
Helps users migrate from the Gemini-based version to the local TinyLlama version
"""

import os
import shutil
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import json

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def backup_existing_data():
    """Backup existing database and configuration"""
    print_header("📦 BACKING UP EXISTING DATA")
    
    backup_dir = Path("./backup_gemini")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "chatbot_users.db",
        ".env",
        "app.py"
    ]
    
    backed_up = []
    
    for file in files_to_backup:
        if Path(file).exists():
            try:
                shutil.copy2(file, backup_dir / file)
                print(f"✅ Backed up: {file}")
                backed_up.append(file)
            except Exception as e:
                print(f"⚠️  Failed to backup {file}: {e}")
    
    if backed_up:
        print(f"\n🎯 Backup created in: {backup_dir.absolute()}")
        print("Your existing data is safe!")
        return True
    else:
        print("ℹ️  No existing data found to backup")
        return False

def check_gemini_version():
    """Check if current version uses Gemini"""
    print_header("🔍 CHECKING CURRENT VERSION")
    
    gemini_indicators = []
    
    # Check .env for Gemini API key
    if Path(".env").exists():
        with open(".env", "r") as f:
            env_content = f.read()
            if "GEMINI_API_KEY" in env_content:
                gemini_indicators.append("Found GEMINI_API_KEY in .env")
    
    # Check app.py for Gemini imports
    if Path("app.py").exists():
        with open("app.py", "r", encoding="utf-8") as f:
            app_content = f.read()
            if "google.generativeai" in app_content:
                gemini_indicators.append("Found google.generativeai import in app.py")
            if "genai.configure" in app_content:
                gemini_indicators.append("Found genai.configure in app.py")
    
    if gemini_indicators:
        print("🔍 Detected Gemini-based version:")
        for indicator in gemini_indicators:
            print(f"  • {indicator}")
        return True
    else:
        print("ℹ️  No Gemini version detected - you may already have TinyLlama!")
        return False

def migrate_environment():
    """Migrate .env file from Gemini to TinyLlama"""
    print_header("🔧 MIGRATING CONFIGURATION")
    
    old_env = {}
    
    # Read existing .env if it exists
    if Path(".env").exists():
        load_dotenv()
        old_env = dict(os.environ)
        print("✅ Read existing .env file")
    
    # Create new .env for TinyLlama
    new_env_content = """# Multi-Model AI Assistant with TinyLlama Configuration
# No API keys needed - everything runs locally!

FLASK_SECRET_KEY={}

# TinyLlama settings
TINYLLAMA_DEVICE=auto
TINYLLAMA_MAX_LENGTH=512
TINYLLAMA_MAX_NEW_TOKENS=256
TINYLLAMA_TEMPERATURE=0.7

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Auto-research settings (crawl4ai runs locally)
ENABLE_WEB_CRAWLING=true

# Database settings
DATABASE_PATH=chatbot_users.db

# Advanced settings (optional)
# TINYLLAMA_MAX_LENGTH=1024  # For more powerful hardware
# TINYLLAMA_MAX_NEW_TOKENS=512  # For longer responses
""".format(old_env.get('FLASK_SECRET_KEY', 'your-super-secret-key-change-this-in-production'))
    
    try:
        with open(".env", "w") as f:
            f.write(new_env_content)
        print("✅ Created new TinyLlama .env configuration")
        
        if 'GEMINI_API_KEY' in old_env:
            print("ℹ️  Removed GEMINI_API_KEY (no longer needed)")
            print("🎉 Your AI is now completely local - no API keys required!")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create new .env: {e}")
        return False

def migrate_database():
    """Ensure database is compatible with TinyLlama version"""
    print_header("💾 CHECKING DATABASE COMPATIBILITY")
    
    db_path = "chatbot_users.db"
    
    if not Path(db_path).exists():
        print("ℹ️  No existing database found - will be created on first run")
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if we need to add TinyLlama-specific columns
        migrations_needed = []
        
        # Check users table for preferred_model column
        cursor.execute("PRAGMA table_info(users)")
        users_columns = [row[1] for row in cursor.fetchall()]
        
        if 'preferred_model' not in users_columns:
            migrations_needed.append("Add preferred_model to users table")
        
        # Check chats table for model_id column
        cursor.execute("PRAGMA table_info(chats)")
        chats_columns = [row[1] for row in cursor.fetchall()]
        
        if 'model_id' not in chats_columns:
            migrations_needed.append("Add model_id to chats table")
        
        # Apply migrations if needed
        if migrations_needed:
            print("🔧 Applying database migrations:")
            for migration in migrations_needed:
                print(f"  • {migration}")
            
            if 'preferred_model' not in users_columns:
                cursor.execute('ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT "chanuth"')
            
            if 'model_id' not in chats_columns:
                cursor.execute('ALTER TABLE chats ADD COLUMN model_id TEXT DEFAULT "chanuth"')
            
            conn.commit()
            print("✅ Database migrations completed")
        else:
            print("✅ Database is already compatible")
        
        # Count existing data
        cursor.execute("SELECT COUNT(*) FROM chats")
        chat_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        if chat_count > 0 or message_count > 0:
            print(f"📊 Found existing data: {chat_count} chats, {message_count} messages")
            print("🎉 All your existing conversations will work with TinyLlama!")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        return False

def install_tinyllama_dependencies():
    """Install TinyLlama dependencies"""
    print_header("📦 INSTALLING TINYLLAMA DEPENDENCIES")
    
    import subprocess
    import sys
    
    dependencies = [
        "torch",
        "transformers",
        "sentence-transformers",
        "accelerate"
    ]
    
    print("Installing TinyLlama dependencies...")
    print("This may take several minutes on first installation.")
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )
            
            if result.returncode == 0:
                print(f"✅ {dep} installed successfully")
            else:
                print(f"⚠️  {dep} installation had issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  {dep} installation timed out")
        except Exception as e:
            print(f"⚠️  {dep} installation failed: {e}")
    
    # Test installations
    print("\n🧪 Testing installations...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 CUDA: Available ({torch.cuda.device_count()} GPU(s))")
        else:
            print("💻 CUDA: Not available (will use CPU)")
    except ImportError:
        print("❌ PyTorch: Not installed")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: Not installed")
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers: Available")
    except ImportError:
        print("❌ Sentence Transformers: Not installed")
    
    return True

def create_model_cache():
    """Create model cache directory"""
    print_header("📁 SETTING UP MODEL CACHE")
    
    cache_dir = Path("./models")
    cache_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created model cache directory: {cache_dir.absolute()}")
    print("📝 Models will be downloaded here on first run (~2GB)")
    
    return True

def update_launch_scripts():
    """Update launch scripts for TinyLlama"""
    print_header("🚀 UPDATING LAUNCH SCRIPTS")
    
    # Linux/Mac script
    linux_script = """#!/bin/bash
# Launch script for Multi-Model AI Assistant with TinyLlama

echo "🤖 Starting Multi-Model AI Assistant with TinyLlama..."
echo "🧠 Running 100% locally - no API keys needed!"
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
    
    # Windows script
    windows_script = """@echo off
echo 🤖 Starting Multi-Model AI Assistant with TinyLlama...
echo 🧠 Running 100% locally - no API keys needed!
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
    
    try:
        with open("start.sh", "w") as f:
            f.write(linux_script)
        os.chmod("start.sh", 0o755)
        print("✅ Updated start.sh")
        
        with open("start.bat", "w") as f:
            f.write(windows_script)
        print("✅ Updated start.bat")
        
        return True
    except Exception as e:
        print(f"❌ Failed to update launch scripts: {e}")
        return False

def show_migration_summary():
    """Show migration summary and next steps"""
    print_header("🎉 MIGRATION COMPLETE!")
    
    print("✨ Successfully migrated to TinyLlama!")
    print("\n🔄 What Changed:")
    print("  ❌ Removed: Gemini API dependency")
    print("  ✅ Added: TinyLlama local AI model")
    print("  ✅ Added: Local embeddings (sentence-transformers)")
    print("  ✅ Added: Enhanced privacy (everything runs locally)")
    print("  ✅ Added: No API costs or rate limits")
    
    print("\n🎯 Benefits of TinyLlama:")
    print("  🔒 Complete Privacy - your data never leaves your machine")
    print("  ⚡ Fast Response - no network latency for AI inference")
    print("  💰 Zero Costs - no API fees or subscriptions")
    print("  🌐 Still has auto-research - gets current web info when needed")
    print("  🚀 Always Available - works offline after initial setup")
    
    print("\n🚀 Next Steps:")
    print("  1. Run: python app.py")
    print("  2. Open: http://localhost:5000")
    print("  3. First run will download TinyLlama model (~2GB)")
    print("  4. After download, everything runs instantly!")
    
    print("\n💡 Tips:")
    print("  • First model download takes 5-10 minutes")
    print("  • Use GPU if available for faster inference")
    print("  • All your existing chats are preserved")
    print("  • Same aggressive AI personalities, now running locally!")
    
    print("\n🆘 If you need to go back:")
    print(f"  Your Gemini version is backed up in: {Path('./backup_gemini').absolute()}")

def main():
    """Main migration function"""
    print("🔄 MIGRATION: Gemini → TinyLlama")
    print("=" * 60)
    print("🧠 Migrating to 100% local AI - no more API dependencies!")
    print("=" * 60)
    
    # Step 1: Check current version
    if not check_gemini_version():
        choice = input("\nDo you want to proceed with TinyLlama setup anyway? (y/n): ")
        if choice.lower() != 'y':
            print("Migration cancelled.")
            return
    
    # Step 2: Backup existing data
    backup_created = backup_existing_data()
    
    # Step 3: Migrate configuration
    if not migrate_environment():
        print("❌ Configuration migration failed")
        return
    
    # Step 4: Migrate database
    if not migrate_database():
        print("❌ Database migration failed")
        return
    
    # Step 5: Install dependencies
    print("\n🤔 Do you want to install TinyLlama dependencies now?")
    print("(This will install PyTorch, Transformers, and other AI libraries)")
    install_deps = input("Install dependencies? (y/n): ").lower() == 'y'
    
    if install_deps:
        install_tinyllama_dependencies()
    else:
        print("⚠️  Skipped dependency installation")
        print("Run this later: pip install torch transformers sentence-transformers")
    
    # Step 6: Create model cache
    create_model_cache()
    
    # Step 7: Update launch scripts
    update_launch_scripts()
    
    # Step 8: Show summary
    show_migration_summary()
    
    print("\n🎊 Migration Complete!")
    print("Your AI assistant is now running 100% locally with TinyLlama!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Migration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during migration: {e}")
        print("Check the error above and try again.")