#!/usr/bin/env python3
"""
Health Check Script for Multi-Model AI Assistant
Verifies all components are working correctly
"""

import os
import sys
import sqlite3
from pathlib import Path
import requests
from dotenv import load_dotenv

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def check_files():
    """Check if all required files exist"""
    print_header("📁 FILE SYSTEM CHECK")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'templates/multi_model_chat.html',
        '.env'
    ]
    
    all_good = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    return all_good

def check_environment():
    """Check environment variables"""
    print_header("🔧 ENVIRONMENT CHECK")
    
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    secret_key = os.getenv('FLASK_SECRET_KEY')
    
    if api_key:
        print(f"✅ GEMINI_API_KEY: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    else:
        print("❌ GEMINI_API_KEY: Missing")
    
    if secret_key:
        print("✅ FLASK_SECRET_KEY: Set")
    else:
        print("❌ FLASK_SECRET_KEY: Missing")
    
    return bool(api_key and secret_key)

def check_python_packages():
    """Check if all required packages are installed"""
    print_header("📦 PYTHON PACKAGES CHECK")
    
    packages = [
        'flask',
        'google-generativeai',
        'numpy',
        'faiss-cpu',
        'requests',
        'python-dotenv',
        'crawl4ai'
    ]
    
    all_installed = True
    for package in packages:
        try:
            if package == 'google-generativeai':
                import google.generativeai
                print(f"✅ {package}")
            elif package == 'faiss-cpu':
                import faiss
                print(f"✅ {package}")
            elif package == 'python-dotenv':
                import dotenv
                print(f"✅ {package}")
            else:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_crawl4ai():
    """Check Crawl4AI and Playwright setup"""
    print_header("🌐 WEB CRAWLING CHECK")
    
    try:
        from crawl4ai import WebCrawler
        print("✅ Crawl4AI: Imported successfully")
        
        # Try to initialize crawler
        crawler = WebCrawler()
        print("✅ WebCrawler: Initialized")
        
        # Test warmup
        try:
            crawler.warmup()
            print("✅ Crawler Warmup: Success")
            return True
        except Exception as e:
            print(f"⚠️  Crawler Warmup: Failed ({str(e)[:50]}...)")
            print("   This might work anyway, but you may need to run:")
            print("   python -m playwright install")
            return True
            
    except ImportError as e:
        print(f"❌ Crawl4AI: Not available ({e})")
        print("   Install with: pip install crawl4ai")
        return False
    except Exception as e:
        print(f"❌ Crawl4AI: Error ({e})")
        return False

def check_gemini_api():
    """Test Gemini API connectivity"""
    print_header("🤖 GEMINI API CHECK")
    
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No API key found")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'API test successful'")
        
        if response and response.text:
            print("✅ Gemini API: Connection successful")
            print(f"   Response: {response.text.strip()}")
            return True
        else:
            print("❌ Gemini API: No response received")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API: Error - {e}")
        print("   Check your API key at: https://makersuite.google.com/app/apikey")
        return False

def check_database():
    """Check database functionality"""
    print_header("💾 DATABASE CHECK")
    
    try:
        # Test SQLite connection
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('SELECT sqlite_version()')
        version = cursor.fetchone()[0]
        conn.close()
        print(f"✅ SQLite: Version {version}")
        
        # Check if main database exists
        if Path('chatbot_users.db').exists():
            print("✅ User Database: Exists")
        else:
            print("ℹ️  User Database: Will be created on first run")
        
        return True
        
    except Exception as e:
        print(f"❌ Database: Error - {e}")
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    print_header("🧪 FUNCTIONALITY TEST")
    
    try:
        # Test imports from main app
        sys.path.insert(0, '.')
        
        print("Testing core components...")
        
        # Test database manager
        from app import DatabaseManager
        db = DatabaseManager(':memory:')
        print("✅ DatabaseManager: OK")
        
        # Test emotion detector
        from app import EmotionDetector
        emotion_detector = EmotionDetector()
        result = emotion_detector.detect_emotion("I'm happy!")
        print(f"✅ EmotionDetector: OK (detected: {result.emotion})")
        
        # Test model manager
        from app import AIModelManager
        model_manager = AIModelManager()
        models = model_manager.get_all_models()
        print(f"✅ AIModelManager: OK ({len(models)} models)")
        
        # Test web crawler
        from app import SmartWebCrawler
        crawler = SmartWebCrawler()
        should_crawl = crawler.should_crawl_for_query("Tell me about Apple company")
        print(f"✅ SmartWebCrawler: OK (would crawl: {should_crawl})")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality Test: Error - {e}")
        return False

def run_full_health_check():
    """Run complete health check"""
    print("🔍 MULTI-MODEL AI ASSISTANT HEALTH CHECK")
    print("🤖 Verifying all components are ready...")
    
    checks = [
        ("Files", check_files),
        ("Environment", check_environment),
        ("Python Packages", check_python_packages),
        ("Crawl4AI", check_crawl4ai),
        ("Gemini API", check_gemini_api),
        ("Database", check_database),
        ("Functionality", test_basic_functionality)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name}: Unexpected error - {e}")
            results[name] = False
    
    # Summary
    print_header("📊 HEALTH CHECK SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS GO!")
        print("Your Multi-Model AI Assistant is ready to rock!")
        print("\n🚀 To start the assistant:")
        print("  python app.py")
        print("  Then open: http://localhost:5000")
        print("\n🌐 Auto-research is enabled - ask about companies, news, or anything!")
    else:
        print(f"\n⚠️  {total - passed} issues need attention")
        print("Please fix the failed checks above before running the assistant")
        
        if not results.get("Gemini API"):
            print("\n💡 Get your Gemini API key:")
            print("  1. Visit: https://makersuite.google.com/app/apikey")
            print("  2. Create an API key")
            print("  3. Add it to your .env file")
        
        if not results.get("Crawl4AI"):
            print("\n💡 Fix Crawl4AI:")
            print("  pip install crawl4ai")
            print("  python -m playwright install")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_full_health_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)