#!/usr/bin/env python3
"""
Health Check Script for Multi-Model AI Assistant with TinyLlama
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
    
    # Check model cache directory
    models_dir = Path('./models')
    if models_dir.exists():
        print(f"✅ ./models/ (cache directory)")
    else:
        print(f"ℹ️  ./models/ - Will be created on first run")
    
    return all_good

def check_environment():
    """Check environment variables"""
    print_header("🔧 ENVIRONMENT CHECK")
    
    load_dotenv()
    
    secret_key = os.getenv('FLASK_SECRET_KEY')
    
    if secret_key:
        print("✅ FLASK_SECRET_KEY: Set")
    else:
        print("❌ FLASK_SECRET_KEY: Missing")
    
    # Check TinyLlama settings
    device = os.getenv('TINYLLAMA_DEVICE', 'auto')
    max_length = os.getenv('TINYLLAMA_MAX_LENGTH', '512')
    
    print(f"ℹ️  TINYLLAMA_DEVICE: {device}")
    print(f"ℹ️  TINYLLAMA_MAX_LENGTH: {max_length}")
    
    return bool(secret_key)

def check_python_packages():
    """Check if all required packages are installed"""
    print_header("📦 PYTHON PACKAGES CHECK")
    
    packages = [
        'flask',
        'torch',
        'transformers',
        'sentence_transformers',
        'numpy',
        'faiss-cpu',
        'requests',
        'python-dotenv',
        'crawl4ai'
    ]
    
    all_installed = True
    for package in packages:
        try:
            if package == 'python-dotenv':
                import dotenv
                print(f"✅ {package}")
            elif package == 'faiss-cpu':
                import faiss
                print(f"✅ {package}")
            elif package == 'sentence_transformers':
                import sentence_transformers
                print(f"✅ {package}")
            else:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_pytorch_cuda():
    """Check PyTorch and CUDA setup"""
    print_header("🔥 PYTORCH & CUDA CHECK")
    
    try:
        import torch
        print(f"✅ PyTorch: Version {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA: Available with {gpu_count} GPU(s)")
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("💻 CUDA: Not available (will use CPU)")
        
        # Test basic tensor operations
        test_tensor = torch.randn(3, 3)
        if torch.cuda.is_available():
            test_tensor_gpu = test_tensor.cuda()
            print("✅ GPU tensor operations: Working")
        
        print("✅ PyTorch: Basic operations working")
        return True
        
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch: Error - {e}")
        return False

def check_tinyllama():
    """Check TinyLlama model availability"""
    print_header("🧠 TINYLLAMA CHECK")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📥 Checking TinyLlama model: {model_name}")
        print(f"🎯 Target device: {device}")
        
        # Check if model is cached
        cache_dir = Path('./models')
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print("✅ Model cache directory exists with content")
        else:
            print("ℹ️  Model cache empty - will download on first run (~2GB)")
        
        # Try to load tokenizer (faster test)
        print("🔤 Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir='./models',
            trust_remote_code=True
        )
        print("✅ Tokenizer: Loaded successfully")
        
        # Test basic tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Tokenizer test: '{test_text}' -> {len(tokens)} tokens")
        
        print("ℹ️  Full model loading will happen on first app run")
        return True
        
    except ImportError as e:
        print(f"❌ TinyLlama dependencies missing: {e}")
        return False
    except Exception as e:
        print(f"❌ TinyLlama test failed: {e}")
        return False

def check_embeddings():
    """Check sentence transformers for embeddings"""
    print_header("🔤 EMBEDDINGS CHECK")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "all-MiniLM-L6-v2"
        print(f"📥 Loading embedding model: {model_name}")
        
        model = SentenceTransformer(model_name)
        
        # Test embedding generation
        test_sentence = "This is a test sentence."
        embedding = model.encode(test_sentence)
        
        print(f"✅ Embedding model: Loaded successfully")
        print(f"✅ Embedding dimension: {len(embedding)}")
        print(f"✅ Test embedding: Generated for test sentence")
        
        return True
        
    except ImportError as e:
        print(f"❌ Sentence transformers missing: {e}")
        return False
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def check_crawl4ai():
    """Check Crawl4AI and Playwright setup"""
    print_header("🌐 WEB CRAWLING CHECK")
    
    try:
        from crawl4ai import AsyncWebCrawler
        print("✅ Crawl4AI: Imported successfully")
        
        # Check if playwright browsers are installed
        try:
            import subprocess
            result = subprocess.run(['python', '-m', 'playwright', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Playwright: {result.stdout.strip()}")
            else:
                print("⚠️  Playwright: Version check failed")
        except Exception:
            print("⚠️  Playwright: Could not check version")
        
        print("✅ Web crawling: Ready for auto-research")
        return True
            
    except ImportError as e:
        print(f"❌ Crawl4AI: Not available ({e})")
        print("   Install with: pip install crawl4ai")
        print("   Then run: python -m playwright install")
        return False
    except Exception as e:
        print(f"❌ Crawl4AI: Error ({e})")
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
        
        # Test TinyLlama manager (without loading full model)
        from app import TinyLlamaManager, TINYLLAMA_AVAILABLE
        if TINYLLAMA_AVAILABLE:
            print("✅ TinyLlamaManager: Dependencies available")
        else:
            print("⚠️  TinyLlamaManager: Dependencies missing")
        
        # Test embedding manager
        from app import LocalEmbeddingManager
        emb_manager = LocalEmbeddingManager()
        if emb_manager.available:
            test_emb = emb_manager.generate_embedding("test")
            print(f"✅ LocalEmbeddingManager: OK (dim: {len(test_emb)})")
        else:
            print("⚠️  LocalEmbeddingManager: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality Test: Error - {e}")
        return False

def run_full_health_check():
    """Run complete health check"""
    print("🔍 MULTI-MODEL AI ASSISTANT HEALTH CHECK")
    print("🧠 Verifying TinyLlama and all components...")
    
    checks = [
        ("Files", check_files),
        ("Environment", check_environment),
        ("Python Packages", check_python_packages),
        ("PyTorch & CUDA", check_pytorch_cuda),
        ("TinyLlama", check_tinyllama),
        ("Embeddings", check_embeddings),
        ("Crawl4AI", check_crawl4ai),
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
        print("Your Multi-Model AI Assistant with TinyLlama is ready!")
        print("\n🚀 To start the assistant:")
        print("  python app.py")
        print("  Then open: http://localhost:5000")
        print("\n🌐 Features ready:")
        print("  🧠 TinyLlama running locally (no API keys needed!)")
        print("  🌐 Auto-research enabled - ask about companies, news, anything!")
        print("  🔒 Completely private - everything runs on your machine!")
        print("\n⚡ First run will download TinyLlama model (~2GB)")
        print("🚀 After that, everything runs instantly offline!")
    else:
        print(f"\n⚠️  {total - passed} issues need attention")
        print("Please fix the failed checks above before running the assistant")
        
        if not results.get("PyTorch & CUDA"):
            print("\n💡 Fix PyTorch:")
            print("  pip install torch torchvision torchaudio")
            print("  Or for GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        if not results.get("TinyLlama"):
            print("\n💡 Fix TinyLlama:")
            print("  pip install transformers sentence-transformers")
            
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