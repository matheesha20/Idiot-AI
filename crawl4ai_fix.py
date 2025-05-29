#!/usr/bin/env python3
"""
Minimal patch for app.py to fix Crawl4AI v0.6+ compatibility
Run this to patch your existing app.py file
"""

def patch_app_py():
    """Apply minimal patches to make app.py work with crawl4ai 0.6+"""
    
    print("🔧 Patching app.py for Crawl4AI v0.6+ compatibility...")
    
    try:
        # Read the current app.py
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        patches_applied = []
        
        # Patch 1: Fix imports
        old_import = """# Import crawl4ai directly
try:
    from crawl4ai import WebCrawler
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI imported successfully")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    print(f"⚠️  Crawl4AI not available: {e}")
    print("Install with: pip install crawl4ai")"""

        new_import = """# Import crawl4ai directly - Updated for v0.6+
try:
    from crawl4ai import AsyncWebCrawler
    import asyncio
    import concurrent.futures
    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI imported successfully")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    print(f"⚠️  Crawl4AI not available: {e}")
    print("Install with: pip install crawl4ai")"""

        if old_import in content:
            content = content.replace(old_import, new_import)
            patches_applied.append("Updated Crawl4AI imports")
        
        # Patch 2: Fix SmartWebCrawler initialization
        old_init = """    def __init__(self):
        self.crawler = None
        self.available = CRAWL4AI_AVAILABLE
        self.crawl_cache = {}  # Simple cache to avoid repeated crawls
        self.rate_limit = {}   # Rate limiting per domain
        
        if self.available:
            self.crawler = WebCrawler()
            # Warm up the crawler
            try:
                self.crawler.warmup()
                print("🌐 Web crawler initialized and warmed up")
            except Exception as e:
                print(f"⚠️  Crawler warmup failed: {e}")
                self.available = False"""

        new_init = """    def __init__(self):
        self.available = CRAWL4AI_AVAILABLE
        self.crawl_cache = {}  # Simple cache to avoid repeated crawls
        self.rate_limit = {}   # Rate limiting per domain
        
        if self.available:
            print("🌐 Web crawler initialized (AsyncWebCrawler)")
        else:
            print("⚠️  Web crawler not available")"""

        if old_init in content:
            content = content.replace(old_init, new_init)
            patches_applied.append("Updated SmartWebCrawler initialization")
        
        # Patch 3: Add concurrent.futures import
        old_imports = """import os
import json
import requests
import numpy as np
import faiss
import re
import sqlite3
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
import random
import asyncio
from urllib.parse import urlparse, urljoin
import time"""

        new_imports = """import os
import json
import requests
import numpy as np
import faiss
import re
import sqlite3
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
import random
import asyncio
from urllib.parse import urlparse, urljoin
import time
import concurrent.futures"""

        if old_imports in content and "import concurrent.futures" not in content:
            content = content.replace(old_imports, new_imports)
            patches_applied.append("Added concurrent.futures import")
        
        # Patch 4: Replace crawl_url method
        old_crawl_start = "    def crawl_url(self, url: str) -> Dict:"
        old_crawl_end = "        except Exception as e:\n            return {\"content\": \"\", \"url\": url, \"error\": str(e), \"success\": False}"
        
        new_crawl_method = """    def crawl_url(self, url: str) -> Dict:
        \"\"\"Crawl a single URL and return structured data\"\"\"
        if not self.available:
            return {"content": "", "url": url, "error": "Crawler not available"}
        
        # Check cache first
        if url in self.crawl_cache:
            cache_time, cached_data = self.crawl_cache[url]
            if time.time() - cache_time < 3600:  # 1 hour cache
                return cached_data
        
        # Check rate limiting
        if self.is_rate_limited(url):
            return {"content": "", "url": url, "error": "Rate limited"}
        
        try:
            # Update rate limit
            domain = urlparse(url).netloc
            self.rate_limit[domain] = time.time()
            
            # Use async crawler with sync wrapper
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._crawl_sync_wrapper, url)
                result = future.result(timeout=30)
            
            if result.get('success'):
                # Cache the result
                self.crawl_cache[url] = (time.time(), result)
                return result
            else:
                return {"content": "", "url": url, "error": "Crawl failed", "success": False}
                
        except Exception as e:
            return {"content": "", "url": url, "error": str(e), "success": False}
    
    def _crawl_sync_wrapper(self, url: str) -> Dict:
        \"\"\"Synchronous wrapper for async crawling\"\"\"
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._crawl_async(url))
        finally:
            try:
                loop.close()
            except:
                pass
    
    async def _crawl_async(self, url: str) -> Dict:
        \"\"\"Async crawling method using AsyncWebCrawler\"\"\"
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                
                if result.success:
                    content = result.markdown or result.cleaned_html or ""
                    # Clean and limit content
                    content = self._clean_content(content)
                    
                    return {
                        "content": content,
                        "url": url,
                        "title": getattr(result, 'title', ''),
                        "success": True
                    }
                else:
                    return {"content": "", "url": url, "error": "Crawl failed", "success": False}
        except Exception as e:
            return {"content": "", "url": url, "error": str(e), "success": False}"""

        # Find and replace the crawl_url method
        import re
        pattern = r'(\s+def crawl_url\(self, url: str\) -> Dict:.*?)(\s+def _clean_content\(self)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            content = content.replace(match.group(1), "\n" + new_crawl_method + "\n")
            patches_applied.append("Updated crawl_url method for AsyncWebCrawler")
        
        # Save the patched file
        if content != original_content:
            # Backup original
            with open("app.py.backup", "w", encoding="utf-8") as f:
                f.write(original_content)
            print("📄 Created backup: app.py.backup")
            
            # Write patched version
            with open("app.py", "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✅ Applied {len(patches_applied)} patches:")
            for patch in patches_applied:
                print(f"   - {patch}")
            
            print("\n🎉 Patching complete!")
            print("🚀 Try running: python app.py")
        else:
            print("ℹ️  No patches needed - app.py appears to already be compatible")
        
        return True
        
    except FileNotFoundError:
        print("❌ app.py not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error patching app.py: {e}")
        return False

def test_import():
    """Test if the patched imports work"""
    print("\n🧪 Testing imports after patch...")
    try:
        from crawl4ai import AsyncWebCrawler
        import asyncio
        import concurrent.futures
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    print("🔧 Minimal App.py Patcher for Crawl4AI v0.6+")
    print("=" * 50)
    
    if patch_app_py():
        if test_import():
            print("\n🎉 SUCCESS! Your app should now work with Crawl4AI v0.6+")
            print("🚀 Run: python app.py")
        else:
            print("\n⚠️  Patching complete but imports still failing")
            print("Try: pip install --upgrade crawl4ai")
    else:
        print("\n❌ Patching failed")

if __name__ == "__main__":
    main()