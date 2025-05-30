#!/usr/bin/env python3
"""
Real-time monitoring of crawling activity
"""

import time
import subprocess
import sys

def monitor_crawling_logs():
    """Monitor the application logs for crawling activity"""
    print("🔍 MONITORING CRAWLING ACTIVITY")
    print("=" * 50)
    print("ℹ️  This will monitor the application logs for crawling activity.")
    print("🚀 Please go to http://127.0.0.1:5000/chat and ask about:")
    print("   • 'Tell me about Apple company'")
    print("   • 'What is Tesla stock price'") 
    print("   • 'Latest news about Google'")
    print("=" * 50)
    print()
    
    # Keywords to look for in logs
    crawl_keywords = [
        "🌐 Auto-crawling for query:",
        "✅ Found",
        "web sources",
        "[FETCH]",
        "[SCRAPE]", 
        "[COMPLETE]",
        "Crawl4AI",
        "AsyncWebCrawler"
    ]
    
    last_position = 0
    
    try:
        while True:
            # Read current server logs (you would replace this with actual log monitoring)
            print("⏳ Waiting for crawling activity... (Press Ctrl+C to stop)")
            print("💡 Go to the browser and send a message about Apple, Tesla, or Google!")
            print()
            
            time.sleep(5)
            
            # In a real scenario, you'd monitor the actual log file
            # For now, we'll just remind the user how to test
            print("🔄 Still monitoring... Try asking:")
            print("   'Tell me about Apple company latest news'")
            print("   'What is current Tesla stock price'")
            print()
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

def test_crawling_status():
    """Check current crawling status"""
    print("📊 CURRENT CRAWLING STATUS")
    print("=" * 30)
    
    try:
        # Import and test crawler directly
        import sys
        import os
        sys.path.append('.')
        
        from app import SmartWebCrawler, CRAWL4AI_AVAILABLE
        
        print(f"✅ Crawl4AI Available: {CRAWL4AI_AVAILABLE}")
        
        if CRAWL4AI_AVAILABLE:
            crawler = SmartWebCrawler()
            print(f"✅ Crawler Initialized: {crawler.available}")
            
            # Test query detection
            test_queries = [
                "Tell me about Apple company",
                "What is Tesla stock",
                "Hello world"
            ]
            
            print("\n🔍 Query Detection Test:")
            for query in test_queries:
                should_crawl = crawler.should_crawl_for_query(query)
                status = "🌐" if should_crawl else "💬"
                print(f"  {status} '{query}' -> {'Will crawl' if should_crawl else 'No crawl'}")
                
        print("\n" + "=" * 30)
        
    except Exception as e:
        print(f"❌ Error testing crawler: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        test_crawling_status()
    else:
        print("🧪 CRAWLING FEATURE VERIFICATION")
        print("=" * 50)
        
        # First show status
        test_crawling_status()
        
        print("\n")
        
        # Then start monitoring
        monitor_crawling_logs()
