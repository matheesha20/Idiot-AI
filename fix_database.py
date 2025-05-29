#!/usr/bin/env python3
"""
Quick Fix Script for Database Issues
Run this if you get "no such column" errors
"""

import sqlite3
import os

def quick_fix_database():
    """Quick fix for database column issues"""
    db_path = 'chatbot_users.db'
    
    if not os.path.exists(db_path):
        print("❌ No database file found. Please run the main app first.")
        return
    
    print("🔧 Quick fixing database issues...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current table structure
        print("📋 Checking current database structure...")
        
        # Check users table
        cursor.execute("PRAGMA table_info(users)")
        users_columns = [row[1] for row in cursor.fetchall()]
        print(f"Users table columns: {users_columns}")
        
        # Add missing columns
        fixes_applied = []
        
        if 'preferred_model' not in users_columns:
            cursor.execute('ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT "chanuth"')
            fixes_applied.append("Added preferred_model to users")
        
        # Check chats table
        cursor.execute("PRAGMA table_info(chats)")
        chats_columns = [row[1] for row in cursor.fetchall()]
        
        if 'model_id' not in chats_columns:
            cursor.execute('ALTER TABLE chats ADD COLUMN model_id TEXT DEFAULT "chanuth"')
            fixes_applied.append("Added model_id to chats")
        
        # Check messages table
        cursor.execute("PRAGMA table_info(messages)")
        messages_columns = [row[1] for row in cursor.fetchall()]
        
        if 'model_id' not in messages_columns:
            cursor.execute('ALTER TABLE messages ADD COLUMN model_id TEXT')
            fixes_applied.append("Added model_id to messages")
        
        if 'emotion_detected' not in messages_columns:
            cursor.execute('ALTER TABLE messages ADD COLUMN emotion_detected TEXT')
            fixes_applied.append("Added emotion_detected to messages")
        
        # Create user_preferences table if missing
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
        if not cursor.fetchone():
            cursor.execute('''
                CREATE TABLE user_preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    preference_key TEXT,
                    preference_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            fixes_applied.append("Created user_preferences table")
        
        conn.commit()
        
        if fixes_applied:
            print("✅ Applied fixes:")
            for fix in fixes_applied:
                print(f"  • {fix}")
        else:
            print("✅ Database is already up to date!")
        
        print("🎉 Database fix completed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        print("You may need to delete the database file and start fresh.")
    finally:
        conn.close()

def delete_database():
    """Delete database to start fresh"""
    db_path = 'chatbot_users.db'
    
    if os.path.exists(db_path):
        confirm = input("⚠️  This will delete ALL your chats and data. Type 'DELETE' to confirm: ")
        if confirm == 'DELETE':
            os.remove(db_path)
            print("🗑️  Database deleted. Run the main app to create a fresh database.")
        else:
            print("❌ Database deletion cancelled.")
    else:
        print("❌ No database file found.")

if __name__ == "__main__":
    print("🛠️  Database Fix Tool")
    print("1. Quick fix database issues")
    print("2. Delete database (start fresh)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        quick_fix_database()
    elif choice == "2":
        delete_database()
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")