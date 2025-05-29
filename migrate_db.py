#!/usr/bin/env python3
"""
Database Migration Script for Multi-Model AI Assistant
Fixes the missing preferred_model column issue
"""

import sqlite3
import os

DATABASE_PATH = 'chatbot_users.db'

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns

def migrate_database():
    """Migrate database to latest schema"""
    print("🔧 Checking database schema...")
    
    if not os.path.exists(DATABASE_PATH):
        print("📝 No existing database found. Will create new one.")
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check and add missing columns to users table
        if not check_column_exists(cursor, 'users', 'preferred_model'):
            print("➕ Adding preferred_model column to users table...")
            cursor.execute('ALTER TABLE users ADD COLUMN preferred_model TEXT DEFAULT "chanuth"')
            print("✅ Added preferred_model column")
        else:
            print("✅ preferred_model column already exists")
        
        # Check and add model_id column to chats table if missing
        if not check_column_exists(cursor, 'chats', 'model_id'):
            print("➕ Adding model_id column to chats table...")
            cursor.execute('ALTER TABLE chats ADD COLUMN model_id TEXT DEFAULT "chanuth"')
            print("✅ Added model_id column to chats")
        else:
            print("✅ model_id column already exists in chats")
        
        # Check and add missing columns to messages table
        if not check_column_exists(cursor, 'messages', 'model_id'):
            print("➕ Adding model_id column to messages table...")
            cursor.execute('ALTER TABLE messages ADD COLUMN model_id TEXT')
            print("✅ Added model_id column to messages")
        else:
            print("✅ model_id column already exists in messages")
        
        if not check_column_exists(cursor, 'messages', 'emotion_detected'):
            print("➕ Adding emotion_detected column to messages table...")
            cursor.execute('ALTER TABLE messages ADD COLUMN emotion_detected TEXT')
            print("✅ Added emotion_detected column to messages")
        else:
            print("✅ emotion_detected column already exists in messages")
        
        # Check if user_preferences table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'")
        if not cursor.fetchone():
            print("➕ Creating user_preferences table...")
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
            print("✅ Created user_preferences table")
        else:
            print("✅ user_preferences table already exists")
        
        conn.commit()
        print("🎉 Database migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()