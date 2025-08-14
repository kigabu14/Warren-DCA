import sqlite3
import datetime
import json
from typing import List, Dict, Any, Optional

class AIDatabase:
    """Database manager for storing AI queries and responses."""
    
    def __init__(self, db_path: str = "ai_queries.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create AI queries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                context_data TEXT,
                session_id TEXT,
                user_id TEXT DEFAULT 'anonymous'
            )
        ''')
        
        # Create index for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON ai_queries(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id ON ai_queries(session_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_query(self, 
                   query: str, 
                   response: str, 
                   context_data: Optional[Dict[str, Any]] = None,
                   session_id: Optional[str] = None,
                   user_id: str = 'anonymous') -> int:
        """Store an AI query and response in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        context_json = json.dumps(context_data) if context_data else None
        
        cursor.execute('''
            INSERT INTO ai_queries (query, response, context_data, session_id, user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response, context_json, session_id, user_id))
        
        query_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return query_id
    
    def get_recent_queries(self, limit: int = 10, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent AI queries, optionally filtered by session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT id, timestamp, query, response, context_data
                FROM ai_queries 
                WHERE session_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))
        else:
            cursor.execute('''
                SELECT id, timestamp, query, response, context_data
                FROM ai_queries 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'query': row[2],
                'response': row[3],
                'context_data': json.loads(row[4]) if row[4] else None
            })
        
        conn.close()
        return results
    
    def search_queries(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search AI queries by content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, query, response, context_data
            FROM ai_queries 
            WHERE query LIKE ? OR response LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (f'%{search_term}%', f'%{search_term}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'timestamp': row[1],
                'query': row[2],
                'response': row[3],
                'context_data': json.loads(row[4]) if row[4] else None
            })
        
        conn.close()
        return results
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total queries
        cursor.execute('SELECT COUNT(*) FROM ai_queries')
        total_queries = cursor.fetchone()[0]
        
        # Queries today
        cursor.execute('''
            SELECT COUNT(*) FROM ai_queries 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        queries_today = cursor.fetchone()[0]
        
        # Most recent query
        cursor.execute('''
            SELECT timestamp FROM ai_queries 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        last_query = cursor.fetchone()
        last_query_time = last_query[0] if last_query else None
        
        conn.close()
        
        return {
            'total_queries': total_queries,
            'queries_today': queries_today,
            'last_query_time': last_query_time
        }
    
    def clear_old_queries(self, days_to_keep: int = 30):
        """Clear queries older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM ai_queries 
            WHERE timestamp < datetime('now', '-{} days')
        '''.format(days_to_keep))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
# ใน AIDatabase class
def store_optimization(self, session_id, context_data, optimizer_result):
    import json, datetime
    conn = self._get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            context_data TEXT,
            result_json TEXT
        )
    """)
    cur.execute("""
        INSERT INTO optimization_results (session_id, timestamp, context_data, result_json)
        VALUES (?, ?, ?, ?)
    """, (
        session_id,
        datetime.datetime.utcnow().isoformat(),
        json.dumps(context_data, ensure_ascii=False),
        json.dumps(optimizer_result, ensure_ascii=False)
    ))
    conn.commit()
    conn.close()

def get_recent_optimizations(self, limit=5):
    conn = self._get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, timestamp, context_data, result_json
        FROM optimization_results
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    import json
    results = []
    for row in rows:
        results.append({
            "session_id": row[0],
            "timestamp": row[1],
            "context_data": json.loads(row[2]) if row[2] else None,
            "result": json.loads(row[3]) if row[3] else None
        })
    return results
