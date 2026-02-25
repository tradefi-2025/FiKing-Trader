"""
PostgreSQL Database Handler - Generic Service
Provides basic database operations with JSON data handling

This service is intentionally generic and does not manipulate data.
Each service should define its own table structure and handle data transformation.

Environment Variables Required:
- POSTGRES_HOST: Database host (default: localhost)
- POSTGRES_PORT: Database port (default: 5432)
- POSTGRES_DATABASE: Database name
- POSTGRES_USER: Database user
- POSTGRES_PASSWORD: Database password
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from decimal import Decimal
import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgreSQLService:
    """Generic PostgreSQL database service for basic operations"""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None,
                 min_connections: int = 1,
                 max_connections: int = 10):
        """
        Initialize PostgreSQL service with connection pooling
        
        Args:
            host: Database host (loads from POSTGRES_HOST if not provided)
            port: Database port (loads from POSTGRES_PORT if not provided)
            database: Database name (loads from POSTGRES_DATABASE if not provided)
            user: Database user (loads from POSTGRES_USER if not provided)
            password: Database password (loads from POSTGRES_PASSWORD if not provided)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5432))
        self.database = database or os.getenv('POSTGRES_DATABASE')
        self.user = user or os.getenv('POSTGRES_USER')
        self.password = password or os.getenv('POSTGRES_PASSWORD')
        
        if not all([self.database, self.user, self.password]):
            raise ValueError("Database name, user, and password are required. "
                           "Set via environment variables or constructor parameters.")
        
        self.connection_pool = None
        self._initialize_pool(min_connections, max_connections)
    
    def _initialize_pool(self, min_conn: int, max_conn: int):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.database}@{self.host}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting database connection from pool
        
        Usage:
            with service.get_connection() as conn:
                # use connection
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Connection error: {str(e)}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        Context manager for getting cursor
        
        Args:
            commit: Whether to commit transaction automatically
            
        Usage:
            with service.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table")
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Query error: {str(e)}")
                raise
            finally:
                cursor.close()
    
    # ==================== Generic Query Methods ====================
    
    def execute_query(self, 
                     query: str, 
                     params: Union[tuple, dict] = None,
                     commit: bool = True) -> bool:
        """
        Execute a SQL query (INSERT, UPDATE, DELETE, CREATE, etc.)
        
        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
            commit: Whether to commit the transaction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor(commit=commit) as cursor:
                cursor.execute(query, params)
                logger.info(f"✅ Query executed successfully")
                return True
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            return False
    
    def fetch_one(self, 
                  query: str, 
                  params: Union[tuple, dict] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from database
        
        Args:
            query: SQL SELECT query
            params: Query parameters
            
        Returns:
            Dictionary representing the row, or None if not found
        """
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Fetch one failed: {str(e)}")
            return None
    
    def fetch_all(self, 
                  query: str, 
                  params: Union[tuple, dict] = None) -> List[Dict[str, Any]]:
        """
        Fetch all rows from database
        
        Args:
            query: SQL SELECT query
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"❌ Fetch all failed: {str(e)}")
            return []
    
    def fetch_many(self, 
                   query: str, 
                   params: Union[tuple, dict] = None,
                   size: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows from database
        
        Args:
            query: SQL SELECT query
            params: Query parameters
            size: Number of rows to fetch
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchmany(size)
                return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"❌ Fetch many failed: {str(e)}")
            return []
    
    # ==================== Generic CRUD Operations ====================
    
    def insert(self, 
               table: str, 
               data: Dict[str, Any],
               returning: str = None) -> Optional[Dict[str, Any]]:
        """
        Insert a single row into table
        
        Args:
            table: Table name
            data: Dictionary with column names and values
            returning: Column name to return (e.g., 'id' or '*')
            
        Returns:
            Dictionary with returned columns if returning is specified, else None
        """
        try:
            columns = list(data.keys())
            values = list(data.values())
            
            # Build query
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(values))
            query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
            
            if returning:
                query += f" RETURNING {returning}"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, values)
                
                if returning:
                    result = cursor.fetchone()
                    return dict(result) if result else None
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Insert failed: {str(e)}")
            return None
    
    def insert_many(self, 
                    table: str, 
                    data_list: List[Dict[str, Any]]) -> int:
        """
        Insert multiple rows into table
        
        Args:
            table: Table name
            data_list: List of dictionaries with column names and values
            
        Returns:
            Number of rows inserted
        """
        if not data_list:
            return 0
        
        try:
            # Assume all dictionaries have the same keys
            columns = list(data_list[0].keys())
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            
            query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
            
            values_list = [list(data.values()) for data in data_list]
            
            with self.get_cursor() as cursor:
                extras.execute_batch(cursor, query, values_list)
                return len(values_list)
                
        except Exception as e:
            logger.error(f"❌ Insert many failed: {str(e)}")
            return 0
    
    def update(self, 
               table: str, 
               data: Dict[str, Any],
               where: Dict[str, Any],
               returning: str = None) -> Optional[Dict[str, Any]]:
        """
        Update rows in table
        
        Args:
            table: Table name
            data: Dictionary with column names and new values
            where: Dictionary with conditions (combined with AND)
            returning: Column name to return (e.g., 'id' or '*')
            
        Returns:
            Dictionary with returned columns if returning is specified, else None
        """
        try:
            # Build SET clause
            set_parts = [f"{col} = %s" for col in data.keys()]
            set_clause = ', '.join(set_parts)
            
            # Build WHERE clause
            where_parts = [f"{col} = %s" for col in where.keys()]
            where_clause = ' AND '.join(where_parts)
            
            # Combine values
            values = list(data.values()) + list(where.values())
            
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            
            if returning:
                query += f" RETURNING {returning}"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, values)
                
                if returning:
                    result = cursor.fetchone()
                    return dict(result) if result else None
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Update failed: {str(e)}")
            return None
    
    def delete(self, 
               table: str, 
               where: Dict[str, Any],
               returning: str = None) -> Optional[Union[int, List[Dict[str, Any]]]]:
        """
        Delete rows from table
        
        Args:
            table: Table name
            where: Dictionary with conditions (combined with AND)
            returning: Column name to return (e.g., 'id' or '*')
            
        Returns:
            Number of deleted rows, or list of returned rows if returning is specified
        """
        try:
            # Build WHERE clause
            where_parts = [f"{col} = %s" for col in where.keys()]
            where_clause = ' AND '.join(where_parts)
            values = list(where.values())
            
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            if returning:
                query += f" RETURNING {returning}"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, values)
                
                if returning:
                    results = cursor.fetchall()
                    return [dict(row) for row in results] if results else []
                
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"❌ Delete failed: {str(e)}")
            return 0 if not returning else []
    
    def select(self, 
               table: str,
               columns: List[str] = None,
               where: Dict[str, Any] = None,
               order_by: str = None,
               limit: int = None,
               offset: int = None) -> List[Dict[str, Any]]:
        """
        Select rows from table
        
        Args:
            table: Table name
            columns: List of column names to select (default: all *)
            where: Dictionary with conditions (combined with AND)
            order_by: ORDER BY clause (e.g., 'created_at DESC')
            limit: Maximum number of rows
            offset: Number of rows to skip
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            # Build SELECT clause
            columns_str = ', '.join(columns) if columns else '*'
            query = f"SELECT {columns_str} FROM {table}"
            values = []
            
            # Build WHERE clause
            if where:
                where_parts = [f"{col} = %s" for col in where.keys()]
                where_clause = ' AND '.join(where_parts)
                query += f" WHERE {where_clause}"
                values = list(where.values())
            
            # Add ORDER BY
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Add LIMIT and OFFSET
            if limit:
                query += f" LIMIT {limit}"
            if offset:
                query += f" OFFSET {offset}"
            
            return self.fetch_all(query, tuple(values) if values else None)
            
        except Exception as e:
            logger.error(f"❌ Select failed: {str(e)}")
            return []
    
    # ==================== Table Management ====================
    
    def table_exists(self, table: str) -> bool:
        """
        Check if table exists
        
        Args:
            table: Table name
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """
        result = self.fetch_one(query, (table,))
        return result.get('exists', False) if result else False
    
    def create_table(self, table: str, schema: str) -> bool:
        """
        Create a table with given schema
        
        Args:
            table: Table name
            schema: Table schema definition (column definitions)
            
        Example:
            schema = '''
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            '''
            
        Returns:
            True if successful, False otherwise
        """
        query = f"CREATE TABLE IF NOT EXISTS {table} ({schema})"
        return self.execute_query(query)
    
    def drop_table(self, table: str, cascade: bool = False) -> bool:
        """
        Drop a table
        
        Args:
            table: Table name
            cascade: Whether to cascade drop
            
        Returns:
            True if successful, False otherwise
        """
        cascade_str = " CASCADE" if cascade else ""
        query = f"DROP TABLE IF EXISTS {table}{cascade_str}"
        return self.execute_query(query)
    
    def get_table_columns(self, table: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table
        
        Args:
            table: Table name
            
        Returns:
            List of dictionaries with column information
        """
        query = """
            SELECT 
                column_name, 
                data_type, 
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' 
            AND table_name = %s
            ORDER BY ordinal_position
        """
        return self.fetch_all(query, (table,))
    
    def list_tables(self) -> List[str]:
        """
        List all tables in database
        
        Returns:
            List of table names
        """
        query = """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        results = self.fetch_all(query)
        return [row['table_name'] for row in results]
    
    # ==================== JSON Specific Methods ====================
    
    def insert_json(self, 
                    table: str, 
                    json_column: str,
                    json_data: Union[Dict, List],
                    additional_columns: Dict[str, Any] = None,
                    returning: str = None) -> Optional[Dict[str, Any]]:
        """
        Insert row with JSON data
        
        Args:
            table: Table name
            json_column: Name of JSONB column
            json_data: JSON data (dict or list)
            additional_columns: Other columns to insert
            returning: Column name to return
            
        Returns:
            Dictionary with returned columns if returning is specified
        """
        data = {json_column: json.dumps(json_data)}
        if additional_columns:
            data.update(additional_columns)
        
        return self.insert(table, data, returning)
    
    def query_json(self,
                   table: str,
                   json_column: str,
                   json_path: str,
                   json_value: Any,
                   operator: str = '=') -> List[Dict[str, Any]]:
        """
        Query rows by JSON field value
        
        Args:
            table: Table name
            json_column: Name of JSONB column
            json_path: JSON path (e.g., 'metadata->>'key'')
            json_value: Value to match
            operator: Comparison operator (=, !=, >, <, etc.)
            
        Returns:
            List of dictionaries representing matching rows
        """
        query = f"SELECT * FROM {table} WHERE {json_column}->>{json_path} {operator} %s"
        return self.fetch_all(query, (json_value,))
    
    # ==================== Transaction Management ====================
    
    @contextmanager
    def transaction(self):
        """
        Context manager for explicit transaction handling
        
        Usage:
            with service.transaction() as cursor:
                cursor.execute("INSERT ...")
                cursor.execute("UPDATE ...")
                # Commits automatically if no exception
                # Rolls back if exception occurs
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            try:
                yield cursor
                conn.commit()
                logger.info("✅ Transaction committed")
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Transaction rolled back: {str(e)}")
                raise
            finally:
                cursor.close()
    
    # ==================== Utility Methods ====================
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user
        }
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database statistics and information"""
        try:
            # Get database size
            size_query = """
                SELECT pg_size_pretty(pg_database_size(%s)) as size
            """
            size_result = self.fetch_one(size_query, (self.database,))
            
            # Get table count
            tables = self.list_tables()
            
            # Get connection count
            conn_query = """
                SELECT count(*) as connections 
                FROM pg_stat_activity 
                WHERE datname = %s
            """
            conn_result = self.fetch_one(conn_query, (self.database,))
            
            return {
                'database': self.database,
                'size': size_result.get('size') if size_result else 'unknown',
                'table_count': len(tables),
                'tables': tables,
                'active_connections': conn_result.get('connections', 0) if conn_result else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database info: {str(e)}")
            return {
                'database': self.database,
                'error': str(e)
            }
    
    def close(self):
        """Close all connections in pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("🔌 PostgreSQL connection pool closed")


# ==================== Example Usage ====================

def test_postgres_service():
    """Test function for PostgreSQL service"""
    try:
        # Initialize service
        service = PostgreSQLService()
        print("✅ PostgreSQL service initialized")
        
        # Get connection info
        info = service.get_connection_info()
        print(f"📋 Connection: {info['user']}@{info['host']}:{info['port']}/{info['database']}")
        
        # Create test table
        print("\n📦 Creating test table...")
        test_table = "test_service_data"
        
        schema = """
            id SERIAL PRIMARY KEY,
            service_name VARCHAR(100) NOT NULL,
            config JSONB,
            metrics JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """
        
        if service.create_table(test_table, schema):
            print(f"  ✅ Table '{test_table}' created")
        
        # Insert test data
        print("\n📝 Inserting test data...")
        test_data = {
            'service_name': 'signaling_service',
            'config': json.dumps({
                'model': 'SignalingModelV1',
                'version': 'v1.0',
                'parameters': {'learning_rate': 0.001}
            }),
            'metrics': json.dumps({
                'accuracy': 0.92,
                'loss': 0.15
            })
        }
        
        result = service.insert(test_table, test_data, returning='id')
        if result:
            print(f"  ✅ Inserted row with id: {result['id']}")
            inserted_id = result['id']
        
        # Select data
        print("\n🔍 Selecting data...")
        rows = service.select(
            test_table,
            where={'service_name': 'signaling_service'},
            order_by='created_at DESC'
        )
        
        for row in rows:
            print(f"  📊 Row {row['id']}: {row['service_name']}")
            print(f"     Config: {row['config']}")
            print(f"     Metrics: {row['metrics']}")
        
        # Update data
        print("\n✏️ Updating data...")
        service.update(
            test_table,
            data={'metrics': json.dumps({'accuracy': 0.95, 'loss': 0.12})},
            where={'id': inserted_id}
        )
        print("  ✅ Data updated")
        
        # Insert multiple rows
        print("\n📝 Inserting multiple rows...")
        multi_data = [
            {
                'service_name': 'agent_1',
                'config': json.dumps({'type': 'trader'}),
                'metrics': json.dumps({'profit': 1500.50})
            },
            {
                'service_name': 'agent_2',
                'config': json.dumps({'type': 'analyzer'}),
                'metrics': json.dumps({'accuracy': 0.88})
            }
        ]
        count = service.insert_many(test_table, multi_data)
        print(f"  ✅ Inserted {count} rows")
        
        # Get database info
        print("\n📊 Database information:")
        db_info = service.get_database_info()
        print(f"  Size: {db_info.get('size')}")
        print(f"  Tables: {db_info.get('table_count')}")
        print(f"  Active connections: {db_info.get('active_connections')}")
        
        # Clean up
        print("\n🧹 Cleaning up...")
        service.drop_table(test_table)
        print(f"  ✅ Dropped table '{test_table}'")
        
        # Close connection
        service.close()
        
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        print("💡 Make sure to set PostgreSQL credentials in .env file")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_postgres_service()
