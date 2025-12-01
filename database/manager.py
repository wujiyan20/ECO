# database/manager.py - Database Connection Manager
"""
Database manager with connection pooling, transaction support,
and comprehensive error handling
"""

import pyodbc
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from threading import Lock

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Simple connection pool for database connections"""
    
    def __init__(self, connection_string: str, pool_size: int = 5):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.connections = []
        self.in_use = set()
        self.lock = Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            for _ in range(self.pool_size):
                conn = pyodbc.connect(self.connection_string)
                self.connections.append(conn)
            logger.info(f"✓ Connection pool initialized with {self.pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        with self.lock:
            for conn in self.connections:
                if conn not in self.in_use:
                    self.in_use.add(conn)
                    return conn
            
            # Pool exhausted, create new connection
            if len(self.connections) < self.pool_size * 2:
                conn = pyodbc.connect(self.connection_string)
                self.connections.append(conn)
                self.in_use.add(conn)
                return conn
            
            raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
    
    def close_all(self):
        """Close all connections"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
            self.in_use.clear()
        logger.info("✓ All connections closed")

class DatabaseManager:
    """
    Database connection and query management with connection pooling
    """
    
    def __init__(self, config, use_pool: bool = True):
        """
        Initialize database manager
        
        Args:
            config: DatabaseConfig instance
            use_pool: Whether to use connection pooling
        """
        self.config = config
        self.connection_string = config.get_connection_string()
        self.use_pool = use_pool
        self.pool = None
        
        if use_pool:
            try:
                self.pool = ConnectionPool(
                    self.connection_string,
                    pool_size=config.pool_size
                )
            except Exception as e:
                logger.warning(f"Failed to create connection pool: {e}")
                self.use_pool = False
        
        # Statistics
        self.query_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Automatically handles commit/rollback and connection cleanup
        """
        conn = None
        start_time = time.time()
        
        try:
            if self.use_pool and self.pool:
                conn = self.pool.get_connection()
            else:
                conn = pyodbc.connect(self.connection_string)
            
            conn.autocommit = False
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            self.error_count += 1
            logger.error(f"Database error: {e}")
            raise
            
        finally:
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.query_count += 1
            
            if conn:
                if self.use_pool and self.pool:
                    self.pool.return_connection(conn)
                else:
                    conn.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, 
                     timeout: Optional[int] = None) -> List[Dict]:
        """
        Execute a SELECT query and return results as list of dictionaries
        
        Args:
            query: SQL SELECT query
            params: Query parameters as tuple
            timeout: Query timeout in seconds (optional)
        
        Returns:
            List of dictionaries with query results
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if timeout:
                    cursor.timeout = timeout
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [column[0] for column in cursor.description] if cursor.description else []
                
                # Fetch results and convert to dictionaries
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                execution_time = time.time() - start_time
                logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(results)} rows")
                
                return results
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            raise
    
    def execute_non_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute INSERT, UPDATE, DELETE queries
        
        Args:
            query: SQL query
            params: Query parameters as tuple
        
        Returns:
            Number of affected rows
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                affected_rows = cursor.rowcount
                conn.commit()
                
                execution_time = time.time() - start_time
                logger.debug(f"Non-query executed in {execution_time:.3f}s, affected {affected_rows} rows")
                
                return affected_rows
                
        except Exception as e:
            logger.error(f"Non-query execution failed: {e}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            raise
    
    def execute_scalar(self, query: str, params: Optional[Tuple] = None) -> Any:
        """
        Execute query and return single value
        
        Args:
            query: SQL query
            params: Query parameters as tuple
        
        Returns:
            Single value result
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Scalar query execution failed: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute query with multiple parameter sets (batch insert/update)
        
        Args:
            query: SQL query
            params_list: List of parameter tuples
        
        Returns:
            Total number of affected rows
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.fast_executemany = True  # Enable fast execution for SQL Server
                
                cursor.executemany(query, params_list)
                affected_rows = cursor.rowcount
                conn.commit()
                
                execution_time = time.time() - start_time
                logger.debug(f"Batch executed in {execution_time:.3f}s, affected {affected_rows} rows")
                
                return affected_rows
                
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise
    
    def execute_stored_procedure(self, proc_name: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        Execute stored procedure
        
        Args:
            proc_name: Stored procedure name
            params: Procedure parameters
        
        Returns:
            Result set as list of dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(f"EXEC {proc_name} {','.join('?' * len(params))}", params)
                else:
                    cursor.execute(f"EXEC {proc_name}")
                
                # Get results if any
                if cursor.description:
                    columns = [column[0] for column in cursor.description]
                    results = []
                    for row in cursor.fetchall():
                        results.append(dict(zip(columns, row)))
                    return results
                
                conn.commit()
                return []
                
        except Exception as e:
            logger.error(f"Stored procedure execution failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful
        """
        try:
            result = self.execute_scalar("SELECT 1")
            if result == 1:
                logger.info("✓ Database connection test successful")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ Database connection test failed: {e}")
            return False
    
    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            return self.execute_scalar(query) or 0
        except:
            return 0
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            query = """
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ?
            """
            count = self.execute_scalar(query, (table_name,))
            return count > 0
        except:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database manager statistics"""
        avg_execution_time = (
            self.total_execution_time / self.query_count 
            if self.query_count > 0 else 0
        )
        
        return {
            'query_count': self.query_count,
            'error_count': self.error_count,
            'total_execution_time': round(self.total_execution_time, 3),
            'average_execution_time': round(avg_execution_time, 3),
            'success_rate': round(
                (self.query_count - self.error_count) / self.query_count * 100 
                if self.query_count > 0 else 0, 2
            ),
            'pool_enabled': self.use_pool,
            'pool_size': self.config.pool_size if self.use_pool else 0
        }
    
    def close(self):
        """Close database manager and cleanup resources"""
        if self.pool:
            self.pool.close_all()
        
        stats = self.get_statistics()
        logger.info(f"Database manager closed. Statistics: {stats}")

# Convenience functions
def create_manager_from_env() -> DatabaseManager:
    """Create database manager from environment variables"""
    from .config import DatabaseConfig
    config = DatabaseConfig.from_env()
    return DatabaseManager(config)

if __name__ == "__main__":
    # Test database manager
    print("=" * 60)
    print("Database Manager Test")
    print("=" * 60)
    
    from config import DatabaseConfig
    
    # Create configuration
    config = DatabaseConfig.from_env()
    print(f"\nConfiguration: {config}")
    
    # Create manager
    print(f"\nCreating database manager...")
    manager = DatabaseManager(config, use_pool=True)
    
    # Test connection
    print(f"\nTesting connection...")
    if manager.test_connection():
        print("✓ Connection successful")
        
        # Test queries
        try:
            # Check if properties table exists
            if manager.table_exists('properties'):
                count = manager.get_table_count('properties')
                print(f"\nProperties table: {count} rows")
            
            # Get statistics
            stats = manager.get_statistics()
            print(f"\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"✗ Query test failed: {e}")
    else:
        print("✗ Connection failed")
    
    # Cleanup
    print(f"\nClosing manager...")
    manager.close()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)
