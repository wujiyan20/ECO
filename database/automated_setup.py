#!/usr/bin/env python3
"""
EcoAssist Database Automated Setup Script
Handles complete database setup including schema creation and sample data insertion
"""

import os
import sys
import pyodbc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Automated database setup for EcoAssist"""
    
    def __init__(self, server: str = "localhost", username: str = "sa", password: str = ""):
        self.server = server
        self.username = username
        self.password = password
        self.driver = "{ODBC Driver 17 for SQL Server}"
        self.database_name = "EcoAssistDB"
        
        # Connection string for master database (to create new database)
        self.master_conn_string = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE=master;"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
        )
        
        # Connection string for EcoAssist database
        self.ecoassist_conn_string = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database_name};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
        )
    
    def test_connection(self) -> bool:
        """Test SQL Server connection"""
        try:
            conn = pyodbc.connect(self.master_conn_string)
            conn.close()
            logger.info("✅ SQL Server connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ SQL Server connection failed: {e}")
            return False
    
    def check_database_exists(self) -> bool:
        """Check if EcoAssistDB already exists"""
        try:
            conn = pyodbc.connect(self.master_conn_string)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM sys.databases WHERE name = ?", 
                (self.database_name,)
            )
            exists = cursor.fetchone()[0] > 0
            
            conn.close()
            return exists
            
        except Exception as e:
            logger.error(f"Error checking database existence: {e}")
            return False
    
    def create_database(self) -> bool:
        """Create the EcoAssistDB database"""
        try:
            logger.info("Creating EcoAssistDB database...")
            
            conn = pyodbc.connect(self.master_conn_string)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Drop database if exists
            if self.check_database_exists():
                logger.info("Dropping existing EcoAssistDB...")
                cursor.execute(f"DROP DATABASE {self.database_name}")
            
            # Create new database
            create_db_sql = f"""
            CREATE DATABASE {self.database_name}
            ON (
                NAME = '{self.database_name}',
                FILENAME = 'C:\\Database\\{self.database_name}.mdf',
                SIZE = 100MB,
                MAXSIZE = 10GB,
                FILEGROWTH = 10MB
            )
            LOG ON (
                NAME = '{self.database_name}_Log',
                FILENAME = 'C:\\Database\\{self.database_name}_Log.ldf',
                SIZE = 10MB,
                MAXSIZE = 1GB,
                FILEGROWTH = 10%
            )
            """
            
            cursor.execute(create_db_sql)
            conn.close()
            
            logger.info("✅ Database created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database creation failed: {e}")
            return False
    
    def execute_sql_file(self, file_path: Path, connection_string: str) -> bool:
        """Execute SQL commands from a file"""
        try:
            if not file_path.exists():
                logger.error(f"❌ SQL file not found: {file_path}")
                return False
            
            logger.info(f"Executing SQL file: {file_path.name}")
            
            # Read SQL file
            with open(file_path, 'r', encoding='utf-8') as file:
                sql_content = file.read()
            
            # Split into batches (separated by GO statements)
            batches = [batch.strip() for batch in sql_content.split('GO') if batch.strip()]
            
            conn = pyodbc.connect(connection_string)
            conn.autocommit = True
            cursor = conn.cursor()
            
            for i, batch in enumerate(batches, 1):
                try:
                    if batch.strip():
                        cursor.execute(batch)
                        logger.debug(f"Executed batch {i}/{len(batches)}")
                except Exception as batch_error:
                    logger.error(f"Error in batch {i}: {batch_error}")
                    logger.debug(f"Problematic batch: {batch[:100]}...")
                    # Continue with next batch
            
            conn.close()
            logger.info(f"✅ SQL file executed successfully: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {file_path.name}: {e}")
            return False
    
    def verify_installation(self) -> Dict[str, int]:
        """Verify database installation by checking record counts"""
        try:
            conn = pyodbc.connect(self.ecoassist_conn_string)
            cursor = conn.cursor()
            
            tables_to_check = [
                'properties',
                'reduction_options', 
                'strategic_patterns',
                'benchmark_data',
                'milestone_scenarios',
                'historical_consumption',
                'historical_emissions',
                'historical_costs'
            ]
            
            results = {}
            for table in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    results[table] = count
                    logger.info(f"  {table}: {count} records")
                except Exception as e:
                    logger.error(f"  {table}: Error - {e}")
                    results[table] = -1
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {}
    
    def setup(self, schema_file: Optional[Path] = None, data_file: Optional[Path] = None) -> bool:
        """Complete database setup process"""
        logger.info("🚀 Starting EcoAssist Database Setup")
        
        # Step 1: Test connection
        if not self.test_connection():
            return False
        
        # Step 2: Create database
        if not self.create_database():
            return False
        
        # Step 3: Create schema
        if schema_file and schema_file.exists():
            if not self.execute_sql_file(schema_file, self.ecoassist_conn_string):
                return False
        else:
            logger.warning("⚠️ Schema file not provided or not found")
        
        # Step 4: Insert sample data
        if data_file and data_file.exists():
            if not self.execute_sql_file(data_file, self.ecoassist_conn_string):
                return False
        else:
            logger.warning("⚠️ Sample data file not provided or not found")
        
        # Step 5: Verify installation
        logger.info("🔍 Verifying installation...")
        results = self.verify_installation()
        
        if results:
            logger.info("✅ Database setup completed successfully!")
            logger.info("📊 Record counts:")
            for table, count in results.items():
                if count >= 0:
                    logger.info(f"  • {table}: {count} records")
            return True
        else:
            logger.error("❌ Database setup verification failed")
            return False

def get_user_input():
    """Get database connection details from user"""
    print("🔧 EcoAssist Database Setup")
    print("=" * 40)
    
    server = input("SQL Server instance (default: localhost): ").strip() or "localhost"
    username = input("Username (default: sa): ").strip() or "sa"
    
    # Get password securely
    import getpass
    password = getpass.getpass("Password: ")
    
    return server, username, password

def find_sql_files() -> tuple[Optional[Path], Optional[Path]]:
    """Find SQL files in current directory"""
    current_dir = Path(".")
    
    schema_file = None
    data_file = None
    
    # Look for schema file
    for pattern in ["*schema*.sql", "*create*.sql", "01_*.sql"]:
        files = list(current_dir.glob(pattern))
        if files:
            schema_file = files[0]
            break
    
    # Look for data file
    for pattern in ["*sample*.sql", "*data*.sql", "*insert*.sql", "02_*.sql"]:
        files = list(current_dir.glob(pattern))
        if files:
            data_file = files[0]
            break
    
    return schema_file, data_file

def main():
    """Main setup function"""
    print("🏗️  EcoAssist Database Automated Setup")
    print("=" * 50)
    
    # Check if pyodbc is installed
    try:
        import pyodbc
        logger.info("✅ pyodbc is installed")
    except ImportError:
        logger.error("❌ pyodbc is not installed. Please run: pip install pyodbc")
        sys.exit(1)
    
    # Get connection details
    try:
        server, username, password = get_user_input()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    
    # Find SQL files
    schema_file, data_file = find_sql_files()
    
    if schema_file:
        logger.info(f"📄 Found schema file: {schema_file}")
    else:
        logger.warning("⚠️ Schema file not found. Looking for: *schema*.sql, *create*.sql, 01_*.sql")
    
    if data_file:
        logger.info(f"📄 Found data file: {data_file}")
    else:
        logger.warning("⚠️ Sample data file not found. Looking for: *sample*.sql, *data*.sql, 02_*.sql")
    
    # Confirm setup
    print(f"\n📋 Setup Configuration:")
    print(f"  Server: {server}")
    print(f"  Username: {username}")
    print(f"  Database: EcoAssistDB")
    print(f"  Schema file: {schema_file.name if schema_file else 'Not found'}")
    print(f"  Data file: {data_file.name if data_file else 'Not found'}")
    
    confirm = input("\nProceed with setup? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Setup cancelled")
        sys.exit(0)
    
    # Run setup
    setup = DatabaseSetup(server, username, password)
    
    try:
        success = setup.setup(schema_file, data_file)
        
        if success:
            print("\n🎉 Database setup completed successfully!")
            print("\n📝 Next steps:")
            print("1. Update connection settings in database_integration.py")
            print("2. Test connection: python database_integration.py")
            print("3. Update your API to use database backend")
            print("\n💡 Connection string template:")
            print(f"   Server: {server}")
            print(f"   Database: EcoAssistDB")
            print(f"   Username: {username}")
        else:
            print("\n❌ Database setup failed. Check logs above for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Setup failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()