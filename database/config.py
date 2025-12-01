# database/config.py - Database Configuration
"""
Database configuration with environment variable support
Supports multiple environments (development, staging, production)
"""

import os
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration with environment variable support"""
    
    server: str
    database: str
    username: str
    password: str
    driver: str
    port: Optional[int] = None
    timeout: int = 30
    pool_size: int = 5
    
    @classmethod
    def from_env(cls, env: str = 'development') -> 'DatabaseConfig':
        """
        Create configuration from environment variables
        
        Environment variables:
            DB_SERVER: Server address (default: localhost)
            DB_NAME: Database name (default: EcoAssistDB)
            DB_USERNAME: Database username (default: sa)
            DB_PASSWORD: Database password (required)
            DB_DRIVER: ODBC driver (default: {ODBC Driver 17 for SQL Server})
            DB_PORT: Port number (optional)
            DB_TIMEOUT: Connection timeout in seconds (default: 30)
            DB_POOL_SIZE: Connection pool size (default: 5)
        """
        # Try to load from .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("✓ Loaded configuration from .env file")
        except ImportError:
            logger.info("python-dotenv not installed, using environment variables only")
        
        server = os.getenv('DB_SERVER', 'localhost')
        database = os.getenv('DB_NAME', 'EcoAssistDB')
        username = os.getenv('DB_USERNAME', 'sa')
        password = os.getenv('DB_PASSWORD', '')
        driver = os.getenv('DB_DRIVER', '{ODBC Driver 17 for SQL Server}')
        port = int(os.getenv('DB_PORT')) if os.getenv('DB_PORT') else None
        timeout = int(os.getenv('DB_TIMEOUT', '30'))
        pool_size = int(os.getenv('DB_POOL_SIZE', '5'))
        
        if not password:
            logger.warning("⚠ DB_PASSWORD not set - database connection may fail")
        
        return cls(
            server=server,
            database=database,
            username=username,
            password=password,
            driver=driver,
            port=port,
            timeout=timeout,
            pool_size=pool_size
        )
    
    @classmethod
    def development(cls) -> 'DatabaseConfig':
        """Development environment configuration"""
        return cls.from_env('development')
    
    @classmethod
    def production(cls) -> 'DatabaseConfig':
        """Production environment configuration"""
        return cls.from_env('production')
    
    @classmethod
    def testing(cls) -> 'DatabaseConfig':
        """Testing environment configuration"""
        config = cls.from_env('testing')
        config.database = os.getenv('TEST_DB_NAME', 'EcoAssistDB_Test')
        return config
    
    def get_connection_string(self, trust_certificate: bool = True) -> str:
        """
        Generate ODBC connection string
        
        Args:
            trust_certificate: If True, adds TrustServerCertificate=yes
        """
        conn_str = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server}"
        )
        
        if self.port:
            conn_str += f",{self.port}"
        
        conn_str += f";DATABASE={self.database};"
        
        # Authentication
        if self.username and self.password:
            conn_str += f"UID={self.username};PWD={self.password};"
        else:
            # Windows Authentication
            conn_str += "Trusted_Connection=yes;"
        
        # Additional parameters
        if trust_certificate:
            conn_str += "TrustServerCertificate=yes;"
        
        conn_str += f"Connection Timeout={self.timeout};"
        
        return conn_str
    
    def get_connection_string_masked(self) -> str:
        """Get connection string with masked password for logging"""
        conn_str = self.get_connection_string()
        if self.password:
            conn_str = conn_str.replace(self.password, "****")
        return conn_str
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.server:
            logger.error("Server address is required")
            return False
        
        if not self.database:
            logger.error("Database name is required")
            return False
        
        if not self.driver:
            logger.error("ODBC driver is required")
            return False
        
        # Check if password is set (unless using Windows auth)
        if self.username and not self.password:
            logger.warning("Username set but password is empty - may fail unless using Windows auth")
        
        return True
    
    def __repr__(self) -> str:
        """String representation with masked password"""
        return (
            f"DatabaseConfig(server='{self.server}', "
            f"database='{self.database}', "
            f"username='{self.username}', "
            f"password='****', "
            f"driver='{self.driver}')"
        )

# Predefined configurations for different SQL Server scenarios
class CommonConfigs:
    """Common database configurations"""
    
    @staticmethod
    def local_sqlexpress() -> DatabaseConfig:
        """Local SQL Server Express configuration"""
        return DatabaseConfig(
            server="localhost\\SQLEXPRESS",
            database="EcoAssistDB",
            username="sa",
            password=os.getenv('DB_PASSWORD', ''),
            driver="{ODBC Driver 17 for SQL Server}"
        )
    
    @staticmethod
    def azure_sql() -> DatabaseConfig:
        """Azure SQL Database configuration"""
        return DatabaseConfig(
            server=os.getenv('AZURE_SQL_SERVER', 'your-server.database.windows.net'),
            database=os.getenv('AZURE_SQL_DATABASE', 'EcoAssistDB'),
            username=os.getenv('AZURE_SQL_USERNAME', ''),
            password=os.getenv('AZURE_SQL_PASSWORD', ''),
            driver="{ODBC Driver 17 for SQL Server}",
            port=1433
        )
    
    @staticmethod
    def docker_sql() -> DatabaseConfig:
        """Docker SQL Server configuration"""
        return DatabaseConfig(
            server=os.getenv('DOCKER_SQL_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'EcoAssistDB'),
            username="sa",
            password=os.getenv('DB_PASSWORD', 'YourPassword123!'),
            driver="{ODBC Driver 17 for SQL Server}",
            port=int(os.getenv('DOCKER_SQL_PORT', '1433'))
        )

# Example .env file content
ENV_FILE_TEMPLATE = """
# EcoAssist Database Configuration
# Copy this to .env and update with your values

# Database Connection
DB_SERVER=localhost
DB_NAME=EcoAssistDB
DB_USERNAME=sa
DB_PASSWORD=YourPassword123!
DB_DRIVER={ODBC Driver 17 for SQL Server}

# Optional Settings
# DB_PORT=1433
# DB_TIMEOUT=30
# DB_POOL_SIZE=5

# Environment
ENVIRONMENT=development

# For Azure SQL
# AZURE_SQL_SERVER=your-server.database.windows.net
# AZURE_SQL_DATABASE=EcoAssistDB
# AZURE_SQL_USERNAME=your-username
# AZURE_SQL_PASSWORD=your-password

# For Docker SQL
# DOCKER_SQL_HOST=localhost
# DOCKER_SQL_PORT=1433
"""

def create_env_template(filepath: str = '.env.template'):
    """Create .env template file"""
    try:
        with open(filepath, 'w') as f:
            f.write(ENV_FILE_TEMPLATE.strip())
        logger.info(f"✓ Created {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to create {filepath}: {e}")
        return False

def test_driver_availability() -> list:
    """Test which ODBC drivers are available"""
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        logger.info(f"Available ODBC drivers: {drivers}")
        return drivers
    except ImportError:
        logger.error("pyodbc not installed")
        return []
    except Exception as e:
        logger.error(f"Error checking drivers: {e}")
        return []

# Default configuration instance
_default_config = None

def get_default_config() -> DatabaseConfig:
    """Get default database configuration"""
    global _default_config
    if _default_config is None:
        _default_config = DatabaseConfig.from_env()
    return _default_config

def set_default_config(config: DatabaseConfig):
    """Set default database configuration"""
    global _default_config
    _default_config = config

if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("Database Configuration Test")
    print("=" * 60)
    
    # Test environment-based config
    config = DatabaseConfig.from_env()
    print(f"\nConfiguration from environment:")
    print(f"  {config}")
    print(f"\nConnection string (masked):")
    print(f"  {config.get_connection_string_masked()}")
    
    # Test validation
    print(f"\nValidation: {'✓ Pass' if config.validate() else '✗ Fail'}")
    
    # Check available drivers
    print(f"\nAvailable ODBC drivers:")
    drivers = test_driver_availability()
    for driver in drivers:
        print(f"  • {driver}")
    
    # Create template
    print(f"\nCreating .env template...")
    create_env_template('.env.template')
    
    print("\n" + "=" * 60)
    print("Configuration test complete")
    print("=" * 60)
