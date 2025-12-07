"""
Database Configuration for EcoAssist
SQL Server connection settings
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    driver: str = 'ODBC Driver 17 for SQL Server'
    server: str = 'jiyan'
    database: str = 'EcoAssist'
    use_windows_auth: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    trust_server_certificate: bool = True
    connection_timeout: int = 30
    command_timeout: int = 60
    pool_size: int = 10
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables"""
        return cls(
            driver=os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server'),
            server=os.getenv('DB_SERVER', 'jiyan'),
            database=os.getenv('DB_DATABASE', 'EcoAssist'),
            use_windows_auth=os.getenv('DB_USE_WINDOWS_AUTH', 'true').lower() == 'true',
            username=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            trust_server_certificate=os.getenv('DB_TRUST_CERT', 'true').lower() == 'true',
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', '30')),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', '60')),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10'))
        )
    
    def get_connection_string(self) -> str:
        """Build connection string for pyodbc"""
        parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.server}",
            f"DATABASE={self.database}",
        ]
        
        if self.use_windows_auth:
            parts.append("Trusted_Connection=yes")
        else:
            if self.username and self.password:
                parts.append(f"UID={self.username}")
                parts.append(f"PWD={self.password}")
            else:
                raise ValueError("Username and password required for SQL authentication")
        
        if self.trust_server_certificate:
            parts.append("TrustServerCertificate=yes")
        
        return ";".join(parts) + ";"


# Default configuration
DEFAULT_CONFIG = DatabaseConfig()

# Configuration for different environments
DEVELOPMENT_CONFIG = DatabaseConfig(
    server='jiyan',
    database='EcoAssist',
    use_windows_auth=True
)

PRODUCTION_CONFIG = DatabaseConfig(
    server=os.getenv('PROD_DB_SERVER', 'production-server'),
    database=os.getenv('PROD_DB_DATABASE', 'EcoAssist'),
    use_windows_auth=False,
    username=os.getenv('PROD_DB_USERNAME'),
    password=os.getenv('PROD_DB_PASSWORD')
)


def get_config(environment: str = 'development') -> DatabaseConfig:
    """
    Get database configuration for specified environment
    
    Args:
        environment: 'development', 'production', or 'from_env'
    
    Returns:
        DatabaseConfig instance
    """
    if environment == 'development':
        return DEVELOPMENT_CONFIG
    elif environment == 'production':
        return PRODUCTION_CONFIG
    elif environment == 'from_env':
        return DatabaseConfig.from_env()
    else:
        return DEFAULT_CONFIG


# Example usage in .env file:
"""
# SQL Server Configuration
DB_SERVER=jiyan
DB_DATABASE=EcoAssist
DB_USE_WINDOWS_AUTH=true
DB_TRUST_CERT=true

# For SQL Server Authentication (if not using Windows Auth)
# DB_USE_WINDOWS_AUTH=false
# DB_USERNAME=ecoassist_user
# DB_PASSWORD=YourPassword123!

# Connection settings
DB_CONNECTION_TIMEOUT=30
DB_COMMAND_TIMEOUT=60
DB_POOL_SIZE=10
"""
