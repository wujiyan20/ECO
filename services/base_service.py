# services/base_service.py - Base Service Class
"""
Base service class providing common functionality for all EcoAssist services.

Features:
- Database connection management
- Transaction handling
- Error handling and logging
- Caching support
- Performance monitoring
- Retry logic for transient failures
"""

import logging
import time
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import traceback
import threading
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic operations
T = TypeVar('T')


# =============================================================================
# SERVICE RESULT CLASSES
# =============================================================================

class ServiceResultStatus(Enum):
    """Status of service operation"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"


@dataclass
class ServiceResult(Generic[T]):
    """
    Standardized result wrapper for all service operations.
    
    Provides consistent error handling and metadata tracking.
    """
    status: ServiceResultStatus
    data: Optional[T] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_success(self) -> bool:
        """Check if operation was successful"""
        return self.status == ServiceResultStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if operation failed"""
        return self.status in [
            ServiceResultStatus.ERROR,
            ServiceResultStatus.VALIDATION_ERROR,
            ServiceResultStatus.TIMEOUT
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def success(cls, data: T, message: str = "Operation successful", 
                metadata: Dict[str, Any] = None) -> 'ServiceResult[T]':
        """Create a success result"""
        return cls(
            status=ServiceResultStatus.SUCCESS,
            data=data,
            message=message,
            metadata=metadata or {}
        )
    
    @classmethod
    def error(cls, message: str, errors: List[str] = None,
              metadata: Dict[str, Any] = None) -> 'ServiceResult[T]':
        """Create an error result"""
        return cls(
            status=ServiceResultStatus.ERROR,
            message=message,
            errors=errors or [message],
            metadata=metadata or {}
        )
    
    @classmethod
    def not_found(cls, message: str = "Resource not found") -> 'ServiceResult[T]':
        """Create a not found result"""
        return cls(
            status=ServiceResultStatus.NOT_FOUND,
            message=message
        )
    
    @classmethod
    def validation_error(cls, errors: List[str], 
                        message: str = "Validation failed") -> 'ServiceResult[T]':
        """Create a validation error result"""
        return cls(
            status=ServiceResultStatus.VALIDATION_ERROR,
            message=message,
            errors=errors
        )


# =============================================================================
# CACHING SUPPORT
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    value: Any
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    hits: int = 0


class SimpleCache:
    """
    Thread-safe in-memory cache with TTL support.
    
    For production, replace with Redis or similar.
    """
    
    def __init__(self, default_ttl_seconds: int = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl_seconds
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if datetime.utcnow() > entry.expires_at:
                # Entry expired
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: int = None) -> None:
        """Set value in cache"""
        with self._lock:
            ttl = ttl_seconds or self._default_ttl
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=datetime.utcnow() + timedelta(seconds=ttl)
            )
            self._stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def clear_expired(self) -> int:
        """Clear expired entries, return count of cleared"""
        with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                k for k, v in self._cache.items() 
                if now > v.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0
            return {
                **self._stats,
                "entries": len(self._cache),
                "hit_rate": hit_rate
            }


# Global cache instance
_service_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """Get the global cache instance"""
    return _service_cache


# =============================================================================
# DECORATORS
# =============================================================================

def measure_time(func: Callable) -> Callable:
    """Decorator to measure execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # If result is ServiceResult, add execution time
        if isinstance(result, ServiceResult):
            result.execution_time_ms = elapsed_ms
        
        logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f}ms")
        return result
    return wrapper


def cached(ttl_seconds: int = 300, key_prefix: str = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Cache time-to-live in seconds
        key_prefix: Prefix for cache key (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or func.__name__
            key_data = json.dumps({
                "args": str(args[1:]),  # Skip self
                "kwargs": str(sorted(kwargs.items()))
            }, sort_keys=True)
            cache_key = f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Check cache
            cache = get_cache()
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache successful results
            if isinstance(result, ServiceResult) and result.is_success:
                cache.set(cache_key, result, ttl_seconds)
            elif not isinstance(result, ServiceResult):
                cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay_seconds: float = 1.0,
          backoff_multiplier: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    Decorator to retry failed operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = delay_seconds
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} "
                        f"failed: {str(e)}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        delay *= backoff_multiplier
            
            # All retries failed
            logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        return wrapper
    return decorator


def transaction(func: Callable) -> Callable:
    """
    Decorator to wrap operation in database transaction.
    
    Automatically commits on success, rolls back on failure.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'db_manager') or self.db_manager is None:
            return func(self, *args, **kwargs)
        
        try:
            # Start transaction
            with self.db_manager.transaction():
                result = func(self, *args, **kwargs)
                return result
        except Exception as e:
            logger.error(f"Transaction failed in {func.__name__}: {str(e)}")
            raise
    return wrapper


def validate_input(*validators: Callable):
    """
    Decorator to validate input parameters.
    
    Args:
        validators: Functions that raise ValueError on invalid input
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            errors = []
            for validator in validators:
                try:
                    validator(*args, **kwargs)
                except ValueError as e:
                    errors.append(str(e))
            
            if errors:
                return ServiceResult.validation_error(errors)
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# BASE SERVICE CLASS
# =============================================================================

class BaseService(ABC):
    """
    Abstract base class for all EcoAssist services.
    
    Provides:
    - Database connection management
    - Caching support
    - Error handling
    - Logging
    - Performance monitoring
    
    Usage:
        class MyService(BaseService):
            def __init__(self, db_manager=None):
                super().__init__(db_manager)
            
            def my_operation(self, data):
                return self._execute(self._do_operation, data)
    """
    
    def __init__(self, db_manager=None, cache: SimpleCache = None):
        """
        Initialize base service.
        
        Args:
            db_manager: Database manager instance (optional)
            cache: Cache instance (optional, uses global cache if not provided)
        """
        self.db_manager = db_manager
        self.cache = cache or get_cache()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._initialization_time = datetime.utcnow()
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    def initialize(self) -> ServiceResult[bool]:
        """
        Initialize service resources.
        Override in subclasses for custom initialization.
        """
        try:
            self._do_initialize()
            self._initialized = True
            self._logger.info(f"{self.__class__.__name__} initialized")
            return ServiceResult.success(True, "Service initialized")
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            return ServiceResult.error(f"Initialization failed: {str(e)}")
    
    def _do_initialize(self) -> None:
        """Override for custom initialization logic"""
        pass
    
    def shutdown(self) -> ServiceResult[bool]:
        """
        Cleanup service resources.
        Override in subclasses for custom cleanup.
        """
        try:
            self._do_shutdown()
            self._initialized = False
            self._logger.info(f"{self.__class__.__name__} shutdown")
            return ServiceResult.success(True, "Service shutdown")
        except Exception as e:
            self._logger.error(f"Shutdown failed: {str(e)}")
            return ServiceResult.error(f"Shutdown failed: {str(e)}")
    
    def _do_shutdown(self) -> None:
        """Override for custom shutdown logic"""
        pass
    
    def health_check(self) -> ServiceResult[Dict[str, Any]]:
        """
        Check service health status.
        
        Returns:
            Health status including database connectivity
        """
        health_data = {
            "service": self.__class__.__name__,
            "initialized": self._initialized,
            "uptime_seconds": (datetime.utcnow() - self._initialization_time).total_seconds(),
            "database_connected": False,
            "cache_stats": self.cache.get_stats()
        }
        
        # Check database connection
        if self.db_manager is not None:
            try:
                health_data["database_connected"] = self.db_manager.test_connection()
            except Exception as e:
                health_data["database_error"] = str(e)
        
        status = ServiceResultStatus.SUCCESS if self._initialized else ServiceResultStatus.ERROR
        return ServiceResult(status=status, data=health_data)
    
    def _execute(self, operation: Callable, *args, **kwargs) -> ServiceResult:
        """
        Execute an operation with standard error handling.
        
        Args:
            operation: The operation to execute
            *args, **kwargs: Arguments for the operation
            
        Returns:
            ServiceResult with operation outcome
        """
        start_time = time.perf_counter()
        
        try:
            result = operation(*args, **kwargs)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            if isinstance(result, ServiceResult):
                result.execution_time_ms = execution_time
                return result
            
            return ServiceResult.success(
                data=result,
                metadata={"execution_time_ms": execution_time}
            )
            
        except ValueError as e:
            self._logger.warning(f"Validation error: {str(e)}")
            return ServiceResult.validation_error([str(e)])
            
        except Exception as e:
            self._logger.error(f"Operation failed: {str(e)}\n{traceback.format_exc()}")
            return ServiceResult.error(
                message=f"Operation failed: {str(e)}",
                metadata={"exception_type": type(e).__name__}
            )
    
    @contextmanager
    def _timed_operation(self, operation_name: str):
        """Context manager for timing operations"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._logger.debug(f"{operation_name} completed in {elapsed:.2f}ms")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache with logging"""
        value = self.cache.get(key)
        if value is not None:
            self._logger.debug(f"Cache hit: {key}")
        return value
    
    def _set_in_cache(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache"""
        self.cache.set(key, value, ttl_seconds)
    
    def _invalidate_cache(self, key: str) -> bool:
        """Invalidate cache entry"""
        return self.cache.delete(key)
    
    def _clear_cache(self) -> None:
        """Clear all cache entries for this service"""
        self.cache.clear()
        self._logger.info("Cache cleared")


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """
    Registry for managing service instances.
    
    Provides dependency injection and lifecycle management.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._services: Dict[str, BaseService] = {}
                cls._instance._initialized = False
            return cls._instance
    
    def register(self, name: str, service: BaseService) -> None:
        """Register a service instance"""
        self._services[name] = service
        logger.info(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[BaseService]:
        """Get a registered service"""
        return self._services.get(name)
    
    def get_all(self) -> Dict[str, BaseService]:
        """Get all registered services"""
        return self._services.copy()
    
    def initialize_all(self) -> Dict[str, ServiceResult]:
        """Initialize all registered services"""
        results = {}
        for name, service in self._services.items():
            results[name] = service.initialize()
        self._initialized = True
        return results
    
    def shutdown_all(self) -> Dict[str, ServiceResult]:
        """Shutdown all registered services"""
        results = {}
        for name, service in self._services.items():
            results[name] = service.shutdown()
        self._initialized = False
        return results
    
    def health_check_all(self) -> Dict[str, ServiceResult]:
        """Health check all registered services"""
        return {
            name: service.health_check() 
            for name, service in self._services.items()
        }


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry"""
    return ServiceRegistry()


# =============================================================================
# SERVICE CONTEXT
# =============================================================================

@dataclass
class ServiceContext:
    """
    Context for service operations.
    
    Carries request-specific information through service calls.
    """
    request_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


# Thread-local storage for context
_context_local = threading.local()


def set_service_context(context: ServiceContext) -> None:
    """Set the current service context"""
    _context_local.context = context


def get_service_context() -> Optional[ServiceContext]:
    """Get the current service context"""
    return getattr(_context_local, 'context', None)


def clear_service_context() -> None:
    """Clear the current service context"""
    if hasattr(_context_local, 'context'):
        delattr(_context_local, 'context')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_service_context(request_id: str = None, user_id: str = None) -> ServiceContext:
    """Create a new service context"""
    import uuid
    return ServiceContext(
        request_id=request_id or str(uuid.uuid4()),
        user_id=user_id
    )


def batch_operation(items: List[T], operation: Callable[[T], ServiceResult],
                   batch_size: int = 100) -> ServiceResult[List[Any]]:
    """
    Execute operation on items in batches.
    
    Args:
        items: List of items to process
        operation: Operation to apply to each item
        batch_size: Size of each batch
        
    Returns:
        ServiceResult with list of results
    """
    results = []
    errors = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        for item in batch:
            result = operation(item)
            if result.is_success:
                results.append(result.data)
            else:
                errors.extend(result.errors)
    
    if errors:
        return ServiceResult(
            status=ServiceResultStatus.PARTIAL,
            data=results,
            errors=errors,
            message=f"Completed {len(results)}/{len(items)} items"
        )
    
    return ServiceResult.success(results)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result classes
    'ServiceResultStatus',
    'ServiceResult',
    
    # Caching
    'SimpleCache',
    'CacheEntry',
    'get_cache',
    
    # Decorators
    'measure_time',
    'cached',
    'retry',
    'transaction',
    'validate_input',
    
    # Base classes
    'BaseService',
    'ServiceRegistry',
    'get_service_registry',
    
    # Context
    'ServiceContext',
    'set_service_context',
    'get_service_context',
    'clear_service_context',
    'create_service_context',
    
    # Utilities
    'batch_operation'
]
