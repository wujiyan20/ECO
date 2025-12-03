# api_core.py - Core API Configuration and Common Models
# EcoAssist AI REST API Layer - Version 2.0
# Comprehensive implementation based on specification Ver2.0

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, constr
from typing import List, Dict, Optional, Union, Any, Literal
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uuid
import time
import hashlib
import logging
import json
import jwt
from collections import defaultdict

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS AND ENUMERATIONS
# =============================================================================

class ApprovalStatus(str, Enum):
    """Approval status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"
    REVISED = "revised"

class AllocationMethod(str, Enum):
    """Target allocation method enumeration"""
    CARBON_INTENSITY_WEIGHTED = "carbon_intensity_weighted"
    PROPORTIONAL = "proportional"
    RETROFIT_POTENTIAL = "retrofit_potential"
    EQUAL_DISTRIBUTION = "equal_distribution"
    AI_OPTIMIZED = "ai_optimized"

class ScenarioType(str, Enum):
    """Scenario type enumeration"""
    STANDARD = "Standard"
    AGGRESSIVE = "Aggressive"
    CONSERVATIVE = "Conservative"
    CUSTOM = "Custom"

class ImplementationStatus(str, Enum):
    """Implementation status enumeration"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class OnTrackStatus(str, Enum):
    """Progress tracking status enumeration"""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    AHEAD = "ahead"

class DataType(str, Enum):
    """Data synchronization types"""
    MILESTONES = "milestones"
    BASELINES = "baselines"
    COSTS = "costs"
    ACTUAL_EMISSIONS = "actual_emissions"
    ACTUAL_COSTS = "actual_costs"
    PERFORMANCE_METRICS = "performance_metrics"

class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    email: str
    role: str = "user"
    permissions: List[str] = Field(default_factory=list)

def verify_token(token: str) -> TokenData:
    """
    Verify JWT token and extract user data
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        email = payload.get("email", "")
        role = payload.get("role", "user")
        permissions = payload.get("permissions", [])
        
        if user_id is None:
            raise ValueError("Invalid token: missing user_id")
        
        return TokenData(
            user_id=user_id,
            email=email,
            role=role,
            permissions=permissions
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> TokenData:
    """
    Dependency to get current authenticated user from token
    """
    token = credentials.credentials
    return verify_token(token)


# API Configuration Constants
API_VERSION = "v1"
BASE_URL = f"/api/{API_VERSION}"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour
REFRESH_TOKEN_EXPIRY_SECONDS = 86400  # 24 hours
SECRET_KEY = "ecoassist-secret-key-change-in-production"  # Should be environment variable
ALGORITHM = "HS256"

# Rate limiting constants
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# =============================================================================
# STANDARDIZED RESPONSE MODELS
# =============================================================================

class APIResponse(BaseModel):
    """
    Standardized API response format for all endpoints
    Follows RFC 9110 HTTP status codes
    """
    status_code: int = Field(
        description="HTTP status code following RFC 9110",
        ge=100,
        le=599
    )
    status_message: str = Field(
        description="Human-readable status message"
    )
    request_id: str = Field(
        description="Unique identifier (UUID v4) for request tracking and logging"
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Response payload data"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status_code": 200,
                "status_message": "Success",
                "request_id": "7f3e8a2b-c4d5-4e1f-9a8b-3c5d7e9f1a2b",
                "data": {}
            }
        }

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(description="Error code for programmatic handling")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context and details"
    )

class ErrorResponse(BaseModel):
    """
    Standardized error response format
    """
    status_code: int = Field(description="HTTP status code")
    status_message: str = Field(description="Human-readable status message")
    request_id: str = Field(description="Request tracking identifier")
    error: ErrorDetail = Field(description="Error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status_code": 400,
                "status_message": "Bad Request",
                "request_id": "7f3e8a2b-c4d5-4e1f-9a8b-3c5d7e9f1a2b",
                "error": {
                    "code": "INVALID_PARAMETER",
                    "message": "Invalid parameter value",
                    "details": {}
                }
            }
        }


class AuthenticationRequest(BaseModel):
    """Authentication request model"""
    username: str = Field(description="Username or email")
    password: str = Field(description="User password")

class AuthenticationResponse(BaseModel):
    """Authentication response model"""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(description="Token expiry time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_info: Dict[str, Any] = Field(description="User information")

# Additional sub-models for API responses
class ImplementationTiming(BaseModel):
    """Implementation timing details"""
    year: int
    quarter: Optional[int] = None

class ConsumptionImpact(BaseModel):
    """Consumption impact details"""
    reduction_kwh: float
    reduction_percentage: float

class EmissionReduction(BaseModel):
    """Emission reduction details"""
    scope1_reduction: float
    scope2_reduction: float
    total_reduction: float
    unit: str = "kg-CO2e"

class CostDetails(BaseModel):
    """Cost breakdown details"""
    capex: float
    opex: float
    total: float
    unit: str = "USD"


# =============================================================================
# COMMON REQUEST/RESPONSE SUB-MODELS
# =============================================================================

class BaselineDataRecord(BaseModel):
    """Baseline emission data record"""
    year: int = Field(
        ge=2000,
        le=2100,
        description="Year of baseline data"
    )
    scope1_emissions: float = Field(
        ge=0,
        description="Scope 1 emissions in kg-CO2e"
    )
    scope2_emissions: float = Field(
        ge=0,
        description="Scope 2 emissions in kg-CO2e"
    )
    total_consumption: float = Field(
        ge=0,
        description="Total energy consumption"
    )
    total_cost: float = Field(
        ge=0,
        description="Total energy cost"
    )
    unit: str = Field(
        default="kg-CO2e",
        description="Emission unit"
    )

class ReductionRate(BaseModel):
    """Custom reduction rate specification"""
    year: int = Field(ge=2025, le=2070)
    target_reduction_percentage: float = Field(
        ge=0,
        le=100,
        description="Target reduction percentage for the year"
    )

class StrategyPreferences(BaseModel):
    """Strategy allocation preferences"""
    renewable_energy_weight: float = Field(
        default=0.33,
        ge=0,
        le=1,
        description="Weight for renewable energy strategies"
    )
    efficiency_improvement_weight: float = Field(
        default=0.33,
        ge=0,
        le=1,
        description="Weight for efficiency improvement strategies"
    )
    behavioral_change_weight: float = Field(
        default=0.34,
        ge=0,
        le=1,
        description="Weight for behavioral change strategies"
    )

class ReductionTarget(BaseModel):
    """Annual reduction target"""
    year: int = Field(description="Target year")
    target_emissions: float = Field(description="Target emission level")
    reduction_from_baseline: float = Field(description="Percentage reduction from baseline")
    cumulative_reduction: float = Field(description="Cumulative reduction amount")
    unit: str = Field(default="kg-CO2e")

class CostProjection(BaseModel):
    """Cost projection for a specific year"""
    year: int = Field(description="Projection year")
    estimated_cost: float = Field(description="Estimated total cost")
    breakdown: Dict[str, float] = Field(
        description="Cost breakdown (capex, opex)"
    )
    unit: str = Field(default="USD")

class StrategyBreakdown(BaseModel):
    """Strategy distribution breakdown"""
    renewable_energy_percentage: float = Field(
        ge=0,
        le=100,
        description="Renewable energy strategy percentage"
    )
    efficiency_improvement_percentage: float = Field(
        ge=0,
        le=100,
        description="Efficiency improvement strategy percentage"
    )
    behavioral_change_percentage: float = Field(
        ge=0,
        le=100,
        description="Behavioral change strategy percentage"
    )

class CalculationMetadata(BaseModel):
    """Metadata about calculation"""
    calculated_at: datetime = Field(description="Calculation timestamp")
    algorithm_version: str = Field(description="Algorithm version used")
    base_year: int = Field(description="Base year for calculations")
    target_years: List[int] = Field(description="Target years")

class ImplementationTiming(BaseModel):
    """Implementation timing information"""
    start_year: int = Field(description="Implementation start year")
    start_quarter: str = Field(description="Start quarter (Q1-Q4)")
    completion_year: int = Field(description="Completion year")
    completion_quarter: str = Field(description="Completion quarter")
    duration_months: int = Field(ge=0, description="Duration in months")

class ConsumptionImpact(BaseModel):
    """Consumption impact details"""
    electricity_reduction: float = Field(ge=0, description="Electricity reduction")
    electricity_unit: str = Field(default="kWh/year")
    gas_reduction: float = Field(ge=0, description="Gas reduction")
    gas_unit: str = Field(default="m³/year")

class EmissionReduction(BaseModel):
    """Emission reduction details"""
    annual_reduction: float = Field(ge=0, description="Annual emission reduction")
    cumulative_reduction: float = Field(ge=0, description="Cumulative reduction")
    unit: str = Field(default="kg-CO2e")

class CostDetails(BaseModel):
    """Detailed cost information"""
    capex: float = Field(ge=0, description="Capital expenditure")
    opex_annual: float = Field(ge=0, description="Annual operating expenditure")
    savings_annual: float = Field(ge=0, description="Annual cost savings")
    roi_years: float = Field(ge=0, description="Return on investment in years")
    currency: str = Field(default="USD")

# =============================================================================
# SECURITY AND AUTHENTICATION
# =============================================================================

security = HTTPBearer()

class TokenData(BaseModel):
    """JWT token data structure"""
    user_id: str
    email: str
    role: str
    permissions: List[str]
    exp: datetime

class AuthenticationRequest(BaseModel):
    """Authentication request model"""
    username: constr(min_length=3) = Field(description="Username or email")
    password: constr(min_length=6) = Field(description="User password")
    client_id: str = Field(default="ecoassist_web", description="Client application ID")
    grant_type: str = Field(default="password", description="OAuth grant type")

class AuthenticationResponse(BaseModel):
    """Authentication response model"""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(description="Token expiry in seconds")
    refresh_token: str = Field(description="Refresh token for obtaining new access token")
    user_info: Dict[str, Any] = Field(description="User information")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=TOKEN_EXPIRY_SECONDS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenData(
            user_id=payload.get("user_id"),
            email=payload.get("email"),
            role=payload.get("role"),
            permissions=payload.get("permissions", []),
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
        return token_data
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    Dependency to get current authenticated user from JWT token
    """
    token = credentials.credentials
    token_data = verify_token(token)
    return token_data

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Advanced rate limiter with per-user tracking and configurable limits
    """
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = RATE_LIMIT_WINDOW
        self.max_requests = RATE_LIMIT_REQUESTS
        self.blocked_clients: Dict[str, datetime] = {}
        self.block_duration = timedelta(minutes=15)
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Remove requests outside the time window"""
        cutoff_time = current_time - self.cleanup_interval
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if timestamp > cutoff_time
        ]
    
    def is_blocked(self, client_id: str) -> bool:
        """Check if client is temporarily blocked"""
        if client_id in self.blocked_clients:
            if datetime.utcnow() < self.blocked_clients[client_id]:
                return True
            else:
                del self.blocked_clients[client_id]
        return False
    
    def block_client(self, client_id: str):
        """Temporarily block a client"""
        self.blocked_clients[client_id] = datetime.utcnow() + self.block_duration
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limits
        Returns True if allowed, False if rate limit exceeded
        """
        if self.is_blocked(client_id):
            logger.warning(f"Blocked client attempted request: {client_id}")
            return False
        
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        
        if len(self.requests[client_id]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            self.block_client(client_id)
            return False
        
        self.requests[client_id].append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get number of remaining requests for client"""
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        return max(0, self.max_requests - len(self.requests[client_id]))
    
    def get_reset_time(self, client_id: str) -> Optional[datetime]:
        """Get time when rate limit resets for client"""
        if not self.requests[client_id]:
            return None
        oldest_request = min(self.requests[client_id])
        reset_time = datetime.fromtimestamp(oldest_request + self.cleanup_interval)
        return reset_time

# Global rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit_dependency(request: Request) -> bool:
    """
    Dependency function for rate limiting
    """
    # Get client identifier (IP address or user ID from token)
    client_id = request.client.host
    
    # Try to get user ID from authorization header for authenticated requests
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            token_data = verify_token(token)
            client_id = token_data.user_id
        except:
            pass  # Use IP address if token verification fails
    
    allowed = await rate_limiter.check_rate_limit(client_id)
    
    if not allowed:
        remaining = rate_limiter.get_remaining_requests(client_id)
        reset_time = rate_limiter.get_reset_time(client_id)
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Rate limit exceeded",
                "limit": rate_limiter.max_requests,
                "window_seconds": rate_limiter.cleanup_interval,
                "remaining": remaining,
                "reset_at": reset_time.isoformat() if reset_time else None
            }
        )
    
    return True

# =============================================================================
# REQUEST ID MIDDLEWARE
# =============================================================================

class RequestIDMiddleware:
    """
    Middleware to add unique request ID to all requests and responses
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())
            scope["request_id"] = request_id
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", request_id.encode()))
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID (UUID v4)"""
    return str(uuid.uuid4())

def generate_calculation_id() -> str:
    """Generate calculation ID with timestamp"""
    timestamp = int(time.time())
    random_suffix = uuid.uuid4().hex[:6]
    return f"CALC_{timestamp}_{random_suffix}"

def get_http_status_message(status_code: int) -> str:
    """Get standard HTTP status message for status code"""
    status_messages = {
        200: "Success",
        201: "Created",
        202: "Accepted",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        409: "Conflict",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout"
    }
    return status_messages.get(status_code, "Unknown Status")

def create_success_response(
    data: Any,
    request_id: str,
    status_code: int = 200,
    message: Optional[str] = None
) -> APIResponse:
    """Create standardized success response"""
    return APIResponse(
        status_code=status_code,
        status_message=message or get_http_status_message(status_code),
        request_id=request_id,
        data=data if isinstance(data, dict) else {"result": data}
    )

def create_error_response(
    error_code: str,
    error_message: str,
    request_id: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        status_code=status_code,
        status_message=get_http_status_message(status_code),
        request_id=request_id,
        error=ErrorDetail(
            code=error_code,
            message=error_message,
            details=details or {}
        )
    )

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

def measure_execution_time(func):
    """Decorator to measure and log execution time"""
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"{func.__name__} executed in {execution_time:.2f}ms")
        
        # Add execution time to response if it's an APIResponse
        if isinstance(result, APIResponse) and hasattr(result, 'data'):
            if result.data is None:
                result.data = {}
            result.data['execution_time_ms'] = round(execution_time, 2)
        
        return result
    
    return wrapper

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_year_range(base_year: int, target_year: int, min_gap: int = 5) -> bool:
    """Validate year range between base and target"""
    if target_year <= base_year:
        raise ValueError(f"Target year must be after base year")
    if target_year - base_year < min_gap:
        raise ValueError(f"Target year must be at least {min_gap} years after base year")
    return True

def validate_property_ids(property_ids: List[str]) -> bool:
    """Validate property ID format (UUID)"""
    for prop_id in property_ids:
        try:
            uuid.UUID(prop_id)
        except ValueError:
            raise ValueError(f"Invalid property ID format: {prop_id}")
    return True

def validate_percentage(value: float, field_name: str) -> bool:
    """Validate percentage value (0-100)"""
    if not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return True

# =============================================================================
# INITIALIZATION
# =============================================================================

logger.info("API Core module initialized successfully")
logger.info(f"API Version: {API_VERSION}")
logger.info(f"Rate Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds")
