# api/api_module4_reoptimization.py - Module 4: Annual Reoptimization APIs
"""
Module 4: Annual Reoptimization APIs

Provides endpoints for:
- API 11: Calculate annual reoptimization based on actual performance
- API 12: Get reoptimization visualization data
- API 13: Register reoptimization adjustments

Author: EcoAssist API Team
Version: 1.0.0
"""

from fastapi import APIRouter, Depends, BackgroundTasks, Path, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

# Import API core utilities
from api_core import (
    APIResponse,
    create_success_response,
    create_error_response,
    get_current_user,
    TokenData,
    generate_request_id,
    measure_execution_time,
    check_rate_limit_dependency
)

# Import services
from services import ReoptimizationService, ServiceResultStatus

# Initialize
router = APIRouter(prefix="/planning/annual", tags=["annual-reoptimization"])
reoptimization_service = ReoptimizationService(db_manager=None, eel_client=None)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class BudgetAdjustmentRequest(BaseModel):
    """Budget adjustment for reoptimization"""
    annual_budget_limit: Optional[float] = Field(None, ge=0, description="Adjusted annual budget limit")
    currency: str = Field("USD", description="Currency code")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for adjustment")


class CalculateReoptimizationRequest(BaseModel):
    """Request for annual reoptimization calculation"""
    plan_id: str = Field(..., description="Long-term plan UUID to reoptimize")
    target_year: int = Field(..., ge=2020, le=2100, description="Year to reoptimize for")
    start_date: str = Field(..., description="Analysis start date (ISO 8601: YYYY-MM-DD)")
    end_date: str = Field(..., description="Analysis end date (ISO 8601: YYYY-MM-DD)")
    frequency: str = Field("quarterly", description="Data aggregation frequency (daily/weekly/monthly/quarterly/yearly)")
    budget_adjustment: Optional[BudgetAdjustmentRequest] = None
    
    @validator('frequency')
    def validate_frequency(cls, v):
        valid = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if v not in valid:
            raise ValueError(f"frequency must be one of: {', '.join(valid)}")
        return v
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("Date must be in ISO 8601 format (YYYY-MM-DD)")
        return v


class ReoptimizationApprovalInfo(BaseModel):
    """Approval information for reoptimization"""
    approved_by: str = Field(..., description="User UUID who approved")
    approval_date: str = Field(..., description="Approval date (ISO 8601 format)")
    comments: Optional[str] = Field(None, max_length=1000, description="Optional approval comments")


class RegisterReoptimizationRequest(BaseModel):
    """Request to register reoptimization adjustments"""
    reoptimization_pattern_id: str = Field(..., description="Reoptimization pattern UUID to register")
    plan_id: str = Field(..., description="Plan UUID to update")
    approval_info: ReoptimizationApprovalInfo


# =============================================================================
# API 11: Calculate Annual Reoptimization
# =============================================================================

@router.post("/reoptimize", response_model=APIResponse)
@measure_execution_time
async def calculate_annual_reoptimization(
    request: CalculateReoptimizationRequest,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 11: Calculate Annual Reoptimization
    
    Calculate annual plan reoptimization based on actual performance data.
    Automatically fetches actual emission and cost data from EEL server.
    
    **Authentication Required:** Bearer Token
    
    **Key Features:**
    - Retrieves actual performance from EEL server (no manual data entry needed)
    - Compares actual vs planned performance
    - Identifies deviations and their causes
    - Generates adjustment recommendations
    - Returns multiple reoptimization strategies
    
    **Request Body:**
    - plan_id: Long-term plan to reoptimize
    - target_year: Year for adjustments
    - start_date/end_date: Analysis period
    - frequency: Data aggregation (daily/monthly/quarterly/yearly)
    - budget_adjustment: Optional budget changes
    
    **Response includes:**
    - Performance comparison (planned vs actual)
    - Deviation analysis
    - Multiple reoptimization patterns
    - Recommended actions and adjustments
    - Impact analysis on long-term goals
    
    **Reoptimization Logic:**
    1. Fetch actual data from EEL server for the period
    2. Compare with planned targets
    3. Calculate variance and trends
    4. If behind target (>5% variance): Generate accelerated pattern
    5. If ahead of target (<-5% variance): Generate momentum pattern
    6. If on track: Generate steady progress pattern
    7. Always include alternative approach pattern
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Calculating reoptimization (request_id: {request_id}, "
            f"plan: {request.plan_id}, year: {request.target_year}, "
            f"period: {request.start_date} to {request.end_date}, "
            f"user: {current_user.user_id})"
        )
        
        # Convert to service request
        from models.reoptimization import (
            ReoptimizationRequest,
            BudgetAdjustment
        )
        
        service_request = ReoptimizationRequest(
            plan_id=request.plan_id,
            target_year=request.target_year,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency,
            budget_adjustment=BudgetAdjustment(
                **request.budget_adjustment.dict()
            ) if request.budget_adjustment else None
        )
        
        # Call service layer
        result = reoptimization_service.calculate_reoptimization(service_request)
        
        if not result.is_success:
            logger.error(f"Reoptimization calculation failed: {result.message}")
            
            if result.status == ServiceResultStatus.VALIDATION_ERROR:
                return create_error_response(
                    error_code="VALIDATION_ERROR",
                    error_message=result.message,
                    request_id=request_id,
                    status_code=400
                )
            
            return create_error_response(
                error_code="REOPTIMIZATION_ERROR",
                error_message=result.message or "Reoptimization calculation failed",
                request_id=request_id,
                status_code=500
            )
        
        # Convert result to API format
        from dataclasses import asdict
        response_data = {
            "reoptimization_patterns": [asdict(p) for p in result.data.reoptimization_patterns],
            "calculation_metadata": result.data.calculation_metadata
        }
        
        logger.info(
            f"Reoptimization calculated successfully (request_id: {request_id}, "
            f"patterns: {len(result.data.reoptimization_patterns)})"
        )
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            message=f"Generated {len(result.data.reoptimization_patterns)} reoptimization patterns successfully"
        )
        
    except Exception as e:
        logger.error(f"Reoptimization calculation error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="REOPTIMIZATION_ERROR",
            error_message=str(e),
            request_id=request_id,
            status_code=500
        )


# =============================================================================
# API 12: Get Reoptimization Visualization
# =============================================================================

@router.get("/visualization/{property_id}/{year}", response_model=APIResponse)
@measure_execution_time
async def get_reoptimization_visualization(
    property_id: str = Path(..., description="Property UUID"),
    year: int = Path(..., ge=2020, le=2100, description="Target year"),
    pattern_index: Optional[int] = Query(None, ge=0, description="Specific pattern index to visualize"),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 12: Get Reoptimization Visualization
    
    Retrieve visualization-ready data for reoptimization analysis.
    
    **Authentication Required:** Bearer Token
    
    **Path Parameters:**
    - property_id: Property UUID
    - year: Target year for reoptimization
    
    **Query Parameters:**
    - pattern_index: Optional index to get specific pattern (0-based)
    
    **Response includes:**
    - Performance trends over time (monthly data)
    - Actual vs target comparison charts
    - Variance analysis
    - Key performance metrics
    - Original vs adjusted plan comparison
    - Cost efficiency improvements
    - Implementation progress tracking
    
    **Use Cases:**
    - Dashboard displays
    - Progress reports
    - Management reviews
    - Stakeholder presentations
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Retrieving reoptimization visualization (request_id: {request_id}, "
            f"property: {property_id}, year: {year}, pattern_index: {pattern_index})"
        )
        
        # Call service layer
        result = reoptimization_service.get_reoptimization_visualization(
            property_id=property_id,
            year=year,
            pattern_index=pattern_index
        )
        
        if not result.is_success:
            logger.error(f"Visualization retrieval failed: {result.message}")
            
            if result.status == ServiceResultStatus.NOT_FOUND:
                return create_error_response(
                    error_code="NOT_FOUND",
                    error_message=f"Reoptimization data for property {property_id} and year {year} not found",
                    request_id=request_id,
                    status_code=404
                )
            
            return create_error_response(
                error_code="VISUALIZATION_ERROR",
                error_message=result.message or "Failed to retrieve visualization data",
                request_id=request_id,
                status_code=500
            )
        
        # Convert to API format
        from dataclasses import asdict
        response_data = asdict(result.data)
        
        logger.info(f"Reoptimization visualization retrieved successfully (request_id: {request_id})")
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            message="Reoptimization visualization data retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Visualization error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="VISUALIZATION_ERROR",
            error_message=str(e),
            request_id=request_id,
            status_code=500
        )


# =============================================================================
# API 13: Register Annual Reoptimization
# =============================================================================

@router.post("/register", response_model=APIResponse)
@measure_execution_time
async def register_annual_reoptimization(
    request: RegisterReoptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 13: Register Annual Reoptimization
    
    Register approved reoptimization adjustments and update the long-term plan.
    
    **Authentication Required:** Bearer Token
    
    **Key Features:**
    - Validates reoptimization pattern exists
    - Records approval metadata
    - Updates long-term plan with adjustments
    - Creates reoptimization record
    - Triggers update workflows
    - Returns change summary
    
    **Business Logic:**
    1. Validates reoptimization_pattern_id exists
    2. Checks user approval permissions
    3. Retrieves adjustment details
    4. Updates associated long-term plan
    5. Creates reoptimization audit record
    6. Sets status to active
    7. Triggers notification workflows
    8. Returns registration confirmation with changes applied
    
    **Impact:**
    - Long-term plan is updated with new actions
    - Budget may be adjusted
    - Timeline may be modified
    - Risk profile may change
    - All changes are tracked in audit trail
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Registering reoptimization (request_id: {request_id}, "
            f"pattern: {request.reoptimization_pattern_id}, "
            f"plan: {request.plan_id}, "
            f"user: {current_user.user_id})"
        )
        
        # Prepare approval info
        approval_info = {
            "approved_by": request.approval_info.approved_by,
            "approval_date": request.approval_info.approval_date,
            "comments": request.approval_info.comments
        }
        
        # Call service layer
        result = reoptimization_service.register_reoptimization(
            reoptimization_pattern_id=request.reoptimization_pattern_id,
            plan_id=request.plan_id,
            approval_info=approval_info
        )
        
        if not result.is_success:
            logger.error(f"Reoptimization registration failed: {result.message}")
            
            if result.status == ServiceResultStatus.VALIDATION_ERROR:
                return create_error_response(
                    error_code="VALIDATION_ERROR",
                    error_message=result.message,
                    request_id=request_id,
                    status_code=400
                )
            
            if result.status == ServiceResultStatus.NOT_FOUND:
                return create_error_response(
                    error_code="NOT_FOUND",
                    error_message=f"Reoptimization pattern {request.reoptimization_pattern_id} not found",
                    request_id=request_id,
                    status_code=404
                )
            
            return create_error_response(
                error_code="REGISTRATION_ERROR",
                error_message=result.message or "Reoptimization registration failed",
                request_id=request_id,
                status_code=500
            )
        
        logger.info(
            f"Reoptimization registered successfully "
            f"(reoptimization_id: {result.data['reoptimization_id']})"
        )
        
        # Optional: Add background task for notifications
        # background_tasks.add_task(send_reoptimization_notification, result.data)
        
        return create_success_response(
            data=result.data,
            request_id=request_id,
            status_code=201,
            message="Annual reoptimization registered successfully"
        )
        
    except Exception as e:
        logger.error(f"Registration error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="REGISTRATION_ERROR",
            error_message=str(e),
            request_id=request_id,
            status_code=500
        )


# =============================================================================
# ROUTER INITIALIZATION
# =============================================================================

logger.info("Module 4: Annual Reoptimization APIs initialized successfully")
