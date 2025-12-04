# api/api_module3_long_term_planning.py - Module 3: Long-term Planning APIs
"""
Module 3: Long-term Planning APIs

Provides endpoints for:
- API 8: Calculate long-term planning strategies
- API 9: Get planning visualization data
- API 10: Register selected planning pattern

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
from services import PlanningService, ServiceResultStatus

# Initialize
router = APIRouter(prefix="/planning/long-term", tags=["long-term-planning"])
planning_service = PlanningService(db_manager=None)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class PlanningHorizonRequest(BaseModel):
    """Planning time period definition"""
    start_year: int = Field(..., ge=2020, le=2070, description="Start year for planning")
    end_year: int = Field(..., ge=2020, le=2100, description="End year for planning")
    evaluation_intervals: str = Field("annual", description="Evaluation frequency")
    
    @validator('end_year')
    def end_after_start(cls, v, values):
        if 'start_year' in values and v <= values['start_year']:
            raise ValueError('end_year must be after start_year')
        return v


class BudgetConstraintsRequest(BaseModel):
    """Financial constraints for planning"""
    total_budget: Optional[float] = Field(None, ge=0, description="Total budget available")
    annual_budget_limit: Optional[float] = Field(None, ge=0, description="Annual spending limit")
    currency: str = Field("USD", description="Currency code")
    cost_escalation_rate: float = Field(3.0, ge=0, le=20, description="Annual cost increase rate (%)")


class StrategyPreferencesRequest(BaseModel):
    """Strategy priority weights"""
    renewable_energy_priority: float = Field(0.33, ge=0, le=1, description="Renewable energy weight")
    energy_efficiency_priority: float = Field(0.33, ge=0, le=1, description="Efficiency improvement weight")
    behavioral_change_priority: float = Field(0.34, ge=0, le=1, description="Behavioral change weight")


class ImplementationConstraintsRequest(BaseModel):
    """Physical and operational constraints"""
    max_simultaneous_projects: Optional[int] = Field(None, ge=1, description="Max concurrent projects")
    required_roi_years: Optional[float] = Field(None, ge=0, description="Required ROI period")
    minimum_reduction_per_action: Optional[float] = Field(None, ge=0, description="Min reduction per action")
    technology_restrictions: List[str] = Field(default_factory=list, description="Restricted technologies")


class CalculateLongTermPlanRequest(BaseModel):
    """Request for long-term planning calculation"""
    scenario_id: str = Field(..., description="Registered milestone scenario UUID")
    allocation_id: str = Field(..., description="Registered target allocation UUID")
    planning_horizon: PlanningHorizonRequest
    budget_constraints: Optional[BudgetConstraintsRequest] = None
    strategy_preferences: Optional[StrategyPreferencesRequest] = None
    implementation_constraints: Optional[ImplementationConstraintsRequest] = None


class RegisterPlanRequest(BaseModel):
    """Request to register a planning pattern"""
    pattern_id: str = Field(..., description="Planning pattern UUID to register")
    plan_name: str = Field(..., min_length=1, max_length=255, description="Name for the plan")
    approved_by: str = Field(..., description="User UUID who approved the plan")
    approval_date: str = Field(..., description="Approval date (ISO 8601 format)")
    comments: Optional[str] = Field(None, max_length=1000, description="Optional approval comments")


# DEBUG: Log model fields at module load time
logger.info(f"RegisterPlanRequest model loaded - Fields: {list(RegisterPlanRequest.__fields__.keys())}")


# =============================================================================
# API 8: Calculate Long-term Plan
# =============================================================================

@router.post("/calculate", response_model=APIResponse)
@measure_execution_time
async def calculate_long_term_plan(
    request: CalculateLongTermPlanRequest,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 8: Calculate Long-term Plan
    
    Generate multi-year implementation plans with specific actions, costs, and ROI analysis.
    Returns multiple planning patterns for comparison.
    
    **Authentication Required:** Bearer Token
    
    **Request Body:**
    - scenario_id: Registered milestone scenario
    - allocation_id: Registered target allocation
    - planning_horizon: Time period (start/end year)
    - budget_constraints: Optional financial limits
    - strategy_preferences: Optional priority weights
    
    **Response includes:**
    - Multiple planning patterns (aggressive, balanced, conservative)
    - Annual action plans with specific measures
    - Financial summaries and ROI analysis
    - Risk assessments
    - Implementation timelines
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Calculating long-term plan (request_id: {request_id}, "
            f"scenario: {request.scenario_id}, allocation: {request.allocation_id}, "
            f"user: {current_user.user_id})"
        )
        
        # Convert to service request
        from models.planning import (
            PlanningRequest,
            PlanningHorizon,
            BudgetConstraints,
            StrategyPreferences,
            ImplementationConstraints
        )
        
        service_request = PlanningRequest(
            scenario_id=request.scenario_id,
            allocation_id=request.allocation_id,
            planning_horizon=PlanningHorizon(
                start_year=request.planning_horizon.start_year,
                end_year=request.planning_horizon.end_year,
                evaluation_intervals=request.planning_horizon.evaluation_intervals
            ),
            budget_constraints=BudgetConstraints(
                **request.budget_constraints.dict()
            ) if request.budget_constraints else None,
            strategy_preferences=StrategyPreferences(
                **request.strategy_preferences.dict()
            ) if request.strategy_preferences else None,
            implementation_constraints=ImplementationConstraints(
                **request.implementation_constraints.dict()
            ) if request.implementation_constraints else None
        )
        
        # Call service layer
        result = planning_service.calculate_long_term_plan(service_request)
        
        if not result.is_success:
            logger.error(f"Planning calculation failed: {result.message}")
            
            if result.status == ServiceResultStatus.VALIDATION_ERROR:
                return create_error_response(
                    error_code="VALIDATION_ERROR",
                    error_message=result.message,
                    request_id=request_id,
                    status_code=400
                )
            
            return create_error_response(
                error_code="PLANNING_ERROR",
                error_message=result.message or "Long-term planning calculation failed",
                request_id=request_id,
                status_code=500
            )
        
        # Convert result to API format
        from dataclasses import asdict
        response_data = {
            "planning_patterns": [asdict(p) for p in result.data.planning_patterns],
            "calculation_metadata": result.data.calculation_metadata
        }
        
        logger.info(
            f"Long-term plan calculated successfully (request_id: {request_id}, "
            f"patterns: {len(result.data.planning_patterns)})"
        )
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            message=f"Generated {len(result.data.planning_patterns)} planning patterns successfully"
        )
        
    except Exception as e:
        logger.error(f"Planning calculation error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="PLANNING_ERROR",
            error_message=str(e),
            request_id=request_id,
            status_code=500
        )


# =============================================================================
# API 9: Get Long-term Plan Visualization
# =============================================================================

@router.get("/visualization/{plan_id}", response_model=APIResponse)
@measure_execution_time
async def get_long_term_plan_visualization(
    plan_id: str = Path(..., description="Plan calculation ID"),
    pattern_index: Optional[int] = Query(None, ge=0, description="Specific pattern index to visualize"),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 9: Get Long-term Plan Visualization
    
    Retrieve visualization-ready data for a calculated planning pattern.
    
    **Authentication Required:** Bearer Token
    
    **Path Parameters:**
    - plan_id: Plan calculation ID
    
    **Query Parameters:**
    - pattern_index: Optional index to get specific pattern (0-based)
    
    **Response includes:**
    - Timeline data for cumulative metrics
    - Action distribution by type and year
    - Cost-benefit analysis
    - Breakeven analysis
    - Investment vs savings trends
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Retrieving plan visualization (request_id: {request_id}, "
            f"plan_id: {plan_id}, pattern_index: {pattern_index})"
        )
        
        # Call service layer
        result = planning_service.get_plan_visualization(plan_id, pattern_index)
        
        if not result.is_success:
            logger.error(f"Visualization retrieval failed: {result.message}")
            
            if result.status == ServiceResultStatus.NOT_FOUND:
                return create_error_response(
                    error_code="NOT_FOUND",
                    error_message=f"Plan {plan_id} not found",
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
        
        logger.info(f"Visualization retrieved successfully (request_id: {request_id})")
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            message="Planning visualization data retrieved successfully"
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
# API 10: Register Long-term Plan
# =============================================================================

@router.post("/register", response_model=APIResponse)
@measure_execution_time
async def register_long_term_plan(
    request: RegisterPlanRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 10: Register Long-term Plan
    
    Register the selected planning pattern as the official long-term implementation plan.
    Persists the plan to database and activates it for tracking.
    
    **Authentication Required:** Bearer Token
    
    **Key Features:**
    - Validates pattern exists
    - Records approval metadata
    - Activates plan for implementation
    - Creates audit trail
    - Returns next steps
    
    **Business Logic:**
    1. Validates pattern_id exists
    2. Checks user approval permissions
    3. Creates plan record in database
    4. Links to scenario and allocation
    5. Sets status to registered
    6. Triggers notification workflows
    7. Returns registration confirmation
    """
    request_id = generate_request_id()
    
    try:
        # DEBUG: Log what we received
        logger.info(
            f"Registering long-term plan (request_id: {request_id}, "
            f"pattern: {request.pattern_id}, user: {current_user.user_id})"
        )
        logger.info(f"DEBUG: Received request fields: {request.dict()}")
        
        # Prepare approval info
        approval_info = {
            "approved_by": request.approved_by,
            "approval_date": request.approval_date,
            "comments": request.comments
        }
        
        # Call service layer
        result = planning_service.register_plan(
            pattern_id=request.pattern_id,
            plan_name=request.plan_name,
            approval_info=approval_info
        )
        
        if not result.is_success:
            logger.error(f"Plan registration failed: {result.message}")
            
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
                    error_message=f"Pattern {request.pattern_id} not found",
                    request_id=request_id,
                    status_code=404
                )
            
            return create_error_response(
                error_code="REGISTRATION_ERROR",
                error_message=result.message or "Plan registration failed",
                request_id=request_id,
                status_code=500
            )
        
        logger.info(f"Plan registered successfully (plan_id: {result.data['plan_id']})")
        
        # Optional: Add background task for notifications
        # background_tasks.add_task(send_plan_registration_notification, result.data)
        
        return create_success_response(
            data=result.data,
            request_id=request_id,
            status_code=201,
            message="Long-term plan registered successfully"
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

logger.info("Module 3: Long-term Planning APIs initialized successfully")
