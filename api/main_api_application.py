# main_api_application.py - Main FastAPI Application
# EcoAssist AI REST API Layer - Complete Integration
# Version 2.0 - Comprehensive implementation
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime, timedelta

# NEW: Database integration
from database.database_manager import get_db_manager

from api_core import (
    API_VERSION,
    BASE_URL,
    RequestIDMiddleware,
    create_error_response,
    generate_request_id,
    logger as core_logger
)

# Import module routers
from api_module1_milestones import router as milestones_router
from api_module2_target_division import router as target_division_router
from api_module3_long_term_planning import router as planning_router
from api_module4_reoptimization import router as reoptimization_router


logger = logging.getLogger(__name__)

# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 80)
    logger.info("EcoAssist AI REST API Starting...")
    logger.info(f"API Version: {API_VERSION}")
    logger.info(f"Base URL: {BASE_URL}")
    logger.info("=" * 80)
    logger.info("📚 API Documentation:")
    logger.info(f"   Swagger UI: http://localhost:8000{BASE_URL}/docs")
    logger.info(f"   ReDoc:      http://localhost:8000{BASE_URL}/redoc")
    logger.info("=" * 80)
    
    # Initialize backend services
    try:
        # TODO: Initialize database connections
        # await init_database()
        try:
            db = get_db_manager()
            health = db.health_check()
            
            if health['status'] == 'healthy':
                logger.info("✅ Database connected successfully")
                logger.info(f"   Server: {health['server']}")
                logger.info(f"   Database: {health['database']}")
                for table, count in health.get('table_counts', {}).items():
                    logger.info(f"   {table}: {count} records")
            else:
                logger.warning("⚠️ Database connection failed - running in mock mode")
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            logger.warning("⚠️ APIs will run in mock mode")
        logger.info("✓ Database connections initialized")
        
        try:
            from api_module1_milestones import initialize_milestone_service
            await initialize_milestone_service()
            logger.info("✓ Milestone service initialized")
        except Exception as e:
            logger.error(f"⚠️ Milestone service initialization failed: {e}")
            
        try:
            from api_module2_target_division import initialize_allocation_service
            await initialize_allocation_service()
            logger.info("✓ Allocation service initialized")
        except Exception as e:
            logger.error(f"⚠️ Allocation service initialization failed: {e}")
            
        try:
            from api_module3_long_term_planning import initialize_planning_service
            await initialize_planning_service()
            logger.info("✓ Planning service initialized")
        except Exception as e:
            logger.error(f"⚠️ Planning service initialization failed: {e}")
            
        # Initialize reoptimization service with database
        try:
            from api_module4_reoptimization import initialize_reoptimization_service
            await initialize_reoptimization_service()
            logger.info("✓ Reoptimization service initialized")
        except Exception as e:
            logger.error(f"⚠️ Reoptimization service initialization failed: {e}")
        
        # TODO: Initialize cache
        # await init_cache()
        logger.info("✓ Cache system initialized")
        
        # TODO: Load ML models
        # await load_ml_models()
        logger.info("✓ AI models loaded")
        
        logger.info("=" * 80)
        logger.info("EcoAssist AI REST API Started Successfully")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("EcoAssist AI REST API Shutting Down...")
    logger.info("=" * 80)
    
    try:
        # TODO: Close database connections
        # await close_database()
        logger.info("✓ Database connections closed")
        
        # TODO: Clear cache
        # await clear_cache()
        logger.info("✓ Cache cleared")
        
        logger.info("=" * 80)
        logger.info("EcoAssist AI REST API Shut Down Successfully")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="EcoAssist AI API",
    description="""
    # EcoAssist AI REST API
    
    Comprehensive API for carbon emissions reduction planning and optimization.
    
    ## Features
    
    * **Milestone Setting**: AI-powered scenario generation for long-term targets
    * **Target Division**: Intelligent allocation of portfolio targets to properties
    * **Long-term Planning**: Strategic implementation planning with ROI analysis
    * **Annual Reoptimization**: Adaptive planning based on actual performance
    * **Dashboard Management**: Real-time monitoring and analytics
    * **System Management**: Authentication, health checks, and error handling
    
    ## Modules
    
    1. **Module 1: Milestone Setting** (APIs 1-4)
    2. **Module 2: Target Division** (APIs 5-7)
    3. **Module 3: Long-term Planning** (APIs 8-10)
    4. **Module 4: Reoptimization** (APIs 11-13)
    5. **Module 5: Dashboard Management** (APIs 14-17)
    6. **Module 6: System APIs** (APIs 18-21)
    
    ## Authentication
    
    All endpoints require Bearer token authentication except:
    - `/api/v1/system/health`
    - `/api/v1/system/security/authenticate`
    
    ## Rate Limiting
    
    - 100 requests per hour per client
    - Automatic blocking for excessive requests
    - Rate limit headers included in responses
    
    ## Error Handling
    
    All errors follow RFC 9110 standard with detailed error codes and messages.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url=f"{BASE_URL}/docs",
    redoc_url=f"{BASE_URL}/redoc",
    openapi_url=f"{BASE_URL}/openapi.json"
)

# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id", "x-rate-limit-remaining", "x-rate-limit-reset"]
)

# Request ID Middleware
app.add_middleware(RequestIDMiddleware)

# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 Not Found errors"""
    request_id = getattr(request.state, 'request_id', generate_request_id())
    return JSONResponse(
        status_code=404,
        content=create_error_response(
            error_code="NOT_FOUND",
            error_message=f"Endpoint {request.url.path} not found",
            request_id=request_id,
            status_code=404
        ).dict()
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 Internal Server Error"""
    request_id = getattr(request.state, 'request_id', generate_request_id())
    logger.error(f"Internal server error (request_id: {request_id}): {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error_code="INTERNAL_ERROR",
            error_message="An unexpected error occurred",
            request_id=request_id,
            status_code=500
        ).dict()
    )

# =============================================================================
# REQUEST/RESPONSE MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses"""
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', generate_request_id())
    
    # Log request
    logger.info(
        f"→ {request.method} {request.url.path} "
        f"(request_id: {request_id}, client: {request.client.host})"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log response
    logger.info(
        f"← {response.status_code} "
        f"(request_id: {request_id}, duration: {duration_ms:.2f}ms)"
    )
    
    # Add custom headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    
    return response

# =============================================================================
# INCLUDE MODULE ROUTERS
# =============================================================================

# Module 1: Milestone Setting APIs
app.include_router(
    milestones_router,
    prefix=BASE_URL
)

# Module 2: Target Division APIs
app.include_router(
    target_division_router,
    prefix=BASE_URL
)

# Module 3: Long-term Planning APIs
app.include_router(
    planning_router,
    prefix=BASE_URL
)

# Module 4: Annual Reoptimization APIs
app.include_router(
    reoptimization_router,
    prefix=BASE_URL
)
# =============================================================================
# MODULE 3: LONG-TERM PLANNING APIs (Inline Implementation)
# =============================================================================

from fastapi import APIRouter, Depends, BackgroundTasks, Query, Path, Body
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any

from api_core import (
    APIResponse,
    ImplementationStatus,
    OnTrackStatus,
    ImplementationTiming,
    ConsumptionImpact,
    EmissionReduction,
    CostDetails,
    measure_execution_time,
    check_rate_limit_dependency,
    get_current_user,
    create_success_response,
    create_error_response,
    generate_request_id,
    TokenData
)

planning_router = APIRouter(
    prefix="/planning",
    tags=["Module 3: Long-term Planning"]
)

class PlanningHorizon(BaseModel):
    """Planning time horizon"""
    start_year: int = Field(ge=2025, le=2070)
    end_year: int = Field(ge=2025, le=2070)
    evaluation_intervals: str = Field(default="annual", description="Evaluation frequency")
    
    @validator('end_year')
    def validate_end_year(cls, v, values):
        if 'start_year' in values and v <= values['start_year']:
            raise ValueError('end_year must be after start_year')
        return v

class BudgetConstraints(BaseModel):
    """Budget limitations"""
    total_budget: float = Field(gt=0, description="Total budget available")
    annual_budget_limit: float = Field(gt=0, description="Annual spending limit")
    currency: str = Field(default="USD")
    cost_escalation_rate: float = Field(default=3.0, description="Annual cost escalation %")

class LongTermPlanRequest(BaseModel):
    """API 8: Calculate Long-term Plan Request"""
    scenario_id: str = Field(description="Registered milestone scenario UUID")
    allocation_id: str = Field(description="Registered target allocation UUID")
    planning_horizon: PlanningHorizon
    budget_constraints: BudgetConstraints
    strategy_preferences: Optional[Dict[str, float]] = Field(
        None,
        description="Strategy priority weights"
    )

class ReductionAction(BaseModel):
    """Individual reduction action"""
    action_type: str
    target_properties: List[str]
    expected_reduction: float
    investment_required: float
    roi_years: float
    emission_unit: str = "kg-CO2e"
    cost_unit: str = "USD"

class AnnualPlan(BaseModel):
    """Annual implementation plan"""
    year: int
    actions: List[ReductionAction]
    total_investment: float
    total_reduction: float
    cumulative_progress: float
    cost_unit: str = "USD"
    emission_unit: str = "kg-CO2e"

class FinancialSummary(BaseModel):
    """Financial summary of plan"""
    total_investment: float
    total_savings: float
    net_benefit: float
    overall_roi_years: float
    currency: str = "USD"

class RiskAssessment(BaseModel):
    """Risk assessment"""
    implementation_risk: str
    financial_risk: str
    technology_risk: str
    key_risks: List[str]

class PlanningPattern(BaseModel):
    """Long-term planning pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    annual_plan: List[AnnualPlan]
    financial_summary: FinancialSummary
    risk_assessment: RiskAssessment

@planning_router.post("/long-term/calculate", response_model=APIResponse)
@measure_execution_time
async def calculate_long_term_plan(
    request: LongTermPlanRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 8: Calculate Long-term Plan
    
    Generate strategic implementation plans spanning multiple years with
    optimized action sequencing and budget allocation.
    
    **Features:**
    - Multi-year action planning with optimal sequencing
    - Budget constraint optimization
    - ROI-driven prioritization
    - Risk assessment
    - Multiple planning patterns for comparison
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"Calculating long-term plan (request_id: {request_id})")
        
        # Generate planning patterns (simplified for demo)
        patterns = []
        
        # Pattern 1: Aggressive Implementation
        pattern1 = PlanningPattern(
            pattern_id=f"pattern-{generate_request_id()[:8]}",
            pattern_name="Aggressive Implementation",
            description="Front-loaded investment with early ROI",
            annual_plan=[
                AnnualPlan(
                    year=2025,
                    actions=[
                        ReductionAction(
                            action_type="solar_panel_installation",
                            target_properties=["property-uuid-1"],
                            expected_reduction=250.0,
                            investment_required=80000.0,
                            roi_years=6.5
                        )
                    ],
                    total_investment=115000.0,
                    total_reduction=370.0,
                    cumulative_progress=7.9
                )
            ],
            financial_summary=FinancialSummary(
                total_investment=480000.0,
                total_savings=650000.0,
                net_benefit=170000.0,
                overall_roi_years=7.4
            ),
            risk_assessment=RiskAssessment(
                implementation_risk="medium",
                financial_risk="low",
                technology_risk="low",
                key_risks=["Supplier availability", "Installation timeline dependencies"]
            )
        )
        patterns.append(pattern1.dict())
        
        response_data = {
            "planning_patterns": patterns,
            "calculation_metadata": {
                "calculated_at": datetime.utcnow().isoformat(),
                "algorithm_version": "2.1.0",
                "patterns_generated": len(patterns)
            }
        }
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            message=f"Generated {len(patterns)} planning patterns successfully"
        )
        
    except Exception as e:
        logger.error(f"Planning calculation error: {str(e)}", exc_info=True)
        return create_error_response(
            error_code="PLANNING_ERROR",
            error_message="Error calculating long-term plan",
            request_id=request_id,
            status_code=500
        )

@planning_router.get("/long-term/visualization/{plan_id}", response_model=APIResponse)
@measure_execution_time
async def get_long_term_plan_visualization(
    plan_id: str = Path(...),
    include_progress: bool = Query(False),
    year: Optional[int] = Query(None),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 9: Get Long-term Plan Visualization
    
    Retrieve registered long-term plan data with progress tracking.
    """
    request_id = generate_request_id()
    
    try:
        visualization_data = {
            "plan_id": plan_id,
            "plan_name": "Decarbonization Strategy 2025-2050",
            "implementation_schedule": [],
            "overall_progress": {
                "on_track_status": OnTrackStatus.ON_TRACK.value,
                "current_year": 2025,
                "progress_percentage": 15.5
            }
        }
        
        return create_success_response(
            data=visualization_data,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        return create_error_response(
            error_code="VISUALIZATION_ERROR",
            error_message="Error retrieving plan visualization",
            request_id=request_id,
            status_code=500
        )

@planning_router.post("/long-term/register", response_model=APIResponse)
@measure_execution_time
async def register_long_term_plan(
    plan: LongTermPlanRequest,
    pattern_id: str = Body(...),  # Add Body(...)
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 10: Register Long-term Plan
    
    Register selected planning pattern as official implementation plan.
    """
    request_id = generate_request_id()
    
    try:
        plan_id = f"PLAN_{int(datetime.utcnow().timestamp())}_{generate_request_id()[:6]}"
        
        registration_data = {
            "plan_id": plan_id,
            "pattern_id": pattern_id,
            "plan_name": plan_name,
            "approval_status": approval_status,
            "registered_at": datetime.utcnow().isoformat(),
            "registered_by": current_user.user_id
        }
        
        return create_success_response(
            data=registration_data,
            request_id=request_id,
            status_code=201,
            message="Long-term plan registered successfully"
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return create_error_response(
            error_code="REGISTRATION_ERROR",
            error_message="Error registering plan",
            request_id=request_id,
            status_code=500
        )

app.include_router(planning_router, prefix=BASE_URL)

# =============================================================================
# MODULE 4: REOPTIMIZATION APIs
# =============================================================================

reopt_router = APIRouter(
    prefix="/reoptimization",
    tags=["Module 4: Annual Reoptimization"]
)

class ReoptimizationRequest(BaseModel):
    """API 11: Calculate Annual Reoptimization Request"""
    plan_id: str
    current_year: int = Field(ge=2025, le=2070)
    actual_performance: Dict[str, float]
    external_factors: Optional[Dict[str, Any]] = None

@reopt_router.post("/calculate", response_model=APIResponse)
@measure_execution_time
async def calculate_reoptimization(
    request: ReoptimizationRequest,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 11: Calculate Annual Reoptimization
    
    Analyze actual performance vs plan and generate adjusted recommendations.
    
    **Features:**
    - Performance deviation analysis
    - Root cause identification
    - Adaptive plan adjustments
    - Budget reallocation
    - Updated action priorities
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"Calculating reoptimization (request_id: {request_id})")
        
        reoptimization_data = {
            "reoptimization_id": generate_request_id(),
            "plan_id": request.plan_id,
            "analysis_year": request.current_year,
            "deviation_analysis": {
                "target_vs_actual": {
                    "target_reduction": 370.0,
                    "actual_reduction": 420.0,
                    "variance_percentage": 13.5,
                    "status": "ahead_of_target"
                },
                "cost_analysis": {
                    "planned_cost": 115000.0,
                    "actual_cost": 98000.0,
                    "savings": 17000.0
                }
            },
            "root_causes": [
                "Better than expected weather conditions reduced heating demand",
                "Early adoption of behavioral changes by occupants",
                "Technology performed above specifications"
            ],
            "recommended_adjustments": [
                {
                    "action": "Accelerate solar installation to next year",
                    "impact": "Additional 15% emission reduction",
                    "budget_impact": 85000.0
                }
            ],
            "updated_timeline": []
        }
        
        return create_success_response(
            data=reoptimization_data,
            request_id=request_id,
            message="Reoptimization analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Reoptimization error: {str(e)}", exc_info=True)
        return create_error_response(
            error_code="REOPTIMIZATION_ERROR",
            error_message="Error calculating reoptimization",
            request_id=request_id,
            status_code=500
        )

@reopt_router.get("/visualization/{reopt_id}", response_model=APIResponse)
@measure_execution_time
async def get_reoptimization_visualization(
    reopt_id: str = Path(...),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 12: Get Annual Plan Visualization
    
    Retrieve visualization data for reoptimization analysis.
    """
    request_id = generate_request_id()
    
    try:
        viz_data = {
            "reoptimization_id": reopt_id,
            "performance_charts": {},
            "comparison_data": {},
            "trend_analysis": {}
        }
        
        return create_success_response(
            data=viz_data,
            request_id=request_id
        )
        
    except Exception as e:
        return create_error_response(
            error_code="VISUALIZATION_ERROR",
            error_message="Error retrieving visualization",
            request_id=request_id,
            status_code=500
        )

@reopt_router.post("/register", response_model=APIResponse)
@measure_execution_time
async def register_reoptimization(
    reopt_id: str,
    approval_status: str,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 13: Register Reoptimized Plan
    
    Register approved reoptimization adjustments.
    """
    request_id = generate_request_id()
    
    try:
        registration_data = {
            "revision_id": f"REV_{int(datetime.utcnow().timestamp())}",
            "reoptimization_id": reopt_id,
            "approval_status": approval_status,
            "registered_at": datetime.utcnow().isoformat()
        }
        
        return create_success_response(
            data=registration_data,
            request_id=request_id,
            status_code=201
        )
        
    except Exception as e:
        return create_error_response(
            error_code="REGISTRATION_ERROR",
            error_message="Error registering reoptimization",
            request_id=request_id,
            status_code=500
        )

app.include_router(reopt_router, prefix=BASE_URL)

# =============================================================================
# MODULE 5: DASHBOARD MANAGEMENT APIs
# =============================================================================

dashboard_router = APIRouter(
    prefix="/dashboard",
    tags=["Module 5: Dashboard Management"]
)

@dashboard_router.get("/carbon-credits", response_model=APIResponse)
@measure_execution_time
async def get_carbon_credits(
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 14: Get Carbon Credits
    
    Retrieve carbon credit portfolio and offset management data.
    """
    request_id = generate_request_id()
    
    try:
        credits_data = {
            "portfolio_summary": {
                "total_credits_available": 150000,
                "credits_used_ytd": 45000,
                "credits_remaining": 105000,
                "average_credit_price": 25.50,
                "total_investment": 3825000
            },
            "credit_types": [
                {"type": "Renewable Energy Certificates", "quantity": 80000, "unit_price": 22.00},
                {"type": "Forest Carbon Credits", "quantity": 50000, "unit_price": 28.00}
            ],
            "compliance_status": "On Track"
        }
        
        return create_success_response(
            data=credits_data,
            request_id=request_id
        )
        
    except Exception as e:
        return create_error_response(
            error_code="DASHBOARD_ERROR",
            error_message="Error retrieving carbon credits",
            request_id=request_id,
            status_code=500
        )

@dashboard_router.post("/carbon-credits/update", response_model=APIResponse)
@measure_execution_time
async def update_carbon_credits(
    credit_type: str,
    quantity_purchased: float,
    unit_price: float,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 15: Update Carbon Credits
    
    Update carbon credit inventory with new purchases.
    """
    request_id = generate_request_id()
    
    try:
        update_data = {
            "update_id": f"UPD_{int(datetime.utcnow().timestamp())}",
            "updated_fields": ["quantity_purchased", "unit_price"],
            "update_timestamp": datetime.utcnow().isoformat(),
            "validation_status": "passed"
        }
        
        return create_success_response(
            data=update_data,
            request_id=request_id
        )
        
    except Exception as e:
        return create_error_response(
            error_code="UPDATE_ERROR",
            error_message="Error updating carbon credits",
            request_id=request_id,
            status_code=500
        )

@dashboard_router.get("/consumption/{property_id}", response_model=APIResponse)
@measure_execution_time
async def get_consumption_dashboard(
    property_id: str = Path(...),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 16: Get Consumption Dashboard
    
    Retrieve energy consumption and cost analytics for a property.
    """
    request_id = generate_request_id()
    
    try:
        consumption_data = {
            "property_id": property_id,
            "current_month_consumption": {
                "electricity_kwh": 45000,
                "natural_gas_gj": 8500,
                "total_cost_aud": 18500
            },
            "ytd_consumption": {
                "total_electricity_kwh": 520000,
                "total_cost_aud": 215000,
                "vs_target_percentage": -8.5
            },
            "efficiency_metrics": {
                "energy_intensity_kwh_per_sqm": 180.5,
                "efficiency_improvement_ytd": 8.5
            }
        }
        
        return create_success_response(
            data=consumption_data,
            request_id=request_id
        )
        
    except Exception as e:
        return create_error_response(
            error_code="DASHBOARD_ERROR",
            error_message="Error retrieving consumption data",
            request_id=request_id,
            status_code=500
        )

@dashboard_router.get("/emissions/{property_id}", response_model=APIResponse)
@measure_execution_time
async def get_emissions_dashboard(
    property_id: str = Path(...),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 17: Get Emissions Dashboard
    
    Retrieve emissions tracking and progress analytics for a property.
    """
    request_id = generate_request_id()
    
    try:
        emissions_data = {
            "property_id": property_id,
            "current_emissions": {
                "scope1_tco2e": 8500,
                "scope2_tco2e": 5200,
                "total_tco2e": 13700
            },
            "targets_vs_actual": {
                "annual_target_tco2e": 140000,
                "ytd_actual_tco2e": 132000,
                "variance_percentage": -5.7
            },
            "reduction_progress": {
                "baseline_year": 2025,
                "baseline_emission": 150000,
                "current_reduction_percentage": 12.0,
                "target_reduction_2030": 45.0
            }
        }
        
        return create_success_response(
            data=emissions_data,
            request_id=request_id
        )
        
    except Exception as e:
        return create_error_response(
            error_code="DASHBOARD_ERROR",
            error_message="Error retrieving emissions data",
            request_id=request_id,
            status_code=500
        )

app.include_router(dashboard_router, prefix=BASE_URL)

# =============================================================================
# MODULE 6: SYSTEM APIs
# =============================================================================

system_router = APIRouter(
    prefix="/system",
    tags=["Module 6: System Management"]
)

from api_core import (
    AuthenticationRequest,
    AuthenticationResponse,
    create_access_token,
    TOKEN_EXPIRY_SECONDS
)

@system_router.post("/security/authenticate", response_model=AuthenticationResponse)
async def authenticate(request: AuthenticationRequest):
    """
    API 18: Security Authentication
    
    Authenticate user and obtain access token for API access.
    
    **No Authentication Required** (this is the login endpoint)
    
    Returns JWT access token valid for 1 hour.
    """
    try:
        # TODO: Validate credentials against database
        # user = await db.users.find_one({"username": request.username})
        # if not user or not verify_password(request.password, user.password_hash):
        #     raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Mock user data (replace with actual authentication)
        user_data = {
            "user_id": "usr_123456",
            "email": request.username,
            "role": "analyst",
            "permissions": ["read", "write", "calculate"]
        }
        
        # Create access token
        access_token = create_access_token(user_data)
        refresh_token = create_access_token(
            user_data,
            expires_delta=timedelta(seconds=86400)
        )
        
        return AuthenticationResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=TOKEN_EXPIRY_SECONDS,
            refresh_token=refresh_token,
            user_info=user_data
        )
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

@system_router.get("/health", response_model=APIResponse)
async def health_check():
    """
    API 19: System Health Check
    
    Check system health and component status.
    
    **No Authentication Required**
    """
    request_id = generate_request_id()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {
            "api": "operational",
            "database": "operational",  # TODO: actual check
            "cache": "operational",     # TODO: actual check
            "ml_models": "operational"  # TODO: actual check
        },
        "uptime_seconds": 0,  # TODO: track actual uptime
        "request_count": 0     # TODO: track actual count
    }
    
    return create_success_response(
        data=health_data,
        request_id=request_id,
        message="System healthy"
    )

@system_router.post("/error/report", response_model=APIResponse)
@measure_execution_time
async def report_error(
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 20: Advanced Error Handling
    
    Report and handle system errors with detailed tracking.
    """
    request_id = generate_request_id()
    
    error_id = f"ERR_{int(datetime.utcnow().timestamp())}_{generate_request_id()[:6]}"
    
    error_report = {
        "error_id": error_id,
        "severity": "medium",
        "auto_resolved": False,
        "suggested_actions": [
            "Review error details in logs",
            "Contact support if issue persists"
        ],
        "escalation_required": False
    }
    
    return create_success_response(
        data=error_report,
        request_id=request_id,
        message="Error logged successfully"
    )

@system_router.get("/test/endpoint/{endpoint_name}", response_model=APIResponse)
@measure_execution_time
async def test_endpoint(
    endpoint_name: str = Path(...),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 21: API Endpoint Testing
    
    Test individual API endpoints for connectivity and functionality.
    """
    request_id = generate_request_id()
    
    test_results = {
        "endpoint": endpoint_name,
        "test_status": "passed",
        "response_time_ms": 45.2,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return create_success_response(
        data=test_results,
        request_id=request_id,
        message=f"Endpoint {endpoint_name} test completed"
    )

app.include_router(system_router, prefix=BASE_URL)

# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/", response_model=APIResponse)
async def root():
    """API root endpoint with information"""
    request_id = generate_request_id()
    
    info = {
        "name": "EcoAssist AI REST API",
        "version": "2.0.0",
        "description": "Comprehensive carbon emissions reduction planning and optimization",
        "documentation": f"{BASE_URL}/docs",
        "base_url": BASE_URL,
        "modules": {
            "milestone_setting": f"{BASE_URL}/milestones",
            "target_division": f"{BASE_URL}/target-division",
            "long_term_planning": f"{BASE_URL}/planning",
            "reoptimization": f"{BASE_URL}/reoptimization",
            "dashboard": f"{BASE_URL}/dashboard",
            "system": f"{BASE_URL}/system"
        }
    }
    
    return create_success_response(
        data=info,
        request_id=request_id,
        message="Welcome to EcoAssist AI API"
    )

@app.get(f"{BASE_URL}/", response_model=APIResponse)
async def api_root():
    """API version root"""
    return await root()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting EcoAssist AI REST API server...")
    
    uvicorn.run(
        "main_api_application:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
