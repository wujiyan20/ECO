# api_module2_target_division.py - Module 2: Target Division APIs
# EcoAssist AI REST API Layer - APIs 5-7
# Comprehensive implementation based on specification Ver2.0

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import logging
import numpy as np

# NEW: Database integration
from database.database_manager import get_db_manager

from api_core import (
    APIResponse,
    ErrorResponse,
    ApprovalStatus,
    AllocationMethod,
    generate_request_id,
    generate_calculation_id,
    create_success_response,
    create_error_response,
    measure_execution_time,
    check_rate_limit_dependency,
    get_current_user,
    validate_property_ids,
    TokenData
)

from services import (
    AllocationService,
    AllocationRequest as ServiceAllocationRequest,
    ServiceResult
)

from services.base_service import ServiceResultStatus

# Initialize allocation service (will be set up in startup)
allocation_service = None

logger = logging.getLogger(__name__)

# Initialize allocation service with database
async def initialize_allocation_service():
    """Initialize allocation service with database connection"""
    global allocation_service
    
    try:
        db = get_db_manager()
        if db and db.test_connection():
            logger.info("✅ Database connected - allocation service in DATABASE mode")
            allocation_service = AllocationService(db_manager=db)
        else:
            logger.warning("⚠️ Database not available - allocation service in MOCK mode")
            allocation_service = AllocationService(db_manager=None)
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        logger.warning("⚠️ Allocation service will run in MOCK mode")
        allocation_service = AllocationService(db_manager=None)

# Create router for target division APIs
router = APIRouter(
    prefix="/target-division",
    tags=["Module 2: Target Division"]
)

# =============================================================================
# REQUEST MODELS FOR MODULE 2
# =============================================================================

class PropertyData(BaseModel):
    """Property information for target allocation"""
    property_id: str = Field(description="Property UUID")
    property_name: str = Field(description="Property display name")
    building_type: str = Field(description="Building classification")
    floor_area: float = Field(gt=0, description="Floor area in square meters")
    baseline_emission: float = Field(ge=0, description="Baseline annual emissions")
    carbon_intensity: float = Field(ge=0, description="Carbon intensity (kg-CO2e/m²)")
    retrofit_potential: float = Field(
        ge=0,
        le=100,
        description="Retrofit potential score (0-100)"
    )
    current_efficiency_rating: Optional[str] = Field(
        None,
        description="Current energy efficiency rating"
    )

class TargetDivisionRequest(BaseModel):
    """
    API 5: Calculate Target Division Request Model
    Allocate portfolio-level targets to individual properties
    """
    scenario_id: str = Field(
        description="UUID of registered milestone scenario"
    )
    target_years: List[int] = Field(
        min_items=1,
        description="Years for which to allocate targets"
    )
    properties: List[PropertyData] = Field(
        min_items=1,
        description="Property data for target allocation"
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.CARBON_INTENSITY_WEIGHTED,
        description="Method for distributing targets across properties"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional constraints (e.g., budget limits, feasibility thresholds)"
    )
    optimization_objectives: Optional[Dict[str, float]] = Field(
        None,
        description="Multi-objective optimization weights (fairness, efficiency, feasibility)"
    )
    
    @validator('scenario_id')
    def validate_scenario_id(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('scenario_id must be a valid UUID')
        return v
    
    @validator('target_years')
    def validate_target_years(cls, v):
        if not v:
            raise ValueError('At least one target year must be specified')
        if any(year < 2025 or year > 2070 for year in v):
            raise ValueError('Target years must be between 2025 and 2070')
        if len(v) != len(set(v)):
            raise ValueError('Target years must be unique')
        return sorted(v)
    
    @validator('properties')
    def validate_properties(cls, v):
        if len(v) < 1:
            raise ValueError('At least one property must be provided')
        
        # Validate property IDs are unique
        property_ids = [p.property_id for p in v]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError('Property IDs must be unique')
        
        # Validate all property IDs are valid UUIDs
        validate_property_ids(property_ids)
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "scenario_id": "a1b2c3d4-e5f6-4a5b-6c7d-8e9f0a1b2c3d",
                "target_years": [2025, 2030, 2050],
                "properties": [
                    {
                        "property_id": "550e8400-e29b-41d4-a716-446655440000",
                        "property_name": "Building A - Office",
                        "building_type": "Commercial Office",
                        "floor_area": 5000.0,
                        "baseline_emission": 1500.0,
                        "carbon_intensity": 0.30,
                        "retrofit_potential": 75.0,
                        "current_efficiency_rating": "C"
                    },
                    {
                        "property_id": "550e8400-e29b-41d4-a716-446655440001",
                        "property_name": "Building B - Retail",
                        "building_type": "Retail",
                        "floor_area": 3000.0,
                        "baseline_emission": 900.0,
                        "carbon_intensity": 0.30,
                        "retrofit_potential": 60.0,
                        "current_efficiency_rating": "D"
                    }
                ],
                "allocation_method": "carbon_intensity_weighted",
                "optimization_objectives": {
                    "fairness": 0.4,
                    "efficiency": 0.4,
                    "feasibility": 0.2
                }
            }
        }

class PropertyTarget(BaseModel):
    """Allocated target for a single property"""
    property_id: str
    property_name: str
    year: int
    allocated_target: float = Field(description="Target emission level (kg-CO2e)")
    reduction_from_baseline: float = Field(description="Percentage reduction from baseline")
    absolute_reduction: float = Field(description="Absolute reduction amount (kg-CO2e)")
    allocation_weight: float = Field(
        ge=0,
        le=1,
        description="Weight used in allocation (normalized)"
    )
    feasibility_score: float = Field(
        ge=0,
        le=100,
        description="Feasibility assessment score"
    )
    estimated_cost: float = Field(ge=0, description="Estimated implementation cost")
    recommended_actions: List[str] = Field(description="Recommended reduction actions")
    unit: str = Field(default="kg-CO2e")

class AllocationMetrics(BaseModel):
    """Allocation quality metrics"""
    fairness_index: float = Field(
        ge=0,
        le=1,
        description="Gini coefficient (0=perfect fairness, 1=maximum inequality)"
    )
    efficiency_score: float = Field(
        ge=0,
        le=100,
        description="Cost-effectiveness score"
    )
    feasibility_score: float = Field(
        ge=0,
        le=100,
        description="Overall feasibility score"
    )
    coverage: float = Field(
        ge=0,
        le=100,
        description="Percentage of portfolio target achieved"
    )

class RegisterTargetDivisionRequest(BaseModel):
    """
    API 7: Register Target Division Request Model
    """
    allocation_id: str = Field(description="UUID of calculated allocation")
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.APPROVED,
        description="Approval status"
    )
    approved_by: Optional[str] = Field(None, description="Approver user ID")
    approval_date: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(None, max_length=1000)
    
    @validator('allocation_id')
    def validate_allocation_id(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('allocation_id must be a valid UUID')
        return v

# =============================================================================
# ALLOCATION ALGORITHMS
# =============================================================================

class TargetAllocationEngine:
    """
    Advanced target allocation engine with multiple methods
    """
    
    @staticmethod
    def allocate_carbon_intensity_weighted(
        properties: List[PropertyData],
        portfolio_target: float
    ) -> Dict[str, float]:
        """
        Allocate targets based on carbon intensity
        Properties with higher carbon intensity get larger reductions
        """
        total_weighted_emission = sum(
            p.baseline_emission * p.carbon_intensity
            for p in properties
        )
        
        allocations = {}
        for prop in properties:
            weight = (prop.baseline_emission * prop.carbon_intensity) / total_weighted_emission
            allocations[prop.property_id] = portfolio_target * weight
        
        return allocations
    
    @staticmethod
    def allocate_proportional(
        properties: List[PropertyData],
        portfolio_target: float
    ) -> Dict[str, float]:
        """
        Simple proportional allocation based on baseline emissions
        """
        total_emission = sum(p.baseline_emission for p in properties)
        
        allocations = {}
        for prop in properties:
            weight = prop.baseline_emission / total_emission
            allocations[prop.property_id] = portfolio_target * weight
        
        return allocations
    
    @staticmethod
    def allocate_retrofit_potential(
        properties: List[PropertyData],
        portfolio_target: float
    ) -> Dict[str, float]:
        """
        Allocate based on retrofit potential
        Properties with higher retrofit potential get larger targets
        """
        total_potential = sum(
            p.baseline_emission * (p.retrofit_potential / 100)
            for p in properties
        )
        
        allocations = {}
        for prop in properties:
            weighted_potential = p.baseline_emission * (prop.retrofit_potential / 100)
            weight = weighted_potential / total_potential
            allocations[prop.property_id] = portfolio_target * weight
        
        return allocations
    
    @staticmethod
    def allocate_equal_distribution(
        properties: List[PropertyData],
        portfolio_target: float
    ) -> Dict[str, float]:
        """
        Equal reduction percentage across all properties
        """
        total_baseline = sum(p.baseline_emission for p in properties)
        reduction_percentage = (portfolio_target / total_baseline) * 100
        
        allocations = {}
        for prop in properties:
            allocations[prop.property_id] = prop.baseline_emission * (reduction_percentage / 100)
        
        return allocations
    
    @staticmethod
    def allocate_ai_optimized(
        properties: List[PropertyData],
        portfolio_target: float,
        objectives: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Multi-objective AI optimization
        Balances fairness, efficiency, and feasibility
        """
        # Normalize objectives
        total_weight = sum(objectives.values())
        normalized_objectives = {k: v / total_weight for k, v in objectives.items()}
        
        # Calculate composite scores
        composite_scores = []
        for prop in properties:
            # Fairness component (inverse of emission share to balance load)
            total_emission = sum(p.baseline_emission for p in properties)
            fairness = 1 - (prop.baseline_emission / total_emission)
            
            # Efficiency component (carbon intensity)
            efficiency = prop.carbon_intensity
            
            # Feasibility component (retrofit potential)
            feasibility = prop.retrofit_potential / 100
            
            # Weighted composite score
            composite = (
                normalized_objectives.get('fairness', 0.33) * fairness +
                normalized_objectives.get('efficiency', 0.33) * efficiency +
                normalized_objectives.get('feasibility', 0.34) * feasibility
            )
            composite_scores.append(composite)
        
        # Normalize composite scores to use as weights
        total_composite = sum(composite_scores)
        
        allocations = {}
        for i, prop in enumerate(properties):
            weight = composite_scores[i] / total_composite
            allocations[prop.property_id] = portfolio_target * weight
        
        return allocations

def calculate_allocation_metrics(
    properties: List[PropertyData],
    allocations: Dict[str, float]
) -> AllocationMetrics:
    """
    Calculate quality metrics for the allocation
    """
    # Calculate Gini coefficient for fairness
    reduction_percentages = []
    for prop in properties:
        allocated = allocations.get(prop.property_id, 0)
        reduction_pct = (allocated / prop.baseline_emission) * 100 if prop.baseline_emission > 0 else 0
        reduction_percentages.append(reduction_pct)
    
    # Gini coefficient calculation
    sorted_reductions = sorted(reduction_percentages)
    n = len(sorted_reductions)
    cumsum = np.cumsum(sorted_reductions)
    gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_reductions))) / (n * sum(sorted_reductions)) - (n + 1) / n
    gini = max(0, min(1, gini))  # Ensure between 0 and 1
    
    # Calculate efficiency score (based on carbon intensity coverage)
    total_intensity = sum(p.carbon_intensity * p.baseline_emission for p in properties)
    allocated_intensity = sum(
        p.carbon_intensity * allocations.get(p.property_id, 0)
        for p in properties
    )
    efficiency = (allocated_intensity / total_intensity * 100) if total_intensity > 0 else 0
    
    # Calculate feasibility score (based on retrofit potential)
    avg_feasibility = np.mean([p.retrofit_potential for p in properties])
    
    # Calculate coverage
    total_target = sum(allocations.values())
    total_baseline = sum(p.baseline_emission for p in properties)
    coverage = (total_target / total_baseline * 100) if total_baseline > 0 else 0
    
    return AllocationMetrics(
        fairness_index=round(1 - gini, 3),  # Convert to fairness (1 = fair, 0 = unfair)
        efficiency_score=round(min(100, efficiency), 2),
        feasibility_score=round(avg_feasibility, 2),
        coverage=round(min(100, coverage), 2)
    )

def generate_recommended_actions(
    property_data: PropertyData,
    allocated_reduction: float
) -> List[str]:
    """
    Generate recommended actions based on property characteristics and reduction target
    """
    actions = []
    reduction_percentage = (allocated_reduction / property_data.baseline_emission * 100) if property_data.baseline_emission > 0 else 0
    
    # High retrofit potential properties
    if property_data.retrofit_potential > 70:
        actions.append("High-efficiency HVAC system upgrade")
        actions.append("LED lighting retrofit with smart controls")
        actions.append("Building envelope improvements (insulation, glazing)")
    
    # Medium retrofit potential
    elif property_data.retrofit_potential > 40:
        actions.append("Energy management system installation")
        actions.append("Variable frequency drives for motors")
        actions.append("Solar PV installation (rooftop or carport)")
    
    # Lower retrofit potential - focus on operational improvements
    else:
        actions.append("Operational schedule optimization")
        actions.append("Behavioral change programs")
        actions.append("Regular maintenance and commissioning")
    
    # Add renewable energy for aggressive targets
    if reduction_percentage > 40:
        actions.append("On-site renewable energy generation (solar/wind)")
        actions.append("Green power purchase agreement")
    
    # Building type specific recommendations
    if "office" in property_data.building_type.lower():
        actions.append("Workspace density optimization")
        actions.append("Smart lighting with occupancy sensors")
    elif "retail" in property_data.building_type.lower():
        actions.append("Refrigeration system upgrade")
        actions.append("Display lighting optimization")
    
    return actions[:5]  # Return top 5 recommendations

# =============================================================================
# API ENDPOINTS
# =============================================================================

# @router.post("/calculate", response_model=APIResponse)
# @measure_execution_time
# async def calculate_target_division(
    # request: TargetDivisionRequest,
    # background_tasks: BackgroundTasks,
    # current_user: TokenData = Depends(get_current_user),
    # _: bool = Depends(check_rate_limit_dependency)
# ):
    # """
    # API 5: Calculate Target Division
    
    # Allocate portfolio-level reduction targets to individual properties using
    # advanced AI algorithms. Supports multiple allocation methods and multi-objective
    # optimization.
    
    # **Authentication Required:** Bearer Token
    
    # **Allocation Methods:**
    # - carbon_intensity_weighted: Prioritizes high-carbon-intensity properties
    # - proportional: Simple proportional to baseline emissions
    # - retrofit_potential: Based on retrofit feasibility
    # - equal_distribution: Equal percentage reduction across all properties
    # - ai_optimized: Multi-objective optimization balancing fairness, efficiency, feasibility
    
    # **Key Features:**
    # - Multi-objective optimization with configurable weights
    # - Feasibility assessment for each property
    # - Cost estimation based on property characteristics
    # - Recommended action generation
    # - Allocation quality metrics (fairness, efficiency, coverage)
    
    # **Business Logic:**
    # 1. Validates scenario exists and retrieves portfolio targets
    # 2. Applies selected allocation algorithm
    # 3. Calculates per-property targets for each target year
    # 4. Estimates implementation costs
    # 5. Generates recommended actions
    # 6. Calculates allocation quality metrics
    # 7. Returns detailed allocation for preview
    
    # **Response:** Property-level targets with:
    # - Allocated emission targets per year
    # - Reduction percentages and absolute amounts
    # - Feasibility scores
    # - Cost estimates
    # - Recommended actions
    # - Overall allocation metrics
    # """
    # request_id = generate_request_id()
    # allocation_id = str(uuid.uuid4())
    
    # try:
        # logger.info(
            # f"Calculating target division (request_id: {request_id}, "
            # f"scenario: {request.scenario_id}, method: {request.allocation_method}, "
            # f"properties: {len(request.properties)})"
        # )
        
        # # TODO: Retrieve scenario from database and validate
        # # scenario = await db.milestone_scenarios.find_one({"scenario_id": request.scenario_id})
        
        # # Mock portfolio targets (in production, from scenario)
        # portfolio_targets = {
            # 2025: 4465.69,  # Total reduction target for 2025
            # 2030: 2820.75,  # Total reduction target for 2030
            # 2050: 940.25    # Total reduction target for 2050
        # }
        
        # # Initialize allocation engine
        # engine = TargetAllocationEngine()
        
        # # Select allocation method
        # allocation_methods = {
            # AllocationMethod.CARBON_INTENSITY_WEIGHTED: engine.allocate_carbon_intensity_weighted,
            # AllocationMethod.PROPORTIONAL: engine.allocate_proportional,
            # AllocationMethod.RETROFIT_POTENTIAL: engine.allocate_retrofit_potential,
            # AllocationMethod.EQUAL_DISTRIBUTION: engine.allocate_equal_distribution,
        # }
        
        # # Get total baseline for calculating portfolio reduction
        # total_baseline = sum(p.baseline_emission for p in request.properties)
        
        # # Store all property targets
        # all_property_targets = []
        
        # for year in request.target_years:
            # # Get portfolio target for this year (or interpolate if not exact match)
            # if year in portfolio_targets:
                # portfolio_reduction = total_baseline - portfolio_targets[year]
            # else:
                # # Interpolate for years not in the scenario
                # sorted_years = sorted(portfolio_targets.keys())
                # if year < sorted_years[0]:
                    # portfolio_reduction = 0
                # elif year > sorted_years[-1]:
                    # portfolio_reduction = total_baseline - portfolio_targets[sorted_years[-1]]
                # else:
                    # # Linear interpolation
                    # for i in range(len(sorted_years) - 1):
                        # if sorted_years[i] < year < sorted_years[i + 1]:
                            # y1 = portfolio_targets[sorted_years[i]]
                            # y2 = portfolio_targets[sorted_years[i + 1]]
                            # x1 = sorted_years[i]
                            # x2 = sorted_years[i + 1]
                            # interpolated_target = y1 + (y2 - y1) * (year - x1) / (x2 - x1)
                            # portfolio_reduction = total_baseline - interpolated_target
                            # break
            
            # # Apply allocation method
            # if request.allocation_method == AllocationMethod.AI_OPTIMIZED:
                # objectives = request.optimization_objectives or {
                    # 'fairness': 0.4,
                    # 'efficiency': 0.4,
                    # 'feasibility': 0.2
                # }
                # allocations = engine.allocate_ai_optimized(
                    # request.properties,
                    # portfolio_reduction,
                    # objectives
                # )
            # else:
                # allocation_func = allocation_methods.get(request.allocation_method)
                # allocations = allocation_func(request.properties, portfolio_reduction)
            
            # # Calculate metrics for this year's allocation
            # metrics = calculate_allocation_metrics(request.properties, allocations)
            
            # # Create property targets
            # for prop in request.properties:
                # allocated_reduction = allocations.get(prop.property_id, 0)
                # target_emission = prop.baseline_emission - allocated_reduction
                # reduction_pct = (allocated_reduction / prop.baseline_emission * 100) if prop.baseline_emission > 0 else 0
                
                # # Estimate cost (simplified model: $150 per tonne CO2e reduced)
                # cost_per_tonne = 150
                # estimated_cost = allocated_reduction * cost_per_tonne
                
                # # Generate recommended actions
                # recommended_actions = generate_recommended_actions(prop, allocated_reduction)
                
                # # Calculate allocation weight
                # total_allocated = sum(allocations.values())
                # allocation_weight = allocated_reduction / total_allocated if total_allocated > 0 else 0
                
                # property_target = PropertyTarget(
                    # property_id=prop.property_id,
                    # property_name=prop.property_name,
                    # year=year,
                    # allocated_target=round(target_emission, 2),
                    # reduction_from_baseline=round(reduction_pct, 2),
                    # absolute_reduction=round(allocated_reduction, 2),
                    # allocation_weight=round(allocation_weight, 4),
                    # feasibility_score=prop.retrofit_potential,
                    # estimated_cost=round(estimated_cost, 2),
                    # recommended_actions=recommended_actions
                # )
                
                # all_property_targets.append(property_target.dict())
        
        # # Calculate overall metrics (using last year's allocation)
        # final_allocations = {
            # prop.property_id: next(
                # pt['absolute_reduction'] for pt in all_property_targets
                # if pt['property_id'] == prop.property_id and pt['year'] == request.target_years[-1]
            # )
            # for prop in request.properties
        # }
        # overall_metrics = calculate_allocation_metrics(request.properties, final_allocations)
        
        # logger.info(
            # f"Target division calculated successfully (allocation_id: {allocation_id}, "
            # f"fairness: {overall_metrics.fairness_index})"
        # )
        
        # response_data = {
            # "allocation_id": allocation_id,
            # "scenario_id": request.scenario_id,
            # "allocation_method": request.allocation_method.value,
            # "property_targets": all_property_targets,
            # "allocation_metrics": overall_metrics.dict(),
            # "summary": {
                # "total_properties": len(request.properties),
                # "target_years": request.target_years,
                # "total_baseline_emission": round(total_baseline, 2),
                # "total_allocated_reduction": round(sum(final_allocations.values()), 2),
                # "average_reduction_percentage": round(
                    # np.mean([pt['reduction_from_baseline'] for pt in all_property_targets]),
                    # 2
                # )
            # },
            # "calculation_metadata": {
                # "calculated_at": datetime.utcnow().isoformat(),
                # "algorithm_version": "2.1.0",
                # "user_id": current_user.user_id
            # }
        # }
        
        # # Background task to cache results
        # # background_tasks.add_task(cache_allocation_results, allocation_id, response_data)
        
        # return create_success_response(
            # data=response_data,
            # request_id=request_id,
            # status_code=200,
            # message=f"Target division calculated successfully for {len(request.properties)} properties"
        # )
        
    # except ValueError as e:
        # logger.error(f"Validation error (request_id: {request_id}): {str(e)}")
        # return create_error_response(
            # error_code="VALIDATION_ERROR",
            # error_message=str(e),
            # request_id=request_id,
            # status_code=400
        # )
    # except Exception as e:
        # logger.error(f"Calculation error (request_id: {request_id}): {str(e)}", exc_info=True)
        # return create_error_response(
            # error_code="ALLOCATION_ERROR",
            # error_message="Error during target division calculation",
            # request_id=request_id,
            # status_code=500
        # )
        
@router.post("/calculate", response_model=APIResponse)
@measure_execution_time
async def calculate_target_allocation(
    request: ServiceAllocationRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """API 5: Calculate Target Allocation"""
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Calculating target allocation (request_id: {request_id}, "
            f"scenario_id: {request.scenario_id}, properties: {len(request.property_ids)})"
        )
        
        method_mapping = {
            "PROPORTIONAL_BASELINE": "PROPORTIONAL",
            "EQUAL_DISTRIBUTION": "EQUAL",
            "CAPACITY_BASED": "RETROFIT_POTENTIAL",
            "COST_OPTIMIZED": "AI_OPTIMIZED",
            "INTENSITY_WEIGHTED": "INTENSITY_WEIGHTED"
        }

        api_method = request.allocation_method.value if hasattr(request.allocation_method, 'value') else str(request.allocation_method)
        service_method = method_mapping.get(api_method, api_method)

        logger.info(f"Mapping allocation method: {api_method} -> {service_method}")
        
        # Create service request
        service_request = ServiceAllocationRequest(
            scenario_id=request.scenario_id,
            property_ids=request.property_ids,
            total_reduction_target=request.total_reduction_target,
            allocation_method=service_method,  # Use mapped method
            constraints={
                "min_reduction": request.constraints.get("min_reduction_per_property") if request.constraints else None,
                "max_reduction": request.constraints.get("max_reduction_per_property") if request.constraints else None,
                "budget_limit": request.constraints.get("total_budget_limit") if request.constraints else None
            } if request.constraints else None
        )
        
        # Call service layer
        result = allocation_service.allocate_targets(service_request)
        
        # ADD DEBUGGING HERE
        logger.info(f"Service call completed: is_success={result.is_success}")
        logger.info(f"Service result message: {result.message}")
        logger.info(f"Service result status: {result.status}")

        # ADD THIS - Check for error details
        if hasattr(result, 'error_details'):
            logger.error(f"Validation errors: {result.error_details}")
        if hasattr(result, 'errors'):
            logger.error(f"Errors list: {result.errors}")

        # Also log what we sent to the service
        logger.info(f"Service request - scenario_id: {service_request.scenario_id}")
        logger.info(f"Service request - property_ids: {service_request.property_ids}")
        logger.info(f"Service request - total_reduction_target: {service_request.total_reduction_target}")
        logger.info(f"Service request - allocation_method: {service_request.allocation_method}")
        logger.info(f"Service request - constraints: {service_request.constraints}")
        
        if not result.is_success:
            return create_error_response(
                error_code="ALLOCATION_ERROR",
                error_message=result.message or "Target allocation failed",
                request_id=request_id,
                status_code=500
            )
        
        # Convert service result to API response
        allocations_data = []
        if hasattr(result.data, 'allocations'):
            for allocation in result.data.allocations:
                alloc_dict = allocation.to_dict() if hasattr(allocation, 'to_dict') else (
                    allocation.__dict__ if hasattr(allocation, '__dict__') else allocation
                )
                allocations_data.append(alloc_dict)
        elif isinstance(result.data, list):
            allocations_data = result.data
        
        # Calculate summary
        total_allocated = sum(
            a.get("allocated_2030_target", 0) + a.get("allocated_2050_target", 0) 
            for a in allocations_data
        )
        
        response_data = {
            "allocations": allocations_data,
            "allocation_summary": {
                "total_properties": len(allocations_data),
                "allocation_method": api_method,  # Use the api_method variable we created earlier
                "total_allocated_2030": sum(a.get("allocated_2030_target", 0) for a in allocations_data),
                "total_allocated_2050": sum(a.get("allocated_2050_target", 0) for a in allocations_data),
                "total_allocated": total_allocated
            }
        }
        
        # Background task
        if allocations_data:
            background_tasks.add_task(
                save_allocation_results,
                allocations_data,
                request_id
            )
        
        return create_success_response(
            data=response_data,
            request_id=request_id,
            status_code=200,
            message=f"Successfully allocated targets to {len(allocations_data)} properties"
        )
        
    except Exception as e:
        logger.error(f"Allocation error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="ALLOCATION_ERROR",
            error_message=f"Allocation failed: {str(e)}",  # Include actual error
            request_id=request_id,
            status_code=500
        )

# @router.get("/visualization/{allocation_id}", response_model=APIResponse)
# @measure_execution_time
# async def get_target_division_visualization(
    # allocation_id: str = Path(..., description="UUID of allocation calculation"),
    # property_id: Optional[str] = Query(None, description="Filter by specific property"),
    # year: Optional[int] = Query(None, ge=2025, le=2070, description="Filter by specific year"),
    # current_user: TokenData = Depends(get_current_user),
    # _: bool = Depends(check_rate_limit_dependency)
# ):
    # """
    # API 6: Get Target Division Visualization
    
    # Retrieve visualization-ready data for target allocation results.
    # Supports filtering by property and year.
    
    # **Authentication Required:** Bearer Token
    
    # **Query Parameters:**
    # - property_id: Filter results for specific property
    # - year: Filter results for specific year
    
    # **Response includes:**
    # - Property-level target breakdowns
    # - Comparative analysis across properties
    # - Timeline visualizations for each property
    # - Allocation fairness metrics
    # - Cost distribution charts
    # """
    # request_id = generate_request_id()
    
    # try:
        # logger.info(
            # f"Retrieving target division visualization (request_id: {request_id}, "
            # f"allocation_id: {allocation_id})"
        # )
        
        # # TODO: Retrieve from database/cache
        # # allocation_data = await db.target_allocations.find_one({"allocation_id": allocation_id})
        
        # visualization_data = {
            # "allocation_id": allocation_id,
            # "visualization_config": {
                # "chart_types": {
                    # "property_comparison": "bar",
                    # "timeline": "line",
                    # "cost_distribution": "treemap",
                    # "fairness_metrics": "radar"
                # },
                # "color_scheme": {
                    # "primary": "#2E8B57",
                    # "secondary": "#FF6B6B",
                    # "tertiary": "#4ECDC4",
                    # "quaternary": "#FFD93D"
                # }
            # },
            # "filters_applied": {
                # "property_id": property_id,
                # "year": year
            # }
        # }
        
        # return create_success_response(
            # data=visualization_data,
            # request_id=request_id,
            # message="Target division visualization data retrieved successfully"
        # )
        
    # except Exception as e:
        # logger.error(f"Visualization error (request_id: {request_id}): {str(e)}", exc_info=True)
        # return create_error_response(
            # error_code="VISUALIZATION_ERROR",
            # error_message="Error retrieving visualization data",
            # request_id=request_id,
            # status_code=500
        # )


@router.get("/visualization/{allocation_id}", response_model=APIResponse)
@measure_execution_time
async def get_target_division_visualization(
    allocation_id: str = Path(..., description="UUID of allocation calculation"),
    property_id: Optional[str] = Query(None, description="Filter by specific property"),
    year: Optional[int] = Query(None, ge=2025, le=2070, description="Filter by specific year"),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 5: Get Target Division Visualization
    
    Retrieve visualization-ready data for target allocation results.
    Supports filtering by property and year.
    
    **Authentication Required:** Bearer Token
    
    **Query Parameters:**
    - property_id: Filter results for specific property
    - year: Filter results for specific year
    
    **Response includes:**
    - Property-level target breakdowns
    - Comparative analysis across properties
    - Timeline visualizations for each property
    - Allocation fairness metrics
    - Cost distribution charts
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Retrieving target division visualization (request_id: {request_id}, "
            f"allocation_id: {allocation_id}, property_id: {property_id}, year: {year})"
        )
        
        # Call service layer
        result = allocation_service.get_allocation_visualization(
            allocation_id=allocation_id,
            property_id=property_id,
            year=year
        )
        
        if not result.is_success:
            logger.error(f"Visualization retrieval failed: {result.message}")
            return create_error_response(
                error_code="VISUALIZATION_ERROR",
                error_message=result.message or "Failed to retrieve visualization data",
                request_id=request_id,
                status_code=500
            )
        
        logger.info(f"Visualization data retrieved successfully (request_id: {request_id})")
        
        return create_success_response(
            data=result.data,
            request_id=request_id,
            message="Target division visualization data retrieved successfully"
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


@router.get("/property/{property_id}", response_model=APIResponse)
@measure_execution_time
async def get_property_allocation(
    property_id: str = Path(..., description="UUID of the property"),
    scenario_id: Optional[str] = Query(None, description="Filter by specific scenario"),
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 6: Get Property Allocation
    
    Retrieve allocation data for a specific property across all target years.
    
    **Authentication Required:** Bearer Token
    
    **Path Parameters:**
    - property_id: UUID of the property
    
    **Query Parameters:**
    - scenario_id: Optional scenario filter
    
    **Response includes:**
    - Property details
    - All allocations for the property (2030, 2050)
    - Recommended actions
    - Implementation timeline
    - Feasibility assessment
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Retrieving property allocation (request_id: {request_id}, "
            f"property_id: {property_id}, scenario_id: {scenario_id})"
        )
        
        # Call service layer
        result = allocation_service.get_property_allocation(
            property_id=property_id,
            scenario_id=scenario_id
        )
        
        if not result.is_success:
            logger.error(f"Property allocation retrieval failed: {result.message}")
            return create_error_response(
                error_code="RETRIEVAL_ERROR",
                error_message=result.message or "Failed to retrieve property allocation",
                request_id=request_id,
                status_code=404 if result.status == ServiceResultStatus.NOT_FOUND else 500
            )
        
        logger.info(f"Property allocation retrieved successfully (request_id: {request_id})")
        
        return create_success_response(
            data=result.data,
            request_id=request_id,
            message=f"Property allocation retrieved for {property_id}"
        )
        
    except Exception as e:
        logger.error(f"Property allocation error (request_id: {request_id}): {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return create_error_response(
            error_code="RETRIEVAL_ERROR",
            error_message=str(e),
            request_id=request_id,
            status_code=500
        )


# @router.post("/register", response_model=APIResponse)
# @measure_execution_time
# async def register_target_division(
    # request: RegisterTargetDivisionRequest,
    # background_tasks: BackgroundTasks,
    # current_user: TokenData = Depends(get_current_user),
    # _: bool = Depends(check_rate_limit_dependency)
# ):
    # """
    # API 7: Register Target Division
    
    # Register the selected target allocation as the official property-level targets.
    # Persists allocations to database and activates them for implementation planning.
    
    # **Authentication Required:** Bearer Token
    
    # **Key Features:**
    # - Validates allocation exists
    # - Records approval metadata
    # - Activates property-level targets
    # - Triggers downstream processes (planning, notifications)
    # - Creates audit trail
    
    # **Business Logic:**
    # 1. Validates allocation_id exists
    # 2. Checks user approval permissions
    # 3. Persists approved allocation
    # 4. Updates property target tables
    # 5. Triggers notification workflow
    # 6. Returns registration confirmation
    # """
    # request_id = generate_request_id()
    
    # try:
        # logger.info(
            # f"Registering target division (request_id: {request_id}, "
            # f"allocation_id: {request.allocation_id}, user: {current_user.user_id})"
        # )
        
        # # TODO: Validate allocation exists
        # # allocation = await db.target_allocations.find_one({"allocation_id": request.allocation_id})
        
        # registration_id = f"REG_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:6]}"
        
        # registration_data = {
            # "registration_id": registration_id,
            # "allocation_id": request.allocation_id,
            # "approval_status": request.approval_status.value,
            # "approved_by": request.approved_by or current_user.user_id,
            # "approval_date": request.approval_date.isoformat(),
            # "notes": request.notes,
            # "registered_at": datetime.utcnow().isoformat(),
            # "registered_by": current_user.user_id
        # }
        
        # # TODO: Persist to database
        # # await db.registered_allocations.insert_one(registration_data)
        
        # logger.info(f"Target division registered successfully (registration_id: {registration_id})")
        
        # return create_success_response(
            # data=registration_data,
            # request_id=request_id,
            # status_code=201,
            # message="Target division registered successfully"
        # )
        
    # except Exception as e:
        # logger.error(f"Registration error (request_id: {request_id}): {str(e)}", exc_info=True)
        # return create_error_response(
            # error_code="REGISTRATION_ERROR",
            # error_message="Error during target division registration",
            # request_id=request_id,
            # status_code=500
        # )





# =============================================================================
# 4. REPLACE API 7 - POST /register
# Replace lines 849-923 with this implementation:
# =============================================================================

@router.post("/register", response_model=APIResponse)
@measure_execution_time
async def register_target_division(
    request: RegisterTargetDivisionRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """
    API 7: Register Target Division
    
    Register the selected target allocation as the official property-level targets.
    Persists allocations to database and activates them for implementation planning.
    
    **Authentication Required:** Bearer Token
    
    **Key Features:**
    - Validates allocation exists
    - Records approval metadata
    - Activates property-level targets
    - Triggers downstream processes (planning, notifications)
    - Creates audit trail
    
    **Business Logic:**
    1. Validates allocation_id exists
    2. Checks user approval permissions
    3. Persists approved allocation
    4. Updates property target tables
    5. Triggers notification workflow
    6. Returns registration confirmation
    """
    request_id = generate_request_id()
    
    try:
        logger.info(
            f"Registering target division (request_id: {request_id}, "
            f"allocation_id: {request.allocation_id}, user: {current_user.user_id})"
        )
        
        # Prepare approval info
        approval_info = {
            "approved_by": request.approved_by or current_user.user_id,
            "approval_date": request.approval_date.isoformat() if request.approval_date else datetime.utcnow().isoformat(),
            "approval_status": request.approval_status.value if hasattr(request.approval_status, 'value') else str(request.approval_status),
            "notes": request.notes
        }
        
        # Call service layer
        result = allocation_service.register_allocation(
            allocation_id=request.allocation_id,
            approval_info=approval_info
        )
        
        if not result.is_success:
            logger.error(f"Registration failed: {result.message}")
            
            # Check if validation error
            if result.status == ServiceResultStatus.VALIDATION_ERROR:
                return create_error_response(
                    error_code="VALIDATION_ERROR",
                    error_message=result.message,
                    request_id=request_id,
                    status_code=400
                )
            
            return create_error_response(
                error_code="REGISTRATION_ERROR",
                error_message=result.message or "Registration failed",
                request_id=request_id,
                status_code=500
            )
        
        logger.info(f"Target division registered successfully (registration_id: {result.data['registration_id']})")
        
        # Optional: Add background task for notifications
        # background_tasks.add_task(send_registration_notification, result.data)
        
        return create_success_response(
            data=result.data,
            request_id=request_id,
            status_code=201,
            message="Target division registered successfully"
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

logger.info("Module 2: Target Division APIs initialized successfully")
