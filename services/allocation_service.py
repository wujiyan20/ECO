# services/allocation_service.py - Target Allocation Service
"""
Service for allocating emission reduction targets across properties.

Features:
- Multiple allocation methods (proportional, intensity-weighted, AI-optimized)
- Fairness and efficiency optimization
- Constraint handling (budget, feasibility)
- Manual adjustment support
- Allocation validation
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached,
    transaction
)

# Import models
try:
    from models import (
        Property,
        PropertyFilter,
        AllocationMethod,
        PropertyRepository,
        generate_uuid,
        validate_percentage,
        validate_positive_number
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Models package not available - using mock mode")

# Import AI functions
try:
    from ai_functions import PropertyTargetAllocator
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI functions not available - using simplified allocation")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

@dataclass
class AllocationRequest:
    """Request for target allocation"""
    scenario_id: str
    property_ids: List[str]
    total_reduction_target: float
    target_years: List[int] = field(default_factory=lambda: [2030, 2050])
    allocation_method: str = "AI_OPTIMIZED"
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None


@dataclass
class PropertyAllocation:
    """Allocation result for a single property"""
    property_id: str
    property_name: str
    year: int
    baseline_emission: float
    allocated_target: float
    reduction_amount: float
    reduction_percentage: float
    allocation_weight: float
    feasibility_score: float
    estimated_cost: float
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class AllocationResult:
    """Complete allocation result"""
    allocation_id: str
    scenario_id: str
    property_allocations: List[PropertyAllocation]
    allocation_metrics: Dict[str, float]
    total_allocated: float
    total_target: float
    coverage_percentage: float


@dataclass
class AllocationAdjustment:
    """Request to adjust allocation"""
    property_id: str
    new_target: float
    reason: Optional[str] = None


# =============================================================================
# ALLOCATION SERVICE
# =============================================================================

class AllocationService(BaseService):
    """
    Service for allocating emission reduction targets across properties.
    
    Provides:
    - Multiple allocation algorithms
    - AI-optimized distribution
    - Fairness and efficiency metrics
    - Manual adjustment handling
    - Constraint validation
    
    Usage:
        service = AllocationService(db_manager)
        request = AllocationRequest(
            scenario_id="SCN-001",
            property_ids=["PROP-001", "PROP-002"],
            total_reduction_target=5000.0
        )
        result = service.allocate_targets(request)
    """
    
    def __init__(self, db_manager=None, property_service=None):
        """
        Initialize allocation service.
        
        Args:
            db_manager: Database manager instance
            property_service: Property service for property data
        """
        super().__init__(db_manager)
        self._property_service = property_service
        self._ai_allocator: Optional[PropertyTargetAllocator] = None
        self._property_repo: Optional[PropertyRepository] = None
    
    def _do_initialize(self) -> None:
        """Initialize AI allocator and repositories"""
        if self.db_manager and MODELS_AVAILABLE:
            self._property_repo = PropertyRepository(self.db_manager)
            self._logger.info("Property repository initialized")
        
        if AI_AVAILABLE:
            self._ai_allocator = PropertyTargetAllocator()
            self._logger.info("AI allocator initialized")
    
    # =========================================================================
    # TARGET ALLOCATION
    # =========================================================================
    
    @measure_time
    def allocate_targets(self, request: AllocationRequest) -> ServiceResult[AllocationResult]:
        """
        Allocate reduction targets across properties.
        
        Args:
            request: Allocation request with parameters
            
        Returns:
            ServiceResult containing allocation results
        """
        # Validate request
        validation_errors = self._validate_allocation_request(request)
        if validation_errors:
            return ServiceResult.validation_error(validation_errors)
        
        return self._execute(self._allocate_targets_impl, request)
    
    def _validate_allocation_request(self, request: AllocationRequest) -> List[str]:
        """Validate allocation request"""
        errors = []
        
        if not request.property_ids:
            errors.append("At least one property_id is required")
        
        if request.total_reduction_target <= 0:
            errors.append("total_reduction_target must be positive")
        
        valid_methods = ["PROPORTIONAL", "INTENSITY_WEIGHTED", "RETROFIT_POTENTIAL", "AI_OPTIMIZED", "EQUAL"]
        if request.allocation_method not in valid_methods:
            errors.append(f"Invalid allocation_method. Must be one of: {valid_methods}")
        
        if not request.target_years:
            errors.append("At least one target_year is required")
        
        return errors
    
    def _allocate_targets_impl(self, request: AllocationRequest) -> ServiceResult[AllocationResult]:
        """Implementation of allocate_targets"""
        allocation_id = f"ALLOC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Get property data
        properties = self._get_properties_data(request.property_ids)
        if not properties:
            return ServiceResult.error("No properties found for allocation")
        
        # Calculate allocations based on method
        allocations = []
        for year in request.target_years:
            year_allocations = self._calculate_allocations(
                properties=properties,
                total_target=request.total_reduction_target,
                year=year,
                method=request.allocation_method,
                constraints=request.constraints
            )
            allocations.extend(year_allocations)
        
        # Calculate metrics
        metrics = self._calculate_allocation_metrics(allocations, properties)
        
        # Build result
        total_allocated = sum(a.reduction_amount for a in allocations)
        result = AllocationResult(
            allocation_id=allocation_id,
            scenario_id=request.scenario_id,
            property_allocations=allocations,
            allocation_metrics=metrics,
            total_allocated=total_allocated,
            total_target=request.total_reduction_target * len(request.target_years),
            coverage_percentage=(total_allocated / (request.total_reduction_target * len(request.target_years))) * 100 if request.total_reduction_target > 0 else 0
        )
        
        return ServiceResult.success(result)
    
    def _get_properties_data(self, property_ids: List[str]) -> List[Dict]:
        """Get property data for allocation"""
        properties = []
        
        if self._property_repo:
            filter_obj = PropertyFilter(property_ids=property_ids) if MODELS_AVAILABLE else None
            db_properties = self._property_repo.get_all(filter_obj)
            for p in db_properties:
                properties.append({
                    "property_id": p.property_id,
                    "name": p.name,
                    "baseline_emission": p.baseline_emission,
                    "area_sqm": p.area_sqm,
                    "carbon_intensity": p.carbon_intensity if hasattr(p, 'carbon_intensity') else p.baseline_emission / p.area_sqm,
                    "retrofit_potential": getattr(p, 'retrofit_potential', 'MEDIUM'),
                    "building_type": str(p.building_type.value if hasattr(p.building_type, 'value') else p.building_type)
                })
        else:
            # Mock properties
            for i, pid in enumerate(property_ids):
                properties.append({
                    "property_id": pid,
                    "name": f"Property {i+1}",
                    "baseline_emission": 1000.0 + i * 200,
                    "area_sqm": 5000.0 + i * 1000,
                    "carbon_intensity": 200.0 - i * 10,
                    "retrofit_potential": "HIGH" if i % 2 == 0 else "MEDIUM",
                    "building_type": "OFFICE"
                })
        
        return properties
    
    def _calculate_allocations(self, properties: List[Dict], total_target: float,
                              year: int, method: str,
                              constraints: Optional[Dict]) -> List[PropertyAllocation]:
        """Calculate allocations using specified method"""
        # Calculate weights based on method
        weights = self._calculate_weights(properties, method)
        
        # Apply constraints if any
        if constraints:
            weights = self._apply_constraints(weights, properties, constraints)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Create allocations
        allocations = []
        for prop in properties:
            pid = prop["property_id"]
            weight = weights.get(pid, 1.0 / len(properties))
            
            baseline = prop["baseline_emission"]
            reduction_amount = total_target * weight
            allocated_target = baseline - reduction_amount
            reduction_pct = (reduction_amount / baseline) * 100 if baseline > 0 else 0
            
            # Calculate feasibility
            feasibility = self._calculate_property_feasibility(prop, reduction_pct)
            
            # Estimate cost
            estimated_cost = self._estimate_allocation_cost(prop, reduction_amount)
            
            # Get recommendations
            recommendations = self._get_property_recommendations(prop, reduction_pct)
            
            allocation = PropertyAllocation(
                property_id=pid,
                property_name=prop["name"],
                year=year,
                baseline_emission=baseline,
                allocated_target=round(allocated_target, 2),
                reduction_amount=round(reduction_amount, 2),
                reduction_percentage=round(reduction_pct, 2),
                allocation_weight=round(weight, 4),
                feasibility_score=feasibility,
                estimated_cost=estimated_cost,
                recommended_actions=recommendations
            )
            allocations.append(allocation)
        
        return allocations
    
    def _calculate_weights(self, properties: List[Dict], method: str) -> Dict[str, float]:
        """Calculate allocation weights based on method"""
        weights = {}
        
        if method == "EQUAL":
            # Equal distribution
            for prop in properties:
                weights[prop["property_id"]] = 1.0
        
        elif method == "PROPORTIONAL":
            # Proportional to baseline emission
            total_emission = sum(p["baseline_emission"] for p in properties)
            for prop in properties:
                weights[prop["property_id"]] = prop["baseline_emission"] / total_emission if total_emission > 0 else 1.0
        
        elif method == "INTENSITY_WEIGHTED":
            # Higher weight to higher carbon intensity
            total_intensity = sum(p["carbon_intensity"] for p in properties)
            for prop in properties:
                weights[prop["property_id"]] = prop["carbon_intensity"] / total_intensity if total_intensity > 0 else 1.0
        
        elif method == "RETROFIT_POTENTIAL":
            # Higher weight to higher retrofit potential
            potential_scores = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
            total_score = sum(potential_scores.get(str(p.get("retrofit_potential", "MEDIUM")).upper(), 2.0) for p in properties)
            for prop in properties:
                score = potential_scores.get(str(prop.get("retrofit_potential", "MEDIUM")).upper(), 2.0)
                weights[prop["property_id"]] = score / total_score if total_score > 0 else 1.0
        
        elif method == "AI_OPTIMIZED":
            # AI-based optimization
            weights = self._ai_optimize_weights(properties)
        
        else:
            # Default to equal
            for prop in properties:
                weights[prop["property_id"]] = 1.0
        
        return weights
    
    def _ai_optimize_weights(self, properties: List[Dict]) -> Dict[str, float]:
        """Use AI to optimize allocation weights"""
        if self._ai_allocator:
            try:
                # Prepare features for AI
                import numpy as np
                features = np.array([
                    [p["baseline_emission"], p["area_sqm"], p["carbon_intensity"]]
                    for p in properties
                ])
                
                # Get AI predictions
                predictions = self._ai_allocator.predict(features)
                
                # Convert to weights
                weights = {}
                total = sum(predictions)
                for i, prop in enumerate(properties):
                    weights[prop["property_id"]] = predictions[i] / total if total > 0 else 1.0 / len(properties)
                
                return weights
            except Exception as e:
                self._logger.warning(f"AI optimization failed, falling back to heuristic: {e}")
        
        # Fallback: Combined heuristic
        weights = {}
        for prop in properties:
            # Combine multiple factors
            intensity_factor = prop["carbon_intensity"] / 200  # Normalize around typical value
            potential_factor = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}.get(
                str(prop.get("retrofit_potential", "MEDIUM")).upper(), 1.0
            )
            emission_factor = prop["baseline_emission"] / 1000  # Normalize
            
            weights[prop["property_id"]] = intensity_factor * potential_factor * 0.6 + emission_factor * 0.4
        
        return weights
    
    def _apply_constraints(self, weights: Dict[str, float], properties: List[Dict],
                          constraints: Dict) -> Dict[str, float]:
        """Apply constraints to allocation weights"""
        adjusted_weights = weights.copy()
        
        # Maximum reduction constraint
        max_reduction_pct = constraints.get("max_reduction_percentage", 100)
        
        # Property-specific constraints
        property_constraints = constraints.get("property_constraints", {})
        
        for prop in properties:
            pid = prop["property_id"]
            
            # Check property-specific max
            prop_max = property_constraints.get(pid, {}).get("max_reduction", max_reduction_pct)
            
            # Adjust weight if needed
            baseline = prop["baseline_emission"]
            current_weight = adjusted_weights.get(pid, 0)
            
            # Calculate implied reduction
            # This is approximate - actual reduction depends on total target
            if current_weight > prop_max / 100:
                adjusted_weights[pid] = prop_max / 100
        
        return adjusted_weights
    
    def _calculate_property_feasibility(self, prop: Dict, reduction_pct: float) -> float:
        """Calculate feasibility score for property allocation"""
        base_score = 100.0
        
        # Reduce score for high reduction targets
        if reduction_pct > 50:
            base_score -= (reduction_pct - 50) * 1.5
        
        # Adjust based on retrofit potential
        potential = str(prop.get("retrofit_potential", "MEDIUM")).upper()
        potential_adjustments = {"HIGH": 10, "MEDIUM": 0, "LOW": -15}
        base_score += potential_adjustments.get(potential, 0)
        
        # Adjust based on carbon intensity (higher intensity = easier to reduce)
        intensity = prop.get("carbon_intensity", 200)
        if intensity > 250:
            base_score += 5
        elif intensity < 150:
            base_score -= 5
        
        return max(0, min(100, round(base_score, 1)))
    
    def _estimate_allocation_cost(self, prop: Dict, reduction_amount: float) -> float:
        """Estimate cost for property allocation"""
        # Base cost per tonne (varies by retrofit potential)
        potential = str(prop.get("retrofit_potential", "MEDIUM")).upper()
        cost_per_tonne = {"HIGH": 120, "MEDIUM": 150, "LOW": 200}.get(potential, 150)
        
        return round(reduction_amount * cost_per_tonne, 2)
    
    def _get_property_recommendations(self, prop: Dict, reduction_pct: float) -> List[str]:
        """Get recommendations for property allocation"""
        recommendations = []
        
        # Based on building type
        building_type = prop.get("building_type", "OFFICE")
        if building_type == "OFFICE":
            recommendations.append("Consider LED lighting upgrade")
            if reduction_pct > 20:
                recommendations.append("Evaluate HVAC optimization")
        elif building_type == "RETAIL":
            recommendations.append("Optimize refrigeration systems")
        elif building_type == "INDUSTRIAL":
            recommendations.append("Assess process efficiency improvements")
        
        # Based on reduction level
        if reduction_pct > 40:
            recommendations.append("Solar PV installation recommended")
            recommendations.append("Consider heat pump conversion")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _calculate_allocation_metrics(self, allocations: List[PropertyAllocation],
                                     properties: List[Dict]) -> Dict[str, float]:
        """Calculate allocation quality metrics"""
        if not allocations:
            return {}
        
        # Fairness (Gini coefficient - lower is more equal)
        weights = [a.allocation_weight for a in allocations]
        gini = self._calculate_gini_coefficient(weights)
        fairness_index = 1 - gini  # Convert to fairness score
        
        # Efficiency (correlation between allocation and potential)
        efficiency = self._calculate_efficiency_score(allocations, properties)
        
        # Feasibility (average feasibility score)
        feasibility = sum(a.feasibility_score for a in allocations) / len(allocations)
        
        # Coverage (how much of total is allocated)
        total_baseline = sum(p["baseline_emission"] for p in properties)
        total_reduction = sum(a.reduction_amount for a in allocations)
        coverage = (total_reduction / total_baseline) * 100 if total_baseline > 0 else 0
        
        return {
            "fairness_index": round(fairness_index, 3),
            "efficiency_score": round(efficiency, 1),
            "feasibility_score": round(feasibility, 1),
            "coverage": round(coverage, 1)
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for fairness measurement"""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        cumsum = 0
        for i, v in enumerate(sorted_values):
            cumsum += (i + 1) * v
        
        total = sum(sorted_values)
        if total == 0:
            return 0.0
        
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        return max(0, min(1, gini))
    
    def _calculate_efficiency_score(self, allocations: List[PropertyAllocation],
                                   properties: List[Dict]) -> float:
        """Calculate efficiency score"""
        # Higher efficiency = allocating more to properties that can reduce more easily
        
        # Map properties by ID
        prop_map = {p["property_id"]: p for p in properties}
        
        efficiency_sum = 0
        for alloc in allocations:
            prop = prop_map.get(alloc.property_id, {})
            potential = str(prop.get("retrofit_potential", "MEDIUM")).upper()
            potential_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(potential, 2)
            
            # Efficiency: weight * potential alignment
            efficiency_sum += alloc.allocation_weight * potential_score
        
        # Normalize to 0-100
        max_possible = 3 * sum(a.allocation_weight for a in allocations)
        efficiency = (efficiency_sum / max_possible) * 100 if max_possible > 0 else 50
        
        return efficiency
    
    # =========================================================================
    # ALLOCATION ADJUSTMENT
    # =========================================================================
    
    @measure_time
    def adjust_allocation(self, allocation_id: str,
                         adjustments: List[AllocationAdjustment]) -> ServiceResult[AllocationResult]:
        """
        Adjust existing allocations manually.
        
        Args:
            allocation_id: Allocation to adjust
            adjustments: List of adjustments
            
        Returns:
            ServiceResult containing updated allocation
        """
        if not adjustments:
            return ServiceResult.validation_error(["No adjustments provided"])
        
        return self._execute(self._adjust_allocation_impl, allocation_id, adjustments)
    
    def _adjust_allocation_impl(self, allocation_id: str,
                               adjustments: List[AllocationAdjustment]) -> ServiceResult[AllocationResult]:
        """Implementation of adjust_allocation"""
        # In production, this would fetch and update from database
        # For now, create a new adjusted allocation
        
        adjustment_map = {a.property_id: a for a in adjustments}
        adjusted_allocations = []
        
        # Apply adjustments
        for adj in adjustments:
            allocation = PropertyAllocation(
                property_id=adj.property_id,
                property_name=f"Property {adj.property_id}",
                year=2030,
                baseline_emission=1000.0,  # Would fetch from original
                allocated_target=adj.new_target,
                reduction_amount=1000.0 - adj.new_target,
                reduction_percentage=((1000.0 - adj.new_target) / 1000.0) * 100,
                allocation_weight=0.0,  # Recalculate
                feasibility_score=85.0,
                estimated_cost=(1000.0 - adj.new_target) * 150
            )
            adjusted_allocations.append(allocation)
        
        result = AllocationResult(
            allocation_id=f"{allocation_id}-ADJ-{datetime.utcnow().strftime('%H%M%S')}",
            scenario_id="",
            property_allocations=adjusted_allocations,
            allocation_metrics={"adjusted": True},
            total_allocated=sum(a.reduction_amount for a in adjusted_allocations),
            total_target=0,
            coverage_percentage=0
        )
        
        return ServiceResult.success(result)
    
    @measure_time
    def rebalance_allocation(self, allocation_id: str,
                            locked_properties: List[str] = None) -> ServiceResult[AllocationResult]:
        """
        Rebalance allocation while keeping some properties locked.
        
        Args:
            allocation_id: Allocation to rebalance
            locked_properties: Properties not to change
            
        Returns:
            ServiceResult containing rebalanced allocation
        """
        locked = locked_properties or []
        
        return self._execute(
            self._rebalance_impl,
            allocation_id, locked
        )
    
    def _rebalance_impl(self, allocation_id: str,
                       locked_properties: List[str]) -> ServiceResult[AllocationResult]:
        """Implementation of rebalance_allocation"""
        # In production, fetch original allocation and rebalance unlocked properties
        
        return ServiceResult.success(AllocationResult(
            allocation_id=f"{allocation_id}-REBAL",
            scenario_id="",
            property_allocations=[],
            allocation_metrics={"rebalanced": True, "locked_count": len(locked_properties)},
            total_allocated=0,
            total_target=0,
            coverage_percentage=0
        ))
    
    # =========================================================================
    # ALLOCATION VALIDATION
    # =========================================================================
    
    @measure_time
    def validate_allocation(self, allocation: AllocationResult) -> ServiceResult[Dict[str, Any]]:
        """
        Validate allocation against constraints and best practices.
        
        Args:
            allocation: Allocation to validate
            
        Returns:
            ServiceResult containing validation results
        """
        errors = []
        warnings = []
        
        for alloc in allocation.property_allocations:
            # Check reduction percentage
            if alloc.reduction_percentage > 80:
                errors.append(
                    f"Property {alloc.property_id}: Reduction of {alloc.reduction_percentage}% "
                    f"exceeds maximum feasible (80%)"
                )
            elif alloc.reduction_percentage > 60:
                warnings.append(
                    f"Property {alloc.property_id}: Reduction of {alloc.reduction_percentage}% "
                    f"is aggressive and may be challenging"
                )
            
            # Check feasibility
            if alloc.feasibility_score < 50:
                warnings.append(
                    f"Property {alloc.property_id}: Low feasibility score ({alloc.feasibility_score})"
                )
        
        # Check overall coverage
        if allocation.coverage_percentage < 90:
            warnings.append(
                f"Allocation only covers {allocation.coverage_percentage}% of target"
            )
        
        return ServiceResult.success({
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": allocation.allocation_metrics
        })
    
    # =========================================================================
    # ALLOCATION PERSISTENCE
    # =========================================================================
    
    @measure_time
    @transaction
    def save_allocation(self, allocation: AllocationResult,
                       saved_by: str = None) -> ServiceResult[str]:
        """
        Save allocation to database.
        
        Args:
            allocation: Allocation to save
            saved_by: User saving the allocation
            
        Returns:
            ServiceResult containing saved allocation ID
        """
        return self._execute(self._save_allocation_impl, allocation, saved_by)
    
    def _save_allocation_impl(self, allocation: AllocationResult,
                             saved_by: str) -> ServiceResult[str]:
        """Implementation of save_allocation"""
        # In production, save to database
        # For now, return success
        
        self._invalidate_cache(f"allocation:{allocation.scenario_id}")
        
        return ServiceResult.success(
            data=allocation.allocation_id,
            message=f"Allocation {allocation.allocation_id} saved successfully"
        )
    
    @measure_time
    @cached(ttl_seconds=300, key_prefix="allocation")
    def get_allocation(self, allocation_id: str) -> ServiceResult[AllocationResult]:
        """
        Get allocation by ID.
        
        Args:
            allocation_id: Allocation ID
            
        Returns:
            ServiceResult containing allocation
        """
        if not allocation_id:
            return ServiceResult.validation_error(["allocation_id is required"])
        
        # In production, fetch from database
        return ServiceResult.not_found(f"Allocation {allocation_id} not found")
    
    @measure_time
    def get_allocation_visualization(self, allocation_id: str, 
                                     property_id: Optional[str] = None,
                                     year: Optional[int] = None) -> ServiceResult[Dict[str, Any]]:
        """
        Get visualization data for an allocation.
        
        Args:
            allocation_id: Allocation ID to visualize
            property_id: Optional property filter
            year: Optional year filter
            
        Returns:
            ServiceResult containing visualization data
        """
        if not allocation_id:
            return ServiceResult.validation_error(["allocation_id is required"])
        
        return self._execute(
            self._get_visualization_impl,
            allocation_id, property_id, year
        )
    
    def _get_visualization_impl(self, allocation_id: str,
                                property_id: Optional[str],
                                year: Optional[int]) -> ServiceResult[Dict[str, Any]]:
        """Implementation of get_allocation_visualization"""
        
        # In production, fetch from database
        # For now, return mock visualization data
        
        # Mock allocation data
        all_allocations = [
            {
                "property_id": "550e8400-e29b-41d4-a716-446655440001",
                "property_name": "Brisbane Plaza",
                "year": 2030,
                "baseline_emission": 1500.0,
                "allocated_target": 825.0,
                "reduction_amount": 675.0,
                "reduction_percentage": 45.0,
                "estimated_cost": 85000.0
            },
            {
                "property_id": "550e8400-e29b-41d4-a716-446655440002",
                "property_name": "Melbourne Tower",
                "year": 2030,
                "baseline_emission": 1800.0,
                "allocated_target": 990.0,
                "reduction_amount": 810.0,
                "reduction_percentage": 45.0,
                "estimated_cost": 102000.0
            },
            {
                "property_id": "550e8400-e29b-41d4-a716-446655440003",
                "property_name": "Sydney Centre",
                "year": 2030,
                "baseline_emission": 1400.0,
                "allocated_target": 770.0,
                "reduction_amount": 630.0,
                "reduction_percentage": 45.0,
                "estimated_cost": 79500.0
            }
        ]
        
        # Apply filters
        filtered_allocations = all_allocations
        if property_id:
            filtered_allocations = [a for a in filtered_allocations if a["property_id"] == property_id]
        if year:
            filtered_allocations = [a for a in filtered_allocations if a["year"] == year]
        
        # Build visualization data
        visualization_data = {
            "allocation_id": allocation_id,
            "property_breakdown": filtered_allocations,
            
            "summary_metrics": {
                "total_properties": len(filtered_allocations),
                "total_baseline": sum(a["baseline_emission"] for a in filtered_allocations),
                "total_reduction": sum(a["reduction_amount"] for a in filtered_allocations),
                "total_cost": sum(a["estimated_cost"] for a in filtered_allocations),
                "avg_reduction_percentage": sum(a["reduction_percentage"] for a in filtered_allocations) / len(filtered_allocations) if filtered_allocations else 0,
                "emission_unit": "kg-CO2e",
                "cost_unit": "USD"
            },
            
            "timeline_data": [
                {
                    "year": 2025,
                    "cumulative_reduction": 0.0,
                    "cumulative_cost": 0.0
                },
                {
                    "year": 2030,
                    "cumulative_reduction": sum(a["reduction_amount"] for a in filtered_allocations),
                    "cumulative_cost": sum(a["estimated_cost"] for a in filtered_allocations)
                }
            ],
            
            "fairness_metrics": {
                "gini_coefficient": 0.15,
                "max_min_ratio": 1.2,
                "variance": 0.08
            },
            
            "cost_distribution": [
                {
                    "property_id": a["property_id"],
                    "property_name": a["property_name"],
                    "cost_share": (a["estimated_cost"] / sum(x["estimated_cost"] for x in filtered_allocations)) * 100,
                    "cost_unit": "USD"
                }
                for a in filtered_allocations
            ],
            
            "filters_applied": {
                "property_id": property_id,
                "year": year
            }
        }
        
        return ServiceResult.success(
            data=visualization_data,
            message="Visualization data retrieved successfully"
        )
    
    @measure_time
    def get_property_allocation(self, property_id: str,
                               scenario_id: Optional[str] = None) -> ServiceResult[Dict[str, Any]]:
        """
        Get allocation data for a specific property.
        
        Args:
            property_id: Property ID
            scenario_id: Optional scenario filter
            
        Returns:
            ServiceResult containing property allocation data
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        return self._execute(
            self._get_property_allocation_impl,
            property_id, scenario_id
        )
    
    def _get_property_allocation_impl(self, property_id: str,
                                      scenario_id: Optional[str]) -> ServiceResult[Dict[str, Any]]:
        """Implementation of get_property_allocation"""
        
        # In production, fetch from database
        # For now, return mock data
        
        property_data = {
            "property_id": property_id,
            "property_name": "Sample Property",
            "allocations": [
                {
                    "allocation_id": "ALLOC-20251204-001",
                    "scenario_id": scenario_id or "SCEN-12345678",
                    "year": 2030,
                    "baseline_emission": 1500.0,
                    "allocated_target": 825.0,
                    "reduction_amount": 675.0,
                    "reduction_percentage": 45.0,
                    "allocation_weight": 0.33,
                    "feasibility_score": 85.0,
                    "estimated_cost": 85000.0,
                    "emission_unit": "kg-CO2e",
                    "cost_unit": "USD"
                },
                {
                    "allocation_id": "ALLOC-20251204-001",
                    "scenario_id": scenario_id or "SCEN-12345678",
                    "year": 2050,
                    "baseline_emission": 1500.0,
                    "allocated_target": 150.0,
                    "reduction_amount": 1350.0,
                    "reduction_percentage": 90.0,
                    "allocation_weight": 0.33,
                    "feasibility_score": 75.0,
                    "estimated_cost": 195000.0,
                    "emission_unit": "kg-CO2e",
                    "cost_unit": "USD"
                }
            ],
            "recommended_actions": [
                "Install solar panels (capacity: 50kW)",
                "Upgrade HVAC system to high-efficiency model",
                "LED lighting retrofit for common areas"
            ],
            "implementation_timeline": [
                {
                    "phase": 1,
                    "year_range": "2025-2027",
                    "actions": ["Solar panel installation", "LED retrofit"],
                    "expected_reduction": 350.0,
                    "estimated_cost": 45000.0
                },
                {
                    "phase": 2,
                    "year_range": "2028-2030",
                    "actions": ["HVAC upgrade"],
                    "expected_reduction": 325.0,
                    "estimated_cost": 40000.0
                }
            ]
        }
        
        return ServiceResult.success(
            data=property_data,
            message=f"Property allocation data retrieved for {property_id}"
        )
    
    @measure_time
    @transaction
    def register_allocation(self, allocation_id: str,
                           approval_info: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """
        Register allocation as official property targets.
        
        Args:
            allocation_id: Allocation to register
            approval_info: Approval metadata (approved_by, approval_date, notes)
            
        Returns:
            ServiceResult containing registration confirmation
        """
        if not allocation_id:
            return ServiceResult.validation_error(["allocation_id is required"])
        
        return self._execute(
            self._register_allocation_impl,
            allocation_id, approval_info
        )
    
    def _register_allocation_impl(self, allocation_id: str,
                                  approval_info: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """Implementation of register_allocation"""
        
        # In production:
        # 1. Validate allocation exists
        # 2. Check if already registered
        # 3. Save to database
        # 4. Update property targets
        # 5. Trigger notifications
        
        registration_id = f"REG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        registration_data = {
            "registration_id": registration_id,
            "allocation_id": allocation_id,
            "status": "registered",
            "approved_by": approval_info.get("approved_by"),
            "approval_date": approval_info.get("approval_date"),
            "approval_status": approval_info.get("approval_status", "approved"),
            "notes": approval_info.get("notes"),
            "registered_at": datetime.utcnow().isoformat(),
            "next_steps": [
                "Review long-term implementation plan",
                "Set up monitoring dashboards",
                "Schedule quarterly review meetings"
            ]
        }
        
        # Invalidate cache
        self._invalidate_cache(f"allocation:{allocation_id}")
        
        return ServiceResult.success(
            data=registration_data,
            message=f"Allocation {allocation_id} registered successfully as {registration_id}"
        )
    
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    @measure_time
    def analyze_allocation_options(self, request: AllocationRequest) -> ServiceResult[Dict[str, Any]]:
        """
        Analyze different allocation methods for the same request.
        
        Args:
            request: Allocation request
            
        Returns:
            ServiceResult containing comparison of methods
        """
        methods = ["PROPORTIONAL", "INTENSITY_WEIGHTED", "RETROFIT_POTENTIAL", "AI_OPTIMIZED"]
        
        results = {}
        for method in methods:
            method_request = AllocationRequest(
                scenario_id=request.scenario_id,
                property_ids=request.property_ids,
                total_reduction_target=request.total_reduction_target,
                target_years=request.target_years,
                allocation_method=method,
                constraints=request.constraints
            )
            
            result = self.allocate_targets(method_request)
            if result.is_success:
                results[method] = {
                    "metrics": result.data.allocation_metrics,
                    "total_cost": sum(a.estimated_cost for a in result.data.property_allocations)
                }
        
        # Recommend best method
        best_method = max(
            results.keys(),
            key=lambda m: results[m]["metrics"].get("efficiency_score", 0) * 0.4 +
                         results[m]["metrics"].get("fairness_index", 0) * 100 * 0.3 +
                         results[m]["metrics"].get("feasibility_score", 0) * 0.3
        )
        
        return ServiceResult.success({
            "method_comparison": results,
            "recommended_method": best_method,
            "recommendation_reason": f"{best_method} provides the best balance of efficiency, fairness, and feasibility"
        })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AllocationService',
    'AllocationRequest',
    'AllocationResult',
    'PropertyAllocation',
    'AllocationAdjustment',
    'get_allocation_visualization',
    'get_property_allocation', 
    'register_allocation'
]
