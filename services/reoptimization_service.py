# services/reoptimization_service.py - Annual Reoptimization Service
"""
Service for annual plan reoptimization based on actual performance.

Features:
- Actual vs planned performance analysis
- Deviation detection and analysis
- Adjustment recommendations
- Multi-pattern reoptimization strategies
- Integration with EEL server for actual data
- Registration and tracking
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

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
    from models.reoptimization import (
        ReoptimizationRequest,
        ReoptimizationResult,
        ReoptimizationPattern,
        AdjustedAnnualPlan,
        AdjustedAction,
        ChangesFromOriginal,
        PerformanceComparison,
        PerformanceMetrics,
        CostMetrics,
        ImpactAnalysis,
        HistoricalPerformance,
        ReoptimizationVisualizationData,
        PerformanceTrends,
        MonthlyPerformanceData,
        PlanComparison,
        KeyMetrics,
        validate_reoptimization_request
    )
    from models.base import generate_uuid
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Reoptimization models not available - using mock mode")

logger = logging.getLogger(__name__)


# =============================================================================
# REOPTIMIZATION SERVICE
# =============================================================================

class ReoptimizationService(BaseService):
    """
    Service for annual plan reoptimization.
    
    Provides:
    - Actual performance data retrieval
    - Deviation analysis
    - Adjustment recommendations
    - Multiple reoptimization strategies
    - Impact analysis
    
    Usage:
        service = ReoptimizationService(db_manager)
        request = ReoptimizationRequest(
            plan_id="PLAN-001",
            target_year=2026,
            start_date="2025-01-01",
            end_date="2025-12-31"
        )
        result = service.calculate_reoptimization(request)
    """
    
    def __init__(self, db_manager=None, eel_client=None):
        """
        Initialize reoptimization service.
        
        Args:
            db_manager: Database manager instance
            eel_client: Client for EEL server (actual performance data)
        """
        super().__init__(db_manager)
        self._eel_client = eel_client
        self._calculation_cache = {}
    
    def _do_initialize(self) -> None:
        """Initialize service resources"""
        self._logger.info("Reoptimization service initialized")
    
    # =========================================================================
    # ACTUAL PERFORMANCE DATA
    # =========================================================================
    
    def get_actual_performance(self, plan_id: str, start_date: str, 
                              end_date: str) -> Dict[str, Any]:
        """
        Retrieve actual performance data from EEL server.
        
        Args:
            plan_id: Plan ID
            start_date: Start date (ISO 8601)
            end_date: End date (ISO 8601)
            
        Returns:
            Dictionary with actual performance data
        """
        # In production, fetch from EEL server
        # For now, return mock data
        
        return {
            "plan_id": plan_id,
            "period": f"{start_date} to {end_date}",
            "actual_emission": 4322.35,
            "planned_emission": 4000.0,
            "variance": 322.35,
            "variance_percentage": 8.06,
            "actual_cost": 108500.0,
            "planned_cost": 112000.0,
            "cost_variance": -3500.0,
            "cost_variance_percentage": -3.1,
            "emission_unit": "kg-CO2e",
            "cost_unit": "USD",
            "monthly_data": [
                {
                    "month": "2025-01",
                    "actual_emission": 380.2,
                    "target_emission": 350.0,
                    "variance": 30.2
                },
                {
                    "month": "2025-02",
                    "actual_emission": 365.8,
                    "target_emission": 350.0,
                    "variance": 15.8
                }
                # ... more months
            ]
        }
    
    # =========================================================================
    # REOPTIMIZATION CALCULATION
    # =========================================================================
    
    @measure_time
    def calculate_reoptimization(self, request: ReoptimizationRequest) -> ServiceResult[ReoptimizationResult]:
        """
        Calculate annual reoptimization strategies.
        
        Args:
            request: Reoptimization request with parameters
            
        Returns:
            ServiceResult containing reoptimization patterns
        """
        # Validate request
        if MODELS_AVAILABLE:
            validation_errors = validate_reoptimization_request(request)
            if validation_errors:
                return ServiceResult.validation_error(validation_errors)
        
        return self._execute(self._calculate_reoptimization_impl, request)
    
    def _calculate_reoptimization_impl(self, request: ReoptimizationRequest) -> ServiceResult[ReoptimizationResult]:
        """Implementation of calculate_reoptimization"""
        
        # Get actual performance data
        actual_data = self.get_actual_performance(
            request.plan_id,
            request.start_date,
            request.end_date
        )
        
        # Analyze performance
        performance_comparison = self._analyze_performance(actual_data)
        
        # Generate reoptimization patterns based on performance
        patterns = []
        
        if actual_data["variance_percentage"] > 5:
            # Behind target - accelerated pattern
            accelerated = self._generate_accelerated_pattern(request, actual_data, performance_comparison)
            patterns.append(accelerated)
        elif actual_data["variance_percentage"] < -5:
            # Ahead of target - maintain momentum pattern
            momentum = self._generate_momentum_pattern(request, actual_data, performance_comparison)
            patterns.append(momentum)
        else:
            # On track - steady pattern
            steady = self._generate_steady_pattern(request, actual_data, performance_comparison)
            patterns.append(steady)
        
        # Always generate alternative pattern
        alternative = self._generate_alternative_pattern(request, actual_data, performance_comparison)
        patterns.append(alternative)
        
        # Build result
        result = ReoptimizationResult(
            reoptimization_patterns=patterns,
            calculation_metadata={
                "calculation_timestamp": datetime.utcnow().isoformat(),
                "plan_id": request.plan_id,
                "target_year": request.target_year,
                "analysis_period": f"{request.start_date} to {request.end_date}",
                "patterns_generated": len(patterns),
                "actual_performance": actual_data
            }
        )
        
        # Cache result
        self._calculation_cache[result.calculation_id] = result
        
        return ServiceResult.success(
            data=result,
            message=f"Generated {len(patterns)} reoptimization patterns successfully"
        )
    
    def _analyze_performance(self, actual_data: Dict[str, Any]) -> PerformanceComparison:
        """Analyze actual vs planned performance"""
        
        performance_metrics = PerformanceMetrics(
            planned_emission=actual_data["planned_emission"],
            actual_emission=actual_data["actual_emission"],
            variance=actual_data["variance"],
            variance_percentage=actual_data["variance_percentage"]
        )
        
        cost_metrics = CostMetrics(
            planned_cost=actual_data["planned_cost"],
            actual_cost=actual_data["actual_cost"],
            variance=actual_data["cost_variance"],
            variance_percentage=actual_data["cost_variance_percentage"]
        )
        
        # Determine status
        if actual_data["variance_percentage"] > 10:
            status = "behind"
        elif actual_data["variance_percentage"] < -5:
            status = "exceeding"
        else:
            status = "on_track"
        
        return PerformanceComparison(
            period=actual_data["period"],
            performance_metrics=performance_metrics,
            cost_metrics=cost_metrics,
            status=status
        )
    
    def _generate_accelerated_pattern(self, request: ReoptimizationRequest, 
                                     actual_data: Dict[str, Any],
                                     performance: PerformanceComparison) -> ReoptimizationPattern:
        """Generate accelerated implementation pattern (for when behind target)"""
        
        adjusted_plans = []
        
        # Add aggressive actions for target year
        recommended_actions = [
            AdjustedAction(
                action_type="solar_panel_installation",
                target_properties=["property-1"],
                expected_reduction=250.0,
                investment_required=65000.0,
                priority="high",
                rationale="Accelerate renewable energy to catch up",
                change_type="added"
            ),
            AdjustedAction(
                action_type="hvac_upgrade",
                target_properties=["property-2"],
                expected_reduction=180.0,
                investment_required=45000.0,
                priority="high",
                rationale="High-impact efficiency improvement",
                change_type="added"
            )
        ]
        
        changes = ChangesFromOriginal(
            actions_added=2,
            actions_removed=0,
            actions_modified=0,
            budget_change=110000.0,
            expected_additional_reduction=430.0
        )
        
        adjusted_plan = AdjustedAnnualPlan(
            year=request.target_year,
            recommended_actions=recommended_actions,
            changes_from_original=changes
        )
        adjusted_plans.append(adjusted_plan)
        
        # Impact analysis
        impact = ImpactAnalysis(
            long_term_target_impact="Positive - Gets back on track to 2030 target",
            financial_impact=f"Increased annual spend by ${changes.budget_change:,.0f}",
            risk_impact="Medium - Requires additional budget approval"
        )
        
        return ReoptimizationPattern(
            pattern_name="Accelerated Catch-up",
            rationale=f"Behind target by {actual_data['variance_percentage']:.1f}%, requiring accelerated action",
            adjusted_annual_plan=adjusted_plans,
            impact_analysis=impact,
            performance_comparison=performance,
            plan_id=request.plan_id
        )
    
    def _generate_momentum_pattern(self, request: ReoptimizationRequest,
                                   actual_data: Dict[str, Any],
                                   performance: PerformanceComparison) -> ReoptimizationPattern:
        """Generate momentum maintenance pattern (for when ahead of target)"""
        
        adjusted_plans = []
        
        # Add stretch goals
        recommended_actions = [
            AdjustedAction(
                action_type="led_lighting_upgrade",
                target_properties=["property-3"],
                expected_reduction=120.0,
                investment_required=22000.0,
                priority="medium",
                rationale="Capitalize on momentum with quick win",
                change_type="added"
            )
        ]
        
        changes = ChangesFromOriginal(
            actions_added=1,
            actions_removed=0,
            actions_modified=0,
            budget_change=22000.0,
            expected_additional_reduction=120.0
        )
        
        adjusted_plan = AdjustedAnnualPlan(
            year=request.target_year,
            recommended_actions=recommended_actions,
            changes_from_original=changes
        )
        adjusted_plans.append(adjusted_plan)
        
        # Impact analysis
        impact = ImpactAnalysis(
            long_term_target_impact="Very Positive - Enables reaching 2030 target ahead of schedule",
            financial_impact=f"Modest increase of ${changes.budget_change:,.0f} with strong ROI",
            risk_impact="Low - Building on proven success"
        )
        
        return ReoptimizationPattern(
            pattern_name="Maintain Momentum",
            rationale=f"Ahead of target by {abs(actual_data['variance_percentage']):.1f}%, opportunity to accelerate",
            adjusted_annual_plan=adjusted_plans,
            impact_analysis=impact,
            performance_comparison=performance,
            plan_id=request.plan_id
        )
    
    def _generate_steady_pattern(self, request: ReoptimizationRequest,
                                 actual_data: Dict[str, Any],
                                 performance: PerformanceComparison) -> ReoptimizationPattern:
        """Generate steady progress pattern (for when on track)"""
        
        adjusted_plans = []
        
        # Minor optimizations
        recommended_actions = [
            AdjustedAction(
                action_type="smart_controls",
                target_properties=["property-1", "property-2"],
                expected_reduction=95.0,
                investment_required=18000.0,
                priority="medium",
                rationale="Fine-tune performance with smart controls",
                change_type="added"
            )
        ]
        
        changes = ChangesFromOriginal(
            actions_added=1,
            actions_removed=0,
            actions_modified=0,
            budget_change=18000.0,
            expected_additional_reduction=95.0
        )
        
        adjusted_plan = AdjustedAnnualPlan(
            year=request.target_year,
            recommended_actions=recommended_actions,
            changes_from_original=changes
        )
        adjusted_plans.append(adjusted_plan)
        
        # Impact analysis
        impact = ImpactAnalysis(
            long_term_target_impact="Stable - Maintains trajectory to 2030 target",
            financial_impact=f"Minor increase of ${changes.budget_change:,.0f}",
            risk_impact="Low - Stay the course with minor enhancements"
        )
        
        return ReoptimizationPattern(
            pattern_name="Steady Progress",
            rationale="On track with current plan, minor optimizations recommended",
            adjusted_annual_plan=adjusted_plans,
            impact_analysis=impact,
            performance_comparison=performance,
            plan_id=request.plan_id
        )
    
    def _generate_alternative_pattern(self, request: ReoptimizationRequest,
                                      actual_data: Dict[str, Any],
                                      performance: PerformanceComparison) -> ReoptimizationPattern:
        """Generate alternative reoptimization pattern"""
        
        adjusted_plans = []
        
        # Focus on different strategy
        recommended_actions = [
            AdjustedAction(
                action_type="behavioral_change_program",
                target_properties=["property-1", "property-2", "property-3"],
                expected_reduction=150.0,
                investment_required=8000.0,
                priority="medium",
                rationale="Complement technical measures with behavior change",
                change_type="added"
            ),
            AdjustedAction(
                action_type="building_insulation",
                target_properties=["property-2"],
                expected_reduction=140.0,
                investment_required=32000.0,
                priority="medium",
                rationale="Address building envelope efficiency",
                change_type="added"
            )
        ]
        
        changes = ChangesFromOriginal(
            actions_added=2,
            actions_removed=0,
            actions_modified=0,
            budget_change=40000.0,
            expected_additional_reduction=290.0
        )
        
        adjusted_plan = AdjustedAnnualPlan(
            year=request.target_year,
            recommended_actions=recommended_actions,
            changes_from_original=changes
        )
        adjusted_plans.append(adjusted_plan)
        
        # Impact analysis
        impact = ImpactAnalysis(
            long_term_target_impact="Positive - Diversifies approach for resilience",
            financial_impact=f"Moderate increase of ${changes.budget_change:,.0f}",
            risk_impact="Low - Lower-cost, proven approaches"
        )
        
        return ReoptimizationPattern(
            pattern_name="Alternative Approach",
            rationale="Diversified strategy combining technical and behavioral measures",
            adjusted_annual_plan=adjusted_plans,
            impact_analysis=impact,
            performance_comparison=performance,
            plan_id=request.plan_id
        )
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    @measure_time
    def get_reoptimization_visualization(self, property_id: str, year: int,
                                        pattern_index: Optional[int] = None) -> ServiceResult[ReoptimizationVisualizationData]:
        """
        Get visualization data for reoptimization.
        
        Args:
            property_id: Property ID
            year: Target year
            pattern_index: Optional pattern index
            
        Returns:
            ServiceResult containing visualization data
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        return self._execute(self._get_reopt_visualization_impl, property_id, year, pattern_index)
    
    def _get_reopt_visualization_impl(self, property_id: str, year: int,
                                      pattern_index: Optional[int]) -> ServiceResult[ReoptimizationVisualizationData]:
        """Implementation of get_reoptimization_visualization"""
        
        # Generate sample visualization data
        monthly_data = []
        for month in range(1, 13):
            monthly_data.append(MonthlyPerformanceData(
                month=f"{year-1}-{month:02d}",
                actual_emission=390.2 - month * 2.5,
                target_emission=400.0 - month * 3.0,
                variance=390.2 - month * 2.5 - (400.0 - month * 3.0)
            ))
        
        performance_trends = PerformanceTrends(
            monthly_data=monthly_data,
            overall_trend="improving",
            trend_analysis="Consistent progress throughout the year with minor variance"
        )
        
        comparison = PlanComparison(
            original_total_reduction=350.0,
            adjusted_total_reduction=470.0,
            difference=120.0,
            original_total_cost=95000.0,
            adjusted_total_cost=110000.0,
            cost_difference=15000.0
        )
        
        key_metrics = KeyMetrics(
            target_achievement_rate=107.9,
            cost_efficiency_improvement=3.1,
            recommended_adjustments=3,
            implementation_progress=78.5
        )
        
        viz_data = ReoptimizationVisualizationData(
            property_id=property_id,
            year=year,
            performance_trends=performance_trends,
            comparison_charts={"original_vs_adjusted": comparison},
            key_metrics=key_metrics
        )
        
        return ServiceResult.success(
            data=viz_data,
            message="Reoptimization visualization data retrieved successfully"
        )
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    @measure_time
    @transaction
    def register_reoptimization(self, reoptimization_pattern_id: str, plan_id: str,
                                approval_info: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """
        Register a reoptimization pattern.
        
        Args:
            reoptimization_pattern_id: Pattern to register
            plan_id: Associated plan ID
            approval_info: Approval metadata
            
        Returns:
            ServiceResult containing registration confirmation
        """
        if not reoptimization_pattern_id:
            return ServiceResult.validation_error(["reoptimization_pattern_id is required"])
        if not plan_id:
            return ServiceResult.validation_error(["plan_id is required"])
        
        return self._execute(
            self._register_reoptimization_impl,
            reoptimization_pattern_id, plan_id, approval_info
        )
    
    def _register_reoptimization_impl(self, reoptimization_pattern_id: str, plan_id: str,
                                      approval_info: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """Implementation of register_reoptimization"""
        
        # In production:
        # 1. Validate pattern exists
        # 2. Update plan with adjustments
        # 3. Create reoptimization record
        # 4. Set status to active
        
        if MODELS_AVAILABLE:
            reoptimization_id = generate_uuid()
        else:
            reoptimization_id = f"REOPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        changes_applied = {
            "actions_added": 2,
            "actions_removed": 0,
            "budget_adjustment": 15000.0,
            "cost_unit": "USD"
        }
        
        registration_data = {
            "reoptimization_id": reoptimization_id,
            "reoptimization_pattern_id": reoptimization_pattern_id,
            "updated_plan_id": plan_id,
            "registration_timestamp": datetime.utcnow().isoformat(),
            "status": "active",
            "approved_by": approval_info.get("approved_by"),
            "approval_date": approval_info.get("approval_date"),
            "comments": approval_info.get("comments"),
            "changes_applied": changes_applied
        }
        
        # Invalidate cache
        self._invalidate_cache(f"reopt:{reoptimization_pattern_id}")
        self._invalidate_cache(f"plan:{plan_id}")
        
        return ServiceResult.success(
            data=registration_data,
            message=f"Reoptimization registered successfully as {reoptimization_id}"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ReoptimizationService'
]
