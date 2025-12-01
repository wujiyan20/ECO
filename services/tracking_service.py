# services/tracking_service.py - Progress Tracking Service
"""
Service for tracking progress against emission reduction targets.

Features:
- Actual vs target comparison
- Deviation analysis
- Trend detection
- Alert generation
- Reoptimization recommendations
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
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
        MilestoneProgress,
        MilestoneAlert,
        OnTrackStatus,
        RiskLevel,
        PriorityLevel,
        UrgencyLevel,
        MilestoneRepository,
        EmissionRepository
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Models package not available - using mock mode")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class ProgressData:
    """Actual progress data point"""
    property_id: str
    year: int
    month: Optional[int] = None
    actual_emission: float = 0.0
    energy_consumption_kwh: float = 0.0
    data_source: str = "manual"
    verified: bool = False


@dataclass
class ProgressSummary:
    """Summary of progress status"""
    scenario_id: str
    year: int
    total_target: float
    total_actual: float
    variance: float
    variance_percentage: float
    on_track_status: str
    properties_on_track: int
    properties_at_risk: int
    properties_off_track: int


@dataclass
class DeviationAnalysis:
    """Detailed deviation analysis"""
    property_id: str
    year: int
    target: float
    actual: float
    deviation: float
    deviation_percentage: float
    root_causes: List[str]
    contributing_factors: Dict[str, float]
    recommended_actions: List[str]
    urgency: str


@dataclass
class ReoptimizationRecommendation:
    """Recommendation for plan adjustment"""
    recommendation_id: str
    property_id: str
    current_target: float
    recommended_target: float
    adjustment_type: str
    rationale: str
    estimated_impact: Dict[str, float]
    priority: str


# =============================================================================
# TRACKING SERVICE
# =============================================================================

class TrackingService(BaseService):
    """
    Service for tracking progress against emission reduction targets.
    
    Provides:
    - Progress monitoring
    - Deviation analysis
    - Trend detection
    - Alert management
    - Reoptimization recommendations
    
    Usage:
        service = TrackingService(db_manager)
        result = service.get_progress_summary("SCN-001", 2025)
    """
    
    def __init__(self, db_manager=None, milestone_service=None):
        """
        Initialize tracking service.
        
        Args:
            db_manager: Database manager instance
            milestone_service: Milestone service for target data
        """
        super().__init__(db_manager)
        self._milestone_service = milestone_service
        self._milestone_repo: Optional[MilestoneRepository] = None
        self._emission_repo: Optional[EmissionRepository] = None
    
    def _do_initialize(self) -> None:
        """Initialize repositories"""
        if self.db_manager and MODELS_AVAILABLE:
            self._milestone_repo = MilestoneRepository(self.db_manager)
            self._emission_repo = EmissionRepository(self.db_manager)
            self._logger.info("Tracking repositories initialized")
    
    # =========================================================================
    # PROGRESS MONITORING
    # =========================================================================
    
    @measure_time
    def record_actual_data(self, data: List[ProgressData]) -> ServiceResult[int]:
        """
        Record actual emission data for tracking.
        
        Args:
            data: List of progress data points
            
        Returns:
            ServiceResult containing count of recorded entries
        """
        if not data:
            return ServiceResult.validation_error(["No data provided"])
        
        return self._execute(self._record_actual_data_impl, data)
    
    def _record_actual_data_impl(self, data: List[ProgressData]) -> ServiceResult[int]:
        """Implementation of record_actual_data"""
        recorded = 0
        errors = []
        
        for entry in data:
            try:
                # Validate entry
                if entry.actual_emission < 0:
                    errors.append(f"{entry.property_id}: Emission cannot be negative")
                    continue
                
                # Save to database (would use repository in production)
                if self._emission_repo:
                    # Create baseline record with actual data
                    pass
                
                recorded += 1
            except Exception as e:
                errors.append(f"{entry.property_id}: {str(e)}")
        
        if errors:
            return ServiceResult(
                status=ServiceResultStatus.PARTIAL,
                data=recorded,
                errors=errors,
                message=f"Recorded {recorded}/{len(data)} entries"
            )
        
        return ServiceResult.success(recorded, f"Recorded {recorded} entries")
    
    @measure_time
    @cached(ttl_seconds=300, key_prefix="progress_summary")
    def get_progress_summary(self, scenario_id: str, year: int) -> ServiceResult[ProgressSummary]:
        """
        Get progress summary for a scenario and year.
        
        Args:
            scenario_id: Scenario ID
            year: Year to analyze
            
        Returns:
            ServiceResult containing progress summary
        """
        if not scenario_id:
            return ServiceResult.validation_error(["scenario_id is required"])
        
        return self._execute(self._get_progress_summary_impl, scenario_id, year)
    
    def _get_progress_summary_impl(self, scenario_id: str, year: int) -> ServiceResult[ProgressSummary]:
        """Implementation of get_progress_summary"""
        # Get scenario targets (would fetch from database)
        targets = self._get_targets_for_year(scenario_id, year)
        actuals = self._get_actuals_for_year(scenario_id, year)
        
        total_target = sum(t["target"] for t in targets)
        total_actual = sum(a["actual"] for a in actuals)
        variance = total_actual - total_target
        variance_pct = (variance / total_target) * 100 if total_target > 0 else 0
        
        # Determine overall status
        if variance_pct <= -5:
            status = "AHEAD"
        elif variance_pct <= 5:
            status = "ON_TRACK"
        elif variance_pct <= 15:
            status = "AT_RISK"
        elif variance_pct <= 25:
            status = "OFF_TRACK"
        else:
            status = "CRITICAL"
        
        # Count properties by status
        on_track = sum(1 for a in actuals if a.get("status") in ["AHEAD", "ON_TRACK"])
        at_risk = sum(1 for a in actuals if a.get("status") == "AT_RISK")
        off_track = sum(1 for a in actuals if a.get("status") in ["OFF_TRACK", "CRITICAL"])
        
        summary = ProgressSummary(
            scenario_id=scenario_id,
            year=year,
            total_target=round(total_target, 2),
            total_actual=round(total_actual, 2),
            variance=round(variance, 2),
            variance_percentage=round(variance_pct, 2),
            on_track_status=status,
            properties_on_track=on_track,
            properties_at_risk=at_risk,
            properties_off_track=off_track
        )
        
        return ServiceResult.success(summary)
    
    def _get_targets_for_year(self, scenario_id: str, year: int) -> List[Dict]:
        """Get targets for a specific year"""
        # Mock data - in production, fetch from database
        return [
            {"property_id": "PROP-001", "target": 800.0},
            {"property_id": "PROP-002", "target": 600.0},
            {"property_id": "PROP-003", "target": 500.0}
        ]
    
    def _get_actuals_for_year(self, scenario_id: str, year: int) -> List[Dict]:
        """Get actual emissions for a specific year"""
        # Mock data - in production, fetch from database
        return [
            {"property_id": "PROP-001", "actual": 750.0, "status": "AHEAD"},
            {"property_id": "PROP-002", "actual": 650.0, "status": "AT_RISK"},
            {"property_id": "PROP-003", "actual": 480.0, "status": "ON_TRACK"}
        ]
    
    @measure_time
    def get_property_progress(self, property_id: str, scenario_id: str,
                             start_year: int = None,
                             end_year: int = None) -> ServiceResult[List[Dict]]:
        """
        Get progress history for a specific property.
        
        Args:
            property_id: Property ID
            scenario_id: Scenario ID
            start_year: Start of date range
            end_year: End of date range
            
        Returns:
            ServiceResult containing progress history
        """
        return self._execute(
            self._get_property_progress_impl,
            property_id, scenario_id, start_year, end_year
        )
    
    def _get_property_progress_impl(self, property_id: str, scenario_id: str,
                                   start_year: int, end_year: int) -> ServiceResult[List[Dict]]:
        """Implementation of get_property_progress"""
        # Mock data - in production, fetch from database
        progress_data = []
        
        start = start_year or 2024
        end = end_year or datetime.now().year
        
        baseline = 1000.0
        target_reduction_annual = 0.05  # 5% per year
        
        for year in range(start, end + 1):
            years_from_start = year - start
            target = baseline * (1 - target_reduction_annual * years_from_start)
            
            # Simulate actual with some variance
            variance = (year - 2024) * 0.02 * (-1 if year % 2 == 0 else 1)
            actual = target * (1 + variance)
            
            deviation = actual - target
            deviation_pct = (deviation / target) * 100 if target > 0 else 0
            
            progress_data.append({
                "property_id": property_id,
                "year": year,
                "target": round(target, 2),
                "actual": round(actual, 2),
                "deviation": round(deviation, 2),
                "deviation_percentage": round(deviation_pct, 2),
                "status": self._determine_status(deviation_pct)
            })
        
        return ServiceResult.success(progress_data)
    
    def _determine_status(self, deviation_pct: float) -> str:
        """Determine status based on deviation percentage"""
        if deviation_pct <= -5:
            return "AHEAD"
        elif deviation_pct <= 5:
            return "ON_TRACK"
        elif deviation_pct <= 15:
            return "AT_RISK"
        elif deviation_pct <= 25:
            return "OFF_TRACK"
        else:
            return "CRITICAL"
    
    # =========================================================================
    # DEVIATION ANALYSIS
    # =========================================================================
    
    @measure_time
    def analyze_deviations(self, scenario_id: str, year: int,
                          threshold_pct: float = 5.0) -> ServiceResult[List[DeviationAnalysis]]:
        """
        Analyze properties with significant deviations.
        
        Args:
            scenario_id: Scenario ID
            year: Year to analyze
            threshold_pct: Minimum deviation to include
            
        Returns:
            ServiceResult containing deviation analyses
        """
        return self._execute(
            self._analyze_deviations_impl,
            scenario_id, year, threshold_pct
        )
    
    def _analyze_deviations_impl(self, scenario_id: str, year: int,
                                threshold_pct: float) -> ServiceResult[List[DeviationAnalysis]]:
        """Implementation of analyze_deviations"""
        # Get progress data
        targets = self._get_targets_for_year(scenario_id, year)
        actuals = self._get_actuals_for_year(scenario_id, year)
        
        # Match targets and actuals
        actual_map = {a["property_id"]: a["actual"] for a in actuals}
        
        analyses = []
        for target_data in targets:
            pid = target_data["property_id"]
            target = target_data["target"]
            actual = actual_map.get(pid, target)
            
            deviation = actual - target
            deviation_pct = (deviation / target) * 100 if target > 0 else 0
            
            # Only include if above threshold
            if abs(deviation_pct) < threshold_pct:
                continue
            
            # Identify root causes
            root_causes = self._identify_root_causes(pid, deviation_pct)
            
            # Contributing factors
            factors = self._analyze_contributing_factors(pid, year)
            
            # Recommendations
            actions = self._generate_deviation_actions(deviation_pct, factors)
            
            # Determine urgency
            urgency = self._determine_urgency(deviation_pct)
            
            analysis = DeviationAnalysis(
                property_id=pid,
                year=year,
                target=target,
                actual=actual,
                deviation=round(deviation, 2),
                deviation_percentage=round(deviation_pct, 2),
                root_causes=root_causes,
                contributing_factors=factors,
                recommended_actions=actions,
                urgency=urgency
            )
            analyses.append(analysis)
        
        return ServiceResult.success(analyses)
    
    def _identify_root_causes(self, property_id: str, deviation_pct: float) -> List[str]:
        """Identify root causes for deviation"""
        causes = []
        
        if deviation_pct > 0:
            # Underperforming
            causes.extend([
                "Implementation delays in planned initiatives",
                "Higher than expected occupancy/usage",
                "Equipment efficiency degradation"
            ])
        else:
            # Overperforming
            causes.extend([
                "Better than expected technology performance",
                "Lower occupancy due to work-from-home policies",
                "Favorable weather conditions"
            ])
        
        return causes[:3]
    
    def _analyze_contributing_factors(self, property_id: str, year: int) -> Dict[str, float]:
        """Analyze factors contributing to deviation"""
        # Mock analysis - in production, analyze actual data
        return {
            "energy_price_changes": 15.0,
            "occupancy_variation": 25.0,
            "weather_impact": 10.0,
            "equipment_performance": 30.0,
            "behavior_changes": 20.0
        }
    
    def _generate_deviation_actions(self, deviation_pct: float,
                                   factors: Dict[str, float]) -> List[str]:
        """Generate recommended actions based on deviation"""
        actions = []
        
        if deviation_pct > 10:
            actions.append("Accelerate planned efficiency improvements")
            actions.append("Review and adjust implementation timeline")
            
            # Based on factors
            if factors.get("equipment_performance", 0) > 20:
                actions.append("Conduct equipment performance audit")
            if factors.get("occupancy_variation", 0) > 20:
                actions.append("Implement occupancy-based controls")
        elif deviation_pct > 0:
            actions.append("Monitor closely and prepare contingency plans")
            actions.append("Identify quick-win opportunities")
        else:
            actions.append("Consider accelerating future targets")
            actions.append("Document success factors for replication")
        
        return actions[:4]
    
    def _determine_urgency(self, deviation_pct: float) -> str:
        """Determine urgency level based on deviation"""
        if deviation_pct > 25:
            return "CRITICAL"
        elif deviation_pct > 15:
            return "HIGH"
        elif deviation_pct > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    
    @measure_time
    def analyze_trends(self, scenario_id: str,
                      property_id: str = None) -> ServiceResult[Dict[str, Any]]:
        """
        Analyze emission trends over time.
        
        Args:
            scenario_id: Scenario ID
            property_id: Optional specific property
            
        Returns:
            ServiceResult containing trend analysis
        """
        return self._execute(self._analyze_trends_impl, scenario_id, property_id)
    
    def _analyze_trends_impl(self, scenario_id: str,
                            property_id: str) -> ServiceResult[Dict[str, Any]]:
        """Implementation of analyze_trends"""
        # Get historical data
        current_year = datetime.now().year
        
        if property_id:
            progress_result = self.get_property_progress(
                property_id, scenario_id,
                start_year=current_year - 3,
                end_year=current_year
            )
            if not progress_result.is_success:
                return progress_result
            data = progress_result.data
        else:
            # Aggregate all properties
            data = self._get_aggregate_trend_data(scenario_id, current_year)
        
        # Calculate trend metrics
        if len(data) >= 2:
            values = [d.get("actual", d.get("value", 0)) for d in data]
            trend_direction = self._calculate_trend_direction(values)
            avg_change = self._calculate_average_change(values)
            volatility = self._calculate_volatility(values)
        else:
            trend_direction = "STABLE"
            avg_change = 0.0
            volatility = 0.0
        
        # Forecast next period
        forecast = self._forecast_next_period(data)
        
        return ServiceResult.success({
            "scenario_id": scenario_id,
            "property_id": property_id,
            "data_points": len(data),
            "trend_direction": trend_direction,
            "average_annual_change_pct": round(avg_change, 2),
            "volatility": round(volatility, 2),
            "forecast": forecast,
            "confidence": "HIGH" if len(data) >= 4 else "MEDIUM" if len(data) >= 2 else "LOW"
        })
    
    def _get_aggregate_trend_data(self, scenario_id: str, current_year: int) -> List[Dict]:
        """Get aggregate trend data for all properties"""
        # Mock data
        data = []
        for year in range(current_year - 3, current_year + 1):
            years_from_start = year - (current_year - 3)
            value = 10000 * (1 - 0.05 * years_from_start)  # 5% annual reduction
            data.append({
                "year": year,
                "value": value,
                "property_count": 10
            })
        return data
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "STABLE"
        
        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "STABLE"
        
        slope = numerator / denominator
        slope_pct = (slope / y_mean) * 100 if y_mean != 0 else 0
        
        if slope_pct < -2:
            return "IMPROVING"
        elif slope_pct > 2:
            return "DECLINING"
        else:
            return "STABLE"
    
    def _calculate_average_change(self, values: List[float]) -> float:
        """Calculate average year-over-year change"""
        if len(values) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = ((values[i] - values[i-1]) / values[i-1]) * 100
                changes.append(change)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation)"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        return (std_dev / mean) * 100
    
    def _forecast_next_period(self, data: List[Dict]) -> Dict[str, Any]:
        """Forecast next period based on trend"""
        if len(data) < 2:
            return {"available": False}
        
        values = [d.get("actual", d.get("value", 0)) for d in data]
        years = [d.get("year", 0) for d in data]
        
        # Simple linear projection
        avg_change = self._calculate_average_change(values)
        last_value = values[-1]
        forecast_value = last_value * (1 + avg_change / 100)
        
        return {
            "available": True,
            "next_year": max(years) + 1 if years else datetime.now().year + 1,
            "forecast_value": round(forecast_value, 2),
            "confidence_interval": {
                "low": round(forecast_value * 0.9, 2),
                "high": round(forecast_value * 1.1, 2)
            }
        }
    
    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================
    
    @measure_time
    def generate_alerts(self, scenario_id: str) -> ServiceResult[List[Dict]]:
        """
        Generate alerts for properties requiring attention.
        
        Args:
            scenario_id: Scenario ID
            
        Returns:
            ServiceResult containing list of alerts
        """
        return self._execute(self._generate_alerts_impl, scenario_id)
    
    def _generate_alerts_impl(self, scenario_id: str) -> ServiceResult[List[Dict]]:
        """Implementation of generate_alerts"""
        current_year = datetime.now().year
        
        # Analyze deviations
        deviations_result = self.analyze_deviations(scenario_id, current_year, threshold_pct=5.0)
        if not deviations_result.is_success:
            return ServiceResult.error("Failed to analyze deviations")
        
        deviations = deviations_result.data
        
        alerts = []
        for dev in deviations:
            severity = self._determine_alert_severity(dev.deviation_percentage)
            
            alert = {
                "alert_id": f"ALERT-{dev.property_id}-{current_year}",
                "property_id": dev.property_id,
                "alert_type": "DEVIATION",
                "severity": severity,
                "title": f"{'Over' if dev.deviation_percentage > 0 else 'Under'} Target Alert",
                "message": f"Property {dev.property_id} is {abs(dev.deviation_percentage):.1f}% "
                          f"{'above' if dev.deviation_percentage > 0 else 'below'} target",
                "deviation_percentage": dev.deviation_percentage,
                "recommended_actions": dev.recommended_actions,
                "created_at": datetime.utcnow().isoformat(),
                "acknowledged": False
            }
            alerts.append(alert)
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))
        
        return ServiceResult.success(alerts)
    
    def _determine_alert_severity(self, deviation_pct: float) -> str:
        """Determine alert severity from deviation"""
        if abs(deviation_pct) > 20:
            return "CRITICAL"
        elif abs(deviation_pct) > 10:
            return "WARNING"
        else:
            return "INFO"
    
    @measure_time
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> ServiceResult[bool]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User acknowledging
            
        Returns:
            ServiceResult indicating success
        """
        # In production, update database
        return ServiceResult.success(True, f"Alert {alert_id} acknowledged")
    
    # =========================================================================
    # REOPTIMIZATION
    # =========================================================================
    
    @measure_time
    def generate_reoptimization_recommendations(self, scenario_id: str,
                                               current_year: int) -> ServiceResult[List[ReoptimizationRecommendation]]:
        """
        Generate recommendations for plan adjustment based on actual performance.
        
        Args:
            scenario_id: Scenario ID
            current_year: Current year
            
        Returns:
            ServiceResult containing recommendations
        """
        return self._execute(
            self._generate_reopt_recommendations_impl,
            scenario_id, current_year
        )
    
    def _generate_reopt_recommendations_impl(self, scenario_id: str,
                                            current_year: int) -> ServiceResult[List[ReoptimizationRecommendation]]:
        """Implementation of generate_reoptimization_recommendations"""
        # Analyze deviations
        deviations_result = self.analyze_deviations(scenario_id, current_year, threshold_pct=10.0)
        if not deviations_result.is_success:
            return ServiceResult.error("Failed to analyze deviations")
        
        deviations = deviations_result.data
        
        recommendations = []
        for i, dev in enumerate(deviations):
            if dev.deviation_percentage > 10:
                # Need to catch up
                adjustment = -0.1  # Increase reduction target by 10%
                adj_type = "INCREASE"
                rationale = "Performance lagging target; increased effort needed"
            elif dev.deviation_percentage < -10:
                # Ahead of target
                adjustment = 0.05  # Can relax by 5%
                adj_type = "DECREASE"
                rationale = "Performance exceeding target; opportunity to redistribute"
            else:
                continue  # Within acceptable range
            
            current_target = dev.target
            recommended_target = current_target * (1 + adjustment)
            
            rec = ReoptimizationRecommendation(
                recommendation_id=f"REOPT-{i+1:03d}",
                property_id=dev.property_id,
                current_target=current_target,
                recommended_target=round(recommended_target, 2),
                adjustment_type=adj_type,
                rationale=rationale,
                estimated_impact={
                    "cost_change_pct": abs(adjustment) * 100 if adj_type == "INCREASE" else -abs(adjustment) * 50,
                    "feasibility_change": -5 if adj_type == "INCREASE" else 5
                },
                priority="HIGH" if abs(dev.deviation_percentage) > 20 else "MEDIUM"
            )
            recommendations.append(rec)
        
        return ServiceResult.success(recommendations)
    
    @measure_time
    def apply_reoptimization(self, scenario_id: str,
                            recommendations: List[str]) -> ServiceResult[Dict[str, Any]]:
        """
        Apply selected reoptimization recommendations.
        
        Args:
            scenario_id: Scenario ID
            recommendations: List of recommendation IDs to apply
            
        Returns:
            ServiceResult with updated scenario info
        """
        if not recommendations:
            return ServiceResult.validation_error(["No recommendations selected"])
        
        # In production, update scenario targets
        return ServiceResult.success({
            "scenario_id": scenario_id,
            "applied_recommendations": recommendations,
            "status": "UPDATED",
            "message": f"Applied {len(recommendations)} recommendations"
        })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TrackingService',
    'ProgressData',
    'ProgressSummary',
    'DeviationAnalysis',
    'ReoptimizationRecommendation',
    'AlertSeverity',
    'TrendDirection'
]
