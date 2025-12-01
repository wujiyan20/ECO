# services/visualization_service.py - Visualization Service
"""
Service for generating charts, reports, and dashboards.

Features:
- Chart data generation
- Report creation
- Dashboard data aggregation
- Export functionality
- Customizable visualizations
"""

import logging
import io
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

class ChartType(Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    STACKED_BAR = "stacked_bar"
    WATERFALL = "waterfall"
    GAUGE = "gauge"
    HEATMAP = "heatmap"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    chart_type: str
    title: str
    x_axis_label: str = ""
    y_axis_label: str = ""
    colors: Optional[List[str]] = None
    show_legend: bool = True
    show_grid: bool = True
    show_values: bool = False
    stacked: bool = False


@dataclass
class ChartData:
    """Data for chart rendering"""
    chart_id: str
    chart_type: str
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSection:
    """Section of a report"""
    section_id: str
    title: str
    content_type: str  # text, chart, table, metric
    content: Any
    order: int = 0


@dataclass
class Report:
    """Complete report"""
    report_id: str
    title: str
    subtitle: str
    generated_at: str
    sections: List[ReportSection]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Dashboard data bundle"""
    dashboard_id: str
    title: str
    last_updated: str
    key_metrics: List[Dict[str, Any]]
    charts: List[ChartData]
    alerts: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLOR_SCHEMES = {
    "default": ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"],
    "sustainability": ["#27ae60", "#2ecc71", "#1abc9c", "#16a085", "#3498db", "#2980b9"],
    "status": ["#2ecc71", "#f39c12", "#e74c3c"],  # Green, Yellow, Red
    "sequential": ["#eef2f7", "#c5d5ea", "#9bb8dd", "#729bd0", "#487ec3", "#1e62b6"],
    "divergent": ["#e74c3c", "#f5a7a0", "#f7f7f7", "#a7d4a7", "#2ecc71"]
}


# =============================================================================
# VISUALIZATION SERVICE
# =============================================================================

class VisualizationService(BaseService):
    """
    Service for generating charts, reports, and dashboards.
    
    Provides:
    - Chart data generation for various chart types
    - Report creation in multiple formats
    - Dashboard data aggregation
    - Export functionality
    
    Usage:
        service = VisualizationService()
        
        # Generate chart
        chart = service.generate_milestone_chart(scenario_id)
        
        # Create report
        report = service.create_progress_report(scenario_id, year)
    """
    
    def __init__(self, db_manager=None, milestone_service=None,
                 tracking_service=None, property_service=None):
        """
        Initialize visualization service.
        
        Args:
            db_manager: Database manager
            milestone_service: Milestone service
            tracking_service: Tracking service
            property_service: Property service
        """
        super().__init__(db_manager)
        self._milestone_service = milestone_service
        self._tracking_service = tracking_service
        self._property_service = property_service
    
    # =========================================================================
    # CHART GENERATION
    # =========================================================================
    
    @measure_time
    def generate_milestone_chart(self, scenario_id: str,
                                config: ChartConfig = None) -> ServiceResult[ChartData]:
        """
        Generate milestone trajectory chart.
        
        Args:
            scenario_id: Scenario ID
            config: Chart configuration
            
        Returns:
            ServiceResult containing chart data
        """
        return self._execute(self._generate_milestone_chart_impl, scenario_id, config)
    
    def _generate_milestone_chart_impl(self, scenario_id: str,
                                      config: ChartConfig) -> ServiceResult[ChartData]:
        """Implementation of generate_milestone_chart"""
        # Get scenario data (would fetch from milestone service)
        scenario = self._get_scenario_data(scenario_id)
        
        targets = scenario.get("reduction_targets", [])
        
        # Prepare chart data
        labels = [str(t["year"]) for t in targets]
        
        datasets = [
            {
                "label": "Target Emissions",
                "data": [t["target_emissions"] for t in targets],
                "borderColor": COLOR_SCHEMES["sustainability"][0],
                "backgroundColor": f"{COLOR_SCHEMES['sustainability'][0]}40",
                "fill": True,
                "tension": 0.4
            },
            {
                "label": "Baseline",
                "data": [scenario.get("baseline_emission", 10000)] * len(targets),
                "borderColor": "#95a5a6",
                "borderDash": [5, 5],
                "fill": False
            }
        ]
        
        # Add actual data if available
        if "actual_data" in scenario:
            datasets.append({
                "label": "Actual Emissions",
                "data": scenario["actual_data"],
                "borderColor": COLOR_SCHEMES["status"][0],
                "pointRadius": 6,
                "fill": False
            })
        
        chart = ChartData(
            chart_id=f"milestone-{scenario_id}",
            chart_type="line",
            title=config.title if config else "Emission Reduction Trajectory",
            labels=labels,
            datasets=datasets,
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "title": {"display": True, "text": "Emissions (kg CO2e)"},
                        "beginAtZero": False
                    },
                    "x": {
                        "title": {"display": True, "text": "Year"}
                    }
                },
                "plugins": {
                    "legend": {"display": True, "position": "top"},
                    "title": {"display": True, "text": "Milestone Trajectory"}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    def _get_scenario_data(self, scenario_id: str) -> Dict[str, Any]:
        """Get scenario data (mock or from service)"""
        # Mock data
        base_year = 2024
        base_emission = 10000
        
        targets = []
        for year in range(base_year, 2051):
            progress = (year - base_year) / (2050 - base_year)
            reduction = progress * 0.9  # 90% reduction by 2050
            targets.append({
                "year": year,
                "target_emissions": base_emission * (1 - reduction),
                "reduction_percentage": reduction * 100
            })
        
        return {
            "scenario_id": scenario_id,
            "baseline_emission": base_emission,
            "reduction_targets": targets
        }
    
    @measure_time
    def generate_allocation_chart(self, allocation_id: str,
                                 chart_type: str = "bar") -> ServiceResult[ChartData]:
        """
        Generate allocation distribution chart.
        
        Args:
            allocation_id: Allocation ID
            chart_type: Type of chart (bar, pie)
            
        Returns:
            ServiceResult containing chart data
        """
        return self._execute(self._generate_allocation_chart_impl, allocation_id, chart_type)
    
    def _generate_allocation_chart_impl(self, allocation_id: str,
                                       chart_type: str) -> ServiceResult[ChartData]:
        """Implementation of generate_allocation_chart"""
        # Mock allocation data
        properties = [
            {"name": "Property A", "allocation": 35},
            {"name": "Property B", "allocation": 25},
            {"name": "Property C", "allocation": 20},
            {"name": "Property D", "allocation": 12},
            {"name": "Property E", "allocation": 8}
        ]
        
        labels = [p["name"] for p in properties]
        values = [p["allocation"] for p in properties]
        
        if chart_type == "pie":
            datasets = [{
                "data": values,
                "backgroundColor": COLOR_SCHEMES["sustainability"][:len(properties)]
            }]
        else:  # bar
            datasets = [{
                "label": "Allocation %",
                "data": values,
                "backgroundColor": COLOR_SCHEMES["sustainability"][0],
                "borderColor": COLOR_SCHEMES["sustainability"][1],
                "borderWidth": 1
            }]
        
        chart = ChartData(
            chart_id=f"allocation-{allocation_id}",
            chart_type=chart_type,
            title="Target Allocation by Property",
            labels=labels,
            datasets=datasets,
            options={
                "responsive": True,
                "plugins": {
                    "legend": {"display": chart_type == "pie"},
                    "title": {"display": True, "text": "Reduction Target Allocation"}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    @measure_time
    def generate_progress_chart(self, scenario_id: str, year: int) -> ServiceResult[ChartData]:
        """
        Generate progress tracking chart.
        
        Args:
            scenario_id: Scenario ID
            year: Year to visualize
            
        Returns:
            ServiceResult containing chart data
        """
        return self._execute(self._generate_progress_chart_impl, scenario_id, year)
    
    def _generate_progress_chart_impl(self, scenario_id: str,
                                     year: int) -> ServiceResult[ChartData]:
        """Implementation of generate_progress_chart"""
        # Mock progress data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        base_target = 800
        targets = [base_target - (i * 2) for i in range(12)]
        actuals = [base_target - (i * 2) + (i % 3 - 1) * 20 for i in range(12)]
        
        datasets = [
            {
                "label": "Target",
                "data": targets,
                "borderColor": COLOR_SCHEMES["sustainability"][2],
                "borderDash": [5, 5],
                "fill": False
            },
            {
                "label": "Actual",
                "data": actuals,
                "borderColor": COLOR_SCHEMES["sustainability"][0],
                "backgroundColor": f"{COLOR_SCHEMES['sustainability'][0]}20",
                "fill": True
            }
        ]
        
        chart = ChartData(
            chart_id=f"progress-{scenario_id}-{year}",
            chart_type="line",
            title=f"Progress Tracking - {year}",
            labels=months,
            datasets=datasets,
            options={
                "responsive": True,
                "scales": {
                    "y": {"title": {"display": True, "text": "Emissions (kg CO2e)"}}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    @measure_time
    def generate_strategy_breakdown_chart(self, scenario_id: str) -> ServiceResult[ChartData]:
        """
        Generate strategy breakdown pie chart.
        
        Args:
            scenario_id: Scenario ID
            
        Returns:
            ServiceResult containing chart data
        """
        # Mock strategy data
        strategies = {
            "Energy Efficiency": 35,
            "Renewable Energy": 30,
            "Electrification": 20,
            "Operational Optimization": 15
        }
        
        chart = ChartData(
            chart_id=f"strategy-{scenario_id}",
            chart_type="pie",
            title="Strategy Breakdown",
            labels=list(strategies.keys()),
            datasets=[{
                "data": list(strategies.values()),
                "backgroundColor": COLOR_SCHEMES["sustainability"][:len(strategies)]
            }],
            options={
                "responsive": True,
                "plugins": {
                    "legend": {"display": True, "position": "right"}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    @measure_time
    def generate_cost_projection_chart(self, scenario_id: str) -> ServiceResult[ChartData]:
        """
        Generate cost projection chart.
        
        Args:
            scenario_id: Scenario ID
            
        Returns:
            ServiceResult containing chart data
        """
        # Mock cost data
        years = list(range(2024, 2035))
        capex = [100000, 150000, 120000, 80000, 60000, 50000, 40000, 30000, 20000, 15000, 10000]
        opex = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 42000, 44000, 45000, 46000]
        
        chart = ChartData(
            chart_id=f"cost-{scenario_id}",
            chart_type="bar",
            title="Cost Projections",
            labels=[str(y) for y in years],
            datasets=[
                {
                    "label": "CAPEX",
                    "data": capex,
                    "backgroundColor": COLOR_SCHEMES["default"][0]
                },
                {
                    "label": "OPEX",
                    "data": opex,
                    "backgroundColor": COLOR_SCHEMES["default"][1]
                }
            ],
            options={
                "responsive": True,
                "scales": {
                    "x": {"stacked": True},
                    "y": {"stacked": True, "title": {"display": True, "text": "Cost (USD)"}}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    @measure_time
    def generate_comparison_chart(self, scenario_ids: List[str]) -> ServiceResult[ChartData]:
        """
        Generate scenario comparison chart.
        
        Args:
            scenario_ids: List of scenarios to compare
            
        Returns:
            ServiceResult containing chart data
        """
        # Mock comparison data
        metrics = ["2030 Reduction", "2050 Reduction", "Cost Efficiency", "Feasibility"]
        
        datasets = []
        for i, sid in enumerate(scenario_ids):
            datasets.append({
                "label": f"Scenario {i+1}",
                "data": [40 + i*5, 90 + i*2, 80 - i*5, 75 + i*3],
                "backgroundColor": f"{COLOR_SCHEMES['default'][i]}80",
                "borderColor": COLOR_SCHEMES['default'][i],
                "borderWidth": 2
            })
        
        chart = ChartData(
            chart_id=f"comparison-{'_'.join(scenario_ids[:3])}",
            chart_type="radar",
            title="Scenario Comparison",
            labels=metrics,
            datasets=datasets,
            options={
                "responsive": True,
                "scales": {
                    "r": {"min": 0, "max": 100}
                }
            }
        )
        
        return ServiceResult.success(chart)
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    @measure_time
    def create_progress_report(self, scenario_id: str, year: int,
                              format: str = "json") -> ServiceResult[Report]:
        """
        Create progress tracking report.
        
        Args:
            scenario_id: Scenario ID
            year: Year to report on
            format: Output format
            
        Returns:
            ServiceResult containing report
        """
        return self._execute(self._create_progress_report_impl, scenario_id, year, format)
    
    def _create_progress_report_impl(self, scenario_id: str, year: int,
                                    format: str) -> ServiceResult[Report]:
        """Implementation of create_progress_report"""
        report_id = f"RPT-{scenario_id}-{year}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        sections = []
        
        # Executive Summary
        sections.append(ReportSection(
            section_id="executive-summary",
            title="Executive Summary",
            content_type="text",
            content={
                "overview": f"This report provides a comprehensive review of emission reduction progress for scenario {scenario_id} during {year}.",
                "key_findings": [
                    "Overall progress is on track with 95% of targets met",
                    "3 properties exceeded their targets",
                    "2 properties require attention due to underperformance"
                ],
                "recommendations": [
                    "Continue current initiatives for on-track properties",
                    "Implement corrective actions for underperforming properties"
                ]
            },
            order=1
        ))
        
        # Key Metrics
        sections.append(ReportSection(
            section_id="key-metrics",
            title="Key Performance Metrics",
            content_type="metric",
            content={
                "metrics": [
                    {"name": "Target Emission", "value": 8500, "unit": "kg CO2e"},
                    {"name": "Actual Emission", "value": 8200, "unit": "kg CO2e"},
                    {"name": "Variance", "value": -3.5, "unit": "%"},
                    {"name": "Cost Spent", "value": 125000, "unit": "USD"},
                    {"name": "Properties On Track", "value": 8, "unit": "out of 10"}
                ]
            },
            order=2
        ))
        
        # Progress Chart
        chart_result = self.generate_progress_chart(scenario_id, year)
        if chart_result.is_success:
            sections.append(ReportSection(
                section_id="progress-chart",
                title="Progress Trajectory",
                content_type="chart",
                content=chart_result.data,
                order=3
            ))
        
        # Property Details
        sections.append(ReportSection(
            section_id="property-details",
            title="Property Performance",
            content_type="table",
            content={
                "headers": ["Property", "Target", "Actual", "Variance", "Status"],
                "rows": [
                    ["Property A", "850", "800", "-5.9%", "Ahead"],
                    ["Property B", "650", "670", "+3.1%", "On Track"],
                    ["Property C", "500", "480", "-4.0%", "On Track"],
                    ["Property D", "400", "450", "+12.5%", "At Risk"],
                    ["Property E", "300", "280", "-6.7%", "Ahead"]
                ]
            },
            order=4
        ))
        
        report = Report(
            report_id=report_id,
            title=f"Progress Report - {year}",
            subtitle=f"Scenario: {scenario_id}",
            generated_at=datetime.utcnow().isoformat(),
            sections=sections,
            summary={
                "total_target": 8500,
                "total_actual": 8200,
                "overall_status": "ON_TRACK",
                "properties_analyzed": 10,
                "period": f"{year}"
            },
            metadata={
                "format": format,
                "version": "1.0"
            }
        )
        
        return ServiceResult.success(report)
    
    @measure_time
    def create_scenario_report(self, scenario_id: str,
                              format: str = "json") -> ServiceResult[Report]:
        """
        Create comprehensive scenario report.
        
        Args:
            scenario_id: Scenario ID
            format: Output format
            
        Returns:
            ServiceResult containing report
        """
        report_id = f"SCN-RPT-{scenario_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        sections = []
        
        # Scenario Overview
        sections.append(ReportSection(
            section_id="overview",
            title="Scenario Overview",
            content_type="text",
            content={
                "scenario_type": "Standard",
                "description": "Balanced approach aligned with SBTi targets",
                "base_year": 2024,
                "target_2030": "40% reduction",
                "target_2050": "90% reduction",
                "total_properties": 10,
                "total_baseline": "10,000 kg CO2e"
            },
            order=1
        ))
        
        # Milestone Chart
        chart_result = self.generate_milestone_chart(scenario_id)
        if chart_result.is_success:
            sections.append(ReportSection(
                section_id="milestone-chart",
                title="Emission Reduction Trajectory",
                content_type="chart",
                content=chart_result.data,
                order=2
            ))
        
        # Strategy Breakdown
        strategy_result = self.generate_strategy_breakdown_chart(scenario_id)
        if strategy_result.is_success:
            sections.append(ReportSection(
                section_id="strategy-chart",
                title="Strategy Allocation",
                content_type="chart",
                content=strategy_result.data,
                order=3
            ))
        
        # Cost Projections
        cost_result = self.generate_cost_projection_chart(scenario_id)
        if cost_result.is_success:
            sections.append(ReportSection(
                section_id="cost-chart",
                title="Cost Projections",
                content_type="chart",
                content=cost_result.data,
                order=4
            ))
        
        report = Report(
            report_id=report_id,
            title=f"Scenario Report - {scenario_id}",
            subtitle="Comprehensive Analysis",
            generated_at=datetime.utcnow().isoformat(),
            sections=sections,
            summary={
                "scenario_type": "Standard",
                "reduction_2030": 40,
                "reduction_2050": 90,
                "total_capex": 750000,
                "feasibility_score": 85
            }
        )
        
        return ServiceResult.success(report)
    
    # =========================================================================
    # DASHBOARD GENERATION
    # =========================================================================
    
    @measure_time
    @cached(ttl_seconds=300, key_prefix="dashboard")
    def get_dashboard_data(self, scenario_id: str = None) -> ServiceResult[DashboardData]:
        """
        Get aggregated dashboard data.
        
        Args:
            scenario_id: Optional scenario filter
            
        Returns:
            ServiceResult containing dashboard data
        """
        return self._execute(self._get_dashboard_data_impl, scenario_id)
    
    def _get_dashboard_data_impl(self, scenario_id: str) -> ServiceResult[DashboardData]:
        """Implementation of get_dashboard_data"""
        # Key metrics
        key_metrics = [
            {
                "id": "total-reduction",
                "label": "Total Reduction",
                "value": 15.5,
                "unit": "%",
                "trend": "up",
                "change": 2.3
            },
            {
                "id": "properties-on-track",
                "label": "Properties On Track",
                "value": 8,
                "unit": "of 10",
                "trend": "stable",
                "change": 0
            },
            {
                "id": "cost-efficiency",
                "label": "Cost per Tonne",
                "value": 125,
                "unit": "USD",
                "trend": "down",
                "change": -5.2
            },
            {
                "id": "carbon-intensity",
                "label": "Avg Carbon Intensity",
                "value": 185,
                "unit": "kg/sqm",
                "trend": "down",
                "change": -3.8
            }
        ]
        
        # Charts
        charts = []
        
        if scenario_id:
            milestone_result = self.generate_milestone_chart(scenario_id)
            if milestone_result.is_success:
                charts.append(milestone_result.data)
            
            progress_result = self.generate_progress_chart(scenario_id, datetime.now().year)
            if progress_result.is_success:
                charts.append(progress_result.data)
        
        # Alerts
        alerts = [
            {
                "id": "alert-1",
                "severity": "warning",
                "title": "Property D Behind Target",
                "message": "Property D is 12.5% behind target. Review recommended.",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": "alert-2",
                "severity": "info",
                "title": "Quarterly Review Due",
                "message": "Q4 progress review scheduled for next week.",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        dashboard = DashboardData(
            dashboard_id=f"DASH-{datetime.utcnow().strftime('%Y%m%d')}",
            title="EcoAssist Dashboard",
            last_updated=datetime.utcnow().isoformat(),
            key_metrics=key_metrics,
            charts=charts,
            alerts=alerts,
            metadata={
                "scenario_id": scenario_id,
                "refresh_interval": 300
            }
        )
        
        return ServiceResult.success(dashboard)
    
    # =========================================================================
    # EXPORT FUNCTIONALITY
    # =========================================================================
    
    @measure_time
    def export_chart(self, chart_data: ChartData, format: str = "json") -> ServiceResult[str]:
        """
        Export chart data in specified format.
        
        Args:
            chart_data: Chart data to export
            format: Export format (json, csv)
            
        Returns:
            ServiceResult containing exported data
        """
        if format == "json":
            import json
            return ServiceResult.success(json.dumps({
                "chart_id": chart_data.chart_id,
                "chart_type": chart_data.chart_type,
                "title": chart_data.title,
                "labels": chart_data.labels,
                "datasets": chart_data.datasets,
                "options": chart_data.options
            }, indent=2))
        
        elif format == "csv":
            lines = []
            # Header
            headers = ["Label"] + [ds.get("label", f"Series {i}") for i, ds in enumerate(chart_data.datasets)]
            lines.append(",".join(headers))
            
            # Data rows
            for i, label in enumerate(chart_data.labels):
                row = [str(label)]
                for ds in chart_data.datasets:
                    data = ds.get("data", [])
                    row.append(str(data[i]) if i < len(data) else "")
                lines.append(",".join(row))
            
            return ServiceResult.success("\n".join(lines))
        
        return ServiceResult.validation_error([f"Unsupported format: {format}"])
    
    @measure_time
    def export_report(self, report: Report, format: str = "json") -> ServiceResult[str]:
        """
        Export report in specified format.
        
        Args:
            report: Report to export
            format: Export format
            
        Returns:
            ServiceResult containing exported data
        """
        if format == "json":
            import json
            
            def serialize(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            return ServiceResult.success(json.dumps({
                "report_id": report.report_id,
                "title": report.title,
                "subtitle": report.subtitle,
                "generated_at": report.generated_at,
                "sections": [serialize(s) for s in report.sections],
                "summary": report.summary,
                "metadata": report.metadata
            }, indent=2, default=serialize))
        
        elif format == "html":
            html = self._generate_html_report(report)
            return ServiceResult.success(html)
        
        return ServiceResult.validation_error([f"Unsupported format: {format}"])
    
    def _generate_html_report(self, report: Report) -> str:
        """Generate HTML version of report"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .metric {{ display: inline-block; padding: 15px; margin: 10px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2ecc71; }}
        .metric-label {{ color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p><em>{report.subtitle}</em></p>
    <p>Generated: {report.generated_at}</p>
"""
        
        for section in sorted(report.sections, key=lambda s: s.order):
            html += f'<div class="section"><h2>{section.title}</h2>'
            
            if section.content_type == "text":
                content = section.content
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, list):
                            html += f"<h3>{key.replace('_', ' ').title()}</h3><ul>"
                            for item in value:
                                html += f"<li>{item}</li>"
                            html += "</ul>"
                        else:
                            html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            
            elif section.content_type == "metric":
                for metric in section.content.get("metrics", []):
                    html += f'''
                    <div class="metric">
                        <div class="metric-value">{metric['value']} {metric.get('unit', '')}</div>
                        <div class="metric-label">{metric['name']}</div>
                    </div>'''
            
            elif section.content_type == "table":
                content = section.content
                html += "<table><thead><tr>"
                for header in content.get("headers", []):
                    html += f"<th>{header}</th>"
                html += "</tr></thead><tbody>"
                for row in content.get("rows", []):
                    html += "<tr>"
                    for cell in row:
                        html += f"<td>{cell}</td>"
                    html += "</tr>"
                html += "</tbody></table>"
            
            html += "</div>"
        
        html += f"""
    <div class="footer">
        <p>Report ID: {report.report_id}</p>
    </div>
</body>
</html>"""
        
        return html


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VisualizationService',
    'ChartType',
    'ReportFormat',
    'ChartConfig',
    'ChartData',
    'ReportSection',
    'Report',
    'DashboardData',
    'COLOR_SCHEMES'
]
