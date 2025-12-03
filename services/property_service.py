# services/property_service.py - Property Management Service
"""
Service for managing properties and portfolios in the EcoAssist system.

Features:
- Property CRUD operations
- Portfolio management
- Property filtering and search
- Baseline data management
- Carbon intensity calculations
- Property validation
- Batch operations
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from models.property import (
    Property,
    Portfolio,
    PropertyFilter,
    PropertyEmissionBreakdown,
    PropertyMetrics
)
from models.enums import (
    BuildingType,
    RetrofitPotential,
    PropertyStatus
)

from .base_service import (
    BaseService,
    ServiceResult,
    ServiceResultStatus,
    measure_time,
    cached,
    transaction,
    retry
)

from models.base import BaseModel, AuditableModel

# Import models (adjust path as needed)
try:
    from models import (
        Property,
        PropertyFilter,
        Portfolio,
        PropertyMetrics,
        PropertyEmissionBreakdown,
        BaselineDataRecord,
        BuildingType,
        RetrofitPotential,
        PropertyStatus,
        PropertyRepository,
        EmissionRepository,
        validate_year,
        validate_positive_number,
        generate_property_id,
        calculate_carbon_intensity
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logging.warning("Models package not available - using mock mode")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TRANSFER OBJECTS
# =============================================================================

@dataclass
class PropertySummary:
    """Summary view of a property"""
    property_id: str
    name: str
    building_type: str
    area_sqm: float
    baseline_emission: float
    carbon_intensity: float
    retrofit_potential: str
    priority_score: float
    status: str


@dataclass
class PortfolioSummary:
    """Summary view of a portfolio"""
    total_properties: int
    total_area_sqm: float
    total_emissions: float
    average_carbon_intensity: float
    properties_by_type: Dict[str, int]
    high_priority_count: int
    status_breakdown: Dict[str, int]


@dataclass
class PropertyUpdateRequest:
    """Request to update property data"""
    property_id: str
    updates: Dict[str, Any]
    updated_by: Optional[str] = None
    reason: Optional[str] = None


# =============================================================================
# PROPERTY SERVICE
# =============================================================================

class PropertyService(BaseService):
    """
    Service for managing properties and portfolios.
    
    Provides:
    - Property CRUD operations
    - Portfolio aggregation
    - Filtering and search
    - Baseline data management
    - Validation and integrity checks
    
    Usage:
        service = PropertyService(db_manager)
        result = service.get_property("PROP-001")
        if result.is_success:
            property = result.data
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize property service.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__(db_manager)
        self._property_repo: Optional[PropertyRepository] = None
        self._emission_repo: Optional[EmissionRepository] = None
    
    def _do_initialize(self) -> None:
        """Initialize repositories"""
        if self.db_manager and MODELS_AVAILABLE:
            self._property_repo = PropertyRepository(self.db_manager)
            self._emission_repo = EmissionRepository(self.db_manager)
            self._logger.info("Property repositories initialized")
    
    # =========================================================================
    # PROPERTY CRUD OPERATIONS
    # =========================================================================
    
    @measure_time
    @cached(ttl_seconds=300, key_prefix="property")
    def get_property(self, property_id: str) -> ServiceResult[Property]:
        """
        Get a property by ID.
        
        Args:
            property_id: Unique property identifier
            
        Returns:
            ServiceResult containing the property or error
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        return self._execute(self._get_property_impl, property_id)
    
    def _get_property_impl(self, property_id: str) -> ServiceResult[Property]:
        """Implementation of get_property"""
        if self._property_repo:
            property_data = self._property_repo.get_by_id(property_id)
            if property_data:
                return ServiceResult.success(property_data)
            return ServiceResult.not_found(f"Property {property_id} not found")
        
        # Mock mode
        return self._mock_get_property(property_id)
    
    def _mock_get_property(self, property_id: str) -> ServiceResult[Property]:
        """Mock property for testing"""
        if MODELS_AVAILABLE:
            mock_property = Property(
                property_id=property_id,
                name=f"Mock Property {property_id}",
                address="123 Test Street",
                city="Test City",
                building_type=BuildingType.OFFICE,
                area_sqm=5000.0,
                baseline_emission=1000.0,
                scope1_emission=300.0,
                scope2_emission=500.0,
                scope3_emission=200.0
            )
            return ServiceResult.success(mock_property)
        return ServiceResult.error("Models not available")
    
    @measure_time
    def get_properties(self, filter_obj: PropertyFilter = None,
                       page: int = 1, page_size: int = 50) -> ServiceResult[List[Property]]:
        """
        Get properties with optional filtering and pagination.
        
        Args:
            filter_obj: Filter criteria
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            ServiceResult containing list of properties
        """
        return self._execute(self._get_properties_impl, filter_obj, page, page_size)
    
    def _get_properties_impl(self, filter_obj: PropertyFilter,
                            page: int, page_size: int) -> ServiceResult[List[Property]]:
        """Implementation of get_properties"""
        if self._property_repo:
            properties = self._property_repo.get_all(filter_obj)
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated = properties[start_idx:end_idx]
            
            return ServiceResult.success(
                data=paginated,
                metadata={
                    "total_count": len(properties),
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (len(properties) + page_size - 1) // page_size
                }
            )
        
        # Mock mode
        return self._mock_get_properties(filter_obj, page, page_size)
    
    def _mock_get_properties(self, filter_obj: PropertyFilter,
                            page: int, page_size: int) -> ServiceResult[List[Property]]:
        """Mock properties for testing"""
        mock_properties = []
        if MODELS_AVAILABLE:
            for i in range(5):
                prop = Property(
                    property_id=f"PROP-{i+1:03d}",
                    name=f"Mock Property {i+1}",
                    address=f"{100 + i} Test Street",
                    city="Test City",
                    building_type=BuildingType.OFFICE if i % 2 == 0 else BuildingType.RETAIL,
                    area_sqm=5000.0 + (i * 1000),
                    baseline_emission=1000.0 + (i * 200),
                    scope1_emission=300.0,
                    scope2_emission=500.0,
                    scope3_emission=200.0
                )
                mock_properties.append(prop)
        
        return ServiceResult.success(
            data=mock_properties,
            metadata={"total_count": 5, "page": 1, "page_size": 50}
        )
    
    @measure_time
    @transaction
    def create_property(self, property_data: Dict[str, Any],
                        created_by: str = None) -> ServiceResult[Property]:
        """
        Create a new property.
        
        Args:
            property_data: Property data dictionary
            created_by: User creating the property
            
        Returns:
            ServiceResult containing created property
        """
        # Validate input
        validation_errors = self._validate_property_data(property_data)
        if validation_errors:
            return ServiceResult.validation_error(validation_errors)
        
        return self._execute(self._create_property_impl, property_data, created_by)
    
    def _validate_property_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate property data"""
        errors = []
        
        # Required fields
        required_fields = ['name', 'building_type', 'area_sqm']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"{field} is required")
        
        # Validate area
        if 'area_sqm' in data:
            try:
                if MODELS_AVAILABLE:
                    validate_positive_number(data['area_sqm'], 'area_sqm')
            except ValueError as e:
                errors.append(str(e))
        
        # Validate building type
        if 'building_type' in data and MODELS_AVAILABLE:
            try:
                BuildingType(data['building_type'])
            except ValueError:
                errors.append(f"Invalid building_type: {data['building_type']}")
        
        return errors
    
    def _create_property_impl(self, property_data: Dict[str, Any],
                             created_by: str) -> ServiceResult[Property]:
        """Implementation of create_property"""
        # Generate ID if not provided
        if 'property_id' not in property_data:
            property_data['property_id'] = generate_property_id() if MODELS_AVAILABLE else f"PROP-{datetime.utcnow().timestamp()}"
        
        if self._property_repo and MODELS_AVAILABLE:
            # Create Property object
            property_obj = Property(**property_data)
            
            # Validate
            is_valid, errors = property_obj.validate()
            if not is_valid:
                return ServiceResult.validation_error(errors)
            
            # Save to database
            created = self._property_repo.create(property_obj)
            
            # Invalidate cache
            self._invalidate_cache(f"portfolio_metrics")
            
            return ServiceResult.success(
                data=created,
                message=f"Property {created.property_id} created successfully"
            )
        
        # Mock mode
        return ServiceResult.success(
            data=property_data,
            message="Property created (mock mode)"
        )
    
    @measure_time
    @transaction
    def update_property(self, property_id: str, updates: Dict[str, Any],
                        updated_by: str = None) -> ServiceResult[Property]:
        """
        Update a property.
        
        Args:
            property_id: Property to update
            updates: Fields to update
            updated_by: User making the update
            
        Returns:
            ServiceResult containing updated property
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        if not updates:
            return ServiceResult.validation_error(["No updates provided"])
        
        return self._execute(self._update_property_impl, property_id, updates, updated_by)
    
    def _update_property_impl(self, property_id: str, updates: Dict[str, Any],
                             updated_by: str) -> ServiceResult[Property]:
        """Implementation of update_property"""
        # Get existing property
        existing_result = self.get_property(property_id)
        if not existing_result.is_success:
            return existing_result
        
        if self._property_repo and MODELS_AVAILABLE:
            existing = existing_result.data
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            
            # Recalculate intensities
            existing.update_intensities()
            
            # Validate
            is_valid, errors = existing.validate()
            if not is_valid:
                return ServiceResult.validation_error(errors)
            
            # Save
            updated = self._property_repo.update(existing)
            
            # Invalidate cache
            self._invalidate_cache(f"property:{property_id}")
            self._invalidate_cache("portfolio_metrics")
            
            return ServiceResult.success(
                data=updated,
                message=f"Property {property_id} updated successfully"
            )
        
        # Mock mode
        return ServiceResult.success(
            data={**existing_result.data.__dict__, **updates} if MODELS_AVAILABLE else updates,
            message="Property updated (mock mode)"
        )
    
    @measure_time
    @transaction
    def delete_property(self, property_id: str, deleted_by: str = None,
                        soft_delete: bool = True) -> ServiceResult[bool]:
        """
        Delete a property.
        
        Args:
            property_id: Property to delete
            deleted_by: User performing deletion
            soft_delete: If True, mark as deleted; if False, permanently remove
            
        Returns:
            ServiceResult indicating success
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        return self._execute(self._delete_property_impl, property_id, deleted_by, soft_delete)
    
    def _delete_property_impl(self, property_id: str, deleted_by: str,
                             soft_delete: bool) -> ServiceResult[bool]:
        """Implementation of delete_property"""
        if self._property_repo:
            success = self._property_repo.delete(property_id, soft_delete=soft_delete)
            if success:
                # Invalidate cache
                self._invalidate_cache(f"property:{property_id}")
                self._invalidate_cache("portfolio_metrics")
                return ServiceResult.success(True, f"Property {property_id} deleted")
            return ServiceResult.not_found(f"Property {property_id} not found")
        
        # Mock mode
        return ServiceResult.success(True, "Property deleted (mock mode)")
    
    # =========================================================================
    # PORTFOLIO OPERATIONS
    # =========================================================================
    
    @measure_time
    @cached(ttl_seconds=600, key_prefix="portfolio_metrics")
    def get_portfolio_metrics(self, property_ids: List[str] = None) -> ServiceResult[PortfolioSummary]:
        """
        Get aggregated portfolio metrics.
        
        Args:
            property_ids: Optional list of property IDs to include
            
        Returns:
            ServiceResult containing portfolio summary
        """
        return self._execute(self._get_portfolio_metrics_impl, property_ids)
    
    def _get_portfolio_metrics_impl(self, property_ids: List[str]) -> ServiceResult[PortfolioSummary]:
        """Implementation of get_portfolio_metrics"""
        # Get properties
        filter_obj = None
        if property_ids and MODELS_AVAILABLE:
            filter_obj = PropertyFilter(property_ids=property_ids)
        
        properties_result = self.get_properties(filter_obj)
        if not properties_result.is_success:
            return ServiceResult.error("Failed to fetch properties")
        
        properties = properties_result.data
        
        if not properties:
            return ServiceResult.success(PortfolioSummary(
                total_properties=0,
                total_area_sqm=0.0,
                total_emissions=0.0,
                average_carbon_intensity=0.0,
                properties_by_type={},
                high_priority_count=0,
                status_breakdown={}
            ))
        
        # Calculate metrics
        total_area = sum(p.area_sqm for p in properties if hasattr(p, 'area_sqm'))
        total_emissions = sum(p.baseline_emission for p in properties if hasattr(p, 'baseline_emission'))
        
        avg_intensity = total_emissions / total_area if total_area > 0 else 0.0
        
        # Count by type
        type_counts = {}
        for p in properties:
            if hasattr(p, 'building_type'):
                bt = str(p.building_type.value if hasattr(p.building_type, 'value') else p.building_type)
                type_counts[bt] = type_counts.get(bt, 0) + 1
        
        # Count high priority
        high_priority = sum(
            1 for p in properties 
            if hasattr(p, 'is_high_priority') and p.is_high_priority()
        )
        
        # Status breakdown
        status_counts = {}
        for p in properties:
            if hasattr(p, 'status'):
                s = str(p.status.value if hasattr(p.status, 'value') else p.status)
                status_counts[s] = status_counts.get(s, 0) + 1
        
        summary = PortfolioSummary(
            total_properties=len(properties),
            total_area_sqm=total_area,
            total_emissions=total_emissions,
            average_carbon_intensity=round(avg_intensity, 2),
            properties_by_type=type_counts,
            high_priority_count=high_priority,
            status_breakdown=status_counts
        )
        
        return ServiceResult.success(summary)
    
    @measure_time
    def get_property_rankings(self, ranking_criteria: str = "carbon_intensity",
                             limit: int = 10,
                             ascending: bool = False) -> ServiceResult[List[PropertySummary]]:
        """
        Get properties ranked by specified criteria.
        
        Args:
            ranking_criteria: Field to rank by (carbon_intensity, baseline_emission, area_sqm)
            limit: Maximum number of results
            ascending: Sort order (False = highest first)
            
        Returns:
            ServiceResult containing ranked property summaries
        """
        return self._execute(
            self._get_property_rankings_impl, 
            ranking_criteria, limit, ascending
        )
    
    def _get_property_rankings_impl(self, criteria: str, limit: int,
                                   ascending: bool) -> ServiceResult[List[PropertySummary]]:
        """Implementation of get_property_rankings"""
        # Get all properties
        properties_result = self.get_properties()
        if not properties_result.is_success:
            return properties_result
        
        properties = properties_result.data
        
        # Sort by criteria
        try:
            sorted_properties = sorted(
                properties,
                key=lambda p: getattr(p, criteria, 0),
                reverse=not ascending
            )
        except AttributeError:
            return ServiceResult.validation_error([f"Invalid ranking criteria: {criteria}"])
        
        # Create summaries
        summaries = []
        for p in sorted_properties[:limit]:
            summary = PropertySummary(
                property_id=p.property_id if hasattr(p, 'property_id') else '',
                name=p.name if hasattr(p, 'name') else '',
                building_type=str(p.building_type.value if hasattr(p.building_type, 'value') else p.building_type) if hasattr(p, 'building_type') else '',
                area_sqm=p.area_sqm if hasattr(p, 'area_sqm') else 0.0,
                baseline_emission=p.baseline_emission if hasattr(p, 'baseline_emission') else 0.0,
                carbon_intensity=p.carbon_intensity if hasattr(p, 'carbon_intensity') else 0.0,
                retrofit_potential=str(p.retrofit_potential.value if hasattr(p.retrofit_potential, 'value') else p.retrofit_potential) if hasattr(p, 'retrofit_potential') else 'MEDIUM',
                priority_score=p.priority_score if hasattr(p, 'priority_score') else 50.0,
                status=str(p.status.value if hasattr(p.status, 'value') else p.status) if hasattr(p, 'status') else 'ACTIVE'
            )
            summaries.append(summary)
        
        return ServiceResult.success(summaries)
    
    # =========================================================================
    # BASELINE DATA OPERATIONS
    # =========================================================================
    
    @measure_time
    def get_baseline_data(self, property_id: str, 
                          start_year: int = None,
                          end_year: int = None) -> ServiceResult[List[BaselineDataRecord]]:
        """
        Get baseline emission data for a property.
        
        Args:
            property_id: Property ID
            start_year: Start year for data range
            end_year: End year for data range
            
        Returns:
            ServiceResult containing baseline records
        """
        if not property_id:
            return ServiceResult.validation_error(["property_id is required"])
        
        return self._execute(
            self._get_baseline_data_impl, 
            property_id, start_year, end_year
        )
    
    def _get_baseline_data_impl(self, property_id: str, start_year: int,
                               end_year: int) -> ServiceResult[List[BaselineDataRecord]]:
        """Implementation of get_baseline_data"""
        if self._emission_repo:
            records = self._emission_repo.get_baseline_data(
                property_id, start_year, end_year
            )
            return ServiceResult.success(records)
        
        # Mock mode
        return self._mock_get_baseline_data(property_id, start_year, end_year)
    
    def _mock_get_baseline_data(self, property_id: str, start_year: int,
                               end_year: int) -> ServiceResult[List[BaselineDataRecord]]:
        """Mock baseline data for testing"""
        mock_records = []
        if MODELS_AVAILABLE:
            start = start_year or 2020
            end = end_year or 2023
            for year in range(start, end + 1):
                record = BaselineDataRecord(
                    property_id=property_id,
                    year=year,
                    scope1_emissions=300.0 - (year - 2020) * 10,
                    scope2_emissions=500.0 - (year - 2020) * 15,
                    scope3_emissions=200.0 - (year - 2020) * 5,
                    electricity_kwh=100000.0,
                    natural_gas_kwh=50000.0
                )
                mock_records.append(record)
        
        return ServiceResult.success(mock_records)
    
    @measure_time
    @transaction
    def save_baseline_data(self, records: List[Dict[str, Any]],
                           created_by: str = None) -> ServiceResult[int]:
        """
        Save baseline emission records.
        
        Args:
            records: List of baseline data records
            created_by: User saving the data
            
        Returns:
            ServiceResult containing count of saved records
        """
        if not records:
            return ServiceResult.validation_error(["No records provided"])
        
        return self._execute(self._save_baseline_data_impl, records, created_by)
    
    def _save_baseline_data_impl(self, records: List[Dict[str, Any]],
                                created_by: str) -> ServiceResult[int]:
        """Implementation of save_baseline_data"""
        saved_count = 0
        errors = []
        
        for record_data in records:
            try:
                if self._emission_repo and MODELS_AVAILABLE:
                    record = BaselineDataRecord(**record_data)
                    self._emission_repo.create_baseline_record(record)
                saved_count += 1
            except Exception as e:
                errors.append(f"Failed to save record: {str(e)}")
        
        if errors:
            return ServiceResult(
                status=ServiceResultStatus.PARTIAL,
                data=saved_count,
                errors=errors,
                message=f"Saved {saved_count}/{len(records)} records"
            )
        
        return ServiceResult.success(
            data=saved_count,
            message=f"Saved {saved_count} baseline records"
        )
    
    # =========================================================================
    # VALIDATION AND ANALYSIS
    # =========================================================================
    
    @measure_time
    def validate_property_data(self, property_data: Dict[str, Any]) -> ServiceResult[Dict[str, Any]]:
        """
        Validate property data without saving.
        
        Args:
            property_data: Property data to validate
            
        Returns:
            ServiceResult with validation results
        """
        errors = self._validate_property_data(property_data)
        warnings = self._get_property_warnings(property_data)
        
        if errors:
            return ServiceResult(
                status=ServiceResultStatus.VALIDATION_ERROR,
                data={"valid": False, "errors": errors, "warnings": warnings},
                errors=errors
            )
        
        return ServiceResult.success({
            "valid": True,
            "errors": [],
            "warnings": warnings
        })
    
    def _get_property_warnings(self, data: Dict[str, Any]) -> List[str]:
        """Get warnings for property data"""
        warnings = []
        
        # Check for missing optional but recommended fields
        recommended = ['retrofit_potential', 'year_built', 'occupancy_rate']
        for field in recommended:
            if field not in data or data[field] is None:
                warnings.append(f"Recommended field '{field}' is not provided")
        
        # Check for potentially incorrect values
        if 'area_sqm' in data and data['area_sqm'] < 100:
            warnings.append("Area seems unusually small (< 100 sqm)")
        
        if 'baseline_emission' in data and 'area_sqm' in data:
            intensity = data['baseline_emission'] / data['area_sqm']
            if intensity > 500:
                warnings.append(f"Carbon intensity ({intensity:.1f} kg/sqm) is very high")
            elif intensity < 10:
                warnings.append(f"Carbon intensity ({intensity:.1f} kg/sqm) is very low")
        
        return warnings
    
    @measure_time
    def calculate_property_metrics(self, property_id: str) -> ServiceResult[PropertyMetrics]:
        """
        Calculate comprehensive metrics for a property.
        
        Args:
            property_id: Property ID
            
        Returns:
            ServiceResult containing property metrics
        """
        # Get property
        property_result = self.get_property(property_id)
        if not property_result.is_success:
            return property_result
        
        prop = property_result.data
        
        # Get baseline data
        baseline_result = self.get_baseline_data(property_id)
        baseline_data = baseline_result.data if baseline_result.is_success else []
        
        # Calculate metrics
        if MODELS_AVAILABLE and hasattr(prop, 'calculate_emission_intensity'):
            metrics = PropertyMetrics(
                property_id=property_id,
                emission_intensity=prop.calculate_emission_intensity(),
                energy_intensity=prop.calculate_energy_intensity() if hasattr(prop, 'calculate_energy_intensity') else 0.0,
                total_emission=prop.calculate_total_emission() if hasattr(prop, 'calculate_total_emission') else prop.baseline_emission,
                benchmark_percentile=self._calculate_benchmark_percentile(prop),
                year_over_year_change=self._calculate_yoy_change(baseline_data)
            )
        else:
            metrics = {
                "property_id": property_id,
                "emission_intensity": 200.0,
                "energy_intensity": 150.0,
                "total_emission": 1000.0,
                "benchmark_percentile": 65.0,
                "year_over_year_change": -5.0
            }
        
        return ServiceResult.success(metrics)
    
    def _calculate_benchmark_percentile(self, prop: Property) -> float:
        """Calculate benchmark percentile for property"""
        # Simplified calculation - in production, compare against industry data
        if hasattr(prop, 'carbon_intensity'):
            intensity = prop.carbon_intensity
            # Lower intensity = better percentile
            if intensity < 100:
                return 90.0
            elif intensity < 200:
                return 70.0
            elif intensity < 300:
                return 50.0
            else:
                return 30.0
        return 50.0
    
    def _calculate_yoy_change(self, baseline_data: List) -> float:
        """Calculate year-over-year emission change"""
        if not baseline_data or len(baseline_data) < 2:
            return 0.0
        
        # Sort by year
        sorted_data = sorted(baseline_data, key=lambda x: x.year if hasattr(x, 'year') else 0)
        
        if len(sorted_data) >= 2:
            latest = sorted_data[-1]
            previous = sorted_data[-2]
            
            if hasattr(latest, 'calculate_total_emissions') and hasattr(previous, 'calculate_total_emissions'):
                current_total = latest.calculate_total_emissions()
                previous_total = previous.calculate_total_emissions()
            else:
                current_total = getattr(latest, 'scope1_emissions', 0) + getattr(latest, 'scope2_emissions', 0)
                previous_total = getattr(previous, 'scope1_emissions', 0) + getattr(previous, 'scope2_emissions', 0)
            
            if previous_total > 0:
                return ((current_total - previous_total) / previous_total) * 100
        
        return 0.0
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    @measure_time
    @transaction
    def batch_update_properties(self, updates: List[PropertyUpdateRequest]) -> ServiceResult[Dict[str, Any]]:
        """
        Batch update multiple properties.
        
        Args:
            updates: List of update requests
            
        Returns:
            ServiceResult with update summary
        """
        if not updates:
            return ServiceResult.validation_error(["No updates provided"])
        
        results = {"successful": [], "failed": []}
        
        for update_req in updates:
            result = self.update_property(
                update_req.property_id,
                update_req.updates,
                update_req.updated_by
            )
            
            if result.is_success:
                results["successful"].append(update_req.property_id)
            else:
                results["failed"].append({
                    "property_id": update_req.property_id,
                    "errors": result.errors
                })
        
        status = ServiceResultStatus.SUCCESS if not results["failed"] else ServiceResultStatus.PARTIAL
        
        return ServiceResult(
            status=status,
            data=results,
            message=f"Updated {len(results['successful'])}/{len(updates)} properties"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PropertyService',
    'PropertySummary',
    'PortfolioSummary',
    'PropertyUpdateRequest'
]
