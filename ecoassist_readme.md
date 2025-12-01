# EcoAssist - AI-Powered Carbon Emission Management System

## Overview

EcoAssist is an enterprise-grade carbon emission management system designed to help organizations plan, track, and optimize their carbon reduction strategies across property portfolios. Version 2.0 introduces a complete architectural redesign with layered architecture, comprehensive API support, and enhanced AI capabilities.

**Architecture**: Layered (API → Services → Models → Database)

## What's New in v2.0

| Feature | v1.x | v2.0 |
|---------|------|------|
| Architecture | Monolithic | Layered (API/Services/Models/Database) |
| API | Embedded in frontend | RESTful FastAPI with OpenAPI docs |
| Database | Basic integration | SQL Server with repositories |
| AI Functions | Standalone | Integrated with models layer |
| Type Safety | Partial | Full type hints & validation |
| Error Handling | Basic | ServiceResult pattern |
| Caching | None | Built-in TTL cache |
| Testing | Manual | Mock mode for all services |

## Project Structure

```
EcoAssist/
├── api/                              # API Layer (FastAPI)
│   ├── main_api_application.py       # Main API entry point, health checks
│   ├── api_module1_milestones.py     # Milestone planning APIs (APIs 1-4)
│   ├── api_module2_target_division.py # Target allocation APIs (APIs 5-7)
│   ├── api_module3_planning.py       # Long-term planning APIs (APIs 8-10) [Planned]
│   └── api_module4_reoptimization.py # Reoptimization APIs (APIs 11-12) [Planned]
│
├── services/                         # Services Layer (Business Logic)
│   ├── __init__.py                   # Package exports & ServiceFactory
│   ├── base_service.py               # Base class, caching, decorators
│   ├── property_service.py           # Property & portfolio management
│   ├── milestone_service.py          # Milestone calculation & scenarios
│   ├── allocation_service.py         # Target allocation algorithms
│   ├── tracking_service.py           # Progress monitoring & alerts
│   ├── ai_service.py                 # AI/ML model management
│   ├── visualization_service.py      # Charts, reports, dashboards
│   └── README.md                     # Services documentation
│
├── models/                           # Models Layer (Data Structures)
│   ├── __init__.py                   # Package exports
│   ├── enums.py                      # All enumerations (30+ enums)
│   ├── base.py                       # Base classes & utilities
│   ├── property.py                   # Property & Portfolio models
│   ├── emission.py                   # Emission & Baseline models
│   ├── milestone.py                  # Milestone & Scenario models
│   ├── reduction.py                  # Reduction options & strategies
│   ├── cost.py                       # Cost & financial models
│   ├── repository.py                 # Data access layer
│   └── README.md                     # Models documentation
│
├── database/                         # Database Layer
│   ├── config.py                     # Database configuration
│   ├── manager.py                    # Connection pooling & transactions
│   └── sql_server_setup.sql          # Database schema
│
├── ai_functions.py                   # AI/ML Algorithms (v3.0)
├── backend_adapter.py                # Integration adapter
├── ecoassist_frontend.py             # Gradio web interface
│
├── docs/                             # Documentation
│   ├── API_SPECIFICATION.md          # Complete API documentation
│   ├── BU_QA_RESPONSE.md            # Business unit Q&A
│   └── ARCHITECTURE.md               # Architecture details
│
└── tests/                            # Test files
    └── test_models.py                # Model layer tests
```

## Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                               │
│              (Gradio UI / External Applications)                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Module 1   │ │  Module 2   │ │  Module 3   │ │  Module 4   │   │
│  │ Milestones  │ │ Allocation  │ │  Planning   │ │  Reopt      │   │
│  │ (APIs 1-4)  │ │ (APIs 5-7)  │ │ (APIs 8-10) │ │ (APIs 11-12)│   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Method calls
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVICES LAYER                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │PropertyService│ │MilestoneService│ │AllocationService│            │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │TrackingService│ │  AIService   │ │VisualizationService│          │
│  └──────────────┘ └──────────────┘ └──────────────┘                │
│                     ↓ ServiceResult<T> pattern                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Repository pattern
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODELS LAYER                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │Property │ │Emission │ │Milestone│ │Reduction│ │  Cost   │      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│                     ↓ Repositories                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ SQL queries
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATABASE LAYER                                │
│                    (SQL Server / PostgreSQL)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Request → API Endpoint → Service → Repository → Database
                              ↓
                         AI Functions (optional)
                              ↓
Response ← API Response ← ServiceResult ← Model
```

## Key Features

### 1. Milestone Planning (Module 1)
- **API 1**: Calculate baseline emissions from property data
- **API 2**: Generate AI-powered reduction scenarios (Standard, Aggressive, Conservative)
- **API 3**: Retrieve and compare milestone scenarios
- **API 4**: Calculate yearly reduction targets with S-curve interpolation

### 2. Target Allocation (Module 2)
- **API 5**: Allocate portfolio targets across properties
- **API 6**: Property-level target division with multiple algorithms
- **API 7**: Calculate reduction rates by property type

**Allocation Methods**:
- `PROPORTIONAL` - Based on baseline emission
- `INTENSITY_WEIGHTED` - Based on carbon intensity
- `RETROFIT_POTENTIAL` - Based on improvement potential
- `AI_OPTIMIZED` - ML-based optimization
- `EQUAL` - Equal distribution

### 3. Long-Term Planning (Module 3)
- **API 8**: Generate strategic patterns (Renewable, Efficiency, Electrification)
- **API 9**: Cost projection with CAPEX/OPEX breakdown
- **API 10**: Risk assessment and mitigation planning

### 4. Annual Reoptimization (Module 4)
- **API 11**: Track progress against targets
- **API 12**: Deviation analysis and reoptimization recommendations

## Installation & Setup

### Prerequisites

- Python 3.10+
- SQL Server 2019+ (or PostgreSQL 14+)
- ODBC Driver 17 for SQL Server

### Required Python Packages

```bash
# Core dependencies
pip install fastapi uvicorn pydantic

# Database
pip install pyodbc sqlalchemy

# AI/ML
pip install numpy pandas scikit-learn scipy

# Visualization
pip install gradio plotly matplotlib seaborn

# Utilities
pip install python-dateutil typing-extensions
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Database Setup

#### 1. Create Database

```sql
-- Execute database/sql_server_setup.sql in SQL Server Management Studio
CREATE DATABASE EcoAssistDB;
GO
USE EcoAssistDB;
-- ... rest of schema
```

#### 2. Configure Connection

Edit `database/config.py`:

```python
class DatabaseConfig:
    SERVER = "localhost"
    DATABASE = "EcoAssistDB"
    USERNAME = "sa"
    PASSWORD = "your-password"
    DRIVER = "ODBC Driver 17 for SQL Server"
```

Or use environment variables:

```bash
export ECOASSIST_DB_SERVER=localhost
export ECOASSIST_DB_NAME=EcoAssistDB
export ECOASSIST_DB_USER=sa
export ECOASSIST_DB_PASSWORD=your-password
```

#### 3. Verify Connection

```bash
python -c "from database.manager import DatabaseManager; dm = DatabaseManager(); print(dm.test_connection())"
```

## Quick Start

### Option 1: Run API Server

```bash
# Start the API server
uvicorn api.main_api_application:app --host 0.0.0.0 --port 8000 --reload

# API documentation available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Option 2: Run Gradio Frontend

```bash
python ecoassist_frontend.py
# Opens at http://localhost:7860
```

### Option 3: Use Services Directly

```python
from services import ServiceFactory, MilestoneCalculationRequest

# Create services
factory = ServiceFactory(db_manager=None)  # None for mock mode
services = factory.create_all()

# Calculate milestones
request = MilestoneCalculationRequest(
    property_ids=["PROP-001", "PROP-002"],
    baseline_emission=10000.0,
    reduction_2030=40.0,
    reduction_2050=90.0
)

result = services['milestone'].calculate_milestones(request)

if result.is_success:
    for scenario in result.data.scenarios:
        print(f"{scenario['name']}: {scenario['reduction_2030']}% by 2030")
```

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
Currently using API key authentication (header: `X-API-Key`)

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/milestones/baseline` | Calculate baseline emissions |
| POST | `/milestones/scenarios/generate` | Generate reduction scenarios |
| GET | `/milestones/scenarios/{id}` | Get scenario details |
| POST | `/milestones/targets/calculate` | Calculate yearly targets |
| POST | `/allocation/portfolio` | Allocate portfolio targets |
| POST | `/allocation/property` | Property-level allocation |
| GET | `/allocation/rates/{property_id}` | Get reduction rates |
| POST | `/planning/strategies` | Generate strategic patterns |
| POST | `/planning/costs` | Calculate cost projections |
| POST | `/tracking/progress` | Track progress |
| POST | `/tracking/reoptimize` | Trigger reoptimization |

### Example API Calls

#### Generate Scenarios
```bash
curl -X POST "http://localhost:8000/api/v1/milestones/scenarios/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "property_ids": ["PROP-001", "PROP-002"],
    "baseline_emission": 10000,
    "base_year": 2024,
    "target_years": {"mid_term": 2030, "long_term": 2050},
    "reduction_targets": {"target_2030": 40, "target_2050": 90}
  }'
```

#### Allocate Targets
```bash
curl -X POST "http://localhost:8000/api/v1/allocation/portfolio" \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "SCN-001",
    "property_ids": ["PROP-001", "PROP-002", "PROP-003"],
    "total_reduction_target": 5000,
    "allocation_method": "AI_OPTIMIZED"
  }'
```

## Services Reference

### ServiceResult Pattern

All services return `ServiceResult<T>`:

```python
@dataclass
class ServiceResult(Generic[T]):
    status: ServiceResultStatus  # SUCCESS, ERROR, PARTIAL, NOT_FOUND, VALIDATION_ERROR
    data: Optional[T]
    message: str
    errors: List[str]
    warnings: List[str]
    execution_time_ms: float

# Usage
result = service.operation(data)
if result.is_success:
    process(result.data)
else:
    handle_error(result.errors)
```

### Available Services

| Service | Purpose | Key Methods |
|---------|---------|-------------|
| `PropertyService` | Property management | `get_property()`, `get_portfolio_metrics()` |
| `MilestoneService` | Scenario generation | `calculate_milestones()`, `compare_scenarios()` |
| `AllocationService` | Target allocation | `allocate_targets()`, `validate_allocation()` |
| `TrackingService` | Progress monitoring | `get_progress_summary()`, `analyze_deviations()` |
| `AIService` | AI model management | `predict()`, `train_model()`, `optimize()` |
| `VisualizationService` | Charts & reports | `generate_milestone_chart()`, `create_report()` |

### Built-in Decorators

```python
from services import measure_time, cached, retry, transaction

class MyService(BaseService):
    @measure_time  # Tracks execution time
    @cached(ttl_seconds=300)  # Caches for 5 minutes
    def get_data(self, id):
        ...
    
    @retry(max_attempts=3, delay_seconds=1.0)  # Retries on failure
    @transaction  # Wraps in database transaction
    def save_data(self, data):
        ...
```

## AI Functions Reference

### Available Algorithms

| Algorithm | Purpose | Key Methods |
|-----------|---------|-------------|
| `MilestoneOptimizer` | Scenario optimization | `train()`, `predict()`, `generate_scenario_recommendations()` |
| `PropertyTargetAllocator` | Allocation optimization | `allocate_targets()`, `optimize()` |
| `StrategicPatternAnalyzer` | Pattern analysis | `analyze_strategy_pattern()` |
| `ReoptimizationEngine` | Performance analysis | `analyze_performance()`, `generate_adjusted_targets()` |

### Usage Example

```python
from ai_functions import MilestoneOptimizer, create_all_ai_components

# Create optimizer
optimizer = MilestoneOptimizer()

# Train with historical data
X = np.array([[baseline, area, intensity, age, potential], ...])
y = np.array([reduction_achieved, ...])
metrics = optimizer.train(X, y)
print(f"Model R²: {metrics.r2:.4f}")

# Generate recommendations
recommendations = optimizer.generate_scenario_recommendations(properties_df)
for rec in recommendations:
    print(f"{rec.scenario_name}: {rec.ai_confidence}% confidence")
```

## Models Reference

### Core Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `Property` | Building/facility | `property_id`, `baseline_emission`, `area_sqm`, `building_type` |
| `MilestoneScenario` | Reduction scenario | `scenario_id`, `targets[]`, `reduction_target_2030/2050` |
| `MilestoneTarget` | Yearly target | `year`, `target_emission`, `reduction_from_baseline` |
| `ReductionOption` | Reduction measure | `option_id`, `reduction_potential`, `capex`, `opex` |
| `CostProjection` | Financial projection | `year`, `capex`, `opex`, `cumulative_cost` |

### Enumerations

```python
from models import (
    BuildingType,      # OFFICE, RETAIL, INDUSTRIAL, RESIDENTIAL, ...
    RetrofitPotential, # HIGH, MEDIUM, LOW
    ScenarioType,      # STANDARD, AGGRESSIVE, CONSERVATIVE, AI_OPTIMIZED
    AllocationMethod,  # PROPORTIONAL, INTENSITY_WEIGHTED, AI_OPTIMIZED, ...
    OnTrackStatus,     # AHEAD, ON_TRACK, AT_RISK, OFF_TRACK, CRITICAL
    RiskLevel,         # LOW, MEDIUM, HIGH, CRITICAL
)
```

## Configuration

### Environment Variables

```bash
# Database
ECOASSIST_DB_SERVER=localhost
ECOASSIST_DB_NAME=EcoAssistDB
ECOASSIST_DB_USER=sa
ECOASSIST_DB_PASSWORD=password

# API
ECOASSIST_API_HOST=0.0.0.0
ECOASSIST_API_PORT=8000
ECOASSIST_API_KEY=your-api-key

# AI
ECOASSIST_AI_MODEL_PATH=/path/to/models
ECOASSIST_AI_CACHE_TTL=300

# Logging
ECOASSIST_LOG_LEVEL=INFO
```

### Default Scenario Configurations

```python
DEFAULT_SCENARIOS = {
    "STANDARD": {
        "reduction_2030": 40.0,
        "reduction_2050": 90.0,
        "cost_multiplier": 1.0,
        "risk_factor": 0.3,
        "description": "Balanced approach aligned with SBTi"
    },
    "AGGRESSIVE": {
        "reduction_2030": 50.0,
        "reduction_2050": 95.0,
        "cost_multiplier": 1.4,
        "risk_factor": 0.5,
        "description": "Accelerated decarbonization"
    },
    "CONSERVATIVE": {
        "reduction_2030": 30.0,
        "reduction_2050": 80.0,
        "cost_multiplier": 0.7,
        "risk_factor": 0.2,
        "description": "Lower risk gradual implementation"
    }
}
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Mock Mode

All services support mock mode for testing without database:

```python
# Services work without database connection
from services import PropertyService

service = PropertyService(db_manager=None)  # Mock mode
service.initialize()

result = service.get_property("PROP-001")
assert result.is_success  # Works with mock data
```

## Troubleshooting

### Common Issues

#### Database Connection Failed
```
Error: Cannot connect to SQL Server
```
**Solution**: 
1. Verify SQL Server is running
2. Check ODBC driver installation: `odbcinst -q -d`
3. Test connection string manually
4. Check firewall settings

#### Import Errors
```
ModuleNotFoundError: No module named 'models'
```
**Solution**:
1. Ensure you're in the project root directory
2. Add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:/path/to/EcoAssist`
3. Or use: `python -m api.main_api_application`

#### AI Functions Not Available
```
Warning: AI functions not available - using mock mode
```
**Solution**: This is normal if scikit-learn is not installed. Install with:
```bash
pip install scikit-learn scipy
```

### Performance Optimization

1. **Enable caching**: Services use built-in caching (default 5 min TTL)
2. **Use pagination**: API endpoints support `page` and `page_size` parameters
3. **Batch operations**: Use batch endpoints for multiple items
4. **Database indexing**: Ensure indexes are created per schema

## Code Statistics

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| Models Layer | 9 | ~3,500 |
| Services Layer | 8 | ~6,400 |
| API Layer | 4 | ~4,100 |
| AI Functions | 1 | ~1,750 |
| **Total** | **22** | **~15,750** |

## Roadmap

### v2.1 (Next Release)
- [ ] Complete Module 3 & 4 API extraction
- [ ] Redis caching support
- [ ] Email notifications
- [ ] Audit logging

### v2.2
- [ ] GraphQL API option
- [ ] Real-time WebSocket updates
- [ ] Multi-tenant support
- [ ] Advanced reporting

### v3.0
- [ ] Kubernetes deployment
- [ ] Microservices architecture
- [ ] Event-driven processing
- [ ] ML model versioning

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Follow code standards (PEP 8, type hints required)
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

## License

Proprietary software. All rights reserved.

## Support

For technical support and feature requests:
- Email: support@ecoassist.dev
- Documentation: https://docs.ecoassist.dev
- Issue Tracker: https://github.com/ecoassist/issues

---

**EcoAssist v2.0** - Empowering organizations to achieve their carbon reduction goals through AI-powered planning and optimization.
