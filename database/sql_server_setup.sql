-- ================================================================================
-- EcoAssist SQL Server Database Setup Script
-- Version: 1.0
-- Compatible with: SQL Server 2019+
-- ================================================================================

-- Create the database
USE master;
GO

IF EXISTS (SELECT name FROM sys.databases WHERE name = 'EcoAssistDB')
BEGIN
    DROP DATABASE EcoAssistDB;
END
GO

CREATE DATABASE EcoAssistDB
ON (
    NAME = 'EcoAssistDB',
    FILENAME = 'C:\Database\EcoAssistDB.mdf',
    SIZE = 100MB,
    MAXSIZE = 10GB,
    FILEGROWTH = 10MB
)
LOG ON (
    NAME = 'EcoAssistDB_Log',
    FILENAME = 'C:\Database\EcoAssistDB_Log.ldf',
    SIZE = 10MB,
    MAXSIZE = 1GB,
    FILEGROWTH = 10%
);
GO

USE EcoAssistDB;
GO

-- ================================================================================
-- CORE DATA TABLES (4 tables)
-- ================================================================================

-- Table 1: properties - Property Master Data
CREATE TABLE properties (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    property_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(500) NULL,
    city VARCHAR(100) NULL,
    state VARCHAR(50) NULL,
    postal_code VARCHAR(20) NULL,
    country VARCHAR(100) DEFAULT 'Australia',
    area_sqm DECIMAL(10,2) NOT NULL CHECK (area_sqm > 0),
    gross_floor_area DECIMAL(10,2) NULL,
    net_lettable_area DECIMAL(10,2) NULL,
    number_of_floors INT DEFAULT 1,
    year_built INT CHECK (year_built >= 1900 AND year_built <= YEAR(GETDATE())),
    year_renovated INT NULL,
    occupancy_rate DECIMAL(5,4) CHECK (occupancy_rate >= 0 AND occupancy_rate <= 1),
    operating_hours_per_day DECIMAL(4,2) DEFAULT 8.0,
    operating_days_per_week INT DEFAULT 5,
    peak_occupancy INT NULL,
    building_type VARCHAR(50) NOT NULL,
    retrofit_potential VARCHAR(20) NOT NULL,
    building_certification VARCHAR(100) NULL,
    energy_star_rating DECIMAL(3,1) NULL,
    nabers_rating DECIMAL(3,1) NULL,
    baseline_emission DECIMAL(12,2) NOT NULL CHECK (baseline_emission >= 0),
    scope1_emission DECIMAL(12,2) DEFAULT 0,
    scope2_emission DECIMAL(12,2) DEFAULT 0,
    scope3_emission DECIMAL(12,2) DEFAULT 0,
    carbon_intensity DECIMAL(8,4) NULL,
    annual_energy_cost DECIMAL(12,2) DEFAULT 0,
    maintenance_cost DECIMAL(12,2) DEFAULT 0,
    insurance_value DECIMAL(15,2) NULL,
    owner VARCHAR(255) NULL,
    manager VARCHAR(255) NULL,
    tenant VARCHAR(255) NULL,
    lease_expiry DATETIME2 NULL,
    portfolio_id VARCHAR(50) NOT NULL,
    business_unit VARCHAR(100) NULL,
    region VARCHAR(100) NULL,
    tags NVARCHAR(MAX) NULL,
    notes NVARCHAR(MAX) NULL,
    is_active BIT DEFAULT 1,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_properties_property_id ON properties(property_id);
CREATE INDEX IX_properties_building_type ON properties(building_type);
CREATE INDEX IX_properties_portfolio_region ON properties(portfolio_id, region);

-- Table 2: reduction_options - CO2 Reduction Options
CREATE TABLE reduction_options (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    option_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description NVARCHAR(1000) NULL,
    category VARCHAR(100) NOT NULL,
    co2_reduction_potential DECIMAL(10,2) NOT NULL CHECK (co2_reduction_potential >= 0),
    co2_reduction_percentage DECIMAL(5,2) NULL,
    energy_savings_kwh DECIMAL(12,2) DEFAULT 0,
    energy_savings_percentage DECIMAL(5,2) NULL,
    capex DECIMAL(15,2) NOT NULL CHECK (capex >= 0),
    opex DECIMAL(12,2) DEFAULT 0,
    maintenance_cost DECIMAL(12,2) DEFAULT 0,
    financing_cost DECIMAL(12,2) DEFAULT 0,
    payback_period_years DECIMAL(6,2) NULL,
    net_present_value DECIMAL(15,2) NULL,
    internal_rate_of_return DECIMAL(8,4) NULL,
    cost_per_tonne_co2 DECIMAL(10,2) NULL,
    priority INT CHECK (priority >= 1 AND priority <= 5),
    implementation_time_months INT DEFAULT 6,
    implementation_complexity VARCHAR(50) DEFAULT 'Medium',
    risk_level VARCHAR(20) NOT NULL,
    technology_type VARCHAR(100) NULL,
    vendor VARCHAR(255) NULL,
    warranty_years INT DEFAULT 5,
    expected_lifetime_years INT DEFAULT 15,
    maintenance_frequency VARCHAR(50) DEFAULT 'Annual',
    minimum_building_size DECIMAL(10,2) DEFAULT 0,
    suitable_building_types NVARCHAR(500) NULL,
    climate_zones NVARCHAR(200) NULL,
    prerequisites NVARCHAR(1000) NULL,
    constraints NVARCHAR(1000) NULL,
    installation_date DATETIME2 NULL,
    commissioning_date DATETIME2 NULL,
    actual_performance NVARCHAR(MAX) NULL,
    performance_variance DECIMAL(8,4) DEFAULT 0,
    regulatory_requirements NVARCHAR(1000) NULL,
    compliance_standards NVARCHAR(500) NULL,
    incentives_available NVARCHAR(500) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_reduction_options_option_id ON reduction_options(option_id);
CREATE INDEX IX_reduction_options_category ON reduction_options(category);
CREATE INDEX IX_reduction_options_priority_risk ON reduction_options(priority, risk_level);

-- Table 3: strategic_patterns - Strategic Patterns
CREATE TABLE strategic_patterns (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    pattern_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description NVARCHAR(1000) NULL,
    pattern_type VARCHAR(50) DEFAULT 'Standard',
    reduction_options NVARCHAR(MAX) NOT NULL,
    implementation_approach VARCHAR(100) DEFAULT 'Sequential',
    implementation_timeline NVARCHAR(1000) NULL,
    estimated_cost DECIMAL(15,2) NOT NULL CHECK (estimated_cost >= 0),
    estimated_capex DECIMAL(15,2) DEFAULT 0,
    estimated_opex DECIMAL(12,2) DEFAULT 0,
    estimated_reduction DECIMAL(12,2) NOT NULL,
    estimated_energy_savings DECIMAL(12,2) DEFAULT 0,
    risk_level VARCHAR(20) NOT NULL,
    risk_factors NVARCHAR(1000) NULL,
    mitigation_strategies NVARCHAR(1000) NULL,
    suitable_building_types NVARCHAR(500) NULL,
    minimum_portfolio_size INT DEFAULT 1,
    geographic_applicability NVARCHAR(300) NULL,
    success_criteria NVARCHAR(1000) NULL,
    kpis NVARCHAR(1000) NULL,
    benchmarks NVARCHAR(1000) NULL,
    required_expertise NVARCHAR(500) NULL,
    recommended_vendors NVARCHAR(500) NULL,
    training_requirements NVARCHAR(500) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_strategic_patterns_pattern_id ON strategic_patterns(pattern_id);
CREATE INDEX IX_strategic_patterns_type_risk ON strategic_patterns(pattern_type, risk_level);

-- Table 4: benchmark_data - Benchmark Data
CREATE TABLE benchmark_data (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    benchmark_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description NVARCHAR(1000) NULL,
    benchmark_type VARCHAR(50) NOT NULL,
    building_type VARCHAR(50) NOT NULL,
    region VARCHAR(100) NULL,
    data_source VARCHAR(255) NOT NULL,
    methodology NVARCHAR(1000) NULL,
    baseline_year INT NOT NULL,
    intensity_baseline DECIMAL(8,4) NOT NULL,
    reduction_potential DECIMAL(5,4) NOT NULL,
    energy_intensity_kwh_sqm DECIMAL(8,2) NULL,
    cost_intensity_aud_sqm DECIMAL(8,2) NULL,
    percentile_25 DECIMAL(8,4) NULL,
    percentile_50 DECIMAL(8,4) NULL,
    percentile_75 DECIMAL(8,4) NULL,
    percentile_90 DECIMAL(8,4) NULL,
    sample_size INT NULL,
    confidence_level DECIMAL(5,2) DEFAULT 95.0,
    effective_date DATETIME2 NOT NULL,
    expiry_date DATETIME2 NULL,
    certification_weight DECIMAL(5,4) DEFAULT 1.0,
    climate_adjustment DECIMAL(5,4) DEFAULT 1.0,
    occupancy_adjustment DECIMAL(5,4) DEFAULT 1.0,
    validation_status VARCHAR(20) DEFAULT 'Draft',
    notes NVARCHAR(MAX) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_benchmark_data_benchmark_id ON benchmark_data(benchmark_id);
CREATE INDEX IX_benchmark_data_type_building ON benchmark_data(benchmark_type, building_type);
CREATE INDEX IX_benchmark_data_region_year ON benchmark_data(region, baseline_year);

-- ================================================================================
-- HISTORICAL DATA TABLES (3 tables)
-- ================================================================================

-- Table 5: historical_consumption - Historical Consumption Data
CREATE TABLE historical_consumption (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    property_id VARCHAR(50) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    consumption_period VARCHAR(10) NOT NULL,
    consumption_amount DECIMAL(15,4) NOT NULL CHECK (consumption_amount >= 0),
    consumption_unit VARCHAR(20) NOT NULL,
    normalized_consumption DECIMAL(12,4) NULL,
    unit_cost DECIMAL(10,4) DEFAULT 0,
    total_cost DECIMAL(12,2) DEFAULT 0,
    demand_charges DECIMAL(10,2) DEFAULT 0,
    emission_factor DECIMAL(8,6) NULL,
    total_emissions DECIMAL(12,4) NULL,
    emission_scope VARCHAR(20) NULL,
    data_source VARCHAR(100) NOT NULL,
    data_quality_score DECIMAL(5,2) DEFAULT 100.0,
    estimated_data BIT DEFAULT 0,
    heating_degree_days DECIMAL(8,2) NULL,
    cooling_degree_days DECIMAL(8,2) NULL,
    weather_normalized BIT DEFAULT 0,
    meter_reading_type VARCHAR(20) DEFAULT 'Actual',
    billing_start_date DATE NULL,
    billing_end_date DATE NULL,
    days_in_period INT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_historical_consumption_property FOREIGN KEY (property_id) REFERENCES properties(property_id)
);

CREATE INDEX IX_historical_consumption_property_period ON historical_consumption(property_id, consumption_period);
CREATE INDEX IX_historical_consumption_fuel_type ON historical_consumption(fuel_type);

-- Table 6: historical_emissions - Historical Emissions Data
CREATE TABLE historical_emissions (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    property_id VARCHAR(50) NOT NULL,
    emission_period VARCHAR(10) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    emission_scope VARCHAR(20) NOT NULL,
    emissions_tco2e DECIMAL(12,4) NOT NULL CHECK (emissions_tco2e >= 0),
    emission_factor DECIMAL(8,6) NOT NULL,
    emission_factor_source VARCHAR(255) NULL,
    consumption_amount DECIMAL(15,4) NULL,
    consumption_unit VARCHAR(20) NULL,
    calculation_method VARCHAR(100) DEFAULT 'Direct',
    uncertainty_percentage DECIMAL(5,2) NULL,
    biogenic_emissions DECIMAL(12,4) DEFAULT 0,
    market_based_emissions DECIMAL(12,4) NULL,
    location_based_emissions DECIMAL(12,4) NULL,
    offset_credits_applied DECIMAL(12,4) DEFAULT 0,
    net_emissions DECIMAL(12,4) NULL,
    verification_status VARCHAR(20) DEFAULT 'Unverified',
    data_source VARCHAR(100) NOT NULL,
    data_quality_score DECIMAL(5,2) DEFAULT 100.0,
    reporting_boundary VARCHAR(100) NULL,
    baseline_year INT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_historical_emissions_property FOREIGN KEY (property_id) REFERENCES properties(property_id)
);

CREATE INDEX IX_historical_emissions_property_period ON historical_emissions(property_id, emission_period);
CREATE INDEX IX_historical_emissions_scope_fuel ON historical_emissions(emission_scope, fuel_type);

-- Table 7: historical_costs - Historical Cost Data
CREATE TABLE historical_costs (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    property_id VARCHAR(50) NOT NULL,
    cost_period VARCHAR(10) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    cost_category VARCHAR(50) NOT NULL,
    fuel_cost DECIMAL(12,2) DEFAULT 0,
    delivery_cost DECIMAL(10,2) DEFAULT 0,
    network_cost DECIMAL(10,2) DEFAULT 0,
    carbon_tax DECIMAL(10,2) DEFAULT 0,
    green_certificates DECIMAL(10,2) DEFAULT 0,
    demand_charges DECIMAL(10,2) DEFAULT 0,
    connection_fees DECIMAL(10,2) DEFAULT 0,
    other_charges DECIMAL(10,2) DEFAULT 0,
    total_cost DECIMAL(12,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'AUD',
    unit_cost DECIMAL(10,4) NULL,
    consumption_amount DECIMAL(15,4) NULL,
    consumption_unit VARCHAR(20) NULL,
    inflation_adjusted BIT DEFAULT 0,
    base_inflation_year INT NULL,
    tax_rate_percentage DECIMAL(5,2) NULL,
    discount_applied DECIMAL(10,2) DEFAULT 0,
    budget_variance DECIMAL(12,2) NULL,
    payment_date DATE NULL,
    invoice_number VARCHAR(100) NULL,
    supplier VARCHAR(255) NULL,
    contract_reference VARCHAR(100) NULL,
    data_source VARCHAR(100) NOT NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_historical_costs_property FOREIGN KEY (property_id) REFERENCES properties(property_id)
);

CREATE INDEX IX_historical_costs_property_period ON historical_costs(property_id, cost_period);
CREATE INDEX IX_historical_costs_fuel_category ON historical_costs(fuel_type, cost_category);

-- ================================================================================
-- AI CALCULATION RESULTS TABLES (5 tables)
-- ================================================================================

-- Table 8: milestone_scenarios - AI-generated Milestone Scenarios
CREATE TABLE milestone_scenarios (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    scenario_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description NVARCHAR(1000) NULL,
    scenario_type VARCHAR(50) DEFAULT 'Standard',
    target_year INT NOT NULL,
    baseline_year INT DEFAULT 2025,
    yearly_targets NVARCHAR(MAX) NOT NULL,
    yearly_percentage_targets NVARCHAR(MAX) NULL,
    scope1_targets NVARCHAR(MAX) NULL,
    scope2_targets NVARCHAR(MAX) NULL,
    scope3_targets NVARCHAR(MAX) NULL,
    total_capex DECIMAL(15,2) NOT NULL,
    total_opex DECIMAL(15,2) NOT NULL,
    annual_capex NVARCHAR(MAX) NULL,
    annual_opex NVARCHAR(MAX) NULL,
    reduction_rate_2030 DECIMAL(5,2) NOT NULL,
    reduction_rate_2050 DECIMAL(5,2) NOT NULL,
    strategy_type VARCHAR(100) DEFAULT 'Balanced',
    key_milestones NVARCHAR(MAX) NULL,
    critical_path_items NVARCHAR(MAX) NULL,
    risk_factors NVARCHAR(MAX) NULL,
    monitoring_frequency VARCHAR(50) DEFAULT 'Quarterly',
    verification_method VARCHAR(100) DEFAULT 'Third-party',
    reporting_schedule NVARCHAR(500) NULL,
    approval_status VARCHAR(20) DEFAULT 'Draft',
    approved_by VARCHAR(255) NULL,
    approval_date DATETIME2 NULL,
    assumptions NVARCHAR(MAX) NULL,
    dependencies NVARCHAR(MAX) NULL,
    ai_confidence_score DECIMAL(5,2) NULL,
    calculation_version VARCHAR(20) DEFAULT '1.0',
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_milestone_scenarios_scenario_id ON milestone_scenarios(scenario_id);
CREATE INDEX IX_milestone_scenarios_target_year ON milestone_scenarios(target_year);
CREATE INDEX IX_milestone_scenarios_approval_status ON milestone_scenarios(approval_status);

-- Table 9: property_targets - Property-level Target Allocation Results
CREATE TABLE property_targets (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    target_id VARCHAR(50) NOT NULL UNIQUE,
    property_id VARCHAR(50) NOT NULL,
    scenario_id VARCHAR(50) NOT NULL,
    target_year INT NOT NULL,
    baseline_emission DECIMAL(12,2) NOT NULL,
    target_emission DECIMAL(12,2) NOT NULL,
    reduction_amount DECIMAL(12,2) NOT NULL,
    reduction_percentage DECIMAL(5,2) NOT NULL,
    allocation_method VARCHAR(100) NOT NULL,
    allocation_weight DECIMAL(8,6) NOT NULL,
    carbon_intensity_baseline DECIMAL(8,4) NULL,
    carbon_intensity_target DECIMAL(8,4) NULL,
    scope1_target DECIMAL(12,2) NULL,
    scope2_target DECIMAL(12,2) NULL,
    scope3_target DECIMAL(12,2) NULL,
    interim_targets NVARCHAR(MAX) NULL,
    investment_allocation DECIMAL(15,2) NULL,
    implementation_priority INT DEFAULT 3,
    feasibility_score DECIMAL(5,2) NULL,
    cost_effectiveness DECIMAL(10,2) NULL,
    technical_readiness VARCHAR(20) NULL,
    regulatory_constraints NVARCHAR(500) NULL,
    success_criteria NVARCHAR(1000) NULL,
    monitoring_plan NVARCHAR(1000) NULL,
    verification_requirements NVARCHAR(500) NULL,
    ai_confidence_score DECIMAL(5,2) NULL,
    calculation_timestamp DATETIME2 NOT NULL,
    status VARCHAR(20) DEFAULT 'Draft',
    approved_by VARCHAR(255) NULL,
    approval_date DATETIME2 NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_property_targets_property FOREIGN KEY (property_id) REFERENCES properties(property_id),
    CONSTRAINT FK_property_targets_scenario FOREIGN KEY (scenario_id) REFERENCES milestone_scenarios(scenario_id)
);

CREATE INDEX IX_property_targets_target_id ON property_targets(target_id);
CREATE INDEX IX_property_targets_property_year ON property_targets(property_id, target_year);
CREATE INDEX IX_property_targets_scenario_id ON property_targets(scenario_id);

-- Table 10: long_term_plans - Long-term Strategic Plans
CREATE TABLE long_term_plans (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    plan_id VARCHAR(50) NOT NULL UNIQUE,
    property_id VARCHAR(50) NOT NULL,
    scenario_id VARCHAR(50) NULL,
    plan_name VARCHAR(255) NOT NULL,
    strategic_pattern VARCHAR(255) NOT NULL,
    planning_horizon VARCHAR(20) NOT NULL,
    implementation_phases NVARCHAR(MAX) NOT NULL,
    yearly_emission_targets NVARCHAR(MAX) NOT NULL,
    yearly_consumption_targets NVARCHAR(MAX) NULL,
    yearly_cost_targets NVARCHAR(MAX) NULL,
    fuel_type_breakdown NVARCHAR(MAX) NULL,
    reduction_options_selected NVARCHAR(MAX) NOT NULL,
    implementation_timeline NVARCHAR(MAX) NOT NULL,
    total_investment_required DECIMAL(15,2) NOT NULL,
    capex_schedule NVARCHAR(MAX) NULL,
    opex_schedule NVARCHAR(MAX) NULL,
    expected_roi DECIMAL(8,4) NULL,
    payback_period_years DECIMAL(6,2) NULL,
    risk_assessment NVARCHAR(1000) NULL,
    mitigation_strategies NVARCHAR(1000) NULL,
    key_performance_indicators NVARCHAR(1000) NULL,
    success_metrics NVARCHAR(1000) NULL,
    monitoring_schedule NVARCHAR(500) NULL,
    review_frequency VARCHAR(50) DEFAULT 'Quarterly',
    stakeholder_requirements NVARCHAR(1000) NULL,
    regulatory_compliance NVARCHAR(500) NULL,
    technology_dependencies NVARCHAR(500) NULL,
    external_factors NVARCHAR(500) NULL,
    contingency_plans NVARCHAR(1000) NULL,
    ai_optimization_score DECIMAL(5,2) NULL,
    algorithm_version VARCHAR(20) DEFAULT '1.0',
    calculation_timestamp DATETIME2 NOT NULL,
    status VARCHAR(20) DEFAULT 'Draft',
    approved_by VARCHAR(255) NULL,
    approval_date DATETIME2 NULL,
    implementation_start_date DATETIME2 NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_long_term_plans_property FOREIGN KEY (property_id) REFERENCES properties(property_id)
);

CREATE INDEX IX_long_term_plans_plan_id ON long_term_plans(plan_id);
CREATE INDEX IX_long_term_plans_property_id ON long_term_plans(property_id);
CREATE INDEX IX_long_term_plans_status ON long_term_plans(status);

-- Table 11: annual_reoptimization - Annual Plan Reoptimization Results
CREATE TABLE annual_reoptimization (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    reoptimization_id VARCHAR(50) NOT NULL UNIQUE,
    property_id VARCHAR(50) NOT NULL,
    plan_id VARCHAR(50) NULL,
    analysis_period VARCHAR(20) NOT NULL,
    deviation_threshold DECIMAL(5,2) NOT NULL,
    actual_performance_data NVARCHAR(MAX) NOT NULL,
    target_performance_data NVARCHAR(MAX) NOT NULL,
    deviation_analysis NVARCHAR(MAX) NOT NULL,
    emission_variance DECIMAL(8,4) NOT NULL,
    cost_variance DECIMAL(8,4) NOT NULL,
    performance_trend VARCHAR(50) NOT NULL,
    reoptimization_required BIT NOT NULL,
    root_cause_analysis NVARCHAR(MAX) NULL,
    contributing_factors NVARCHAR(1000) NULL,
    external_influences NVARCHAR(1000) NULL,
    updated_targets NVARCHAR(MAX) NULL,
    revised_timeline NVARCHAR(MAX) NULL,
    additional_measures NVARCHAR(1000) NULL,
    budget_adjustments NVARCHAR(1000) NULL,
    recommendations NVARCHAR(MAX) NOT NULL,
    priority_actions NVARCHAR(1000) NULL,
    implementation_urgency VARCHAR(20) NOT NULL,
    expected_improvement DECIMAL(8,4) NULL,
    confidence_level DECIMAL(5,2) NULL,
    monitoring_adjustments NVARCHAR(500) NULL,
    next_review_date DATETIME2 NULL,
    stakeholder_notifications NVARCHAR(500) NULL,
    approval_workflow NVARCHAR(500) NULL,
    ai_analysis_version VARCHAR(20) DEFAULT '1.0',
    calculation_timestamp DATETIME2 NOT NULL,
    status VARCHAR(20) DEFAULT 'Pending',
    approved_by VARCHAR(255) NULL,
    approval_date DATETIME2 NULL,
    implementation_date DATETIME2 NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_annual_reoptimization_property FOREIGN KEY (property_id) REFERENCES properties(property_id)
);

CREATE INDEX IX_annual_reoptimization_reopt_id ON annual_reoptimization(reoptimization_id);
CREATE INDEX IX_annual_reoptimization_property_period ON annual_reoptimization(property_id, analysis_period);
CREATE INDEX IX_annual_reoptimization_status ON annual_reoptimization(status);

-- Table 12: dashboard_metrics - KPIs for Dashboard Visualization
CREATE TABLE dashboard_metrics (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    metric_id VARCHAR(50) NOT NULL UNIQUE,
    metric_name VARCHAR(255) NOT NULL,
    metric_category VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(50) NULL,
    entity_type VARCHAR(50) NULL,
    reporting_period VARCHAR(20) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    target_value DECIMAL(15,4) NULL,
    baseline_value DECIMAL(15,4) NULL,
    variance_from_target DECIMAL(8,4) NULL,
    trend_indicator VARCHAR(20) NULL,
    performance_rating VARCHAR(20) NULL,
    benchmark_comparison DECIMAL(8,4) NULL,
    confidence_level DECIMAL(5,2) DEFAULT 95.0,
    data_quality_score DECIMAL(5,2) DEFAULT 100.0,
    calculation_method VARCHAR(255) NOT NULL,
    data_sources NVARCHAR(500) NULL,
    assumptions NVARCHAR(1000) NULL,
    limitations NVARCHAR(500) NULL,
    metadata NVARCHAR(MAX) NULL,
    visualization_config NVARCHAR(1000) NULL,
    alert_thresholds NVARCHAR(300) NULL,
    notification_settings NVARCHAR(300) NULL,
    access_permissions NVARCHAR(200) NULL,
    refresh_frequency VARCHAR(50) DEFAULT 'Daily',
    last_calculated DATETIME2 NOT NULL,
    next_calculation DATETIME2 NULL,
    calculation_status VARCHAR(20) DEFAULT 'Current',
    version VARCHAR(10) DEFAULT '1.0',
    archived BIT DEFAULT 0,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_dashboard_metrics_metric_id ON dashboard_metrics(metric_id);
CREATE INDEX IX_dashboard_metrics_category_period ON dashboard_metrics(metric_category, reporting_period);
CREATE INDEX IX_dashboard_metrics_entity_type ON dashboard_metrics(entity_type, entity_id);
CREATE INDEX IX_dashboard_metrics_reporting_period ON dashboard_metrics(reporting_period);


CREATE TABLE carbon_credit_prices (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    credit_type VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    price_per_tonne DECIMAL(10,2) NOT NULL CHECK (price_per_tonne >= 0),
    currency VARCHAR(10) DEFAULT 'AUD',
    market_region VARCHAR(100) NOT NULL,
    vintage_year INT NULL,
    certification_standard VARCHAR(100) NULL,
    project_type VARCHAR(100) NULL,
    trading_volume DECIMAL(15,2) NULL,
    price_volatility DECIMAL(5,2) NULL,
    co_benefits NVARCHAR(500) NULL,
    data_source VARCHAR(255) NOT NULL,
    data_quality_score DECIMAL(5,2) DEFAULT 100.0,
    notes NVARCHAR(MAX) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_carbon_credit_prices_year_type ON carbon_credit_prices(year, credit_type);
CREATE INDEX IX_carbon_credit_prices_region ON carbon_credit_prices(market_region);

-- Table 18: renewable_energy_prices - Renewable Energy Pricing Data
CREATE TABLE renewable_energy_prices (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    energy_type VARCHAR(50) NOT NULL,
    year INT NOT NULL,
    price_per_mwh DECIMAL(10,2) NULL,
    price_per_kw_installed DECIMAL(10,2) NULL,
    currency VARCHAR(10) DEFAULT 'AUD',
    region VARCHAR(100) NOT NULL,
    technology_specification NVARCHAR(500) NULL,
    capacity_range VARCHAR(50) NULL,
    capex DECIMAL(12,2) NULL,
    opex_annual DECIMAL(10,2) NULL,
    capacity_factor DECIMAL(5,2) NULL,
    expected_lifetime_years INT DEFAULT 25,
    degradation_rate DECIMAL(5,4) NULL,
    incentive_stc_value DECIMAL(10,2) NULL,
    incentive_lgc_value DECIMAL(10,2) NULL,
    feed_in_tariff DECIMAL(8,4) NULL,
    installation_cost DECIMAL(10,2) NULL,
    grid_connection_cost DECIMAL(10,2) NULL,
    data_source VARCHAR(255) NOT NULL,
    confidence_level DECIMAL(5,2) DEFAULT 95.0,
    notes NVARCHAR(MAX) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_renewable_energy_prices_year_type ON renewable_energy_prices(year, energy_type);
CREATE INDEX IX_renewable_energy_prices_region ON renewable_energy_prices(region);

-- Table 19: renewable_fuel_prices - Renewable Fuel Pricing Data
CREATE TABLE renewable_fuel_prices (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    fuel_type VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    price_per_unit DECIMAL(10,4) NOT NULL CHECK (price_per_unit >= 0),
    unit_type VARCHAR(20) NOT NULL,
    currency VARCHAR(10) DEFAULT 'AUD',
    region VARCHAR(100) NOT NULL,
    feedstock_type VARCHAR(100) NULL,
    production_pathway VARCHAR(100) NULL,
    energy_content_mj DECIMAL(10,2) NULL,
    emission_factor DECIMAL(10,6) NULL,
    carbon_intensity_score DECIMAL(10,2) NULL,
    blend_ratio VARCHAR(50) NULL,
    purity_grade VARCHAR(50) NULL,
    fossil_fuel_price_equivalent DECIMAL(10,4) NULL,
    price_premium_percentage DECIMAL(5,2) NULL,
    sustainability_certification VARCHAR(100) NULL,
    lcfs_credit_value DECIMAL(10,2) NULL,
    feedstock_availability VARCHAR(50) NULL,
    storage_requirements NVARCHAR(500) NULL,
    equipment_compatibility NVARCHAR(500) NULL,
    data_source VARCHAR(255) NOT NULL,
    data_quality_score DECIMAL(5,2) DEFAULT 100.0,
    notes NVARCHAR(MAX) NULL,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX IX_renewable_fuel_prices_year_type ON renewable_fuel_prices(year, fuel_type);
CREATE INDEX IX_renewable_fuel_prices_region_feedstock ON renewable_fuel_prices(region, feedstock_type);

-- Triggers for automatic timestamp updates
CREATE TRIGGER tr_carbon_credit_prices_updated
ON carbon_credit_prices
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE carbon_credit_prices 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;
GO

CREATE TRIGGER tr_renewable_energy_prices_updated
ON renewable_energy_prices
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE renewable_energy_prices 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;
GO

CREATE TRIGGER tr_renewable_fuel_prices_updated
ON renewable_fuel_prices
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE renewable_fuel_prices 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;
GO

-- ================================================================================
-- VIEWS FOR COMMON QUERIES
-- ================================================================================

-- Portfolio Summary View
CREATE VIEW vw_portfolio_summary AS
SELECT
    p.portfolio_id,
    COUNT(*) as total_properties,
    SUM(p.area_sqm) as total_area_sqm,
    SUM(p.baseline_emission) as total_baseline_emission,
    AVG(p.carbon_intensity) as avg_carbon_intensity,
    SUM(p.annual_energy_cost) as total_energy_cost
FROM properties p
WHERE p.is_active = 1
GROUP BY p.portfolio_id;

-- Current Performance View
CREATE VIEW vw_current_performance AS
SELECT
    p.property_id,
    p.name,
    p.building_type,
    YEAR(GETDATE()) as current_year,
    COALESCE(SUM(he.emissions_tco2e), 0) as ytd_emissions,
    COALESCE(SUM(hc.total_cost), 0) as ytd_costs,
    p.baseline_emission,
    ((p.baseline_emission - COALESCE(SUM(he.emissions_tco2e), 0)) / p.baseline_emission) * 100 as reduction_achieved
FROM properties p
LEFT JOIN historical_emissions he ON p.property_id = he.property_id
    AND he.emission_period LIKE CONCAT(YEAR(GETDATE()), '-%')
LEFT JOIN historical_costs hc ON p.property_id = hc.property_id
    AND hc.cost_period LIKE CONCAT(YEAR(GETDATE()), '-%')
WHERE p.is_active = 1
GROUP BY p.property_id, p.name, p.building_type, p.baseline_emission;

-- ================================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ================================================================================

-- Create triggers for updated_at timestamp
CREATE TRIGGER tr_properties_updated
ON properties
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE properties 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;

CREATE TRIGGER tr_reduction_options_updated
ON reduction_options
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE reduction_options 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;

CREATE TRIGGER tr_strategic_patterns_updated
ON strategic_patterns
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE strategic_patterns 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;

-- Add similar triggers for other tables
CREATE TRIGGER tr_milestone_scenarios_updated
ON milestone_scenarios
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE milestone_scenarios 
    SET updated_at = GETDATE()
    WHERE id IN (SELECT id FROM inserted);
END;

-- ================================================================================
-- DATA VALIDATION TRIGGERS
-- ================================================================================

-- Validate emission scope values
CREATE TRIGGER tr_validate_emission_scope
ON historical_emissions
AFTER INSERT, UPDATE
AS
BEGIN
    IF EXISTS (SELECT 1 FROM inserted WHERE emission_scope NOT IN ('Scope 1', 'Scope 2', 'Scope 3'))
    BEGIN
        RAISERROR('Invalid emission scope. Must be Scope 1, 2, or 3.', 16, 1);
        ROLLBACK TRANSACTION;
    END
END;

PRINT 'Database schema created successfully!';
GO