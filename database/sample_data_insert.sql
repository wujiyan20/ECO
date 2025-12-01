-- ================================================================================
-- EcoAssist Sample Data Insertion Scripts
-- Version: 1.0
-- Description: Insert sample data for prototype verification and testing
-- ================================================================================

USE EcoAssistDB;
GO

-- ================================================================================
-- SAMPLE DATA: CORE DATA TABLES
-- ================================================================================

-- Insert Sample Properties (9 properties as specified in documentation)
INSERT INTO properties (
    property_id, name, address, city, state, postal_code, area_sqm, net_lettable_area,
    year_built, occupancy_rate, building_type, retrofit_potential, baseline_emission,
    scope1_emission, scope2_emission, carbon_intensity, annual_energy_cost,
    portfolio_id, business_unit, region
) VALUES
('BP01', 'Brisbane Plaza', '123 Queen Street', 'Brisbane', 'QLD', '4000', 1500.00, 1400.00, 
 2010, 0.95, 'Office', 'High', 150000.00, 45000.00, 105000.00, 100.00, 185000.00, 
 'PORTFOLIO_A', 'Commercial Real Estate', 'QLD'),

('CB01', 'City Building 01', '456 Collins Street', 'Melbourne', 'VIC', '3000', 2200.00, 2000.00,
 2015, 0.88, 'Mixed Use', 'Medium', 185000.00, 55500.00, 129500.00, 84.09, 225000.00,
 'PORTFOLIO_A', 'Commercial Real Estate', 'VIC'),

('CB02', 'City Building 02', '789 Bourke Street', 'Melbourne', 'VIC', '3001', 1800.00, 1650.00,
 2008, 0.92, 'Retail', 'High', 120000.00, 36000.00, 84000.00, 66.67, 145000.00,
 'PORTFOLIO_A', 'Retail', 'VIC'),

('CB03', 'City Building 03', '321 Flinders Street', 'Melbourne', 'VIC', '3002', 2500.00, 2300.00,
 2012, 0.90, 'Office', 'Medium', 200000.00, 60000.00, 140000.00, 80.00, 240000.00,
 'PORTFOLIO_B', 'Commercial Real Estate', 'VIC'),

('CB04', 'City Building 04', '654 Russell Street', 'Melbourne', 'VIC', '3003', 1600.00, 1450.00,
 2005, 0.85, 'Industrial', 'Critical', 180000.00, 108000.00, 72000.00, 112.50, 195000.00,
 'PORTFOLIO_B', 'Industrial', 'VIC'),

('HA1', 'Harbour Apartments', '987 Circular Quay', 'Sydney', 'NSW', '2000', 3200.00, 2800.00,
 2018, 0.94, 'Residential', 'Low', 95000.00, 19000.00, 76000.00, 29.69, 115000.00,
 'PORTFOLIO_C', 'Residential', 'NSW'),

('MP', 'Melbourne Plaza', '147 Spencer Street', 'Melbourne', 'VIC', '3008', 2800.00, 2600.00,
 2009, 0.91, 'Mixed Use', 'High', 220000.00, 66000.00, 154000.00, 78.57, 285000.00,
 'PORTFOLIO_A', 'Mixed Use', 'VIC'),

('TC01', 'Technology Campus 01', '258 University Avenue', 'Adelaide', 'SA', '5000', 1900.00, 1750.00,
 2020, 0.96, 'Educational', 'Medium', 85000.00, 17000.00, 68000.00, 44.74, 105000.00,
 'PORTFOLIO_C', 'Educational', 'SA'),

('WH1', 'Warehouse 01', '369 Industrial Drive', 'Perth', 'WA', '6000', 4500.00, 4200.00,
 2014, 0.87, 'Warehouse', 'High', 160000.00, 96000.00, 64000.00, 35.56, 175000.00,
 'PORTFOLIO_B', 'Logistics', 'WA');

-- Insert Sample Reduction Options (13 options as specified)
INSERT INTO reduction_options (
    option_id, name, description, category, co2_reduction_potential, co2_reduction_percentage,
    energy_savings_kwh, capex, opex, priority, implementation_time_months, 
    implementation_complexity, risk_level, technology_type
) VALUES
('SOLAR_PV_001', 'Solar PV Installation', 'Rooftop solar photovoltaic system installation', 'Renewable Energy', 
 4500.00, 15.0, 180000, 200000.00, 5000.00, 1, 6, 'Medium', 'Low', 'Solar Technology'),

('LED_UPG_001', 'LED Lighting Upgrade', 'Complete LED lighting retrofit with smart controls', 'Energy Efficiency',
 1200.00, 8.0, 95000, 45000.00, 2000.00, 2, 3, 'Low', 'Low', 'Lighting Technology'),

('HVAC_UPG_001', 'HVAC System Upgrade', 'High-efficiency HVAC system with variable frequency drives', 'Energy Efficiency',
 3500.00, 20.0, 145000, 150000.00, 8000.00, 1, 8, 'High', 'Medium', 'HVAC Technology'),

('INSUL_001', 'Building Insulation', 'Comprehensive building envelope insulation upgrade', 'Energy Efficiency',
 2800.00, 12.0, 120000, 80000.00, 1500.00, 2, 4, 'Medium', 'Low', 'Building Envelope'),

('SMART_BMS_001', 'Smart Building Management', 'IoT-enabled building management and automation system', 'Smart Technology',
 2200.00, 10.0, 85000, 120000.00, 12000.00, 3, 10, 'High', 'Medium', 'IoT/Automation'),

('CARBON_CRED_001', 'Carbon Credits Purchase', 'Verified carbon offset credits from renewable projects', 'Carbon Offsetting',
 5000.00, 0.0, 0, 0.00, 125000.00, 4, 1, 'Low', 'Low', 'Carbon Credits'),

('BIOFUEL_001', 'Biofuel Replacement', 'Replace fossil fuels with sustainable biofuel alternatives', 'Fuel Switching',
 3200.00, 18.0, 0, 75000.00, 25000.00, 2, 6, 'Medium', 'Medium', 'Alternative Fuels'),

('WIND_001', 'Wind Energy System', 'Small-scale wind turbine installation', 'Renewable Energy',
 2800.00, 12.0, 115000, 180000.00, 8000.00, 3, 8, 'High', 'High', 'Wind Technology'),

('HEAT_PUMP_001', 'Heat Pump Installation', 'High-efficiency air source heat pump system', 'Energy Efficiency',
 4200.00, 25.0, 165000, 95000.00, 6000.00, 1, 5, 'Medium', 'Low', 'Heat Pump Technology'),

('ENERGY_STOR_001', 'Energy Storage System', 'Battery storage system for renewable energy integration', 'Energy Storage',
 1800.00, 8.0, 75000, 250000.00, 15000.00, 3, 7, 'High', 'Medium', 'Battery Technology'),

('EV_FLEET_001', 'Electric Vehicle Fleet', 'Transition fleet vehicles to electric with charging infrastructure', 'Transportation',
 3600.00, 0.0, 0, 320000.00, 35000.00, 2, 12, 'High', 'Medium', 'Electric Vehicles'),

('HYDROGEN_001', 'Hydrogen Fuel System', 'Hydrogen fuel cell system for backup power and heating', 'Alternative Energy',
 5500.00, 22.0, 0, 450000.00, 45000.00, 4, 18, 'Critical', 'High', 'Hydrogen Technology'),

('GREEN_ROOF_001', 'Green Roof Installation', 'Living roof system with vegetation and insulation benefits', 'Passive Design',
 800.00, 3.0, 25000, 65000.00, 8000.00, 4, 4, 'Medium', 'Low', 'Green Infrastructure');

-- Insert Sample Strategic Patterns (5 patterns as specified)
INSERT INTO strategic_patterns (
    pattern_id, name, description, pattern_type, reduction_options, implementation_approach,
    estimated_cost, estimated_capex, estimated_opex, estimated_reduction, risk_level
) VALUES
('ACTIVE_RE_001', 'Active Installation of RE', 'Aggressive renewable energy deployment with solar and wind systems',
 'Renewable Focus', '["SOLAR_PV_001", "WIND_001", "ENERGY_STOR_001", "SMART_BMS_001"]', 'Parallel',
 680000.00, 630000.00, 50000.00, 12800.00, 'Medium'),

('GREEN_FUEL_001', 'Replace to Green Fuels', 'Comprehensive fuel switching strategy with biofuels and hydrogen',
 'Fuel Transition', '["BIOFUEL_001", "HYDROGEN_001", "EV_FLEET_001", "HEAT_PUMP_001"]', 'Sequential',
 940000.00, 870000.00, 70000.00, 16500.00, 'High'),

('CARBON_PRIOR_001', 'Carbon Credit Priority', 'Carbon offset-focused approach with efficiency improvements',
 'Offset Strategy', '["CARBON_CRED_001", "LED_UPG_001", "INSUL_001", "GREEN_ROOF_001"]', 'Immediate',
 315000.00, 190000.00, 125000.00, 10000.00, 'Low'),

('LOW_ENERGY_001', 'Low Energy Equipment', 'Energy efficiency and conservation focused strategy',
 'Efficiency Focus', '["LED_UPG_001", "HVAC_UPG_001", "INSUL_001", "SMART_BMS_001"]', 'Sequential',
 395000.00, 275000.00, 120000.00, 9700.00, 'Low'),

('BALANCED_001', 'Balanced Strategy', 'Comprehensive approach combining multiple technologies and strategies',
 'Balanced', '["SOLAR_PV_001", "HVAC_UPG_001", "LED_UPG_001", "HEAT_PUMP_001", "SMART_BMS_001"]', 'Phased',
 610000.00, 490000.00, 120000.00, 15100.00, 'Medium');

-- Insert Sample Benchmark Data
INSERT INTO benchmark_data (
    benchmark_id, name, description, benchmark_type, building_type, region, data_source,
    baseline_year, intensity_baseline, reduction_potential, energy_intensity_kwh_sqm,
    cost_intensity_aud_sqm, percentile_25, percentile_50, percentile_75, percentile_90,
    sample_size, effective_date
) VALUES
('BENCH_OFF_001', 'Office Buildings - Australia', 'National benchmark for office buildings', 'Industry', 'Office', 'Australia',
 'NABERS/Green Building Council', 2024, 85.5, 0.45, 195.0, 125.0, 68.2, 85.5, 112.8, 145.6, 
 1250, '2024-01-01'),

('BENCH_RET_001', 'Retail Buildings - Australia', 'National benchmark for retail buildings', 'Industry', 'Retail', 'Australia',
 'NABERS/Green Building Council', 2024, 92.3, 0.40, 220.0, 135.0, 75.1, 92.3, 118.5, 158.2,
 850, '2024-01-01'),

('BENCH_IND_001', 'Industrial Buildings - Australia', 'National benchmark for industrial buildings', 'Industry', 'Industrial', 'Australia',
 'Australian Industry Group', 2024, 125.8, 0.35, 285.0, 95.0, 98.7, 125.8, 165.2, 212.4,
 650, '2024-01-01');

-- ================================================================================
-- SAMPLE DATA: HISTORICAL DATA TABLES
-- ================================================================================

-- Insert Sample Historical Consumption Data (2023-2024 data for first 3 properties)
INSERT INTO historical_consumption (
    property_id, fuel_type, consumption_period, consumption_amount, consumption_unit,
    unit_cost, total_cost, emission_factor, data_source
) VALUES
-- BP01 - 2023 Data
('BP01', 'Electricity', '2023-01', 45250.00, 'kWh', 0.28, 12670.00, 0.82, 'Energy Retailer'),
('BP01', 'Electricity', '2023-02', 42180.00, 'kWh', 0.28, 11810.40, 0.82, 'Energy Retailer'),
('BP01', 'Natural Gas', '2023-01', 850.00, 'GJ', 12.50, 10625.00, 51.4, 'Gas Provider'),
('BP01', 'Natural Gas', '2023-02', 780.00, 'GJ', 12.50, 9750.00, 51.4, 'Gas Provider'),

-- CB01 - 2023 Data
('CB01', 'Electricity', '2023-01', 58500.00, 'kWh', 0.29, 16965.00, 0.82, 'Energy Retailer'),
('CB01', 'Electricity', '2023-02', 55200.00, 'kWh', 0.29, 16008.00, 0.82, 'Energy Retailer'),
('CB01', 'Natural Gas', '2023-01', 1250.00, 'GJ', 13.20, 16500.00, 51.4, 'Gas Provider'),
('CB01', 'Natural Gas', '2023-02', 1180.00, 'GJ', 13.20, 15576.00, 51.4, 'Gas Provider'),

-- CB02 - 2023 Data
('CB02', 'Electricity', '2023-01', 38750.00, 'kWh', 0.27, 10462.50, 0.82, 'Energy Retailer'),
('CB02', 'Electricity', '2023-02', 35800.00, 'kWh', 0.27, 9666.00, 0.82, 'Energy Retailer'),
('CB02', 'Natural Gas', '2023-01', 650.00, 'GJ', 11.80, 7670.00, 51.4, 'Gas Provider'),
('CB02', 'Natural Gas', '2023-02', 590.00, 'GJ', 11.80, 6962.00, 51.4, 'Gas Provider');

-- Insert Sample Historical Emissions Data
INSERT INTO historical_emissions (
    property_id, emission_period, fuel_type, emission_scope, emissions_tco2e, 
    emission_factor, consumption_amount, consumption_unit, data_source
) VALUES
-- BP01 - 2023 Emissions
('BP01', '2023-01', 'Electricity', 'Scope 2', 37.105, 0.82, 45250.00, 'kWh', 'Calculated'),
('BP01', '2023-01', 'Natural Gas', 'Scope 1', 43.69, 51.4, 850.00, 'GJ', 'Calculated'),
('BP01', '2023-02', 'Electricity', 'Scope 2', 34.587, 0.82, 42180.00, 'kWh', 'Calculated'),
('BP01', '2023-02', 'Natural Gas', 'Scope 1', 40.092, 51.4, 780.00, 'GJ', 'Calculated'),

-- CB01 - 2023 Emissions
('CB01', '2023-01', 'Electricity', 'Scope 2', 47.97, 0.82, 58500.00, 'kWh', 'Calculated'),
('CB01', '2023-01', 'Natural Gas', 'Scope 1', 64.25, 51.4, 1250.00, 'GJ', 'Calculated'),
('CB01', '2023-02', 'Electricity', 'Scope 2', 45.264, 0.82, 55200.00, 'kWh', 'Calculated'),
('CB01', '2023-02', 'Natural Gas', 'Scope 1', 60.652, 51.4, 1180.00, 'GJ', 'Calculated');

-- Insert Sample Historical Costs Data
INSERT INTO historical_costs (
    property_id, cost_period, fuel_type, cost_category, fuel_cost, delivery_cost,
    network_cost, carbon_tax, total_cost, currency, data_source
) VALUES
-- BP01 - 2023 Costs
('BP01', '2023-01', 'Electricity', 'Energy', 10500.00, 850.00, 1200.00, 120.00, 12670.00, 'AUD', 'Energy Bill'),
('BP01', '2023-01', 'Natural Gas', 'Energy', 9200.00, 650.00, 500.00, 275.00, 10625.00, 'AUD', 'Gas Bill'),
('BP01', '2023-02', 'Electricity', 'Energy', 9750.00, 820.00, 1120.00, 120.40, 11810.40, 'AUD', 'Energy Bill'),
('BP01', '2023-02', 'Natural Gas', 'Energy', 8450.00, 620.00, 480.00, 200.00, 9750.00, 'AUD', 'Gas Bill'),

-- CB01 - 2023 Costs
('CB01', '2023-01', 'Electricity', 'Energy', 14200.00, 1250.00, 1350.00, 165.00, 16965.00, 'AUD', 'Energy Bill'),
('CB01', '2023-01', 'Natural Gas', 'Energy', 14200.00, 850.00, 750.00, 700.00, 16500.00, 'AUD', 'Gas Bill'),
('CB01', '2023-02', 'Electricity', 'Energy', 13450.00, 1180.00, 1210.00, 168.00, 16008.00, 'AUD', 'Energy Bill'),
('CB01', '2023-02', 'Natural Gas', 'Energy', 13450.00, 800.00, 720.00, 606.00, 15576.00, 'AUD', 'Gas Bill');

-- ================================================================================
-- SAMPLE DATA: AI CALCULATION RESULTS
-- ================================================================================

-- Insert Sample Milestone Scenarios
INSERT INTO milestone_scenarios (
    scenario_id, name, description, target_year, yearly_targets, total_capex, total_opex,
    reduction_rate_2030, reduction_rate_2050, strategy_type
) VALUES
('MS_CONS_001', 'Conservative Approach', 'Gradual emission reduction with proven technologies and manageable investment levels',
 2050, '{"2025": 1450000, "2030": 1160000, "2040": 725000, "2050": 290000}', 2500000.00, 850000.00, 
 20.0, 80.0, 'conservative'),

('MS_AGG_001', 'Aggressive Decarbonisation', 'Front-loaded reduction with advanced technologies and higher initial investment',
 2050, '{"2025": 1450000, "2030": 1015000, "2040": 580000, "2050": 290000}', 3750000.00, 1200000.00,
 30.0, 80.0, 'aggressive'),

('MS_BAL_001', 'Balanced Strategy', 'Balanced approach combining multiple technologies with moderate risk and investment',
 2050, '{"2025": 1450000, "2030": 1087500, "2040": 652500, "2050": 290000}', 3100000.00, 980000.00,
 25.0, 80.0, 'balanced'),

('MS_TECH_001', 'Technology Focus', 'Emphasis on cutting-edge technologies with higher risk but maximum potential reduction',
 2050, '{"2025": 1450000, "2030": 1015000, "2040": 507500, "2050": 217500}', 4500000.00, 1450000.00,
 30.0, 85.0, 'technology');

-- Insert Sample Property Targets
INSERT INTO property_targets (
    target_id, property_id, scenario_id, target_year, baseline_emission, target_emission,
    reduction_amount, reduction_percentage, allocation_method, allocation_weight, calculation_timestamp
) VALUES
('PT_BP01_001', 'BP01', 'MS_AGG_001', 2030, 150000.00, 82500.00, 67500.00, 45.0, 'carbon_intensity_weighted', 0.1034, GETDATE()),
('PT_CB01_001', 'CB01', 'MS_AGG_001', 2030, 185000.00, 129500.00, 55500.00, 30.0, 'carbon_intensity_weighted', 0.1276, GETDATE()),
('PT_CB02_001', 'CB02', 'MS_AGG_001', 2030, 120000.00, 72000.00, 48000.00, 40.0, 'carbon_intensity_weighted', 0.0828, GETDATE()),
('PT_CB03_001', 'CB03', 'MS_AGG_001', 2030, 200000.00, 120000.00, 80000.00, 40.0, 'carbon_intensity_weighted', 0.1379, GETDATE()),
('PT_CB04_001', 'CB04', 'MS_AGG_001', 2030, 180000.00, 126000.00, 54000.00, 30.0, 'carbon_intensity_weighted', 0.1241, GETDATE());

-- Insert Sample Long-term Plans
INSERT INTO long_term_plans (
    plan_id, property_id, plan_name, strategic_pattern, planning_horizon,
    implementation_phases, yearly_emission_targets, total_investment_required,
    calculation_timestamp
) VALUES
('LTP_BP01_001', 'BP01', 'Brisbane Plaza Decarbonisation Plan', 'Active Installation of RE', '2025-2050',
 '[{"phase": "Planning & Design", "duration": 6, "start": "2025-Q1"}, {"phase": "Implementation", "duration": 24, "start": "2025-Q3"}, {"phase": "Monitoring & Optimization", "duration": 60, "start": "2027-Q3"}]',
 '{"2025": 150000, "2030": 82500, "2035": 60000, "2040": 45000, "2045": 35000, "2050": 30000}',
 680000.00, GETDATE()),

('LTP_CB01_001', 'CB01', 'City Building 01 Sustainability Plan', 'Balanced Strategy', '2025-2050',
 '[{"phase": "Assessment", "duration": 3, "start": "2025-Q1"}, {"phase": "Phase 1 Implementation", "duration": 18, "start": "2025-Q2"}, {"phase": "Phase 2 Implementation", "duration": 24, "start": "2026-Q4"}]',
 '{"2025": 185000, "2030": 129500, "2035": 100000, "2040": 75000, "2045": 55000, "2050": 37000}',
 850000.00, GETDATE());

-- Insert Sample Dashboard Metrics
INSERT INTO dashboard_metrics (
    metric_id, metric_name, metric_category, metric_type, entity_id, entity_type,
    reporting_period, metric_value, metric_unit, target_value, baseline_value,
    calculation_method, last_calculated
) VALUES
('DM_EMIT_001', 'Total Portfolio Emissions', 'Environmental', 'KPI', 'PORTFOLIO_A', 'Portfolio',
 '2024-Q1', 1285000.00, 'tCO2e', 1200000.00, 1450000.00, 'Sum of property emissions', GETDATE()),

('DM_INTENS_001', 'Average Carbon Intensity', 'Environmental', 'KPI', 'PORTFOLIO_A', 'Portfolio',
 '2024-Q1', 78.5, 'kgCO2e/m2', 70.0, 95.2, 'Weighted average by floor area', GETDATE()),

('DM_COST_001', 'Annual Energy Cost', 'Financial', 'KPI', 'PORTFOLIO_A', 'Portfolio',
 '2024-Q1', 2850000.00, 'AUD', 2650000.00, 3200000.00, 'Sum of energy costs', GETDATE()),

('DM_RED_001', 'Emissions Reduction Progress', 'Environmental', 'Target', 'PORTFOLIO_A', 'Portfolio',
 '2024-Q1', 11.4, 'Percent', 15.0, 0.0, 'Percentage reduction from baseline', GETDATE());

PRINT 'Sample data inserted successfully!';
PRINT 'Database setup complete with:';
PRINT '  - 9 Properties';
PRINT '  - 13 Reduction Options';
PRINT '  - 5 Strategic Patterns';
PRINT '  - 3 Benchmark Records';
PRINT '  - Historical data for 2023 (consumption, emissions, costs)';
PRINT '  - 4 Milestone Scenarios';
PRINT '  - 5 Property Target Allocations';
PRINT '  - 2 Long-term Plans';
PRINT '  - 4 Dashboard Metrics';


INSERT INTO carbon_credit_prices (
    credit_type, year, price_per_tonne, currency, market_region, vintage_year,
    certification_standard, project_type, trading_volume, price_volatility,
    co_benefits, data_source, data_quality_score
) VALUES
-- Renewable Energy Certificates
('Renewable Energy Certificate', 2023, 22.50, 'AUD', 'Australia', 2023, 'VCS', 'Renewable Energy', 150000, 8.5, 
 '["Clean Energy", "Grid Decarbonization"]', 'Australian Carbon Market Report', 95.0),
('Renewable Energy Certificate', 2024, 24.00, 'AUD', 'Australia', 2024, 'VCS', 'Renewable Energy', 175000, 9.2,
 '["Clean Energy", "Grid Decarbonization"]', 'Australian Carbon Market Report', 95.0),
('Renewable Energy Certificate', 2025, 25.80, 'AUD', 'Australia', 2025, 'VCS', 'Renewable Energy', 200000, 10.1,
 '["Clean Energy", "Grid Decarbonization"]', 'Market Forecast', 85.0),

-- Forest Carbon Credits
('Forest Carbon Credit', 2023, 28.00, 'AUD', 'Australia', 2022, 'Gold Standard', 'Afforestation', 80000, 12.3,
 '["Biodiversity", "Community Development"]', 'Australian Carbon Market Report', 90.0),
('Forest Carbon Credit', 2024, 30.50, 'AUD', 'Australia', 2023, 'Gold Standard', 'Afforestation', 95000, 13.5,
 '["Biodiversity", "Community Development"]', 'Australian Carbon Market Report', 90.0),
('Forest Carbon Credit', 2025, 33.20, 'AUD', 'Australia', 2024, 'Gold Standard', 'Afforestation', 110000, 14.8,
 '["Biodiversity", "Community Development"]', 'Market Forecast', 80.0),

-- Technology Carbon Credits
('Technology Carbon Credit', 2023, 35.00, 'AUD', 'Global', 2023, 'VCS', 'Carbon Capture', 50000, 15.5,
 '["Innovation", "Industrial Decarbonization"]', 'Global Carbon Market Data', 85.0),
('Technology Carbon Credit', 2024, 38.50, 'AUD', 'Global', 2024, 'VCS', 'Carbon Capture', 60000, 16.8,
 '["Innovation", "Industrial Decarbonization"]', 'Global Carbon Market Data', 85.0),
('Technology Carbon Credit', 2025, 42.00, 'AUD', 'Global', 2025, 'VCS', 'Carbon Capture', 75000, 18.2,
 '["Innovation", "Industrial Decarbonization"]', 'Market Forecast', 75.0);

-- Insert Sample Renewable Energy Prices
INSERT INTO renewable_energy_prices (
    energy_type, year, price_per_mwh, price_per_kw_installed, currency, region,
    technology_specification, capacity_range, capex, opex_annual, capacity_factor,
    expected_lifetime_years, degradation_rate, incentive_stc_value, feed_in_tariff,
    installation_cost, data_source, confidence_level
) VALUES
-- Solar PV
('solar_pv', 2023, 65.00, 1200.00, 'AUD', 'Australia', 
 'Monocrystalline panels, 20% efficiency', 'commercial', 1200.00, 15.00, 22.5, 25, 0.005, 38.00, 0.12, 180.00,
 'Clean Energy Council', 92.0),
('solar_pv', 2024, 58.00, 1050.00, 'AUD', 'Australia',
 'Monocrystalline panels, 21% efficiency', 'commercial', 1050.00, 14.00, 23.0, 25, 0.005, 35.00, 0.10, 160.00,
 'Clean Energy Council', 92.0),
('solar_pv', 2025, 52.00, 920.00, 'AUD', 'Australia',
 'Monocrystalline panels, 22% efficiency', 'commercial', 920.00, 13.00, 23.5, 25, 0.004, 32.00, 0.09, 145.00,
 'Market Forecast', 85.0),

-- Wind Energy
('wind', 2023, 75.00, 1800.00, 'AUD', 'Australia',
 '2.5 MW turbine, onshore', 'utility', 1800.00, 35.00, 35.0, 25, 0.002, NULL, NULL, 250.00,
 'Wind Energy Association', 88.0),
('wind', 2024, 70.00, 1650.00, 'AUD', 'Australia',
 '3.0 MW turbine, onshore', 'utility', 1650.00, 32.00, 36.0, 25, 0.002, NULL, NULL, 230.00,
 'Wind Energy Association', 88.0),
('wind', 2025, 65.00, 1500.00, 'AUD', 'Australia',
 '3.5 MW turbine, onshore', 'utility', 1500.00, 30.00, 37.0, 25, 0.002, NULL, NULL, 210.00,
 'Market Forecast', 82.0),

-- Battery Storage
('battery_storage', 2023, NULL, 800.00, 'AUD', 'Australia',
 'Lithium-ion, 4-hour duration', 'commercial', 800.00, 20.00, NULL, 15, 0.025, NULL, NULL, 120.00,
 'Energy Storage Alliance', 85.0),
('battery_storage', 2024, NULL, 680.00, 'AUD', 'Australia',
 'Lithium-ion, 4-hour duration', 'commercial', 680.00, 18.00, NULL, 15, 0.020, NULL, NULL, 100.00,
 'Energy Storage Alliance', 85.0),
('battery_storage', 2025, NULL, 580.00, 'AUD', 'Australia',
 'Lithium-ion, 4-hour duration', 'commercial', 580.00, 16.00, NULL, 15, 0.018, NULL, NULL, 85.00,
 'Market Forecast', 80.0);

-- Insert Sample Renewable Fuel Prices
INSERT INTO renewable_fuel_prices (
    fuel_type, year, price_per_unit, unit_type, currency, region, feedstock_type,
    production_pathway, energy_content_mj, emission_factor, carbon_intensity_score,
    blend_ratio, fossil_fuel_price_equivalent, price_premium_percentage,
    sustainability_certification, feedstock_availability, data_source, data_quality_score
) VALUES
-- Biodiesel
('biodiesel', 2023, 1.85, 'liter', 'AUD', 'Australia', 'waste_oil', 
 '2nd generation', 33.0, 0.012, 18.5, 'B100', 1.45, 27.6, 'ISCC', 'abundant',
 'Biofuels Australia Report', 90.0),
('biodiesel', 2024, 1.78, 'liter', 'AUD', 'Australia', 'waste_oil',
 '2nd generation', 33.0, 0.011, 17.8, 'B100', 1.48, 20.3, 'ISCC', 'abundant',
 'Biofuels Australia Report', 90.0),
('biodiesel', 2025, 1.72, 'liter', 'AUD', 'Australia', 'waste_oil',
 '2nd generation', 33.0, 0.010, 17.0, 'B100', 1.52, 13.2, 'ISCC', 'abundant',
 'Market Forecast', 85.0),

-- Green Hydrogen
('hydrogen', 2023, 8.50, 'kg', 'AUD', 'Australia', 'electrolysis',
 'green_hydrogen', 120.0, 0.000, 0.0, 'H2_100', 3.20, 165.6, 'CertifHy', 'limited',
 'Hydrogen Industry Report', 82.0),
('hydrogen', 2024, 7.20, 'kg', 'AUD', 'Australia', 'electrolysis',
 'green_hydrogen', 120.0, 0.000, 0.0, 'H2_100', 3.35, 114.9, 'CertifHy', 'growing',
 'Hydrogen Industry Report', 82.0),
('hydrogen', 2025, 6.00, 'kg', 'AUD', 'Australia', 'electrolysis',
 'green_hydrogen', 120.0, 0.000, 0.0, 'H2_100', 3.50, 71.4, 'CertifHy', 'growing',
 'Market Forecast', 75.0),

-- Renewable Natural Gas (Biogas)
('biogas', 2023, 18.50, 'GJ', 'AUD', 'Australia', 'organic_waste',
 'anaerobic_digestion', 38.0, 0.008, 12.5, 'RNG_100', 12.50, 48.0, 'RSB', 'seasonal',
 'Biogas Industry Data', 88.0),
('biogas', 2024, 17.80, 'GJ', 'AUD', 'Australia', 'organic_waste',
 'anaerobic_digestion', 38.0, 0.007, 11.8, 'RNG_100', 13.00, 36.9, 'RSB', 'seasonal',
 'Biogas Industry Data', 88.0),
('biogas', 2025, 17.00, 'GJ', 'AUD', 'Australia', 'organic_waste',
 'anaerobic_digestion', 38.0, 0.006, 11.0, 'RNG_100', 13.50, 25.9, 'RSB', 'improving',
 'Market Forecast', 80.0),

-- Sustainable Aviation Fuel
('sustainable_aviation_fuel', 2023, 2.85, 'liter', 'AUD', 'Global', 'waste_oil',
 'HEFA', 35.5, 0.015, 22.0, 'SAF_50', 1.20, 137.5, 'RSB', 'limited',
 'Aviation Fuel Report', 85.0),
('sustainable_aviation_fuel', 2024, 2.65, 'liter', 'AUD', 'Global', 'waste_oil',
 'HEFA', 35.5, 0.014, 20.5, 'SAF_50', 1.25, 112.0, 'RSB', 'limited',
 'Aviation Fuel Report', 85.0),
('sustainable_aviation_fuel', 2025, 2.45, 'liter', 'AUD', 'Global', 'waste_oil',
 'HEFA', 35.5, 0.013, 19.0, 'SAF_50', 1.30, 88.5, 'RSB', 'growing',
 'Market Forecast', 78.0);

PRINT 'Pricing data inserted successfully!';
PRINT '  - Carbon Credit Prices: 9 records';
PRINT '  - Renewable Energy Prices: 9 records';
PRINT '  - Renewable Fuel Prices: 15 records';

GO