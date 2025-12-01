import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ecoassist_backend import EcoAssistBackend

class EcoAssistFrontend:
    """Frontend interface for EcoAssist AI system"""
    
    def __init__(self):
        self.backend = EcoAssistBackend()
        self.selected_scenario = "Aggressive Decarbonisation"
        self.selected_strategy = "Active Installation of RE"
        self.current_property = "BP01"
    
    def create_milestone_plot(self, scenarios, target_year):
        """Create enhanced milestone visualization"""
        fig = go.Figure()
        
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, scenario in enumerate(scenarios):
            years = list(scenario.yearly_targets.keys())
            emissions = list(scenario.yearly_targets.values())
            
            # Convert to more readable units (thousands)
            emissions_k = [e/1000 for e in emissions]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=emissions_k,
                mode='lines+markers',
                name=f"{scenario.name}",
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6),
                hovertemplate=f"<b>{scenario.name}</b><br>" +
                            "Year: %{x}<br>" +
                            "Emissions: %{y:.0f}k tCO2e<br>" +
                            f"2030 Target: {scenario.reduction_rate_2030}%<br>" +
                            f"2050 Target: {scenario.reduction_rate_2050}%<extra></extra>"
            ))
        
        fig.update_layout(
            title="Milestone Recommendation: CO₂ Emission Reduction Pathways",
            xaxis_title="Year",
            yaxis_title="Total CO₂ Emissions (thousands tCO₂e)",
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_budget_plot(self, budget_data, property_id, target_property):
        """Create comprehensive budget visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'Annual Budget for {property_id} ({target_property.building_type})', 
                'Cumulative Investment Progression',
                'Budget vs Emission Reduction Correlation'
            ),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Annual budget bars
        fig.add_trace(
            go.Bar(x=budget_data['years'], y=budget_data['capex_data'], 
                   name='CAPEX', marker_color='#FF6B6B', opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=budget_data['years'], y=budget_data['opex_data'], 
                   name='OPEX', marker_color='#4ECDC4', opacity=0.8),
            row=1, col=1
        )
        
        # Cumulative budget progression
        fig.add_trace(
            go.Scatter(x=budget_data['years'], y=budget_data['cumulative_total'], 
                      name='Total Cumulative', line=dict(color='#2E8B57', width=4), fill='tonexty'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=budget_data['years'], y=budget_data['cumulative_capex'], 
                      name='CAPEX Cumulative', line=dict(color='#FF6B6B', width=2)),
            row=2, col=1
        )
        
        # Budget efficiency analysis
        fig.add_trace(
            go.Bar(x=budget_data['years'], y=budget_data['total_budgets'], 
                   name='Total Budget', marker_color='#45B7D1', opacity=0.6),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=budget_data['years'], y=budget_data['emission_reduction'], 
                      name='Est. CO₂ Reduction (tCO₂e)', line=dict(color='#00FF00', width=3), 
                      mode='lines+markers'),
            row=3, col=1, secondary_y=True
        )
        
        fig.update_layout(
            height=800,
            title=f"Comprehensive Budget Analysis - {property_id}",
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Budget (AUD $)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Investment (AUD $)", row=2, col=1)
        fig.update_yaxes(title_text="Annual Budget (AUD $)", row=3, col=1)
        fig.update_yaxes(title_text="CO₂ Reduction (tCO₂e)", row=3, col=1, secondary_y=True)
        
        return fig
    
    def create_reoptimization_plot(self, plot_data, property_id):
        """Create the annual plan re-optimization visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Emission Target vs Actual', 
                'Cost Target vs Actual'
            ),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            vertical_spacing=0.1
        )
        
        months = plot_data['months']
        current_month = plot_data['current_month_index']
        
        # Emission chart (top)
        fig.add_trace(
            go.Bar(
                x=months, 
                y=plot_data['total_variance'],
                name='Total variance',
                marker_color='lightgray',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=plot_data['emission_target'],
                mode='lines+markers',
                name='Original target',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=plot_data['emission_actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add refined target if reoptimization occurred
        if plot_data['needs_reoptimization']:
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=plot_data['emission_refined_target'],
                    mode='lines+markers',
                    name='Refined target',
                    line=dict(color='gray', width=2, dash='dash'),
                    marker=dict(size=4)
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Cost chart (bottom)
        fig.add_trace(
            go.Scatter(
                x=months,
                y=plot_data['cost_target'],
                mode='lines+markers',
                name='Original target',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=plot_data['cost_actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add refined cost target if reoptimization occurred
        if plot_data['needs_reoptimization']:
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=plot_data['cost_refined_target'],
                    mode='lines+markers',
                    name='Refined target',
                    line=dict(color='gray', width=2, dash='dash'),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            title=f"2026 annual planning for {property_id}",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Emission (tCO2)", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Total variance (tCO2)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="kAUD", row=2, col=1)
        
        return fig
    
    def run_milestone_recommendation(self, target_year, reduction_2030, reduction_2050, 
                                   custom_name="", custom_2030=None, custom_2050=None):
        """Enhanced milestone recommendation with detailed table display"""
        try:
            print("DEBUG: Starting milestone recommendation...")
            target_year = int(target_year)
            reduction_2030 = float(reduction_2030)
            reduction_2050 = float(reduction_2050)
            print(f"DEBUG: Parameters - Year: {target_year}, 2030: {reduction_2030}%, 2050: {reduction_2050}%")
            
            # Generate standard scenarios
            scenarios = self.backend.generate_milestone_scenarios(target_year, reduction_2030, reduction_2050)
            print(f"DEBUG: Generated {len(scenarios)} scenarios")
            
            # Add custom scenario if provided
            if custom_name and custom_2030 is not None and custom_2050 is not None:
                custom_scenario = self.backend.create_custom_scenario(custom_name, float(custom_2030), float(custom_2050), target_year)
                scenarios.insert(0, custom_scenario)
                print(f"DEBUG: Added custom scenario: {custom_name}")
            
            # Create enhanced visualization
            fig = self.create_milestone_plot(scenarios, target_year)
            print("DEBUG: Created milestone plot")
            
            # Get table data
            table_data = self.backend.get_milestone_table_data()
            print(f"DEBUG: Got table data for {len(table_data)} scenarios")
            
            # Create enhanced HTML display
            scenarios_html = self.create_detailed_scenarios_display(table_data)
            print(f"DEBUG: Created HTML display with length: {len(scenarios_html)}")
            
            return fig, scenarios_html, "Enhanced milestone scenarios generated successfully!"
            
        except Exception as e:
            print(f"DEBUG: Error in milestone recommendation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, "", f"Error: {str(e)}"
    
    def create_detailed_scenarios_display(self, table_data):
        """Create detailed HTML display for milestone scenarios"""
        print("DEBUG: Creating detailed scenarios display...")
        
        html_content = """
        <div style="font-family: Arial, sans-serif; margin: 20px 0; background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
            <h2 style="color: #2c3e50; margin-bottom: 20px; text-align: center;">📊 Milestone Scenario Analysis</h2>
        """
        
        # Action buttons section
        html_content += """
        <div style="margin-bottom: 25px; padding: 15px; background-color: #e8f4fd; border-radius: 8px; border-left: 4px solid #3498db;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">Milestone Target Setting Actions</h4>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background-color: #3498db; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;">Recommendation</span>
                <span style="background-color: #e74c3c; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;">Add Custom Scenario</span>
                <span style="background-color: #27ae60; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;">Register</span>
            </div>
        </div>
        """
        
        scenario_counter = 1
        for scenario_data in table_data:
            scenario_name = scenario_data["scenario_name"]
            rows = scenario_data["rows"]
            
            print(f"DEBUG: Processing scenario {scenario_counter}: {scenario_name}")
            
            # Determine border style for custom scenarios
            is_custom = "Custom" in scenario_name
            border_color = "#e74c3c" if is_custom else "#3498db"
            bg_color = "#fff5f5" if is_custom else "#ffffff"
            
            html_content += f"""
            <div style="margin-bottom: 30px; border: 3px solid {border_color}; border-radius: 10px; padding: 20px; background-color: {bg_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                
                <!-- Scenario Header -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1;">
                    <h3 style="margin: 0; color: #2c3e50; font-size: 24px; font-weight: bold;">
                        {scenario_counter}. {scenario_name}
                    </h3>
                    <div style="background-color: #27ae60; color: white; padding: 12px 25px; border-radius: 6px; font-weight: bold; font-size: 16px;">
                        SELECT
                    </div>
                </div>
                
                <!-- Data Table -->
                <div style="overflow-x: auto; border-radius: 8px; border: 2px solid #bdc3c7;">
                    <table style="width: 100%; border-collapse: collapse; background-color: white; font-size: 14px;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <th style="padding: 15px 10px; text-align: left; font-weight: bold; font-size: 16px;">Item</th>
                                <th style="padding: 15px 10px; text-align: center; font-weight: bold; font-size: 16px;">Base Year (2025)</th>
                                <th style="padding: 15px 10px; text-align: center; font-weight: bold; font-size: 16px;">Target Year (2030)</th>
                                <th style="padding: 15px 10px; text-align: center; font-weight: bold; font-size: 16px;">Reduction Rate</th>
                                <th style="padding: 15px 10px; text-align: center; font-weight: bold; font-size: 16px;">Target Year (2050)</th>
                                <th style="padding: 15px 10px; text-align: center; font-weight: bold; font-size: 16px;">Reduction Rate</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            row_colors = ["#ffffff", "#f8f9fa"]
            for i, row in enumerate(rows):
                row_bg = row_colors[i % 2]
                
                html_content += f"""
                            <tr style="background-color: {row_bg};">
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; font-weight: bold; color: #2c3e50; font-size: 14px;">
                                    {row["item"]}
                                </td>
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; text-align: center; background-color: #fff3cd; font-weight: bold;">
                                    <div style="font-size: 14px; color: #2c3e50;">{row["base_year_2025"]}</div>
                                    <div style="font-size: 11px; color: #7f8c8d;">{row["base_unit_2025"]}</div>
                                </td>
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; text-align: center; color: #2c3e50;">
                                    <div style="font-size: 14px; font-weight: bold;">{row["target_year_2030"]}</div>
                                    <div style="font-size: 11px; color: #7f8c8d;">{row["target_unit_2030"]}</div>
                                </td>
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; text-align: center; background-color: #d4edda; font-weight: bold;">
                                    <div style="font-size: 14px; color: #2c3e50;">{row["reduction_rate_2030"]}</div>
                                    <div style="font-size: 11px; color: #7f8c8d;">{row["reduction_unit_2030"]}</div>
                                </td>
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; text-align: center; color: #2c3e50;">
                                    <div style="font-size: 14px; font-weight: bold;">{row["target_year_2050"]}</div>
                                    <div style="font-size: 11px; color: #7f8c8d;">{row["target_unit_2050"]}</div>
                                </td>
                                <td style="padding: 12px 10px; border: 1px solid #ecf0f1; text-align: center; background-color: #d4edda; font-weight: bold;">
                                    <div style="font-size: 14px; color: #2c3e50;">{row["reduction_rate_2050"]}</div>
                                    <div style="font-size: 11px; color: #7f8c8d;">{row["reduction_unit_2050"]}</div>
                                </td>
                            </tr>
                """
            
            html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div style="margin-top: 15px; text-align: right; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <small style="color: #6c757d; font-style: italic;">
                        Click SELECT above to choose this scenario for detailed planning
                    </small>
                </div>
            </div>
            """
            scenario_counter += 1
        
        html_content += """
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #d1ecf1; border-radius: 8px; border-left: 4px solid #0ea5e9;">
            <h4 style="margin: 0 0 5px 0; color: #0c4a6e;">Scenarios Generated Successfully!</h4>
            <p style="margin: 0; color: #0c4a6e; font-size: 14px;">
                Review the scenarios above and select one to proceed with detailed planning.
            </p>
        </div>
        """
        
        print(f"DEBUG: Final HTML length: {len(html_content)}")
        return html_content
    
    def run_property_breakdown_view(self, selected_scenario_name="Aggressive Decarbonisation"):
        """Generate property-level breakdown for yearly targets"""
        breakdown_df, status = self.backend.calculate_property_breakdown(selected_scenario_name)
        if breakdown_df is not None:
            return breakdown_df, f"{status}"
        else:
            return None, f"{status}"
    
    def run_strategic_pattern_selection(self, pattern_name):
        """Handle strategic pattern selection and show detailed analysis"""
        details_df, summary_info, status = self.backend.analyze_strategic_pattern(pattern_name)
        
        if details_df is not None and summary_info is not None:
            # Create HTML summary display
            summary_html = f"""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>Impact on Emission Target:</h4>
                <p><strong>Strategy Name:</strong> {summary_info['strategy_name']}</p>
                <p><strong>Description:</strong> {summary_info['description']}</p>
                <p><strong>Estimated Strategy Cost:</strong> {summary_info['estimated_cost']}</p>
                <p><strong>Estimated Total Reduction:</strong> {summary_info['estimated_reduction']}</p>
                <p><strong>2030 Target Gap:</strong> {summary_info['target_gap_2030']}</p>
                <p><strong>2050 Target Gap:</strong> {summary_info['target_gap_2050']}</p>
                <p><strong>Implementation Approach:</strong> {summary_info['implementation']}</p>
                <p><strong>Risk Level:</strong> {summary_info['risk_level']}</p>
            </div>
            """
            return details_df, summary_html, f"{status}"
        else:
            return None, "", f"{status}"
    
    def run_annual_reoptimization(self, property_id, deviation_threshold=5.0):
        """Run annual plan re-optimization analysis"""
        try:
            threshold_decimal = float(deviation_threshold) / 100.0
            
            plot_data, consumptions_df, analysis_summary, status = self.backend.reoptimize_annual_plan(
                property_id, threshold_decimal
            )
            
            if plot_data is None:
                return None, None, "", f"{status}"
            
            # Create the re-optimization plot
            fig = self.create_reoptimization_plot(plot_data, property_id)
            
            # Create analysis summary HTML
            summary_html = f"""
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>Re-optimization Analysis for {analysis_summary['property_id']}</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <h5>Emission Performance</h5>
                        <p><strong>YTD Actual:</strong> {analysis_summary['ytd_emission_actual']}</p>
                        <p><strong>YTD Target:</strong> {analysis_summary['ytd_emission_target']}</p>
                        <p><strong>Deviation:</strong> <span style="color: {'red' if '+' in analysis_summary['emission_deviation'] else 'green'}">{analysis_summary['emission_deviation']}</span></p>
                    </div>
                    <div>
                        <h5>Cost Performance</h5>
                        <p><strong>YTD Actual:</strong> {analysis_summary['ytd_cost_actual']}</p>
                        <p><strong>YTD Target:</strong> {analysis_summary['ytd_cost_target']}</p>
                        <p><strong>Deviation:</strong> <span style="color: {'red' if '+' in analysis_summary['cost_deviation'] else 'green'}">{analysis_summary['cost_deviation']}</span></p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <h5>Status</h5>
                    <p><strong>Reoptimization Needed:</strong> <span style="color: {'red' if analysis_summary['needs_reoptimization'] else 'green'}">{'Yes' if analysis_summary['needs_reoptimization'] else 'No'}</span></p>
                    <p><strong>Threshold:</strong> ±{analysis_summary['deviation_threshold']}</p>
                </div>
                <div style="margin-top: 15px;">
                    <h5>Recommendations</h5>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        {"".join([f"<li>{rec}</li>" for rec in analysis_summary['recommendations']])}
                    </ul>
                </div>
            </div>
            """
            
            return fig, consumptions_df, summary_html, f"{status}"
            
        except Exception as e:
            return None, None, "", f"Error: {str(e)}"
    
    def execute_replanning(self, property_id):
        """Execute the re-planning process"""
        try:
            replanning_result, status = self.backend.execute_replanning(property_id)
            
            if replanning_result:
                # Update the display with new information
                success_html = f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h4>Re-planning Completed Successfully!</h4>
                    <p><strong>Property:</strong> {replanning_result['property_id']}</p>
                    <p><strong>Optimization Date:</strong> {replanning_result['optimization_date']}</p>
                    <p><strong>Previous Deviation:</strong> {replanning_result['previous_deviation']}</p>
                    <p>New targets have been calculated and applied. Please refresh the analysis to see updated projections.</p>
                </div>
                """
                return success_html, "Re-planning completed successfully!"
            else:
                info_html = f"""
                <div style="background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h4>No Re-planning Required</h4>
                    <p>{status}</p>
                </div>
                """
                return info_html, status
                
        except Exception as e:
            error_html = f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>Re-planning Error</h4>
                <p>Error during re-planning execution: {str(e)}</p>
            </div>
            """
            return error_html, f"Error: {str(e)}"
    
    def generate_budget_visualization(self, property_id="BP01"):
        """Generate comprehensive budget visualization"""
        # Get property for context
        target_property = None
        for prop in self.backend.properties:
            if prop.property_id == property_id:
                target_property = prop
                break
        
        if not target_property:
            return None, None, "", f"Property {property_id} not found"
        
        budget_data, budget_df, summary_info, status = self.backend.generate_budget_data(property_id)
        
        if budget_data is not None:
            fig = self.create_budget_plot(budget_data, property_id, target_property)
            
            # Create summary HTML
            summary_html = f"""
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>Budget Analysis for {summary_info['property_id']}</h4>
                <p><strong>Building Type:</strong> {summary_info['building_type']}</p>
                <p><strong>Total Planning Budget:</strong> {summary_info['total_planning_budget']}</p>
                <p><strong>Total CAPEX:</strong> {summary_info['total_capex']}</p>
                <p><strong>Total OPEX:</strong> {summary_info['total_opex']}</p>
                <p><strong>Retrofit Potential:</strong> {summary_info['retrofit_potential']}</p>
                <p><strong>Total CO₂ Reduction:</strong> {summary_info['total_co2_reduction']}</p>
                <p><strong>Average Cost per Tonne:</strong> {summary_info['avg_cost_per_tonne']}</p>
                <p><strong>Investment Efficiency:</strong> {summary_info['investment_efficiency']}</p>
            </div>
            """
            
            return fig, budget_df, summary_html, f"{status}"
        else:
            return None, None, "", f"{status}"
    
    def create_interface(self):
        """Create the enhanced Gradio interface"""
        
        with gr.Blocks(
            title="EcoAssist AI - Enhanced Functions",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1800px;
                }
                .main-header {
                    background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
                    padding: 25px;
                    border-radius: 12px;
                    margin-bottom: 25px;
                    color: white;
                    text-align: center;
                }
                .function-section {
                    background: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border: 2px solid #90EE90;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }
                .scenario-card {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    background: #f9f9f9;
                }
                .selected-scenario {
                    border-color: #2E8B57;
                    background: #f0f8f5;
                }
                .tab-content {
                    padding: 20px;
                }
            """
        ) as app:
            
            # Enhanced Header
            gr.HTML("""
            <div class="main-header">
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 15px;">
                    <div style="width: 45px; height: 45px; background: #ffffff; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #2E8B57; font-size: 20px;">E</div>
                    <h1 style="font-size: 32px; font-weight: 600; margin: 0;">EcoAssist AI - Enhanced Functions</h1>
                </div>
                
                <div style="display: flex; justify-content: center; gap: 40px; margin: 20px 0;">
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 5px;">🎯</div>
                        <div style="font-size: 12px; opacity: 0.9;">Milestone Planning</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 5px;">📊</div>
                        <div style="font-size: 12px; opacity: 0.9;">Target Division</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 5px;">⚡</div>
                        <div style="font-size: 12px; opacity: 0.9;">Strategic Planning</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 5px;">🔄</div>
                        <div style="font-size: 12px; opacity: 0.9;">Re-optimization</div>
                    </div>
                </div>
                
                <h2 style="font-size: 20px; font-weight: 500; margin: 15px 0 10px 0;">Enhanced AI-Powered CO₂ Emission Reduction Planning</h2>
                
                <p style="font-size: 14px; opacity: 0.95; line-height: 1.6; max-width: 900px; margin: 0 auto;">
                    Advanced milestone recommendation, strategic pattern configuration, property-level targeting, 
                    and comprehensive budget planning with real-time re-optimization for sustainable emission reduction.
                </p>
            </div>
            """)
            
            # State variables for UI interaction
            selected_scenario = gr.State("Aggressive Decarbonisation")
            selected_strategy = gr.State("Active Installation of RE")
            current_property = gr.State("BP01")
            
            # Main Tab Interface
            with gr.Tabs():
                
                # Tab 1: Enhanced Milestone Target Setting
                with gr.TabItem("🎯 Milestone Target Setting", elem_classes="tab-content"):
                    gr.HTML('<h3 style="color: #2E8B57; margin-bottom: 15px;">Milestone Target Setting for All Properties</h3>')
                    
                    with gr.Tabs():
                        # Milestone Targets Sub-tab
                        with gr.TabItem("Milestone Targets"):
                            gr.Markdown("Set long-term emission reduction targets and generate milestone scenarios.")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    target_year_input = gr.Number(
                                        label="Target Year",
                                        value=2050,
                                        minimum=2025,
                                        maximum=2070
                                    )
                                    reduction_2030_input = gr.Slider(
                                        label="Reduction Target by 2030 (%)",
                                        minimum=10,
                                        maximum=70,
                                        value=30,
                                        step=5
                                    )
                                    reduction_2050_input = gr.Slider(
                                        label="Reduction Target by 2050 (%)",
                                        minimum=50,
                                        maximum=100,
                                        value=80,
                                        step=5
                                    )
                                    
                                    # Custom scenario inputs
                                    gr.HTML('<h4 style="color: #228B22; margin-top: 20px;">Create Custom Scenario</h4>')
                                    custom_name_input = gr.Textbox(
                                        label="Custom Scenario Name",
                                        placeholder="e.g., Company Specific Target"
                                    )
                                    custom_2030_input = gr.Number(
                                        label="Custom 2030 Target (%)",
                                        value=35,
                                        minimum=0,
                                        maximum=100
                                    )
                                    custom_2050_input = gr.Number(
                                        label="Custom 2050 Target (%)",
                                        value=85,
                                        minimum=0,
                                        maximum=100
                                    )
                                    
                                    milestone_btn = gr.Button("🎯 Generate Milestone Scenarios", variant="primary", size="lg")
                                
                                with gr.Column(scale=2):
                                    milestone_plot = gr.Plot(label="Milestone Pathways")
                            
                            milestone_results = gr.HTML(label="Milestone Scenario Details")
                            milestone_status = gr.Markdown("")
                            
                            # Scenario selection
                            with gr.Row():
                                scenario_selector = gr.Radio(
                                    choices=["Slow Decarbonisation", "Aggressive Decarbonisation", "SBTi Aligned", "Net Zero by 2050"],
                                    value="Aggressive Decarbonisation",
                                    label="Select Scenario for Detailed Planning"
                                )
                                select_scenario_btn = gr.Button("Select Scenario", variant="secondary")
                        
                        # Yearly Targets Sub-tab
                        with gr.TabItem("Yearly Targets"):
                            gr.Markdown("Review and confirm yearly emission targets for all properties.")
                            
                            yearly_summary_display = gr.HTML("""
                            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                <h4>Yearly Targets for All Properties</h4>
                                <p><strong>Total Baseline Emission:</strong> 1,515,000 tCO₂e</p>
                                <p><strong>Total NLA:</strong> 11,334 m²</p>
                                <p><strong>Portfolio Diversity:</strong> 9 properties across 7 building types</p>
                            </div>
                            """)
                            
                            # Property breakdown section
                            gr.HTML('<h4 style="color: #228B22; margin-top: 25px;">Property-Level Breakdown</h4>')
                            property_breakdown_btn = gr.Button("Generate Property Breakdown", variant="secondary")
                            property_breakdown_table = gr.Dataframe(label="Property-Level Yearly Targets")
                            property_breakdown_status = gr.Markdown("")
                
                # Tab 2: Property Management
                with gr.TabItem("🏢 Property Management", elem_classes="tab-content"):
                    gr.HTML('<h3 style="color: #2E8B57; margin-bottom: 15px;">Building-Level Target Management</h3>')
                    
                    with gr.Row():
                        property_selector = gr.Dropdown(
                            choices=["BP01", "CB01", "CB02", "CB03", "CB04", "HA1", "MP", "TC01", "WH1"],
                            value="BP01",
                            label="Select Property"
                        )
                    
                    with gr.Tabs():
                        # Yearly Targets for Property
                        with gr.TabItem("Yearly Targets"):
                            property_yearly_display = gr.HTML("""
                            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                <h4>BP01 Yearly Targets:</h4>
                                <p><strong>Baseline Emission:</strong> 150,000 tCO₂e</p>
                                <p><strong>NLA:</strong> 1,380 m²</p>
                                <p><strong>Building Type:</strong> Office</p>
                                <p><strong>Retrofit Potential:</strong> High</p>
                            </div>
                            """)
                        
                        # Strategic Patterns
                        with gr.TabItem("Reduction Strategies"):
                            gr.Markdown("Configure emission reduction strategies for this property.")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.HTML('<h4 style="color: #228B22;">Preset Strategies</h4>')
                                    
                                    strategy_options = [pattern.name for pattern in self.backend.strategic_patterns]
                                    strategy_selector = gr.Radio(
                                        choices=strategy_options,
                                        value="Active Installation of RE",
                                        label="Select Strategy Pattern"
                                    )
                                    
                                    select_strategy_btn = gr.Button("Load Strategy Details", variant="secondary")
                                    
                                    gr.HTML('<h4 style="color: #228B22; margin-top: 20px;">Create Custom Strategy</h4>')
                                    custom_strategy_name = gr.Textbox(label="Strategy Name", placeholder="My Custom Strategy")
                                    create_strategy_btn = gr.Button("+ Add Custom Strategy", variant="outline")
                                
                                with gr.Column(scale=2):
                                    gr.HTML('<h4>Strategy Details</h4>')
                                    strategy_details_table = gr.Dataframe(label="Reduction Options and Priorities")
                                    strategy_summary = gr.HTML("")
                            
                            strategy_status = gr.Markdown("")
                        
                        # Budget Setting
                        with gr.TabItem("Budget Setting"):
                            gr.Markdown("Configure budget constraints and view cost projections.")
                            
                            budget_display = gr.HTML("""
                            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                                <h4>Budget Analysis for Selected Property</h4>
                                <p><strong>Total Planning Budget:</strong> Dynamic based on property characteristics</p>
                                <p><strong>Investment Efficiency:</strong> Calculated cost per tonne CO₂e reduced</p>
                                <p><strong>Retrofit Potential Impact:</strong> High potential properties receive larger budgets</p>
                            </div>
                            """)
                            
                            generate_budget_btn = gr.Button("Generate Budget Analysis", variant="secondary")
                            budget_plot = gr.Plot(label="Budget Timeline and Projections")
                            budget_table = gr.Dataframe(label="Annual Budget Breakdown")
                            budget_status = gr.Markdown("")
                
                # Tab 3: Annual Plan Re-optimization
                with gr.TabItem("🔄 Annual Re-optimization", elem_classes="tab-content"):
                    gr.HTML('<h3 style="color: #2E8B57; margin-bottom: 15px;">Annual Plan Re-optimization</h3>')
                    gr.Markdown("Monitor actual vs target performance and re-optimize plans when deviations exceed thresholds.")
                    
                    with gr.Row():
                        reopt_property_selector = gr.Dropdown(
                            choices=["BP01", "CB01", "CB02", "CB03", "CB04", "HA1", "MP", "TC01", "WH1"],
                            value="BP01",
                            label="Select Property for Re-optimization"
                        )
                        deviation_threshold_input = gr.Slider(
                            label="Deviation Threshold (%)",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("📊 Analyze Current Performance", variant="secondary", size="lg")
                        replanning_btn = gr.Button("🔄 Execute Re-planning", variant="primary", size="lg")
                    
                    # Re-optimization results
                    reopt_plot = gr.Plot(label="Annual Planning Analysis")
                    reopt_analysis = gr.HTML("")
                    
                    # Consumptions table
                    gr.HTML('<h4 style="color: #228B22; margin-top: 25px;">Consumptions Breakdown</h4>')
                    consumptions_table = gr.Dataframe(label="Monthly Consumption Data")
                    
                    # Re-planning results
                    replanning_results = gr.HTML("")
                    reopt_status = gr.Markdown("")
                
                # Tab 4: System Overview
                with gr.TabItem("📊 System Overview", elem_classes="tab-content"):
                    gr.HTML('<h3 style="color: #2E8B57; margin-bottom: 15px;">Portfolio Overview & Analytics</h3>')
                    
                    with gr.Tabs():
                        # Portfolio Properties
                        with gr.TabItem("Properties"):
                            property_data = self.backend.get_properties_data()
                            
                            gr.Dataframe(
                                value=property_data,
                                headers=["Property ID", "Area (m²)", "Occupancy", "Baseline (tCO₂e)", 
                                       "Scope 1 (tCO₂e)", "Scope 2 (tCO₂e)", "Scope 3 (tCO₂e)", 
                                       "Retrofit Potential", "Building Type", "Year Built"],
                                label="Portfolio Properties Database"
                            )
                        
                        # Reduction Options
                        with gr.TabItem("Reduction Options"):
                            option_data = self.backend.get_reduction_options_data()
                            
                            gr.Dataframe(
                                value=option_data,
                                headers=["Option ID", "Name", "CO₂ Reduction (tCO₂e)", "CAPEX", "OPEX", 
                                       "Priority", "Implementation Time", "Risk Level"],
                                label="Available Reduction Options"
                            )
                        
                        # Strategic Patterns
                        with gr.TabItem("Strategic Patterns"):
                            pattern_data = self.backend.get_strategic_patterns_data()
                            
                            gr.Dataframe(
                                value=pattern_data,
                                headers=["Strategy Name", "Description", "Implementation", 
                                       "Estimated Cost", "Estimated Reduction", "Risk Level"],
                                label="Available Strategic Patterns"
                            )
            
            # Event Handlers
            milestone_btn.click(
                fn=self.run_milestone_recommendation,
                inputs=[target_year_input, reduction_2030_input, reduction_2050_input, 
                       custom_name_input, custom_2030_input, custom_2050_input],
                outputs=[milestone_plot, milestone_results, milestone_status]
            )
            
            select_scenario_btn.click(
                fn=lambda x: gr.update(value=x),
                inputs=[scenario_selector],
                outputs=[selected_scenario]
            )
            
            property_breakdown_btn.click(
                fn=self.run_property_breakdown_view,
                inputs=[selected_scenario],
                outputs=[property_breakdown_table, property_breakdown_status]
            )
            
            select_strategy_btn.click(
                fn=self.run_strategic_pattern_selection,
                inputs=[strategy_selector],
                outputs=[strategy_details_table, strategy_summary, strategy_status]
            )
            
            generate_budget_btn.click(
                fn=self.generate_budget_visualization,
                inputs=[property_selector],
                outputs=[budget_plot, budget_table, budget_display, budget_status]
            )
            
            # Annual Re-optimization Event Handlers
            analyze_btn.click(
                fn=self.run_annual_reoptimization,
                inputs=[reopt_property_selector, deviation_threshold_input],
                outputs=[reopt_plot, consumptions_table, reopt_analysis, reopt_status]
            )
            
            replanning_btn.click(
                fn=self.execute_replanning,
                inputs=[reopt_property_selector],
                outputs=[replanning_results, reopt_status]
            )
            
            # Property selector change handlers
            property_selector.change(
                fn=lambda prop_id: gr.update(value=prop_id),
                inputs=[property_selector],
                outputs=[current_property]
            )
            
            reopt_property_selector.change(
                fn=self._update_property_display,
                inputs=[reopt_property_selector],
                outputs=[]
            )
        
        return app
    
    def _update_property_display(self, property_id):
        """Helper method to update property-specific displays"""
        # This method can be used for dynamic property information updates
        return None

def main():
    """Main function to launch the application"""
    frontend = EcoAssistFrontend()
    app = frontend.create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()