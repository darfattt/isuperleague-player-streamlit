import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_loader import get_player_metric_categories
#from ai import GemmaAnalyst, AnalysisTypes
#import traceback

# Define negative metrics (where lower values are better)
NEGATIVE_METRICS = ['Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target']

def show_player_comparison(filtered_df):
    """Main player comparison page using pre-filtered data"""
    st.header("🔄 Player Comparison")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return
    
    # Player selection section
    
    # Show current filter summary
    st.info(
    f"📊 Comparing players is more accurate when you filter by position.  \n"
    f"   **Available Players**: {len(filtered_df)} players from {filtered_df['Team'].nunique()} teams"
)
    # Create player options with team and position info
    player_options = []
    for _, player in filtered_df.iterrows():
        option = f"{player['Player Name']} ({player['Team']} - {player['Position']})"
        player_options.append(option)
    
    selected_players = st.multiselect(
        "Select players to compare (2-4 players recommended):",
        options=player_options,
        default=[],
        max_selections=4,
        help="Search by typing player names. Up to 4 players can be compared at once."
    )
    
    if len(selected_players) < 2:
        st.info("👥 Please select at least 2 players to compare.")
        st.subheader("📊 Available Players")
        
        # Show sample of available players
        display_df = filtered_df[['Player Name', 'Team', 'Position', 'Age', 'Appearances', 'Goal', 'Assist']].head(20)
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True)
        return
    
    # Extract selected player data
    selected_player_names = []
    for option in selected_players:
        # Extract player name from "Player Name (Team - Position)" format
        player_name = option.split(' (')[0]
        selected_player_names.append(player_name)
    
    # Filter to selected players
    comparison_df = filtered_df[filtered_df['Player Name'].isin(selected_player_names)]
    
    if len(comparison_df) != len(selected_players):
        st.error("❌ Some selected players could not be found. Please try selecting again.")
        return
    
    # Show comparison visualizations
    st.subheader("📊 Player Comparison Analysis")
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["🎯 Radar Chart", "📊 Bar Charts", "📋 Statistics Table"])
    
    with tab1:
        st.subheader("🎯 Multi-Player Radar Chart")
        show_radar_comparison(comparison_df, filtered_df)
    
    with tab2:
        st.subheader("📊 Performance Bar Charts")
        show_bar_comparison(comparison_df, filtered_df)
    
    with tab3:
        st.subheader("📋 Detailed Statistics Comparison")
        show_table_comparison(comparison_df)
    
    # AI Insights Section
    #st.markdown("---")
    #st.subheader("🤖 AI-Powered Analysis")
    #if st.button("🧠 Generate AI Insights", type="primary", help="Generate AI-powered professional analysis of the player comparison"):
    #    show_ai_insights(comparison_df, filtered_df)
    
        
            

def show_radar_comparison(comparison_df, filtered_df):
    """Create radar chart for player comparison with normalized values"""
    
    # Get all metric columns (exclude info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in comparison_df.columns if col not in info_columns]
    
    # Create normalized data for radar chart based on filtered dataset
    normalized_df = comparison_df.copy()
    
    for col in metric_columns:
        col_min = filtered_df[col].min()
        col_max = filtered_df[col].max()
        
        if col_max == col_min:  # Handle case where all values are the same
            normalized_df[col] = 50  # Set to middle value
        else:
            if col in NEGATIVE_METRICS:
                # For negative metrics, invert normalization (lower values = better = higher score)
                normalized_df[col] = 100 - ((comparison_df[col] - col_min) / (col_max - col_min) * 100)
            else:
                # For positive metrics, normal normalization (higher values = better = higher score)
                normalized_df[col] = (comparison_df[col] - col_min) / (col_max - col_min) * 100
    
    fig = go.Figure()
    
    # Color palette for players
    colors = ['#00D4FF', '#FF6B9D', '#4ECDC4', '#FFD93D', '#6BCF7F', '#FF8C42']
    
    def hex_to_rgba(hex_color, alpha=0.1):
        """Convert hex color to RGBA format"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    for i, (idx, player_data) in enumerate(normalized_df.iterrows()):
        original_player_data = comparison_df.loc[idx]
        
        # Create hover text with original values
        hover_text = []
        for metric in metric_columns:
            actual_val = original_player_data[metric]
            normalized_val = player_data[metric]
            actual_str = "0" if actual_val == 0 else f"{actual_val:.1f}"
            hover_text.append(f"{metric}: {actual_str} (score: {normalized_val:.1f})")
        
        fig.add_trace(go.Scatterpolar(
            r=player_data[metric_columns].values,
            theta=metric_columns,
            fill='toself',
            name=original_player_data['Player Name'],
            line=dict(
                color=colors[i % len(colors)],
                width=3
            ),
            fillcolor=hex_to_rgba(colors[i % len(colors)], 0.1),
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{fullData.name}</b><br>%{text}<extra></extra>',
            text=hover_text
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linecolor='#4a90a4',
                gridcolor='#4a90a4',
                gridwidth=1,
                tickfont=dict(color='#e6e6e6', size=10),
                tickcolor='#4a90a4'
            ),
            angularaxis=dict(
                showline=True,
                linecolor='#4a90a4',
                gridcolor='#4a90a4',
                gridwidth=1,
                tickfont=dict(color='#ffffff', size=11, family='Arial'),
                tickcolor='#4a90a4'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        height=600,
        title=dict(
            text="Player Performance Radar Comparison",
            font=dict(color='#ffffff', size=16),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        legend=dict(
            font=dict(color='#ffffff', size=12),
            bgcolor='rgba(0,0,0,0.4)',
            bordercolor='#4a90a4',
            borderwidth=1,
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.05, yanchor="top"
        ),
        margin=dict(t=70, b=70, l=70, r=70),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="player_radar_comparison")
    
    # Add explanation
    st.info("💡 **Radar Chart Guide**: All 22 metrics displayed with 0-100 normalization based on filtered players. " +
            "For negative metrics (Own Goal, Yellow Card, Foul, Shoot Off Target), lower actual values appear as higher scores. " +
            "Hover over data points to see actual values and normalized scores.")

def display_player_header(player_data):
    """Display player image and info header"""
    
    # Create columns for image and info
    img_col, info_col = st.columns([1, 3])
    
    with img_col:
        # Display player image
        if player_data.get('Picture Url') and str(player_data['Picture Url']).strip() and str(player_data['Picture Url']) != 'nan':
            try:
                st.image(player_data['Picture Url'], width=80, caption="")
            except:
                # Fallback if image fails to load
                st.write("🏃‍♂️")
        else:
            # Default placeholder
            st.write("🏃‍♂️")
    
    with info_col:
        # Player name as main header
        st.markdown(f"**{player_data['Player Name']}**")
        # Line 2: Team and Position
        st.caption(f"{player_data['Team']} | {player_data['Position']}")
        # Line 3: Country, Age, and Appearances
        st.caption(f"{player_data['Country']} | {player_data['Age']} Years | {player_data['Appearances']} Apps")

def show_bar_comparison(comparison_df, filtered_df):
    """Show horizontal bar charts organized by metric categories"""
    
    # Get metric categories
    metric_categories = get_player_metric_categories()
    
    # Create columns for each player
    num_players = len(comparison_df)
    if num_players == 2:
        cols = st.columns(2)
    elif num_players == 3:
        cols = st.columns(3)
    else:
        cols = st.columns(4)
    
    # Create bar chart for each player
    for idx, (_, player_data) in enumerate(comparison_df.iterrows()):
        if idx < len(cols):
            with cols[idx]:
                # Display player image and info header
                display_player_header(player_data)
                # Create the performance chart
                create_player_performance_bar_chart(comparison_df, filtered_df, player_data, metric_categories)

def create_player_performance_bar_chart(_comparison_df, filtered_df, player_data, metric_categories):
    """Create individual player performance chart with horizontal bars organized by categories"""
    
    player_name = player_data['Player Name']
    
    # Prepare data for the chart
    chart_data = []
    category_order = {'Attack': 0, 'Defense': 1, 'Progression': 2, 'Discipline': 3, 'Goalkeeper': 4}
    
    for category, metrics in metric_categories.items():
        for metric_idx, metric in enumerate(metrics):
            if metric in player_data.index:
                current_value = player_data[metric]
                
                # Calculate percentile using filtered dataset
                filtered_metric_values = filtered_df[metric].values
                percentile = (filtered_metric_values < current_value).sum() / len(filtered_metric_values) * 100
                
                if metric in NEGATIVE_METRICS:
                    percentile = 100 - percentile  # Invert for negative metrics
                
                # Handle zero values specially
                if current_value == 0:
                    bar_length = 0  # Zero values show as 0% bars
                    color = '#cccccc'  # Gray color for zero values
                else:
                    # Use percentile for bar length, with minimum of 1 for visibility
                    bar_length = max(percentile, 1)  # Ensure minimum bar length for visibility
                    
                    # Determine color based on percentile
                    if percentile >= 81:
                        color = '#1a9641'  # Dark green
                    elif percentile >= 61:
                        color = '#73c378'  # Light green
                    elif percentile >= 41:
                        color = '#f9d057'  # Yellow
                    elif percentile >= 21:
                        color = '#fc8d59'  # Orange
                    else:
                        color = '#d73027'  # Red
                
                chart_data.append({
                    'Metric': metric,
                    'Value': current_value,
                    'BarLength': bar_length,
                    'Percentile': percentile,
                    'Color': color,
                    'Category': category,
                    'CategoryOrder': category_order.get(category, 5),
                    'MetricOrder': metric_idx,
                    'HoverText': f"{metric}<br>Value: {int(current_value)}<br>Percentile: {percentile:.0f}%"
                })
    
    # Create DataFrame and sort
    df_chart = pd.DataFrame(chart_data)
    df_chart = df_chart.sort_values(['CategoryOrder', 'MetricOrder'], ascending=[True, True])
    
    if df_chart.empty:
        st.warning(f"No data available for {player_name}")
        return
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for each category
    for category in metric_categories.keys():
        category_df = df_chart[df_chart['Category'] == category]
        if not category_df.empty:
            fig.add_trace(go.Bar(
                x=category_df['BarLength'],
                y=category_df['Metric'],
                orientation='h',
                marker=dict(
                    color=category_df['Color'],
                    line=dict(width=0.5, color='white')
                ),
                text=[f"{int(val)}" for val in category_df['Value']],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial Black'),
                hovertext=category_df['HoverText'],
                hoverinfo='text',
                name=category,
                showlegend=False
            ))
    
    # Add 50% percentile reference line
    fig.add_shape(
        type="line",
        x0=50, y0=-0.5,
        x1=50, y1=len(df_chart) - 0.5,
        line=dict(color="#888888", width=1, dash="dash"),
        opacity=0.6,
        layer="below"
    )
    
    # Add annotation for 50% line (at top)
    fig.add_annotation(
        x=50,
        y=len(df_chart) - 0.3,
        text="50%",
        showarrow=False,
        font=dict(size=10, color="#666666"),
        xanchor="center",
        yanchor="bottom"
    )
    
    
    # Add category dividers
    prev_category = None
    for i, (_, row) in enumerate(df_chart.iterrows()):
        if prev_category is not None and row['Category'] != prev_category:
            y_pos = i - 0.5
            fig.add_shape(
                type="line",
                x0=0, y0=y_pos,
                x1=df_chart['BarLength'].max() * 1.1, y1=y_pos,
                line=dict(color="#888888", width=0.8, dash="solid"),
                opacity=0.3,
                layer="below"
            )
        prev_category = row['Category']
    
    # Update layout
    fig.update_layout(
        title="Performance Metrics",
        xaxis=dict(
            title="Performance Score",
            showgrid=True,
            gridcolor='rgba(136, 136, 136, 0.2)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color='#666666')
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",
            tickfont=dict(size=10, color='#333333')
        ),
        margin=dict(l=10, r=10, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        barmode='stack',
        bargap=0.15,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"player_bar_{player_name.lower().replace(' ', '_')}")

def get_performance_color(value, all_values, is_negative_metric=False):
    """
    Get background color for a cell based on performance relative to other players
    
    Args:
        value: The metric value for the current player
        all_values: All metric values for comparison
        is_negative_metric: If True, lower values are better (e.g., fouls, cards)
    
    Returns:
        CSS color string for background
    """
    if len(all_values) <= 1:
        return 'background-color: #f0f0f0'  # Neutral gray for single player
    
    # Calculate percentile rank
    percentile = (all_values < value).sum() / len(all_values) * 100
    
    if is_negative_metric:
        percentile = 100 - percentile  # Invert for negative metrics
    
    # Map percentile to color
    if percentile >= 80:
        return 'background-color: #1a9641; color: white; font-weight: bold'  # Dark green
    elif percentile >= 60:
        return 'background-color: #73c378; color: white; font-weight: bold'  # Light green
    elif percentile >= 40:
        return 'background-color: #f9d057; color: black; font-weight: bold'  # Yellow
    elif percentile >= 20:
        return 'background-color: #fc8d59; color: white; font-weight: bold'  # Orange
    else:
        return 'background-color: #d73027; color: white; font-weight: bold'  # Red

def show_table_comparison(comparison_df):
    """Show detailed statistics comparison table"""
    
    # Get all metric columns
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in comparison_df.columns if col not in info_columns]
    
    # Create comparison table
    st.subheader("📊 Detailed Statistics")
    
    # Player info summary
    st.write("**Player Information:**")
    info_df = comparison_df[['Player Name', 'Team', 'Position', 'Age', 'Appearances']].copy()
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.write("**Performance Metrics:**")
    
    # Get metric categories for organization
    metric_categories = get_player_metric_categories()
    
    # Create tabs for each category
    category_tabs = st.tabs(list(metric_categories.keys()))
    
    for tab, (category, metrics) in zip(category_tabs, metric_categories.items()):
        with tab:
            # Filter metrics that exist in data
            available_metrics = [m for m in metrics if m in metric_columns]
            
            if available_metrics:
                # Create comparison dataframe for this category
                category_df = comparison_df[['Player Name'] + available_metrics].copy()
                
                # Transpose for better comparison view
                category_df_t = category_df.set_index('Player Name').T
                
                # Apply color styling to the transposed table
                def style_metric_cells(df_t):
                    """Apply color styling to metric cells based on performance"""
                    # Apply styling to each metric row
                    def apply_row_styling(row):
                        metric_name = row.name
                        is_negative = metric_name in NEGATIVE_METRICS
                        all_values = row.values
                        
                        # Create style for each cell in the row
                        return [get_performance_color(val, all_values, is_negative) for val in all_values]
                    
                    return df_t.style.apply(apply_row_styling, axis=1)
                
                styled_table = style_metric_cells(category_df_t)
                
                # Display the styled table
                st.dataframe(styled_table, use_container_width=True)
                
                # Add best performer summary for this category
                st.write(f"**Best Performers in {category}:**")
                for metric in available_metrics[:3]:  # Show top 3 metrics
                    if metric in NEGATIVE_METRICS:
                        best_player = comparison_df.loc[comparison_df[metric].idxmin(), 'Player Name']
                        best_value = comparison_df[metric].min()
                    else:
                        best_player = comparison_df.loc[comparison_df[metric].idxmax(), 'Player Name']
                        best_value = comparison_df[metric].max()
                    
                    st.write(f"- **{metric}**: {best_player} ({best_value:.1f})")

# def show_ai_insights(comparison_df, filtered_df):
#     """Generate and display AI-powered insights for player comparison"""
    
#     if len(comparison_df) != 2:
#         st.error("❌ AI insights work best with exactly 2 players. Please select 2 players for comparison.")
#         return
    
#     # Create progress placeholder
#     progress_placeholder = st.empty()
#     result_placeholder = st.empty()
    
#     try:
#         with progress_placeholder:
#             with st.spinner("🤖 Initializing AI analyst..."):
#                 # Get token from session state (primary source)
#                 active_token = st.session_state.get('hf_token')
                
#                 # Show token status
#                 if active_token:
#                     st.info("🔗 Using HuggingFace token from session")
#                 else:
#                     st.warning("⚠️ No HuggingFace token found - using public models")
#                     st.info("💡 Visit the AI Analyst page to configure authentication")
                
#                 # Initialize AI analyst with token
#                 analyst = GemmaAnalyst(token=active_token)
                
#                 # Check system resources first
#                 system_status = analyst.get_system_status()
#                 print(system_status)
#                 if not system_status['memory_status']['sufficient']:
#                     st.error("❌ **Insufficient System Memory for AI Analysis**")
#                     st.write(f"**Available Memory**: {system_status['memory_status']['available_gb']:.1f}GB")
#                     st.write(f"**Required Memory**: {system_status['memory_status']['required_gb']:.1f}GB")
                    
#                     if system_status['recommendations']:
#                         st.write("**Recommendations:**")
#                         for rec in system_status['recommendations']:
#                             st.write(f"• {rec}")
                    
#                     # Show fallback analysis
#                     with st.expander("📊 Statistical Analysis (Fallback)", expanded=True):
#                         show_statistical_fallback_analysis(comparison_df, filtered_df)
#                     return
        
#         with progress_placeholder:
#             with st.spinner("🧠 Generating professional AI analysis..."):
                
#                 # Get the two players
#                 player1_data = comparison_df.iloc[0]
#                 player2_data = comparison_df.iloc[1]
                
#                 # Prepare comparison metrics
#                 info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
#                 metric_columns = [col for col in comparison_df.columns if col not in info_columns]
                
#                 # Prepare comparison data manually for better control
#                 comparison_data = prepare_comparison_data(player1_data, player2_data, metric_columns)
#                 st.info(f"📊 Comparison data prepared ({len(comparison_data)} characters, {len(metric_columns)} metrics)")
                
#                 # Create comparison prompt using the analyst's template
#                 try:
#                     prompt = analyst.prompt_templates.get_player_comparison_prompt(
#                         player1_name=player1_data.get('Player Name', 'Player 1'),
#                         player2_name=player2_data.get('Player Name', 'Player 2'),
#                         comparison_data=comparison_data,
#                         context="Indonesia Super League player comparison analysis for tactical decision-making"
#                     )
#                     st.info(f"✅ Prompt generated successfully ({len(prompt)} characters)")
#                 except Exception as prompt_error:
#                     st.error(f"❌ **Prompt Generation Error**: {str(prompt_error)}")
#                     st.info("📊 **Falling back to statistical analysis**")
#                     show_statistical_fallback_analysis(comparison_df, filtered_df)
#                     return
                
#                 # Generate AI analysis with proper configuration
#                 try:
#                     st.info("🔄 Generating AI analysis...")
#                     ai_analysis = analyst.generate_analysis(
#                         prompt=prompt,
#                         max_length=512,
#                         temperature=0.7,
#                         top_p=0.9
#                     )
#                     st.info(f"✅ AI analysis completed ({len(ai_analysis) if ai_analysis else 0} characters)")
#                 except Exception as analysis_error:
#                     st.error(f"❌ **AI Generation Error**: {str(analysis_error)}")
#                     ai_analysis = None
        
#         # Clear progress and show results
#         progress_placeholder.empty()
        
#         with result_placeholder:
#             # AI Analysis Results
#             st.success("✅ **AI Analysis Complete**")
            
#             with st.expander("🎯 **Professional AI Insights**", expanded=True):
#                 # Format and display AI analysis
#                 if ai_analysis and len(ai_analysis.strip()) > 50:
#                     st.success(f"✅ **AI Analysis Generated** ({len(ai_analysis.strip())} characters)")
#                     # Split analysis into sections if it contains markdown headers
#                     if '##' in ai_analysis or '**' in ai_analysis:
#                         st.markdown(ai_analysis)
#                     else:
#                         # Format as readable text if no markdown
#                         st.write(ai_analysis)
#                 elif ai_analysis:
#                     # Show short analysis but warn
#                     st.warning(f"⚠️ **Short AI Analysis** ({len(ai_analysis.strip())} characters)")
#                     st.write(ai_analysis)
#                     st.info("📊 **Also showing statistical analysis for completeness**")
#                     show_statistical_fallback_analysis(comparison_df, filtered_df)
#                 else:
#                     # Fallback if AI analysis is empty or None
#                     st.error("❌ **AI Analysis Failed**: No content generated")
#                     st.info("📊 **Showing statistical analysis instead**")
#                     show_statistical_fallback_analysis(comparison_df, filtered_df)
            
#             # Additional context
#             with st.expander("ℹ️ **Analysis Details**"):
#                 st.info(f"""
#                 **Analysis Method**: AI-powered using {analyst.model_name}
#                 **Players Analyzed**: {player1_data['Player Name']} vs {player2_data['Player Name']}
#                 **Metrics Considered**: {len(metric_columns)} performance indicators
#                 **Context**: Professional tactical and transfer analysis for Indonesia Super League
#                 """)
    
#     except Exception as e:
#         progress_placeholder.empty()
        
#         # Show error and fallback with better context
#         error_message = str(e)
#         st.error(f"❌ **AI Analysis Error**: {error_message}")
        
#         # Provide specific guidance based on error type
#         if "token" in error_message.lower():
#             st.warning("🔑 **Token Issue Detected**")
#             st.info("""
#             **Possible Solutions:**
#             - Check if your HuggingFace token is valid
#             - Visit the AI Analyst page to configure authentication
#             - Ensure token has proper permissions
#             """)
#         elif "memory" in error_message.lower() or "cuda" in error_message.lower():
#             st.warning("💾 **Memory Issue Detected**")
#             st.info("""
#             **Possible Solutions:**
#             - Close other applications to free memory
#             - Try using the main AI Analyst page for better resource management
#             - System needs at least 4GB available RAM
#             """)
#         elif "connection" in error_message.lower() or "network" in error_message.lower():
#             st.warning("🌐 **Network Issue Detected**")
#             st.info("""
#             **Possible Solutions:**
#             - Check your internet connection
#             - HuggingFace servers might be temporarily unavailable
#             - Try again in a few moments
#             """)
        
#         # Show detailed error in expander for debugging
#         with st.expander("🔧 Technical Details"):
#             st.code(traceback.format_exc())
        
#         # Always show statistical fallback
#         st.info("📊 **Falling back to statistical analysis**")
#         show_statistical_fallback_analysis(comparison_df, filtered_df)

# def show_statistical_fallback_analysis(comparison_df, _filtered_df):
#     """Show statistical analysis when AI is not available"""
    
#     if len(comparison_df) != 2:
#         st.error("Statistical analysis requires exactly 2 players.")
#         return
    
#     player1 = comparison_df.iloc[0]
#     player2 = comparison_df.iloc[1]
    
#     # Basic comparison info
#     st.markdown(f"### 📊 Statistical Comparison: {player1['Player Name']} vs {player2['Player Name']}")
    
#     # Player info comparison
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown(f"**{player1['Player Name']}**")
#         st.write(f"• Team: {player1['Team']}")
#         st.write(f"• Position: {player1['Position']}")
#         st.write(f"• Age: {player1['Age']}")
#         st.write(f"• Appearances: {player1['Appearances']}")
    
#     with col2:
#         st.markdown(f"**{player2['Player Name']}**")
#         st.write(f"• Team: {player2['Team']}")
#         st.write(f"• Position: {player2['Position']}")
#         st.write(f"• Age: {player2['Age']}")
#         st.write(f"• Appearances: {player2['Appearances']}")
    
#     # Performance category comparison
#     metric_categories = get_player_metric_categories()
    
#     st.markdown("### 📈 Performance Category Winners")
    
#     category_winners = {}
#     for category, metrics in metric_categories.items():
#         available_metrics = [m for m in metrics if m in comparison_df.columns]
        
#         if available_metrics:
#             # Calculate category totals for each player
#             player1_total = sum(player1.get(metric, 0) for metric in available_metrics)
#             player2_total = sum(player2.get(metric, 0) for metric in available_metrics)
            
#             if category in NEGATIVE_METRICS or any(m in NEGATIVE_METRICS for m in available_metrics):
#                 # For discipline metrics, lower is better
#                 winner = player1['Player Name'] if player1_total < player2_total else player2['Player Name']
#                 winner_value = min(player1_total, player2_total)
#             else:
#                 # For other metrics, higher is better
#                 winner = player1['Player Name'] if player1_total > player2_total else player2['Player Name']
#                 winner_value = max(player1_total, player2_total)
            
#             category_winners[category] = {
#                 'winner': winner,
#                 'value': winner_value,
#                 'p1_total': player1_total,
#                 'p2_total': player2_total
#             }
    
#     # Display category winners
#     for category, result in category_winners.items():
#         if result['p1_total'] != result['p2_total']:  # Only show if there's a difference
#             st.write(f"**{category}**: {result['winner']} ({result['p1_total']:.1f} vs {result['p2_total']:.1f})")
    
#     # Key insights
#     st.markdown("### 💡 Key Statistical Insights")
    
#     insights = []
    
#     # Age comparison
#     if player1['Age'] != player2['Age']:
#         younger = player1['Player Name'] if player1['Age'] < player2['Age'] else player2['Player Name']
#         age_diff = abs(player1['Age'] - player2['Age'])
#         insights.append(f"• **Age Advantage**: {younger} is {age_diff} years younger")
    
#     # Experience comparison
#     if player1['Appearances'] != player2['Appearances']:
#         more_experienced = player1['Player Name'] if player1['Appearances'] > player2['Appearances'] else player2['Player Name']
#         exp_diff = abs(player1['Appearances'] - player2['Appearances'])
#         insights.append(f"• **Experience**: {more_experienced} has {exp_diff} more appearances")
    
#     # Goals comparison
#     if 'Goal' in comparison_df.columns:
#         if player1['Goal'] != player2['Goal']:
#             better_scorer = player1['Player Name'] if player1['Goal'] > player2['Goal'] else player2['Player Name']
#             goal_diff = abs(player1['Goal'] - player2['Goal'])
#             insights.append(f"• **Goals**: {better_scorer} has {goal_diff} more goals")
    
#     # Show insights
#     for insight in insights:
#         st.write(insight)
    
#     if not insights:
#         st.info("Players show similar statistical profiles across major metrics.")
    
#     # Recommendations
#     st.markdown("### 🎯 Statistical Recommendations")
#     st.info("""
#     **Note**: This is basic statistical analysis. For detailed tactical insights, professional scouting advice, 
#     and playing style analysis, try the AI analysis when sufficient system memory is available.
    
#     **For AI Analysis**: Ensure at least 4GB available RAM and try closing other applications.
#     """)

# def format_ai_analysis_display(analysis_text: str) -> str:
#     """Format AI analysis text for better display"""
#     if not analysis_text:
#         return "No analysis available."
    
#     # Clean up the text
#     cleaned = analysis_text.strip()
    
#     # Add markdown formatting if not present
#     if not any(marker in cleaned for marker in ['##', '**', '*', '-']):
#         # Simple text - add some structure
#         paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
#         formatted_paragraphs = []
        
#         for i, paragraph in enumerate(paragraphs):
#             if i == 0:
#                 # First paragraph as introduction
#                 formatted_paragraphs.append(f"**Analysis Overview:**\n{paragraph}")
#             elif 'recommend' in paragraph.lower() or 'conclusion' in paragraph.lower():
#                 # Recommendations/conclusions
#                 formatted_paragraphs.append(f"**Recommendations:**\n{paragraph}")
#             else:
#                 # Regular paragraph
#                 formatted_paragraphs.append(paragraph)
        
#         return '\n\n'.join(formatted_paragraphs)
    
#     return cleaned


# def prepare_comparison_data(player1_data: pd.Series, player2_data: pd.Series, metrics: list) -> str:
#     """Prepare player comparison data for AI analysis"""
#     comparison_lines = ["Player Comparison:"]
    
#     comparison_lines.append(f"\n{player1_data.get('Player Name', 'Player 1')} vs {player2_data.get('Player Name', 'Player 2')}")
#     comparison_lines.append(f"Teams: {player1_data.get('Team', 'N/A')} vs {player2_data.get('Team', 'N/A')}")
#     comparison_lines.append(f"Ages: {player1_data.get('Age', 'N/A')} vs {player2_data.get('Age', 'N/A')}")
#     comparison_lines.append(f"Positions: {player1_data.get('Position', 'N/A')} vs {player2_data.get('Position', 'N/A')}")
#     comparison_lines.append(f"Appearances: {player1_data.get('Appearances', 'N/A')} vs {player2_data.get('Appearances', 'N/A')}")
    
#     comparison_lines.append(f"\nPerformance Metrics Comparison:")
#     metrics_added = 0
#     for metric in metrics:
#         if metric in player1_data.index and metric in player2_data.index:
#             val1 = player1_data.get(metric, 0)
#             val2 = player2_data.get(metric, 0)
            
#             # Format values nicely
#             val1_str = f"{val1:.1f}" if isinstance(val1, (int, float)) else str(val1)
#             val2_str = f"{val2:.1f}" if isinstance(val2, (int, float)) else str(val2)
            
#             comparison_lines.append(f"  {metric}: {val1_str} vs {val2_str}")
#             metrics_added += 1
    
#     if metrics_added == 0:
#         comparison_lines.append("  No matching metrics found for comparison")
#     else:
#         comparison_lines.append(f"\nTotal metrics compared: {metrics_added}")
    
#     return "\n".join(comparison_lines)