import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_loader import get_player_metric_categories

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
                create_player_performance_bar_chart(comparison_df, filtered_df, player_data, metric_categories)

def create_player_performance_bar_chart(comparison_df, filtered_df, player_data, metric_categories):
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
                
                # Use percentile for bar length (0-100 scale)
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
                    'HoverText': f"{metric}<br>Value: {current_value:.1f}<br>Percentile: {percentile:.0f}%"
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
                text=[f"{val:.1f}" if val > 0 else "0" for val in category_df['Value']],
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial Black'),
                hovertext=category_df['HoverText'],
                hoverinfo='text',
                name=category,
                showlegend=False
            ))
    
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
        title=f"<b>{player_name}</b>",
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
                    # Create a DataFrame for styling
                    styled_df = df_t.copy()
                    
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
            else:
                st.info(f"No metrics available for {category} category.")