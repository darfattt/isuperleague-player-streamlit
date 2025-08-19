import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def show_player_screener(filtered_df):
    """
    Advanced Player Screener with metric-based filtering and heatmap visualization
    
    Features:
    - Multi-metric selection with customizable ranges
    - Dynamic slider controls for filtering
    - Results table with heatmap styling for selected metrics
    - Real-time filtering with immediate feedback
    """
    st.header("🔍 Player Screener")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return
    
    # Get available metric columns (exclude info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    available_metrics = [col for col in filtered_df.columns if col not in info_columns and filtered_df[col].dtype in ['int64', 'float64']]
    
    st.info(f"📊 **Screening Ready**: {len(filtered_df)} players available from {filtered_df['Team'].nunique()} teams")
    
    # Metric Selection Interface
    st.subheader("📈 Select Metrics to Filter")
    
    # Organize metrics by category for better UX
    metric_categories = {
        "⚽ Attacking": ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
        "🎯 Playmaking": ['Passing', 'Create Chance', 'Cross', 'Free Kick', 'Assist'],
        "🛡️ Defensive": ['Tackle', 'Intercept', 'Clearance', 'Block', 'Block Cross', 'Header Won'],
        "🏃 Physical": ['Ball Recovery', 'Dribble Success', 'Fouled', 'Foul'],
        "📊 General": ['Appearances', 'Yellow Card', 'Own Goal', 'Saves']
    }
    
    # Filter categories to only include available metrics
    available_categories = {}
    for category, metrics in metric_categories.items():
        available_metrics_in_category = [m for m in metrics if m in available_metrics]
        if available_metrics_in_category:
            available_categories[category] = available_metrics_in_category
    
    # Add any remaining metrics to a "Other" category
    used_metrics = set()
    for metrics in available_categories.values():
        used_metrics.update(metrics)
    
    remaining_metrics = [m for m in available_metrics if m not in used_metrics]
    if remaining_metrics:
        available_categories["📋 Other"] = remaining_metrics
    
    # Metric selection with categories
    selected_metrics = []
    
    with st.expander("🎯 Select Metrics by Category", expanded=True):
        tabs = st.tabs(list(available_categories.keys()))
        
        for tab, (category, metrics) in zip(tabs, available_categories.items()):
            with tab:
                category_selections = st.multiselect(
                    f"Select {category.split(' ', 1)[1] if ' ' in category else category} metrics:",
                    metrics,
                    key=f"metrics_{category}",
                    help=f"Choose metrics from {category.lower()} to filter players"
                )
                selected_metrics.extend(category_selections)
    
    # Remove duplicates while preserving order
    selected_metrics = list(dict.fromkeys(selected_metrics))
    
    if not selected_metrics:
        st.warning("⚠️ Please select at least one metric to start filtering players.")
        return
    
    st.success(f"✅ **Selected Metrics**: {', '.join(selected_metrics)}")
    
    # Range Filtering Controls
    st.subheader("🎚️ Set Value Ranges")
    
    filtered_players_df = filtered_df.copy()
    range_filters = {}
    
    # Create columns for sliders (max 3 per row)
    metrics_per_row = 3
    metric_rows = [selected_metrics[i:i + metrics_per_row] for i in range(0, len(selected_metrics), metrics_per_row)]
    
    for row_metrics in metric_rows:
        cols = st.columns(len(row_metrics))
        
        for col, metric in zip(cols, row_metrics):
            with col:
                # Get metric statistics
                metric_data = filtered_df[metric].dropna()
                if len(metric_data) > 0:
                    min_val = float(metric_data.min())
                    max_val = float(metric_data.max())
                    mean_val = float(metric_data.mean())
                    
                    # Create slider with default range
                    range_values = st.slider(
                        f"{metric}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=(max_val - min_val) / 100 if max_val > min_val else 0.1,
                        key=f"range_{metric}",
                        help=f"Filter by {metric} range. League average: {mean_val:.1f}"
                    )
                    
                    range_filters[metric] = range_values
                    
                    # Apply filter to dataframe
                    filtered_players_df = filtered_players_df[
                        (filtered_players_df[metric] >= range_values[0]) & 
                        (filtered_players_df[metric] <= range_values[1])
                    ]
    
    # Display filtering results
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players", len(filtered_df))
    with col2:
        st.metric("Filtered Players", len(filtered_players_df))
    with col3:
        filter_percentage = (len(filtered_players_df) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Filter Rate", f"{filter_percentage:.1f}%")
    
    if len(filtered_players_df) == 0:
        st.warning("⚠️ No players match your current filter criteria. Try adjusting the ranges.")
        return
    
    # Results Table with Heatmap
    st.subheader("📋 Filtered Results")
    
    # Prepare display dataframe with core info + selected metrics
    display_columns = ['Player Name', 'Team', 'Position', 'Age', 'Country', 'Appearances'] + selected_metrics
    available_display_columns = [col for col in display_columns if col in filtered_players_df.columns]
    
    display_df = filtered_players_df[available_display_columns].copy()
    
    # Sort by first selected metric (descending)
    if selected_metrics:
        display_df = display_df.sort_values(by=selected_metrics[0], ascending=False)
    
    # Create styled dataframe with heatmap for selected metrics only
    styled_df = create_heatmap_table(display_df, selected_metrics, filtered_df)
    
    # Display the table
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary insights
    if len(filtered_players_df) > 0:
        st.subheader("💡 Filter Summary")
        
        insights = []
        
        # Top performers in first metric
        if selected_metrics:
            top_metric = selected_metrics[0]
            top_performers = display_df.head(3)[['Player Name', 'Team', top_metric]]
            top_players_text = ', '.join([f"{row['Player Name']} ({row[top_metric]:.1f})" for _, row in top_performers.iterrows()])
            insights.append(f"**Top {top_metric}**: {top_players_text}")
        
        # Age distribution
        avg_age = display_df['Age'].mean() if 'Age' in display_df.columns else 0
        insights.append(f"**Average Age**: {avg_age:.1f} years")
        
        # Team distribution
        team_counts = display_df['Team'].value_counts().head(3) if 'Team' in display_df.columns else pd.Series()
        if len(team_counts) > 0:
            top_teams = ', '.join([f"{team} ({count})" for team, count in team_counts.items()])
            insights.append(f"**Most Represented Teams**: {top_teams}")
        
        # Position distribution
        if 'Position' in display_df.columns:
            pos_counts = display_df['Position'].value_counts().head(3)
            top_positions = ', '.join([f"{pos} ({count})" for pos, count in pos_counts.items()])
            insights.append(f"**Most Common Positions**: {top_positions}")
        
        for insight in insights:
            st.write(f"• {insight}")

def create_heatmap_table(df: pd.DataFrame, heatmap_columns: List[str], reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create a styled dataframe with heatmap coloring for specified columns"""
    
    def style_heatmap_columns(styler):
        # Apply heatmap styling only to selected metric columns
        for col in heatmap_columns:
            if col in df.columns:
                # Calculate percentiles for coloring
                col_values = df[col].dropna()
                if len(col_values) > 0:
                    # Create color mapping based on percentiles
                    def color_metric(val):
                        if pd.isna(val):
                            return 'background-color: #f8f9fa'
                        
                        # Calculate percentile
                        percentile = (col_values < val).sum() / len(col_values) * 100
                        
                        # Color gradient from red (low) to green (high)
                        if percentile >= 90:
                            return 'background-color: #28a745; color: white; font-weight: bold'  # Dark Green
                        elif percentile >= 75:
                            return 'background-color: #6bc547; color: white; font-weight: bold'  # Light Green
                        elif percentile >= 60:
                            return 'background-color: #9fd246; color: black; font-weight: bold'  # Yellow-Green
                        elif percentile >= 40:
                            return 'background-color: #ffc107; color: black; font-weight: bold'  # Yellow
                        elif percentile >= 25:
                            return 'background-color: #fd7e14; color: white; font-weight: bold'  # Orange
                        elif percentile >= 10:
                            return 'background-color: #e55353; color: white; font-weight: bold'  # Light Red
                        else:
                            return 'background-color: #dc3545; color: white; font-weight: bold'  # Dark Red
                    
                    styler = styler.applymap(color_metric, subset=[col])
        
        return styler
    
    # Apply styling
    styled = df.style.pipe(style_heatmap_columns)
    
    # Format numeric columns to 1 decimal place
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    format_dict = {col: '{:.1f}' for col in numeric_columns}
    styled = styled.format(format_dict)
    
    return styled

def calculate_percentile_rank(value: float, reference_series: pd.Series) -> float:
    """Calculate percentile rank of a value within a reference series"""
    if pd.isna(value):
        return 0
    
    clean_series = reference_series.dropna()
    if len(clean_series) == 0:
        return 0
    
    rank = (clean_series < value).sum()
    return (rank / len(clean_series)) * 100