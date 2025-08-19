import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

def show_scatter_analysis(filtered_df):
    """
    Advanced scatter plot analysis for player performance data
    
    Features:
    - X/Y axis selection from available metrics
    - Multiple highlighting options (players, teams, performance, age)
    - Interactive hover with player details
    - Trend lines and median lines
    - Professional styling consistent with dashboard theme
    """
    st.header("🎯 Advanced Scatter Analysis")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return
    
    # Get available metric columns (exclude info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in filtered_df.columns if col not in info_columns]
    
    # Show current data summary
    # st.info(
    #     f"📊 **Analysis Ready**  \n\n"
    #     f"   **Available Players**: {len(filtered_df)} players from {filtered_df['Team'].nunique()} teams  \n\n"
    #     f"   **Available Metrics**: {len(metric_columns)} performance metrics"
    # )
    
    # Create the main interface
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        x_axis = st.selectbox(
            "Select X-axis metric:",
            metric_columns,
            index=metric_columns.index('Goal') if 'Goal' in metric_columns else 0,
            help="Choose the metric for the horizontal axis"
        )
    
    with col2:
        y_axis = st.selectbox(
            "Select Y-axis metric:", 
            metric_columns,
            index=metric_columns.index('Assist') if 'Assist' in metric_columns else (1 if len(metric_columns) > 1 else 0),
            help="Choose the metric for the vertical axis"
        )
    
    with col3:
        show_trendline = st.checkbox(
            "Show trend line", 
            value=False,
            help="Add linear regression trend line"
        )
    
    with col4:
        show_median_lines = st.checkbox(
            "Show median lines", 
            value=True,
            help="Add median reference lines for both axes"
        )
    
    with col5:
        show_names = st.selectbox(
            "Show Names",
            ["Never", "Only Highlighted", "Always"],
            index=1,
            help="When to display player names on the chart"
        )
        
    
    st.markdown("---")
    
    # Advanced highlighting options
    with st.expander("🎨 Highlighting Options", expanded=True):
        highlighted_players = set()
        highlight_colors = {}
        highlight_reasons = {}
        
        # Players & Teams Selection
        st.markdown("**👥 Players & Teams**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏷️ Select Specific Players**")
            selected_players = st.multiselect(
                "Choose players to highlight:",
                options=sorted(filtered_df['Player Name'].tolist()),
                default=[],
                help="Select individual players to highlight on the chart"
            )
            
            if selected_players:
                for player in selected_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#FF6B9D'  # Pink for selected players
                    highlight_reasons[player] = 'Selected Player'
        
        with col2:
            st.markdown("**🏟️ Select Teams**")
            available_teams = sorted(filtered_df['Team'].unique().tolist())
            selected_teams = st.multiselect(
                "Choose teams to highlight:",
                options=available_teams,
                default=[],
                help="Select teams to highlight all their players"
            )
            
            if selected_teams:
                team_players = filtered_df[filtered_df['Team'].isin(selected_teams)]['Player Name'].tolist()
                for player in team_players:
                    if player not in highlighted_players:  # Don't override individual selections
                        highlighted_players.add(player)
                        highlight_colors[player] = '#4ECDC4'  # Cyan for team selections
                        highlight_reasons[player] = f'Team: {filtered_df[filtered_df["Player Name"] == player]["Team"].iloc[0]}'
        
        st.markdown("---")
        
        # Performance-Based Highlighting
        st.markdown("**📊 Performance-Based Highlighting**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 X-Axis Performance**")
            
            top_x = st.checkbox(f"Top 10 {x_axis}", help=f"Highlight players with highest {x_axis} values")
            bottom_x = st.checkbox(f"Bottom 10 {x_axis}", help=f"Highlight players with lowest {x_axis} values")
            
            if top_x:
                top_x_players = filtered_df.nlargest(10, x_axis)['Player Name'].tolist()
                for player in top_x_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#1a9641'  # Green for top performers
                        highlight_reasons[player] = f'Top 10 {x_axis}'
            
            if bottom_x:
                bottom_x_players = filtered_df.nsmallest(10, x_axis)['Player Name'].tolist()
                for player in bottom_x_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#d73027'  # Red for bottom performers
                        highlight_reasons[player] = f'Bottom 10 {x_axis}'
        
        with col2:
            st.markdown("**📊 Y-Axis Performance**")
            
            top_y = st.checkbox(f"Top 10 {y_axis}", help=f"Highlight players with highest {y_axis} values")
            bottom_y = st.checkbox(f"Bottom 10 {y_axis}", help=f"Highlight players with lowest {y_axis} values")
            
            if top_y:
                top_y_players = filtered_df.nlargest(10, y_axis)['Player Name'].tolist()
                for player in top_y_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#1a9641'  # Green for top performers
                        highlight_reasons[player] = f'Top 10 {y_axis}'
            
            if bottom_y:
                bottom_y_players = filtered_df.nsmallest(10, y_axis)['Player Name'].tolist()
                for player in bottom_y_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#d73027'  # Red for bottom performers
                        highlight_reasons[player] = f'Bottom 10 {y_axis}'
        
        st.markdown("**🎯 Combined Performance**")
        col1, col2 = st.columns(2)
        
        with col1:
            top_combined = st.checkbox(
                "Top 10 Combined", 
                help=f"Highlight players with highest average of {x_axis} and {y_axis}"
            )
        
        with col2:
            bottom_combined = st.checkbox(
                "Bottom 10 Combined", 
                help=f"Highlight players with lowest average of {x_axis} and {y_axis}"
            )
        
        if top_combined:
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['combined_score'] = (filtered_df_copy[x_axis] + filtered_df_copy[y_axis]) / 2
            top_combined_players = filtered_df_copy.nlargest(10, 'combined_score')['Player Name'].tolist()
            for player in top_combined_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#FFD93D'  # Gold for top combined
                    highlight_reasons[player] = f'Top 10 Combined ({x_axis} + {y_axis})'
        
        if bottom_combined:
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['combined_score'] = (filtered_df_copy[x_axis] + filtered_df_copy[y_axis]) / 2
            bottom_combined_players = filtered_df_copy.nsmallest(10, 'combined_score')['Player Name'].tolist()
            for player in bottom_combined_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#FF8C42'  # Orange for bottom combined
                    highlight_reasons[player] = f'Bottom 10 Combined ({x_axis} + {y_axis})'
        
        st.markdown("---")
        
        # Age-Based Highlighting
        st.markdown("**🎂 Age-Based Highlighting**")
        col1, col2 = st.columns(2)
        
        with col1:
            highlight_u23 = st.checkbox(
                "U23 Players", 
                help="Highlight players under 23 years old"
            )
        
        with col2:
            highlight_u20 = st.checkbox(
                "U20 Players", 
                help="Highlight players under 20 years old"
            )
        
        if highlight_u23:
            u23_players = filtered_df[filtered_df['Age'] < 23]['Player Name'].tolist()
            for player in u23_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#6BCF7F'  # Light green for young talent
                    highlight_reasons[player] = f'U23 Player (Age: {filtered_df[filtered_df["Player Name"] == player]["Age"].iloc[0]})'
        
        if highlight_u20:
            u20_players = filtered_df[filtered_df['Age'] < 20]['Player Name'].tolist()
            for player in u20_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#00D4FF'  # Bright blue for very young talent
                    highlight_reasons[player] = f'U20 Player (Age: {filtered_df[filtered_df["Player Name"] == player]["Age"].iloc[0]})'
        
        st.markdown("---")

        st.markdown("⚙️ Display Controls")
        col1, col2 = st.columns(2)
        with col1:
            default_opacity = st.slider(
                    "Default Player Opacity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Opacity for non-highlighted players"
            )
            
        with col2:
            highlight_opacity = st.slider(
                    "Highlighted Player Opacity",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="Opacity for highlighted players"
            )
            
    st.markdown("---")
    
    # Show debug info about name display after highlighting logic
    # if show_names == "Always":
    #     st.info(f"📝 **Name Display**: Showing top 30 players by combined score")
    # elif show_names == "Only Highlighted":
    #     highlighted_count = len(highlighted_players)
    #     if highlighted_count > 0:
    #         st.info(f"📝 **Name Display**: Showing {highlighted_count} highlighted players")
    #     else:
    #         st.warning(f"📝 **Name Display**: No players highlighted - select highlighting options above to see names")
    # else:  # Never
    #     highlighted_count = len(highlighted_players)
    #     if highlighted_count > 0:
    #         st.info(f"📝 **Name Display**: Auto-showing {highlighted_count} highlighted players")
    #     else:
    #         st.info(f"📝 **Name Display**: No names shown (select highlighting options to auto-show names)")
    
    # Create the scatter plot
    fig = create_advanced_scatter_plot(
        filtered_df, 
        x_axis, 
        y_axis, 
        highlighted_players, 
        highlight_colors,
        highlight_reasons,
        show_trendline, 
        show_median_lines,
        show_names,
        default_opacity,
        highlight_opacity
    )
    
    st.plotly_chart(fig, use_container_width=True, key="advanced_scatter_plot")
    
    # Summary statistics
    show_scatter_summary(filtered_df, x_axis, y_axis, highlighted_players, highlight_reasons)

def create_advanced_scatter_plot(df, x_axis, y_axis, highlighted_players, highlight_colors, 
                                highlight_reasons, show_trendline, show_median_lines, 
                                show_names, default_opacity, highlight_opacity):
    """Create the advanced scatter plot with all highlighting options"""
    
    fig = go.Figure()
    
    # Default color for non-highlighted players
    default_color = '#888888'
    
    # Add non-highlighted players first (so they appear behind highlighted ones)
    non_highlighted_df = df[~df['Player Name'].isin(highlighted_players)]
    
    if len(non_highlighted_df) > 0:
        fig.add_trace(go.Scatter(
            x=non_highlighted_df[x_axis],
            y=non_highlighted_df[y_axis],
            mode='markers',
            marker=dict(
                color=default_color,
                size=8,
                opacity=default_opacity,
                line=dict(width=0.5, color='white')
            ),
            text=[
                f"<b>{row['Player Name']}</b><br>"
                f"Team: {row['Team']}<br>"
                f"{x_axis}: {row[x_axis]:.1f}<br>"
                f"{y_axis}: {row[y_axis]:.1f}"
                for _, row in non_highlighted_df.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Other Players',
            showlegend=True
        ))
    
    # Add highlighted players by color group
    color_groups = {}
    for player in highlighted_players:
        color = highlight_colors.get(player, default_color)
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(player)
    
    # Create a trace for each color group
    color_names = {
        '#FF6B9D': 'Selected Players',
        '#4ECDC4': 'Team Players',
        '#1a9641': 'Top Performers',
        '#d73027': 'Bottom Performers',
        '#FFD93D': 'Top Combined',
        '#FF8C42': 'Bottom Combined',
        '#6BCF7F': 'U23 Players',
        '#00D4FF': 'U20 Players'
    }
    
    for color, players in color_groups.items():
        highlighted_subset = df[df['Player Name'].isin(players)]
        
        fig.add_trace(go.Scatter(
            x=highlighted_subset[x_axis],
            y=highlighted_subset[y_axis],
            mode='markers',
            marker=dict(
                color=color,
                size=12,
                opacity=highlight_opacity,
                line=dict(width=1, color='white')
            ),
            text=[
                f"<b>{row['Player Name']}</b><br>"
                f"Team: {row['Team']}<br>"
                f"{x_axis}: {row[x_axis]:.1f}<br>"
                f"{y_axis}: {row[y_axis]:.1f}<br>"
                f"Highlight: {highlight_reasons.get(row['Player Name'], 'Unknown')}"
                for _, row in highlighted_subset.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name=color_names.get(color, 'Highlighted'),
            showlegend=True
        ))
    
    # Add trend line if requested
    if show_trendline:
        x_vals = df[x_axis].values
        y_vals = df[y_axis].values
        
        # Remove any NaN values
        mask = ~(pd.isna(x_vals) | pd.isna(y_vals))
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]
        
        if len(x_clean) > 1:
            # Calculate simple linear regression
            coef = np.polyfit(x_clean, y_clean, 1)
            line_x = np.array([x_clean.min(), x_clean.max()])
            line_y = coef[0] * line_x + coef[1]
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ))
    
    # Add median lines if requested
    if show_median_lines:
        x_median = df[x_axis].median()
        y_median = df[y_axis].median()
        
        # Add vertical median line (X-axis median)
        fig.add_vline(
            x=x_median,
            line=dict(color='gray', width=2, dash='dot'),
            annotation_text=f"X Median: {x_median:.1f}",
            annotation_position="top"
        )
        
        # Add horizontal median line (Y-axis median)  
        fig.add_hline(
            y=y_median,
            line=dict(color='gray', width=2, dash='dot'),
            annotation_text=f"Y Median: {y_median:.1f}",
            annotation_position="right"
        )
    
    # Add player name labels based on show_names setting
    if show_names == "Always":
        # Limit to top 30 players by combined score to prevent performance issues
        df_copy = df.copy()
        df_copy['combined_score'] = (df_copy[x_axis] + df_copy[y_axis]) / 2
        players_to_label = df_copy.nlargest(30, 'combined_score')
    elif show_names == "Only Highlighted":
        # Debug: Check if highlighted_players has values
        if highlighted_players:
            players_to_label = df[df['Player Name'].isin(highlighted_players)]
        else:
            players_to_label = pd.DataFrame()  # Empty if no highlights
    else:  # Never - but still show highlighted players automatically
        if highlighted_players:
            players_to_label = df[df['Player Name'].isin(highlighted_players)]
        else:
            players_to_label = pd.DataFrame()  # Empty if no highlights
    
    if len(players_to_label) > 0:
        # Limit total annotations for performance (max 50)
        max_annotations = min(50, len(players_to_label))
        players_to_process = players_to_label.head(max_annotations)
        
        for i, (_, player_row) in enumerate(players_to_process.iterrows()):
            # Alternate positioning to reduce overlap
            x_offset = 15 if i % 2 == 0 else -15
            y_offset = 12 if i % 4 < 2 else -12
            
            # Ensure we have valid coordinates
            if pd.isna(player_row[x_axis]) or pd.isna(player_row[y_axis]):
                continue
                
            fig.add_annotation(
                x=player_row[x_axis],
                y=player_row[y_axis],
                text=player_row['Player Name'],
                showarrow=False,
                font=dict(
                    size=9,
                    color='#333333',
                    family='Arial Bold'
                ),
                xshift=x_offset,
                yshift=y_offset
            )
    
    # Update layout
    fig.update_layout(
        title=f"{x_axis} vs {y_axis}",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        height=600,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def show_scatter_summary(df, x_axis, y_axis, highlighted_players, highlight_reasons):
    """Show summary statistics and highlighted players info"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Correlation info
        correlation = df[x_axis].corr(df[y_axis])
        st.metric(
            "Correlation Coefficient",
            f"{correlation:.3f}",
            help=f"Correlation between {x_axis} and {y_axis}"
        )
    
    with col2:
        # Highlighted players count
        st.metric(
            "Highlighted Players",
            len(highlighted_players),
            help="Number of players currently highlighted"
        )
    
    with col3:
        # Total players
        st.metric(
            "Total Players",
            len(df),
            help="Total players in current view"
        )
    
    # Show highlighted players summary if any
    if highlighted_players:
        st.subheader("🎨 Currently Highlighted Players")
        
        highlighted_data = []
        for player in highlighted_players:
            player_info = df[df['Player Name'] == player].iloc[0]
            highlighted_data.append({
                'Player': player,
                'Team': player_info['Team'],
                'Position': player_info['Position'],
                'Age': player_info['Age'],
                f'{x_axis}': player_info[x_axis],
                f'{y_axis}': player_info[y_axis],
                'Highlight Reason': highlight_reasons.get(player, 'Unknown')
            })
        
        highlighted_df = pd.DataFrame(highlighted_data)
        st.dataframe(highlighted_df, use_container_width=True, hide_index=True)
    
    # Quick insights
    st.subheader("📊 Quick Insights")
    
    # Find interesting relationships
    insights = []
    
    # Top performers in both metrics
    df_copy = df.copy()
    df_copy['combined_rank'] = df_copy[x_axis].rank(ascending=False) + df_copy[y_axis].rank(ascending=False)
    top_combined = df_copy.nsmallest(3, 'combined_rank')
    
    insights.append(f"**Best Combined Performance**: {', '.join(top_combined['Player Name'].tolist())}")
    
    # Correlation strength
    if abs(correlation) > 0.7:
        correlation_strength = "Strong"
    elif abs(correlation) > 0.4:
        correlation_strength = "Moderate"
    else:
        correlation_strength = "Weak"
    
    correlation_direction = "positive" if correlation > 0 else "negative"
    insights.append(f"**Correlation**: {correlation_strength} {correlation_direction} relationship between {x_axis} and {y_axis}")
    
    # Outliers
    x_q75, x_q25 = df[x_axis].quantile([0.75, 0.25])
    y_q75, y_q25 = df[y_axis].quantile([0.75, 0.25])
    x_iqr = x_q75 - x_q25
    y_iqr = y_q75 - y_q25
    
    x_outliers = df[(df[x_axis] < x_q25 - 1.5 * x_iqr) | (df[x_axis] > x_q75 + 1.5 * x_iqr)]
    y_outliers = df[(df[y_axis] < y_q25 - 1.5 * y_iqr) | (df[y_axis] > y_q75 + 1.5 * y_iqr)]
    
    if len(x_outliers) > 0:
        insights.append(f"**{x_axis} Outliers**: {', '.join(x_outliers['Player Name'].head(3).tolist())}")
    
    if len(y_outliers) > 0:
        insights.append(f"**{y_axis} Outliers**: {', '.join(y_outliers['Player Name'].head(3).tolist())}")
    
    for insight in insights:
        st.write(f"• {insight}")