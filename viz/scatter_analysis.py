import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

def detect_theme():
    """Detect if the current theme is light or dark"""
    try:
        # Try to detect theme from Streamlit's session state or config
        if hasattr(st, '_config') and hasattr(st._config, 'get_option'):
            theme = st._config.get_option('theme.base')
            if theme == 'dark':
                return 'dark'
        
        # Check if user has set a preference in session state
        if 'theme' in st.session_state:
            return st.session_state.theme
        
        # Default to light theme
        return 'light'
    except:
        # Fallback to light theme if detection fails
        return 'light'

def get_theme_colors(theme='light'):
    """Get color scheme based on current theme"""
    if theme == 'dark':
        return {
            'background': 'rgba(14, 17, 23, 1)',  # Dark background
            'grid': '#404040',                    # Medium gray for grid
            'text': '#FFFFFF',                    # White text
            'bg_points': '#666666',               # Lighter gray for visibility on dark
            'point_border': '#888888',            # Medium gray border
            'subtitle_text': '#B0B0B0'           # Light gray for subtitles
        }
    else:  # light theme
        return {
            'background': 'white',                # White background
            'grid': '#F8F9FA',                   # Very light gray for grid  
            'text': '#2C3E50',                   # Dark blue-gray text
            'bg_points': '#9E9E9E',              # Medium gray points
            'point_border': 'white',             # White border
            'subtitle_text': '#7F8C8D'           # Gray for subtitles
        }

def show_scatter_analysis(filtered_df):
    """
    Advanced scatter plot analysis for player performance data

    Features:
    - X/Y axis selection from available metrics
    - Multiple highlighting options (players, teams, performance, age)
    - Interactive hover with player details
    - Trend lines and median lines
    - Professional styling consistent with dashboard theme
    - Includes players with 0-value metrics (not filtered out)
    - Groups overlapping players with smart labeling
    """
    st.header("🎯 Advanced Scatter Analysis")

    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return

    # Ensure 0-value metrics are explicitly preserved in analysis
    # Note: filtered_df should include players with 0 values for comprehensive analysis
    
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
                    highlight_colors[player] = '#DC143C'  # Crimson red for selected players
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
                        highlight_colors[player] = '#1E90FF'  # Blue for team selections
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
                        highlight_colors[player] = '#228B22'  # Forest green for top performers
                        highlight_reasons[player] = f'Top 10 {x_axis}'
            
            if bottom_x:
                bottom_x_players = filtered_df.nsmallest(10, x_axis)['Player Name'].tolist()
                for player in bottom_x_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#FF4500'  # Orange red for bottom performers
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
                        highlight_colors[player] = '#228B22'  # Forest green for top performers
                        highlight_reasons[player] = f'Top 10 {y_axis}'
            
            if bottom_y:
                bottom_y_players = filtered_df.nsmallest(10, y_axis)['Player Name'].tolist()
                for player in bottom_y_players:
                    if player not in highlighted_players:
                        highlighted_players.add(player)
                        highlight_colors[player] = '#FF4500'  # Orange red for bottom performers
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
                    highlight_colors[player] = '#DAA520'  # Goldenrod for top combined
                    highlight_reasons[player] = f'Top 10 Combined ({x_axis} + {y_axis})'
        
        if bottom_combined:
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['combined_score'] = (filtered_df_copy[x_axis] + filtered_df_copy[y_axis]) / 2
            bottom_combined_players = filtered_df_copy.nsmallest(10, 'combined_score')['Player Name'].tolist()
            for player in bottom_combined_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#FF6347'  # Tomato for bottom combined
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
                    highlight_colors[player] = '#32CD32'  # Lime green for young talent
                    highlight_reasons[player] = f'U23 Player (Age: {filtered_df[filtered_df["Player Name"] == player]["Age"].iloc[0]})'
        
        if highlight_u20:
            u20_players = filtered_df[filtered_df['Age'] < 20]['Player Name'].tolist()
            for player in u20_players:
                if player not in highlighted_players:
                    highlighted_players.add(player)
                    highlight_colors[player] = '#4169E1'  # Royal blue for very young talent
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
    """Create enhanced scatter plot with professional styling and smart labeling"""
    
    fig = go.Figure()
    
    # Detect current theme and get appropriate colors
    current_theme = detect_theme()
    theme_colors = get_theme_colors(current_theme)
    
    # Theme-aware color scheme
    default_color = theme_colors['bg_points']      # Background points color
    grid_color = theme_colors['grid']              # Grid color
    text_color = theme_colors['text']              # Text color
    subtitle_color = theme_colors['subtitle_text'] # Subtitle color
    point_border = theme_colors['point_border']    # Point border color
    bg_color = theme_colors['background']          # Background color

    # Smart labeling with overlap prevention and grouping (moved to beginning)
    players_to_label = get_players_for_labeling(df, highlighted_players, show_names, x_axis, y_axis)

    # Create grouped player information for enhanced hover templates
    player_groups = {}
    grouped_labels = {}

    if len(players_to_label) > 0:
        # Group overlapping players
        player_groups = group_overlapping_players(players_to_label, x_axis, y_axis)

        # Create grouped labels
        grouped_labels = create_grouped_labels(player_groups, players_to_label, highlighted_players, x_axis, y_axis)

    # Add non-highlighted players first (so they appear behind highlighted ones)
    non_highlighted_df = df[~df['Player Name'].isin(highlighted_players)]
    
    if len(non_highlighted_df) > 0:
        fig.add_trace(go.Scatter(
            x=non_highlighted_df[x_axis],
            y=non_highlighted_df[y_axis],
            mode='markers',
            marker=dict(
                color=default_color,
                size=15,  # Keep size 15 as requested
                opacity=0.4,  # Fill opacity for clear visibility of distribution
                line=dict(width=1, color=point_border)  # Theme-aware border color
            ),
            text=[
                create_enhanced_hover_text(row, x_axis, y_axis, player_groups, theme_colors)
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
    
    # Professional color scheme for highlights
    color_names = {
        '#DC143C': 'Selected Players',      # Crimson red - primary highlight
        '#1E90FF': 'Team Players',          # Dodger blue - team selections
        '#228B22': 'Top Performers',        # Forest green - top performers
        '#FF4500': 'Bottom Performers',     # Orange red - needs improvement
        '#DAA520': 'Top Combined',          # Goldenrod - excellent combined
        '#FF6347': 'Bottom Combined',       # Tomato - combined improvement needed
        '#32CD32': 'U23 Players',           # Lime green - young talent
        '#4169E1': 'U20 Players'            # Royal blue - very young talent
    }
    
    # Update highlight_colors to use new professional colors
    professional_color_mapping = {
        '#FF6B9D': '#DC143C',  # Selected Players
        '#4ECDC4': '#1E90FF',  # Team Players
        '#1a9641': '#228B22',  # Top Performers
        '#d73027': '#FF4500',  # Bottom Performers
        '#FFD93D': '#DAA520',  # Top Combined
        '#FF8C42': '#FF6347',  # Bottom Combined
        '#6BCF7F': '#32CD32',  # U23 Players
        '#00D4FF': '#4169E1'   # U20 Players
    }
    
    for color, players in color_groups.items():
        highlighted_subset = df[df['Player Name'].isin(players)]
        
        # Map old colors to new professional colors
        professional_color = professional_color_mapping.get(color, color)
        
        fig.add_trace(go.Scatter(
            x=highlighted_subset[x_axis],
            y=highlighted_subset[y_axis],
            mode='markers',
            marker=dict(
                color=professional_color,
                size=15,  # Keep size 15 as requested
                opacity=0.9,  # Slightly reduced fill opacity for better border definition
                line=dict(width=2, color=point_border)  # Theme-aware border color
            ),
            text=[
                create_enhanced_hover_text(row, x_axis, y_axis, player_groups, theme_colors) +
                f"<br><span style='color: {professional_color}; font-weight: bold'>Highlight: {highlight_reasons.get(row['Player Name'], 'Unknown')}</span>"
                for _, row in highlighted_subset.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name=color_names.get(professional_color, 'Highlighted'),
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
    
    # Add annotations for each group's primary player (using previously computed grouped_labels)
    if len(grouped_labels) > 0:
        for primary_player, label_info in grouped_labels.items():
            # Use red color for highlighted players, theme-aware color for others
            label_color = '#DC143C' if primary_player in highlighted_players else text_color

            fig.add_annotation(
                x=label_info['x_pos'],
                y=label_info['y_pos'],
                text=label_info['label_text'],
                showarrow=False,
                font=dict(
                    size=10,
                    color=label_color,
                    family='Arial'
                ),
                xshift=label_info['x_offset'],
                yshift=label_info['y_offset']
            )
    
    # Create contextual subtitle with data information
    position_info = df['Position'].unique()
    team_count = df['Team'].nunique()
    age_range = f"{df['Age'].min()}-{df['Age'].max()}"
    
    subtitle = f"Players aged {age_range} | {team_count} teams | Positions: {', '.join(position_info[:3])}"
    if len(position_info) > 3:
        subtitle += f" (+{len(position_info)-3} more)"
    
    # Update layout with theme-aware styling
    fig.update_layout(
        title=dict(
            text=f"<b style='font-size: 20px; color: {text_color}'>{x_axis} vs {y_axis}</b><br>"
                 f"<span style='font-size: 14px; color: {subtitle_color}'>{subtitle}</span>",
            x=0.05,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(
                text=f"<b>{x_axis}</b>",
                font=dict(size=14, color=text_color)
            ),
            gridcolor=grid_color,
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=text_color)
        ),
        yaxis=dict(
            title=dict(
                text=f"<b>{y_axis}</b>",
                font=dict(size=14, color=text_color)
            ),
            gridcolor=grid_color,
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            tickfont=dict(size=12, color=text_color)
        ),
        height=650,
        hovermode='closest',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=text_color),
            bgcolor=f'rgba(128,128,128,0.1)' if current_theme == 'dark' else 'rgba(255,255,255,0.8)',
            bordercolor=f'rgba(255,255,255,0.2)' if current_theme == 'dark' else 'rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=100)
    )
    
    return fig

def get_players_for_labeling(df, highlighted_players, show_names, x_axis, y_axis):
    """Determine which players should get labels based on show_names setting"""
    if show_names == "Always":
        # Limit to top 20 players by combined score to prevent performance issues
        df_copy = df.copy()
        df_copy['combined_score'] = (df_copy[x_axis] + df_copy[y_axis]) / 2
        return df_copy.nlargest(20, 'combined_score')
    elif show_names == "Only Highlighted":
        if highlighted_players:
            return df[df['Player Name'].isin(highlighted_players)]
        else:
            return pd.DataFrame()  # Empty if no highlights
    else:  # Never - but still show highlighted players automatically
        if highlighted_players:
            return df[df['Player Name'].isin(highlighted_players)]
        else:
            return pd.DataFrame()  # Empty if no highlights

def calculate_smart_label_positions(players_df, x_axis, y_axis):
    """Calculate smart label positions to minimize overlaps"""
    if len(players_df) == 0:
        return {}
    
    positions = {}
    
    # Define possible label positions (offsets from point)
    label_offsets = [
        (20, 15),   # Top-right
        (-20, 15),  # Top-left
        (20, -15),  # Bottom-right
        (-20, -15), # Bottom-left
        (25, 0),    # Right
        (-25, 0),   # Left
        (0, 20),    # Top
        (0, -20),   # Bottom
    ]
    
    # For each player, find the best label position
    for _, player_row in players_df.iterrows():
        player_name = player_row['Player Name']
        x_pos, y_pos = player_row[x_axis], player_row[y_axis]
        
        if pd.isna(x_pos) or pd.isna(y_pos):
            continue
        
        # Find the best offset that minimizes conflicts
        best_offset = label_offsets[0]  # Default to top-right
        min_conflicts = float('inf')
        
        for offset in label_offsets:
            conflicts = 0
            label_x = x_pos + offset[0]
            label_y = y_pos + offset[1]
            
            # Check conflicts with other players' positions
            for _, other_row in players_df.iterrows():
                if other_row['Player Name'] == player_name:
                    continue
                    
                other_x, other_y = other_row[x_axis], other_row[y_axis]
                if pd.isna(other_x) or pd.isna(other_y):
                    continue
                
                # Calculate distance between label position and other points
                distance = ((label_x - other_x) ** 2 + (label_y - other_y) ** 2) ** 0.5
                if distance < 30:  # Threshold for conflict
                    conflicts += 1
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_offset = offset
        
        positions[player_name] = best_offset
    
    return positions

def group_overlapping_players(players_df, x_axis, y_axis, threshold_percentage=0.02):
    """
    Group players that are at similar coordinates to prevent label overlap

    Args:
        players_df: DataFrame of players to label
        x_axis, y_axis: The metrics being plotted
        threshold_percentage: Percentage of axis range to consider as "overlapping" (default 2%)

    Returns:
        Dict mapping group_id to list of player names in that group
    """
    if len(players_df) == 0:
        return {}

    # Calculate thresholds based on data range
    x_range = players_df[x_axis].max() - players_df[x_axis].min()
    y_range = players_df[y_axis].max() - players_df[y_axis].min()

    x_threshold = x_range * threshold_percentage if x_range > 0 else 1
    y_threshold = y_range * threshold_percentage if y_range > 0 else 1

    # Group players by proximity
    groups = {}
    group_id = 0
    assigned_players = set()

    for _, player_row in players_df.iterrows():
        player_name = player_row['Player Name']
        if player_name in assigned_players:
            continue

        x_pos, y_pos = player_row[x_axis], player_row[y_axis]
        if pd.isna(x_pos) or pd.isna(y_pos):
            continue

        # Find all players within threshold
        current_group = [player_name]
        assigned_players.add(player_name)

        for _, other_row in players_df.iterrows():
            other_name = other_row['Player Name']
            if other_name in assigned_players:
                continue

            other_x, other_y = other_row[x_axis], other_row[y_axis]
            if pd.isna(other_x) or pd.isna(other_y):
                continue

            # Check if within threshold
            if (abs(x_pos - other_x) <= x_threshold and
                abs(y_pos - other_y) <= y_threshold):
                current_group.append(other_name)
                assigned_players.add(other_name)

        groups[group_id] = current_group
        group_id += 1

    return groups

def get_primary_player_from_group(group_players, players_df, highlighted_players, x_axis, y_axis):
    """
    Select the primary player to represent a group of overlapping players

    Priority:
    1. Highlighted players (if any in group)
    2. Player with highest combined score
    3. Alphabetical order (as fallback)
    """
    group_df = players_df[players_df['Player Name'].isin(group_players)]

    # Priority 1: Highlighted players
    highlighted_in_group = [p for p in group_players if p in highlighted_players]
    if highlighted_in_group:
        # If multiple highlighted, pick the one with highest combined score
        highlighted_df = group_df[group_df['Player Name'].isin(highlighted_in_group)]
        highlighted_df = highlighted_df.copy()
        highlighted_df['combined_score'] = (highlighted_df[x_axis] + highlighted_df[y_axis]) / 2
        return highlighted_df.loc[highlighted_df['combined_score'].idxmax(), 'Player Name']

    # Priority 2: Highest combined score
    group_df = group_df.copy()
    group_df['combined_score'] = (group_df[x_axis] + group_df[y_axis]) / 2
    return group_df.loc[group_df['combined_score'].idxmax(), 'Player Name']

def create_grouped_labels(player_groups, players_df, highlighted_players, x_axis, y_axis):
    """
    Create label text and positions for grouped players

    Returns:
        Dict mapping primary_player_name to (label_text, x_pos, y_pos, x_offset, y_offset)
    """
    grouped_labels = {}

    # Define possible label positions (offsets from point)
    label_offsets = [
        (20, 15),   # Top-right
        (-20, 15),  # Top-left
        (20, -15),  # Bottom-right
        (-20, -15), # Bottom-left
        (25, 0),    # Right
        (-25, 0),   # Left
        (0, 20),    # Top
        (0, -20),   # Bottom
    ]

    used_positions = []

    for group_id, group_players in player_groups.items():
        if len(group_players) == 0:
            continue

        # Get primary player for this group
        primary_player = get_primary_player_from_group(
            group_players, players_df, highlighted_players, x_axis, y_axis
        )

        # Get position from primary player
        primary_row = players_df[players_df['Player Name'] == primary_player].iloc[0]
        x_pos, y_pos = primary_row[x_axis], primary_row[y_axis]

        if pd.isna(x_pos) or pd.isna(y_pos):
            continue

        # Create label text
        if len(group_players) == 1:
            label_text = f"<b>{primary_player}</b>"
        else:
            additional_count = len(group_players) - 1
            label_text = f"<b>{primary_player}</b><br><span style='font-size: 8px'>+{additional_count} more player{'s' if additional_count > 1 else ''}</span>"

        # Find best offset to avoid conflicts with other labels
        best_offset = label_offsets[0]
        min_conflicts = float('inf')

        for offset in label_offsets:
            conflicts = 0
            test_pos = (x_pos + offset[0], y_pos + offset[1])

            for used_pos in used_positions:
                distance = ((test_pos[0] - used_pos[0]) ** 2 + (test_pos[1] - used_pos[1]) ** 2) ** 0.5
                if distance < 40:  # Increased threshold for label separation
                    conflicts += 1

            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_offset = offset

        used_positions.append((x_pos + best_offset[0], y_pos + best_offset[1]))

        grouped_labels[primary_player] = {
            'label_text': label_text,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'x_offset': best_offset[0],
            'y_offset': best_offset[1],
            'group_players': group_players,
            'is_primary': True
        }

    return grouped_labels

def create_enhanced_hover_text(row, x_axis, y_axis, player_groups, theme_colors):
    """
    Create enhanced hover text that shows grouped players if applicable

    Args:
        row: DataFrame row for the current player
        x_axis, y_axis: The metrics being plotted
        player_groups: Dict of grouped players
        theme_colors: Theme color dictionary

    Returns:
        Enhanced hover text string
    """
    player_name = row['Player Name']
    text_color = theme_colors['text']
    subtitle_color = theme_colors['subtitle_text']

    # Find if this player is part of a group
    player_group = None
    for group_id, group_players in player_groups.items():
        if player_name in group_players:
            player_group = group_players
            break

    # Base player information
    hover_text = (
        f"<b style='color: {text_color}'>{player_name}</b><br>"
        f"<span style='color: {subtitle_color}'>Team: {row['Team']}</span><br>"
        f"<span style='color: {subtitle_color}'>Position: {row['Position']}</span><br>"
        f"<span style='color: {subtitle_color}'>Age: {row['Age']}</span><br>"
        f"<span style='color: {subtitle_color}'>{x_axis}: {row[x_axis]:.1f}</span><br>"
        f"<span style='color: {subtitle_color}'>{y_axis}: {row[y_axis]:.1f}</span>"
    )

    # Add grouped player information if applicable
    if player_group and len(player_group) > 1:
        other_players = [p for p in player_group if p != player_name]
        if len(other_players) <= 3:
            # Show all players if 3 or fewer
            hover_text += f"<br><br><span style='color: {subtitle_color}; font-size: 10px'><b>Nearby Players:</b><br>"
            hover_text += "<br>".join(other_players)
            hover_text += "</span>"
        else:
            # Show first 2 and count
            hover_text += f"<br><br><span style='color: {subtitle_color}; font-size: 10px'><b>Nearby Players:</b><br>"
            hover_text += "<br>".join(other_players[:2])
            hover_text += f"<br>+{len(other_players) - 2} more players</span>"

    return hover_text

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