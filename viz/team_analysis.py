import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any

# Define negative metrics for player data (lower values are better)
PLAYER_NEGATIVE_METRICS = ['Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target']

def detect_theme():
    """Detect if the current theme is light or dark"""
    try:
        if hasattr(st, '_config') and hasattr(st._config, 'get_option'):
            theme = st._config.get_option('theme.base')
            if theme == 'dark':
                return 'dark'
        if 'theme' in st.session_state:
            return st.session_state.theme
        return 'light'
    except:
        return 'light'

def get_theme_colors():
    """Get theme-appropriate colors"""
    theme = detect_theme()
    if theme == 'dark':
        return {
            'background': '#0E1117',
            'paper': '#262730',
            'text': '#FAFAFA',
            'grid': '#404040',
            'primary': '#FF4B4B',
            'secondary': '#00D4AA',
            'tertiary': '#FFD23F',
            'quaternary': '#9D4EDD'
        }
    else:
        return {
            'background': '#FFFFFF',
            'paper': '#FFFFFF',
            'text': '#262730',
            'grid': '#E5E5E5',
            'primary': '#FF4B4B',
            'secondary': '#00D4AA',
            'tertiary': '#FFD23F',
            'quaternary': '#9D4EDD'
        }

def get_position_groups():
    """Get position groupings for analysis"""
    return {
        'Goalkeepers': ['P. GAWANG'],
        'Defenders': ['BELAKANG'],
        'Midfielders': ['TENGAH'],
        'Forwards': ['DEPAN']
    }

def calculate_team_strengths_weaknesses(team_df: pd.DataFrame, league_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate team strengths and weaknesses based on league percentiles"""
    
    # Key metrics for analysis
    attack_metrics = ['Goal', 'Assist', 'Shoot On Target', 'Create Chance']
    defense_metrics = ['Tackle', 'Intercept', 'Block', 'Clearance']
    possession_metrics = ['Passing', 'Cross', 'Dribble Success']
    
    all_metrics = attack_metrics + defense_metrics + possession_metrics
    
    strengths = []
    weaknesses = []
    
    for metric in all_metrics:
        if metric in team_df.columns and metric in league_df.columns:
            team_avg = team_df[metric].mean()
            league_avg = league_df[metric].mean()
            league_std = league_df[metric].std()
            
            if league_std > 0:
                z_score = (team_avg - league_avg) / league_std
                percentile = (league_df[metric] <= team_avg).mean() * 100
                
                if z_score > 1.0:  # Above 84th percentile
                    strengths.append({
                        'metric': metric,
                        'percentile': percentile,
                        'team_avg': team_avg,
                        'league_avg': league_avg
                    })
                elif z_score < -1.0:  # Below 16th percentile
                    weaknesses.append({
                        'metric': metric,
                        'percentile': percentile,
                        'team_avg': team_avg,
                        'league_avg': league_avg
                    })
    
    # Sort by percentile
    strengths.sort(key=lambda x: x['percentile'], reverse=True)
    weaknesses.sort(key=lambda x: x['percentile'])
    
    return {
        'strengths': strengths[:5],  # Top 5 strengths
        'weaknesses': weaknesses[:5]  # Top 5 weaknesses
    }

def get_players_to_watch(team_df: pd.DataFrame, league_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify key players to watch based on performance metrics"""
    
    key_metrics = ['Goal', 'Assist', 'Passing', 'Tackle', 'Intercept']
    players_to_watch = []
    
    for _, player in team_df.iterrows():
        player_score = 0
        reasons = []
        
        for metric in key_metrics:
            if metric in team_df.columns and metric in league_df.columns:
                player_val = player[metric]
                league_percentile = (league_df[metric] <= player_val).mean() * 100
                
                if league_percentile >= 90:
                    player_score += 3
                    reasons.append(f"Top 10% in {metric}")
                elif league_percentile >= 75:
                    player_score += 2
                elif league_percentile >= 50:
                    player_score += 1
        
        if player_score >= 5:  # Minimum threshold
            players_to_watch.append({
                'name': player['Player Name'],
                'position': player['Position'],
                'age': player.get('Age', 'N/A'),
                'score': player_score,
                'reasons': reasons[:3]  # Top 3 reasons
            })
    
    # Sort by score and return top players
    players_to_watch.sort(key=lambda x: x['score'], reverse=True)
    return players_to_watch[:8]  # Top 8 players

def create_position_distribution_chart(teams_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create positional distribution chart"""
    colors = get_theme_colors()
    position_groups = get_position_groups()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(position_groups.keys()),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    team_colors = [colors['primary'], colors['secondary'], colors['tertiary']]
    
    for i, (group_name, positions) in enumerate(position_groups.items()):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        for j, (team_name, team_df) in enumerate(teams_data.items()):
            position_counts = team_df[team_df['Position'].isin(positions)]['Position'].value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=position_counts.index,
                    y=position_counts.values,
                    name=team_name,
                    marker_color=team_colors[j % len(team_colors)],
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font_color=colors['text'],
        title_text="Position Distribution Analysis",
        title_x=0.5
    )
    
    return fig

def create_age_distribution_chart(teams_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create age distribution analysis"""
    colors = get_theme_colors()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Age Distribution", "Average Age by Position"],
        specs=[[{"type": "histogram"}, {"type": "bar"}]]
    )
    
    team_colors = [colors['primary'], colors['secondary'], colors['tertiary']]
    
    # Age histogram
    for i, (team_name, team_df) in enumerate(teams_data.items()):
        if 'Age' in team_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=team_df['Age'],
                    name=team_name,
                    marker_color=team_colors[i % len(team_colors)],
                    opacity=0.7,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Average age by position
    for i, (team_name, team_df) in enumerate(teams_data.items()):
        if 'Age' in team_df.columns:
            avg_age_by_pos = team_df.groupby('Position')['Age'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=avg_age_by_pos.index,
                    y=avg_age_by_pos.values,
                    name=f"{team_name} Avg Age",
                    marker_color=team_colors[i % len(team_colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
    
    fig.update_layout(
        height=400,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font_color=colors['text']
    )
    
    return fig

def create_performance_radar(teams_data: Dict[str, pd.DataFrame], league_df: pd.DataFrame) -> go.Figure:
    """Create team performance radar chart"""
    colors = get_theme_colors()
    
    # Key performance categories
    performance_metrics = {
        'Attack': ['Goal', 'Assist'],
        'Creativity': ['Create Chance'],
        'Possession': ['Passing'],
        'Defense': ['Tackle', 'Intercept'],
        'Physicality': ['Appearances']
    }
    
    fig = go.Figure()
    team_colors = [colors['primary'], colors['secondary'], colors['tertiary']]
    
    for i, (team_name, team_df) in enumerate(teams_data.items()):
        categories = []
        percentiles = []
        
        for category, metrics in performance_metrics.items():
            category_percentiles = []
            
            for metric in metrics:
                if metric in team_df.columns and metric in league_df.columns:
                    team_avg = team_df[metric].mean()
                    percentile = (league_df[metric] <= team_avg).mean() * 100
                    category_percentiles.append(percentile)
            
            if category_percentiles:
                categories.append(category)
                percentiles.append(np.mean(category_percentiles))
        
        # Close the radar chart
        categories.append(categories[0])
        percentiles.append(percentiles[0])
        
        fig.add_trace(go.Scatterpolar(
            r=percentiles,
            theta=categories,
            fill='toself',
            name=team_name,
            line_color=team_colors[i % len(team_colors)],
            fillcolor=team_colors[i % len(team_colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%"
            )
        ),
        showlegend=True,
        height=500,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font_color=colors['text']
    )
    
    return fig

def show_squad_list_table(team_df: pd.DataFrame, league_df: pd.DataFrame):
    """Show comprehensive squad list with all players and their metrics"""
    st.subheader("📋 Complete Squad List")
    
    # Get all metric columns (exclude info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in team_df.columns if col not in info_columns]
    
    # Create display dataframe with essential info + all metrics (remove duplicates)
    essential_info = ['Player Name', 'Position', 'Age', 'Team', 'Country', 'Appearances']
    display_columns = essential_info + metric_columns
    
    # Remove duplicates while preserving order
    seen = set()
    display_columns = [col for col in display_columns if not (col in seen or seen.add(col))]
    
    # Filter to only existing columns
    available_columns = [col for col in display_columns if col in team_df.columns]
    squad_df = team_df[available_columns].copy()
    
    # Clean and validate the dataframe data types safely
    for col in squad_df.columns:
        if col not in squad_df.columns:
            continue  # Skip non-existent columns
            
        try:
            if col in ['Player Name', 'Position', 'Team', 'Country']:
                # String columns - convert to string and fill NaN
                squad_df[col] = squad_df[col].astype(str).replace('nan', 'Unknown').fillna('Unknown')
            elif col in ['Age', 'Appearances'] or col in metric_columns:
                # Ensure we have a valid Series before pd.to_numeric
                if isinstance(squad_df[col], pd.Series) and len(squad_df[col]) > 0:
                    squad_df[col] = pd.to_numeric(squad_df[col], errors='coerce').fillna(0)
                else:
                    squad_df[col] = 0  # Default value for invalid columns
        except Exception as e:
            # Set safe default value for problematic columns
            if col in ['Player Name', 'Position', 'Team', 'Country']:
                squad_df[col] = 'Unknown'
            else:
                squad_df[col] = 0
    
    # Remove any completely empty columns
    squad_df = squad_df.dropna(axis=1, how='all')
    
    # Sort by position order
    position_order = {'P. GAWANG': 0, 'BELAKANG': 1, 'TENGAH': 2, 'DEPAN': 3}
    squad_df['_position_order'] = squad_df['Position'].map(position_order).fillna(4)
    squad_df = squad_df.sort_values(['_position_order', 'Player Name']).drop('_position_order', axis=1)
    squad_df = squad_df.reset_index(drop=True)
    
    # Display metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(squad_df))
    with col2:
        avg_age = squad_df['Age'].mean() if 'Age' in squad_df.columns else 0
        st.metric("Average Age", f"{avg_age:.1f}")
    with col3:
        total_goals = squad_df['Goal'].sum() if 'Goal' in squad_df.columns else 0
        st.metric("Squad Goals", int(total_goals))
    with col4:
        total_assists = squad_df['Assist'].sum() if 'Assist' in squad_df.columns else 0
        st.metric("Squad Assists", int(total_assists))
    
    # Validate dataframe before display
    if squad_df.empty:
        st.warning("No player data available for this team")
        return
    
    # Display the comprehensive table with error handling
    try:
        st.dataframe(
            squad_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Player Name": st.column_config.TextColumn("Player", width="medium"),
                "Position": st.column_config.TextColumn("Pos", width="small"),
                "Age": st.column_config.NumberColumn("Age", width="small"),
                "Appearances": st.column_config.NumberColumn("Apps", width="small")
            }
        )
    except Exception as e:
        st.warning(f"Display issue with advanced formatting. Showing basic table instead.")
        # Fallback to basic dataframe without column configuration
        st.dataframe(squad_df, use_container_width=True, hide_index=True)
    
    # Position breakdown
    st.write("### 👥 Squad Composition by Position")
    position_counts = squad_df['Position'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        # Position distribution chart
        fig = px.pie(
            values=position_counts.values, 
            names=position_counts.index,
            title="Position Distribution"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Position breakdown table
        position_breakdown = pd.DataFrame({
            'Position': position_counts.index,
            'Count': position_counts.values,
            'Percentage': (position_counts.values / len(squad_df) * 100).round(1)
        })
        st.dataframe(position_breakdown, hide_index=True, use_container_width=True)

def show_squad_heatmap(team_df: pd.DataFrame, league_df: pd.DataFrame):
    """Show squad heatmap with players sorted by position and metrics by category"""
    from utils.data_loader import get_player_metric_categories
    
    st.subheader("🔥 Squad Performance Heatmap")
    
    # Get metric categories
    try:
        metric_categories = get_player_metric_categories()
    except:
        # Fallback if function not available
        metric_categories = {
            'Attack': ['Goal', 'Assist', 'Shoot On Target'],
            'Defense': ['Tackle', 'Intercept', 'Block'],
            'Progression': ['Passing', 'Cross'],
            'Discipline': ['Yellow Card', 'Foul'],
            'Goalkeeper': ['Saves']
        }
    
    # Order metrics by category
    category_order = ['Goalkeeper', 'Defense', 'Discipline', 'Progression', 'Attack']
    ordered_metrics = []
    
    for category in category_order:
        if category in metric_categories:
            for metric in metric_categories[category]:
                if metric in team_df.columns:
                    ordered_metrics.append(metric)
    
    # Add any remaining metrics not in categories
    all_metric_columns = [col for col in team_df.columns 
                         if col not in ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']]
    for metric in all_metric_columns:
        if metric not in ordered_metrics:
            ordered_metrics.append(metric)
    
    if not ordered_metrics:
        st.warning("No metrics available for heatmap display")
        return
    
    # Sort players by position
    position_order = {'P. GAWANG': 0, 'BELAKANG': 1, 'TENGAH': 2, 'DEPAN': 3}
    team_df_sorted = team_df.copy()
    team_df_sorted['_position_order'] = team_df_sorted['Position'].map(position_order).fillna(4)
    team_df_sorted = team_df_sorted.sort_values(['_position_order', 'Player Name'])
    
    # Create normalized data with per-metric scaling
    normalized_data = pd.DataFrame(index=team_df_sorted.index, columns=ordered_metrics)
    
    for col in ordered_metrics:
        col_min = team_df_sorted[col].min()
        col_max = team_df_sorted[col].max()
        
        if col_max == col_min:  # Handle case where all values are the same
            normalized_data[col] = 50  # Set to middle value
        else:
            if col in PLAYER_NEGATIVE_METRICS:
                # For negative metrics, invert normalization (lower values = better = higher score)
                normalized_data[col] = 100 - ((team_df_sorted[col] - col_min) / (col_max - col_min) * 100)
            else:
                # For positive metrics, normal normalization (higher values = better = higher score)
                normalized_data[col] = (team_df_sorted[col] - col_min) / (col_max - col_min) * 100
    
    # Create heatmap
    fig = px.imshow(
        normalized_data.values,
        labels=dict(x="Metrics", y="Players", color="Normalized Score (0-100)"),
        x=ordered_metrics,
        y=team_df_sorted['Player Name'],
        aspect="auto",
        color_continuous_scale='RdYlGn',  # Red (poor) to Green (good)
        zmin=0,
        zmax=100,
        text_auto=False
    )
    
    # Add text annotations with actual values
    actual_values = team_df_sorted[ordered_metrics].values
    text_annotations = [[f"{int(val)}" for val in row] for row in actual_values]
    
    fig.update_traces(
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 8, "color": "black"},
        hovertemplate="<b>%{y}</b><br>%{x}<br>Value: %{text}<br>Normalized: %{z:.0f}<extra></extra>"
    )
    
    fig.update_layout(
        title=f"Squad Performance Heatmap - {team_df_sorted['Team'].iloc[0]}",
        height=max(500, len(team_df_sorted) * 30),  # Adjust height based on number of players
        xaxis_title="Performance Metrics (by Category)",
        yaxis_title="Players (by Position)"
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.info("💡 **Squad Heatmap Guide**: Players are sorted by position (GK → DEF → MID → FWD), " +
            "metrics organized by category (Goalkeeper → Defense → Discipline → Progression → Attack). " +
            "Green = Better performance, Red = Poorer performance. " +
            "For negative metrics (fouls, cards, etc.), lower actual values appear green. " +
            "Hover for detailed values.")

def render_team_analysis(df: pd.DataFrame):
    """Main function to render team analysis"""
    
    st.header("⚽ Team Analysis")
    st.markdown("*Professional football scouting and tactical intelligence*")
    
    # Team selection
    available_teams = sorted(df['Team'].unique())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_teams = st.multiselect(
            "Select teams (1-3 teams)",
            available_teams,
            default=[available_teams[0]] if available_teams else [],
            max_selections=3
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Team Profile", "Comparative Analysis", "Tactical Intelligence"]
        )
    
    if not selected_teams:
        st.warning("Please select at least one team to analyze.")
        return
    
    # Filter data for selected teams
    teams_data = {}
    for team in selected_teams:
        teams_data[team] = df[df['Team'] == team]
    
    # Main analysis based on type
    if len(selected_teams) == 1:
        # Single team analysis
        team_name = selected_teams[0]
        team_df = teams_data[team_name]
        
        if analysis_type == "Team Profile":
            render_team_profile(team_name, team_df, df)
        elif analysis_type == "Comparative Analysis":
            st.info("💡 Select multiple teams to enable comparative analysis")
            render_team_profile(team_name, team_df, df)
        else:  # Tactical Intelligence
            render_tactical_intelligence(team_name, team_df, df)
    
    else:
        # Multiple team comparison (always comparative analysis)
        if analysis_type != "Comparative Analysis":
            st.info(f"💡 Switching to Comparative Analysis mode for multiple teams")
        
        render_comparative_analysis(teams_data, df)

def get_players_to_watch_by_position(team_df: pd.DataFrame, league_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Get the best player in each position (1 per position only)"""
    position_groups = get_position_groups()
    key_metrics = ['Goal', 'Assist', 'Passing', 'Tackle', 'Intercept', 'Create Chance']
    
    position_players = {}
    
    for group_name, positions in position_groups.items():
        group_players = team_df[team_df['Position'].isin(positions)]
        
        if len(group_players) == 0:
            continue
            
        best_player = None
        best_score = 0
        
        for _, player in group_players.iterrows():
            player_score = 0
            player_reasons = []
            player_stats = []
            
            for metric in key_metrics:
                if metric in team_df.columns and metric in league_df.columns:
                    player_val = player[metric]
                    league_percentile = (league_df[metric] <= player_val).mean() * 100
                    
                    player_stats.append({
                        'metric': metric,
                        'value': player_val,
                        'percentile': league_percentile
                    })
                    
                    if league_percentile >= 90:
                        player_score += 3
                        player_reasons.append(f"Top 10% in {metric} ({player_val:.1f})")
                    elif league_percentile >= 75:
                        player_score += 2
                        player_reasons.append(f"Top 25% in {metric} ({player_val:.1f})")
                    elif league_percentile >= 50:
                        player_score += 1
            
            if player_score > best_score:
                best_score = player_score
                best_player = {
                    'name': player['Player Name'],
                    'position': player['Position'],
                    'age': player.get('Age', 'N/A'),
                    'appearances': player.get('Appearances', 'N/A'),
                    'team': player.get('Team', 'N/A'),
                    'score': player_score,
                    'reasons': player_reasons[:3],
                    'stats': player_stats,
                    'top_metrics': sorted([s for s in player_stats if s['percentile'] >= 75], 
                                         key=lambda x: x['percentile'], reverse=True)[:3]
                }
        
        if best_player and best_score >= 3:  # Minimum threshold
            position_players[group_name] = best_player
    
    return position_players

def get_top_players_by_metrics(team_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Get top players by key metrics for team facts"""
    top_players = {}
    
    metrics_info = {
        'Goal': {'name': 'Top Scorer', 'icon': '⚽'},
        'Assist': {'name': 'Top Assister', 'icon': '🎯'},
        'Passing': {'name': 'Most Passes', 'icon': '📈'},
        'Tackle': {'name': 'Best Defender', 'icon': '🛡️'},
        'Create Chance': {'name': 'Most Creative', 'icon': '💡'},
        'Appearances': {'name': 'Most Experienced', 'icon': '👥'}
    }
    
    for metric, info in metrics_info.items():
        if metric in team_df.columns:
            top_player_idx = team_df[metric].idxmax()
            top_player = team_df.loc[top_player_idx]
            
            top_players[metric] = {
                'name': top_player['Player Name'],
                'value': top_player[metric],
                'position': top_player['Position'],
                'age': top_player.get('Age', 'N/A'),
                'icon': info['icon'],
                'title': info['name']
            }
    
    return top_players

def create_team_performance_bars(team_df: pd.DataFrame, league_df: pd.DataFrame) -> go.Figure:
    """Create team performance bar chart similar to player comparison"""
    from utils.data_loader import get_player_metric_categories
    
    colors = get_theme_colors()
    metric_categories = get_player_metric_categories()
    
    # Calculate team averages and percentiles
    team_data = []
    
    for category, metrics in metric_categories.items():
        for metric in metrics:
            if metric in team_df.columns and metric in league_df.columns:
                team_avg = team_df[metric].mean()
                percentile = (league_df[metric] <= team_avg).mean() * 100
                
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
                
                team_data.append({
                    'metric': metric,
                    'category': category,
                    'team_avg': team_avg,
                    'percentile': percentile,
                    'color': color
                })
    
    if not team_data:
        return go.Figure()
    
    df_chart = pd.DataFrame(team_data)
    df_chart = df_chart.sort_values('percentile', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_chart['metric'],
        x=df_chart['percentile'],
        orientation='h',
        marker=dict(
            color=df_chart['color'],
            line=dict(width=0.5, color='white')
        ),
        text=[f"{val:.1f}" for val in df_chart['team_avg']],
        textposition='inside',
        textfont=dict(color='white', size=11, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Team Average: %{text}<br>League Percentile: %{x:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Team Performance vs League",
        xaxis=dict(
            title="League Percentile",
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(136, 136, 136, 0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=10)
        ),
        height=600,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font_color=colors['text'],
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def render_team_profile(team_name: str, team_df: pd.DataFrame, league_df: pd.DataFrame):
    """Render Team Profile analysis type"""
    st.subheader(f"📋 {team_name} - Complete Team Profile")
    
    # Enhanced team facts with top players
    st.write("### 🏆 Team Facts & Top Performers")
    
    # Basic metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Squad Size", len(team_df))
    with col2:
        avg_age = team_df['Age'].mean() if 'Age' in team_df.columns else 0
        st.metric("Average Age", f"{avg_age:.1f}")
    with col3:
        total_goals = team_df['Goal'].sum() if 'Goal' in team_df.columns else 0
        st.metric("Total Goals", int(total_goals))
    with col4:
        total_assists = team_df['Assist'].sum() if 'Assist' in team_df.columns else 0
        st.metric("Total Assists", int(total_assists))
    
    # Top players by metrics
    top_players = get_top_players_by_metrics(team_df)
    if top_players:
        st.write("### 🌟 Top Performers by Category")
        cols = st.columns(3)
        
        for i, (metric, player_info) in enumerate(top_players.items()):
            with cols[i % 3]:
                st.metric(
                    label=f"{player_info['icon']} {player_info['title']}",
                    value=f"{player_info['name']}",
                    delta=f"{player_info['value']:.1f} ({player_info['position']})"
                )
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👥 Players by Position", 
        "📋 Squad List",
        "🔥 Squad Heatmap",
        "📊 Performance Analysis", 
        "📈 Strengths & Weaknesses",
        "📋 Squad Overview"
    ])
    
    with tab1:
        st.subheader("🎯 Key Player in Each Position")
        position_players = get_players_to_watch_by_position(team_df, league_df)
        
        if position_players:
            for position_group, player in position_players.items():
                with st.expander(f"{position_group}: {player['name']} ({player['position']})", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Age**: {player['age']}")
                        st.write(f"**Appearances**: {player['appearances']}")
                        st.write(f"**Performance Score**: {player['score']}")
                    
                    with col2:
                        st.write("**Key Strengths:**")
                        for reason in player['reasons']:
                            st.write(f"• {reason}")
                        
                        if player['top_metrics']:
                            st.write("**Top 25% League Metrics:**")
                            for metric_info in player['top_metrics']:
                                st.write(f"• {metric_info['metric']}: {metric_info['value']:.1f} ({metric_info['percentile']:.0f}th percentile)")
        else:
            st.info("No standout players identified in specific positions based on current criteria.")
    
    with tab2:
        show_squad_list_table(team_df, league_df)
    
    with tab3:
        show_squad_heatmap(team_df, league_df)
    
    with tab4:
        st.subheader("📊 Detailed Performance Analysis")
        fig = create_team_performance_bars(team_df, league_df)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Performance Guide**: Green bars indicate top performance (75%+ percentile), while red bars show areas needing improvement (<25% percentile).")
        else:
            st.warning("No performance data available for visualization.")
    
    with tab5:
        st.subheader("⚖️ Strengths & Weaknesses Analysis")
        analysis = calculate_team_strengths_weaknesses(team_df, league_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 💪 Team Strengths")
            if analysis['strengths']:
                for strength in analysis['strengths']:
                    st.success(f"**{strength['metric']}** - {strength['percentile']:.0f}th percentile")
                    st.caption(f"Team avg: {strength['team_avg']:.1f} vs League avg: {strength['league_avg']:.1f}")
            else:
                st.info("No significant strengths identified")
        
        with col2:
            st.write("### ⚠️ Areas for Improvement")
            if analysis['weaknesses']:
                for weakness in analysis['weaknesses']:
                    st.error(f"**{weakness['metric']}** - {weakness['percentile']:.0f}th percentile")
                    st.caption(f"Team avg: {weakness['team_avg']:.1f} vs League avg: {weakness['league_avg']:.1f}")
            else:
                st.info("No significant weaknesses identified")
    
    with tab6:
        st.subheader("📋 Squad Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_position_distribution_chart({team_name: team_df}), use_container_width=True)
        with col2:
            st.plotly_chart(create_age_distribution_chart({team_name: team_df}), use_container_width=True)

def render_tactical_intelligence(team_name: str, team_df: pd.DataFrame, league_df: pd.DataFrame):
    """Render Tactical Intelligence analysis type"""
    st.subheader(f"🎯 {team_name} - Tactical Intelligence Report")
    
    # Advanced tactical analysis
    tab1, tab2, tab3 = st.tabs([
        "🔍 Formation Analysis",
        "⚡ Playing Style", 
        "🎪 Tactical Recommendations"
    ])
    
    with tab1:
        st.write("### 📐 Likely Formation & Structure")
        
        # Position distribution analysis
        position_groups = get_position_groups()
        formation_analysis = {}
        
        for group, positions in position_groups.items():
            count = len(team_df[team_df['Position'].isin(positions)])
            formation_analysis[group] = count
        
        # Display formation
        col1, col2 = st.columns(2)
        with col1:
            for group, count in formation_analysis.items():
                if count > 0:
                    st.metric(group, count)
        
        with col2:
            st.plotly_chart(create_position_distribution_chart({team_name: team_df}), use_container_width=True)
    
    with tab2:
        st.write("### ⚡ Playing Style Analysis")
        
        # Performance radar for tactical style
        st.plotly_chart(create_performance_radar({team_name: team_df}, league_df), use_container_width=True)
        
        # Style indicators
        style_metrics = {
            'Attacking': ['Goal', 'Assist', 'Create Chance'],
            'Possession': ['Passing'],
            'Defensive': ['Tackle', 'Intercept']
        }
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (style, metrics) in enumerate(style_metrics.items()):
            with cols[i]:
                style_score = 0
                valid_metrics = 0
                
                for metric in metrics:
                    if metric in team_df.columns and metric in league_df.columns:
                        team_avg = team_df[metric].mean()
                        percentile = (league_df[metric] <= team_avg).mean() * 100
                        style_score += percentile
                        valid_metrics += 1
                
                if valid_metrics > 0:
                    avg_percentile = style_score / valid_metrics
                    
                    if avg_percentile >= 75:
                        st.success(f"**{style} Oriented**")
                        st.write(f"{avg_percentile:.0f}th percentile")
                    elif avg_percentile >= 50:
                        st.info(f"**Moderate {style}**")
                        st.write(f"{avg_percentile:.0f}th percentile")
                    else:
                        st.warning(f"**Low {style}**")
                        st.write(f"{avg_percentile:.0f}th percentile")
    
    with tab3:
        st.write("### 🎪 Tactical Recommendations")
        
        analysis = calculate_team_strengths_weaknesses(team_df, league_df)
        
        recommendations = []
        
        # Generate recommendations based on strengths and weaknesses
        if analysis['strengths']:
            top_strength = analysis['strengths'][0]
            recommendations.append(f"✅ **Leverage {top_strength['metric']}**: This is your strongest area ({top_strength['percentile']:.0f}th percentile). Build tactics around this strength.")
        
        if analysis['weaknesses']:
            top_weakness = analysis['weaknesses'][0]
            recommendations.append(f"⚠️ **Address {top_weakness['metric']}**: Priority area for improvement ({top_weakness['percentile']:.0f}th percentile). Consider targeted training or transfers.")
        
        # Formation-based recommendations
        def_count = len(team_df[team_df['Position'].isin(['BELAKANG'])])
        mid_count = len(team_df[team_df['Position'].isin(['TENGAH'])])
        att_count = len(team_df[team_df['Position'].isin(['DEPAN'])])
        
        if def_count >= mid_count and def_count >= att_count:
            recommendations.append("🛡️ **Defensive Setup**: Strong defensive squad - consider defensive formations like 5-3-2 or 4-5-1.")
        elif att_count >= mid_count:
            recommendations.append("⚽ **Attacking Setup**: Forward-heavy squad - consider attacking formations like 4-3-3 or 3-4-3.")
        elif mid_count > def_count and mid_count > att_count:
            recommendations.append("🎯 **Midfield Control**: Midfield-dominant squad - consider formations like 4-5-1 or 3-5-2.")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        if not recommendations:
            st.info("No specific tactical recommendations available based on current data.")

def render_comparative_analysis(teams_data: Dict[str, pd.DataFrame], league_df: pd.DataFrame):
    """Render Comparative Analysis for multiple teams"""
    st.subheader("🔄 Multi-Team Comparison Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Squad Comparison",
        "📈 Performance Comparison", 
        "🎯 Head-to-Head Analysis"
    ])
    
    with tab1:
        # Comparative squad metrics
        comparison_data = []
        for team_name, team_df in teams_data.items():
            comparison_data.append({
                'Team': team_name,
                'Squad Size': len(team_df),
                'Average Age': team_df['Age'].mean() if 'Age' in team_df.columns else 0,
                'Total Goals': team_df['Goal'].sum() if 'Goal' in team_df.columns else 0,
                'Total Assists': team_df['Assist'].sum() if 'Assist' in team_df.columns else 0,
                'Total Passes': team_df['Passing'].sum() if 'Passing' in team_df.columns else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparisons
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_position_distribution_chart(teams_data), use_container_width=True)
        with col2:
            st.plotly_chart(create_age_distribution_chart(teams_data), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_performance_radar(teams_data, league_df), use_container_width=True)
    
    with tab3:
        st.subheader("🏆 Team-by-Team Intelligence")
        
        for team_name, team_df in teams_data.items():
            with st.expander(f"📋 {team_name} - Scouting Report"):
                analysis = calculate_team_strengths_weaknesses(team_df, league_df)
                position_players = get_players_to_watch_by_position(team_df, league_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Strengths:**")
                    for strength in analysis['strengths'][:3]:
                        st.write(f"• {strength['metric']} ({strength['percentile']:.0f}th percentile)")
                
                with col2:
                    st.write("**Key Players by Position:**")
                    for pos_group, player in position_players.items():
                        st.write(f"• **{pos_group}**: {player['name']} ({player['position']})")

if __name__ == "__main__":
    # This would be called from the main app
    pass