import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple

def show_player_report(filtered_df):
    """
    Comprehensive Player Report with position-based role analysis
    
    Features:
    - Player selection and search
    - Position-based role mapping and analysis
    - Role-specific rating bars with percentiles
    - Position-specific radar chart
    - Complete stats table with league rankings
    """
    st.header("👤 Player Report")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return
    
    # Player Selection Interface
    st.subheader("🔍 Select Player")
    
    # Single column player selector (global filters already handle team/position filtering)
    selected_player = st.selectbox(
        "Search and select a player:",
        options=sorted(filtered_df['Player Name'].tolist()),
        help="Type to search for a specific player (use sidebar filters to narrow down options)"
    )
    
    # Get selected player data
    if selected_player:
        player_data = filtered_df[filtered_df['Player Name'] == selected_player].iloc[0]
        
        # Display basic player info
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("**Player**", player_data['Player Name'])
        with col2:
            st.metric("**Team**", player_data['Team'])
        with col3:
            st.metric("**Position**", player_data['Position'])
        with col4:
            st.metric("**Age**", f"{player_data['Age']} years")
        with col5:
            st.metric("**Country**", player_data['Country'])
        
        st.markdown("---")
        
        # Position-Based Role Analysis
        st.subheader("🎯 Position-Based Role Analysis")
        
        # Position selection for detailed analysis
        actual_position = st.selectbox(
            "Select specific position for detailed role analysis:",
            options=get_position_options(player_data['Position']),
            help="Choose the most accurate position to get relevant role analysis"
        )
        
        if actual_position:
            # Generate role analysis
            role_analysis = generate_role_analysis(player_data, actual_position, filtered_df)
            
            # Display role ratings
            st.subheader("📊 Role Suitability Ratings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Role rating bars
                fig_bars = create_role_rating_bars(role_analysis['roles'])
                st.plotly_chart(fig_bars, use_container_width=True)
            
            with col2:
                # Position-specific radar
                fig_radar = create_position_radar(player_data, actual_position, filtered_df)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Best role insight
            best_role = max(role_analysis['roles'], key=lambda x: x['percentile'])
            st.success(f"🏆 **Best Role**: {best_role['name']} ({best_role['percentile']:.1f}th percentile)")
            
        st.markdown("---")
        
        # Comprehensive Stats Table with Rankings
        st.subheader("📋 Complete Statistics & League Rankings")
        
        create_comprehensive_stats_table(player_data, filtered_df)
        
        # Performance summary
        st.subheader("⚡ Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Attack metrics
            attack_avg = calculate_category_average(player_data, ['Goal', 'Assist', 'Shoot On Target'])
            st.metric("Attack Rating", f"{attack_avg:.1f}/10", help="Based on goals, assists, and shots on target")
        
        with col2:
            # Playmaking metrics
            playmaking_avg = calculate_category_average(player_data, ['Passing', 'Create Chance', 'Cross'])
            st.metric("Playmaking Rating", f"{playmaking_avg:.1f}/10", help="Based on passing, chance creation, and crossing")
        
        with col3:
            # Defensive metrics
            defensive_avg = calculate_category_average(player_data, ['Tackle', 'Intercept', 'Clearance'])
            st.metric("Defensive Rating", f"{defensive_avg:.1f}/10", help="Based on tackles, interceptions, and clearances")
        
        # Add explanation at the end
        show_role_calculation_explanation()

def get_position_options(general_position: str) -> List[str]:
    """Get specific position options based on general position"""
    position_mapping = {
        'Defender': [
            'Center Back',
            'Full Back', 
            'Wing Back'
        ],
        'Midfielder': [
            'Deep Midfielder',
            'Central Midfielder',
            'Attacking Midfielder',
            'Wide Midfielder'
        ],
        'Forward': [
            'Center Forward',
            'Winger',
            'Second Striker'
        ]
    }
    
    # Try to match the general position, otherwise return generic options
    for key in position_mapping.keys():
        if key.lower() in general_position.lower():
            return position_mapping[key]
    
    # Fallback - return all options if no match
    all_options = []
    for options in position_mapping.values():
        all_options.extend(options)
    return all_options

def generate_role_analysis(player_data: pd.Series, position: str, league_df: pd.DataFrame) -> Dict:
    """Generate role-specific analysis based on position with position-filtered comparisons"""
    
    role_definitions = {
        'Center Back': {
            'Ball-Playing Defender': {
                'metrics': ['Passing', 'Ball Recovery', 'Clearance', 'Intercept'],
                'weights': [0.35, 0.25, 0.25, 0.15]
            },
            'Central Defender': {
                'metrics': ['Clearance', 'Header Won', 'Tackle', 'Block'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Full Back': {
            'Attacking Full Back': {
                'metrics': ['Cross', 'Assist', 'Create Chance', 'Passing'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Defensive Full Back': {
                'metrics': ['Tackle', 'Intercept', 'Clearance', 'Ball Recovery'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Wing Back': {
            'Attacking Wing Back': {
                'metrics': ['Cross', 'Assist', 'Create Chance', 'Dribble Success'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Defensive Wing Back': {
                'metrics': ['Tackle', 'Intercept', 'Ball Recovery', 'Cross'],
                'weights': [0.30, 0.25, 0.20, 0.25]
            }
        },
        'Deep Midfielder': {
            'Deep Playmaker': {
                'metrics': ['Passing', 'Ball Recovery', 'Create Chance', 'Assist'],
                'weights': [0.35, 0.25, 0.25, 0.15]
            },
            'Ball Winner': {
                'metrics': ['Tackle', 'Intercept', 'Ball Recovery', 'Passing'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Central Midfielder': {
            'Box-to-Box': {
                'metrics': ['Passing', 'Tackle', 'Goal', 'Assist'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Playmaker': {
                'metrics': ['Passing', 'Create Chance', 'Assist', 'Dribble Success'],
                'weights': [0.35, 0.30, 0.20, 0.15]
            }
        },
        'Attacking Midfielder': {
            'Creative Playmaker': {
                'metrics': ['Create Chance', 'Assist', 'Passing', 'Goal'],
                'weights': [0.35, 0.25, 0.25, 0.15]
            },
            'Advanced Midfielder': {
                'metrics': ['Goal', 'Assist', 'Shoot On Target', 'Create Chance'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Wide Midfielder': {
            'Winger': {
                'metrics': ['Cross', 'Dribble Success', 'Assist', 'Create Chance'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Wide Playmaker': {
                'metrics': ['Assist', 'Create Chance', 'Cross', 'Passing'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Center Forward': {
            'Target Man': {
                'metrics': ['Goal', 'Header Won', 'Shoot On Target', 'Penalty Goal'],
                'weights': [0.40, 0.25, 0.25, 0.10]
            },
            'Poacher': {
                'metrics': ['Goal', 'Penalty Goal', 'Shoot On Target', 'Create Chance'],
                'weights': [0.45, 0.20, 0.25, 0.10]
            }
        },
        'Winger': {
            'Pace Winger': {
                'metrics': ['Dribble Success', 'Cross', 'Assist', 'Goal'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Inverted Winger': {
                'metrics': ['Goal', 'Shoot On Target', 'Dribble Success', 'Create Chance'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            }
        },
        'Second Striker': {
            'False 9': {
                'metrics': ['Create Chance', 'Assist', 'Goal', 'Passing'],
                'weights': [0.30, 0.25, 0.25, 0.20]
            },
            'Support Striker': {
                'metrics': ['Goal', 'Assist', 'Create Chance', 'Shoot On Target'],
                'weights': [0.35, 0.25, 0.25, 0.15]
            }
        }
    }
    
    # Filter league data by same position for fair comparison
    player_position = player_data['Position']
    position_filtered_df = league_df[league_df['Position'] == player_position]
    
    roles = []
    if position in role_definitions:
        for role_name, role_data in role_definitions[position].items():
            # Calculate weighted score
            weighted_score = 0
            valid_metrics = 0
            
            for metric, weight in zip(role_data['metrics'], role_data['weights']):
                if metric in player_data.index and not pd.isna(player_data[metric]):
                    # Calculate percentile vs position-filtered league
                    percentile = calculate_percentile(player_data[metric], position_filtered_df[metric])
                    weighted_score += percentile * weight
                    valid_metrics += weight
            
            if valid_metrics > 0:
                final_score = weighted_score / valid_metrics
                roles.append({
                    'name': role_name,
                    'percentile': final_score,
                    'metrics': role_data['metrics'],
                    'weights': role_data['weights']
                })
    
    return {'roles': sorted(roles, key=lambda x: x['percentile'], reverse=True)}

def calculate_percentile(value: float, league_values: pd.Series) -> float:
    """Calculate percentile rank of a value within league distribution"""
    if pd.isna(value):
        return 0
    
    rank = (league_values < value).sum()
    total = len(league_values.dropna())
    
    if total == 0:
        return 0
    
    return (rank / total) * 100

def categorize_metric(metric: str) -> str:
    """Categorize metrics for sorting and organization"""
    metric_categories = {
        "⚽ Attack": ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
        "🎯 Progression": ['Passing', 'Cross', 'Dribble Success', 'Ball Recovery', 'Free Kick'],
        "🛡️ Defense": ['Tackle', 'Intercept', 'Clearance', 'Block', 'Block Cross', 'Header Won'],
        "📊 General": ['Appearances', 'Yellow Card', 'Own Goal', 'Saves', 'Fouled', 'Foul']
    }
    
    for category, metrics in metric_categories.items():
        if metric in metrics:
            return category
    
    return "📊 General"  # Default category

def create_role_rating_bars(roles: List[Dict]) -> go.Figure:
    """Create horizontal bar chart for role ratings"""
    
    if not roles:
        # Return empty figure if no roles
        fig = go.Figure()
        fig.add_annotation(
            text="No role data available for this position",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(title="Role Suitability Ratings", height=400)
        return fig
    
    # Prepare data
    role_names = [role['name'] for role in roles]
    percentiles = [role['percentile'] for role in roles]
    
    # Color coding based on percentile
    colors = []
    for p in percentiles:
        if p >= 80:
            colors.append('#1a9641')  # Green - Excellent
        elif p >= 60:
            colors.append('#a6d96a')  # Light Green - Good  
        elif p >= 40:
            colors.append('#ffffbf')  # Yellow - Average
        elif p >= 20:
            colors.append('#fdae61')  # Orange - Below Average
        else:
            colors.append('#d73027')  # Red - Poor
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=role_names,
        x=percentiles,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in percentiles],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Arial Bold')
    ))
    
    fig.update_layout(
        title="Role Suitability Ratings (League Percentiles)",
        xaxis_title="Percentile",
        yaxis_title="Role",
        height=max(300, len(roles) * 50),
        xaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig

def create_position_radar(player_data: pd.Series, position: str, league_df: pd.DataFrame) -> go.Figure:
    """Create position-specific radar chart with position-filtered comparisons"""
    
    # Position-specific key metrics
    position_metrics = {
        'Center Back': ['Clearance', 'Header Won', 'Tackle', 'Intercept', 'Passing', 'Ball Recovery'],
        'Full Back': ['Cross', 'Tackle', 'Assist', 'Intercept', 'Passing', 'Dribble Success'],
        'Wing Back': ['Cross', 'Assist', 'Tackle', 'Create Chance', 'Ball Recovery', 'Dribble Success'],
        'Deep Midfielder': ['Passing', 'Tackle', 'Ball Recovery', 'Intercept', 'Create Chance', 'Assist'],
        'Central Midfielder': ['Passing', 'Tackle', 'Assist', 'Goal', 'Create Chance', 'Ball Recovery'],
        'Attacking Midfielder': ['Create Chance', 'Assist', 'Goal', 'Passing', 'Shoot On Target', 'Dribble Success'],
        'Wide Midfielder': ['Cross', 'Assist', 'Dribble Success', 'Create Chance', 'Passing', 'Tackle'],
        'Center Forward': ['Goal', 'Shoot On Target', 'Header Won', 'Assist', 'Create Chance', 'Penalty Goal'],
        'Winger': ['Dribble Success', 'Cross', 'Assist', 'Goal', 'Create Chance', 'Shoot On Target'],
        'Second Striker': ['Goal', 'Assist', 'Create Chance', 'Shoot On Target', 'Passing', 'Dribble Success']
    }
    
    metrics = position_metrics.get(position, ['Goal', 'Assist', 'Passing', 'Tackle', 'Create Chance', 'Dribble Success'])
    
    # Filter league data by same position for fair comparison
    player_position = player_data['Position']
    position_filtered_df = league_df[league_df['Position'] == player_position]
    
    # Calculate percentiles for each metric
    categories = []
    values = []
    
    for metric in metrics:
        if metric in player_data.index and metric in position_filtered_df.columns:
            percentile = calculate_percentile(player_data[metric], position_filtered_df[metric])
            categories.append(metric)
            values.append(percentile)
    
    if not values:
        # Return empty radar if no valid metrics
        fig = go.Figure()
        fig.add_annotation(
            text="No radar data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(title=f"{position} Performance Radar", height=400)
        return fig
    
    # Close the radar by repeating first value
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name=player_data['Player Name'],
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )
        ),
        title=f"{position} Performance Radar (League Percentiles)",
        showlegend=True,
        height=400
    )
    
    return fig

def create_comprehensive_stats_table(player_data: pd.Series, league_df: pd.DataFrame):
    """Create comprehensive stats table with Club Rank and position-based filtering"""
    
    # Get available metric columns (exclude info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in league_df.columns if col not in info_columns]
    
    # Filter league data by same position
    player_position = player_data['Position']
    position_filtered_df = league_df[league_df['Position'] == player_position]
    
    # Filter club data by same team and position
    player_team = player_data['Team']
    club_filtered_df = league_df[
        (league_df['Team'] == player_team) & 
        (league_df['Position'] == player_position)
    ]
    
    stats_data = []
    
    for metric in metric_columns:
        if metric in player_data.index:
            value = player_data[metric]
            
            if not pd.isna(value):
                # League calculations (position-filtered)
                position_metric_data = position_filtered_df[metric].dropna()
                if len(position_metric_data) > 0:
                    league_better_players = (position_metric_data > value).sum()
                    league_total_players = len(position_metric_data)
                    league_rank = league_better_players + 1
                    league_percentile = calculate_percentile(value, position_metric_data)
                    league_avg = position_metric_data.mean()
                    league_rank_percentile = (1 - (league_rank - 1) / league_total_players) * 100
                else:
                    league_rank = league_total_players = league_percentile = league_avg = league_rank_percentile = 0
                
                # Club calculations (team + position filtered)
                club_metric_data = club_filtered_df[metric].dropna()
                if len(club_metric_data) > 1:  # Need at least 2 players for meaningful comparison
                    club_better_players = (club_metric_data > value).sum()
                    club_total_players = len(club_metric_data)
                    club_rank = club_better_players + 1
                    club_rank_percentile = (1 - (club_rank - 1) / club_total_players) * 100
                else:
                    club_rank = club_total_players = club_rank_percentile = 0
                
                stats_data.append({
                    'Metric': metric,
                    'Player Value': f"{value:.1f}",
                    'Club Rank': f"{club_rank} / {club_total_players}" if club_total_players > 1 else "Only player",
                    'League Rank': f"{league_rank} / {league_total_players}" if league_total_players > 0 else "N/A",
                    'Percentile': league_percentile,
                    'League Average': f"{league_avg:.1f}" if league_avg > 0 else "N/A",
                    'vs Average': f"{((value - league_avg) / league_avg * 100):+.1f}%" if league_avg > 0 else "N/A",
                    'League Rank Percentile': league_rank_percentile,
                    'Club Rank Percentile': club_rank_percentile,
                    'Category': categorize_metric(metric)
                })
    
    df = pd.DataFrame(stats_data)
    
    # Sort by category and then by metric name
    if len(df) > 0:
        # Define category order
        category_order = ["⚽ Attack", "🎯 Progression", "🛡️ Defense", "📊 General"]
        df['Category_Order'] = df['Category'].map({cat: i for i, cat in enumerate(category_order)})
        df = df.sort_values(['Category_Order', 'Metric']).reset_index(drop=True)
        
        st.markdown("### 📈 Performance Statistics")
        st.markdown("*Comparing against players in the same position*")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Standard Table", "🎨 Visual Dashboard"])
        
        with tab1:
            # Prepare display DataFrame with additional formatting
            display_df = df[['Category', 'Metric', 'Player Value', 'Club Rank', 'League Rank', 'Percentile', 'League Average', 'vs Average']].copy()
            
            # Keep percentile as numeric for ProgressColumn
            display_df['Percentile_Value'] = df['Percentile']
            
            # Add color-coded rank columns
            display_df['Club_Rank_Color'] = df.apply(lambda row: get_colored_rank_text(row['Club Rank'], row['Club Rank Percentile']), axis=1)
            display_df['League_Rank_Color'] = df.apply(lambda row: get_colored_rank_text(row['League Rank'], row['League Rank Percentile']), axis=1)
            
            # Display the enhanced table
            st.dataframe(
                display_df[['Category', 'Metric', 'Player Value', 'Club_Rank_Color', 'League_Rank_Color', 'Percentile_Value', 'League Average', 'vs Average']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="small"),
                    "Metric": st.column_config.TextColumn("Metric", width="medium"), 
                    "Player Value": st.column_config.NumberColumn("Value", width="small", format="%.1f"),
                    "Club_Rank_Color": st.column_config.TextColumn("Club Rank", width="small"),
                    "League_Rank_Color": st.column_config.TextColumn("League Rank", width="small"),
                    "Percentile_Value": st.column_config.ProgressColumn(
                        "Percentile",
                        help="Performance percentile vs same position players",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                        width="small"
                    ),
                    "League Average": st.column_config.NumberColumn("League Avg", width="small", format="%.1f"),
                    "vs Average": st.column_config.TextColumn("vs Average", width="small")
                }
            )
        
        with tab2:
            # Create visual dashboard
            create_visual_dashboard(player_data, df)


def get_colored_rank_text(rank_text: str, rank_percentile: float) -> str:
    """Get colored rank text based on percentile"""
    if rank_percentile >= 75:
        return f"🟢 {rank_text}"  # Green - Top 25%
    elif rank_percentile >= 25:
        return f"🟡 {rank_text}"  # Yellow - Middle 50%
    else:
        return f"🔴 {rank_text}"  # Red - Bottom 25%

def create_visual_dashboard(player_data: pd.Series, stats_df: pd.DataFrame):
    """Create visual dashboard with player info and charts"""
    
    # Player Info Card
    st.markdown("### 👤 Player Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**🏆 {player_data['Player Name']}**")
        st.markdown(f"⚽ **{player_data['Team']}**")
    with col2:
        st.markdown(f"📍 **Position**: {player_data['Position']}")
        st.markdown(f"🌍 **Country**: {player_data['Country']}")
    with col3:
        st.markdown(f"🎂 **Age**: {player_data['Age']} years")
    with col4:
        # Overall performance indicator
        overall_percentile = stats_df['Percentile'].mean()
        st.metric("Overall Percentile", f"{overall_percentile:.1f}%")
    
    st.markdown("---")
    
    # Category-wise Performance Charts
    st.markdown("### 📊 Performance by Category")
    
    # Group stats by category
    category_performance = {}
    categories = ["⚽ Attack", "🎯 Progression", "🛡️ Defense", "📊 General"]
    
    for category in categories:
        category_data = stats_df[stats_df['Category'] == category]
        if not category_data.empty:
            avg_percentile = category_data['Percentile'].mean()
            category_performance[category] = {
                'percentile': avg_percentile,
                'metrics': category_data[['Metric', 'Percentile', 'Club Rank Percentile', 'League Rank Percentile']].to_dict('records')
            }
    
    # Create category comparison chart
    if category_performance:
        fig_categories = create_category_comparison_chart(category_performance)
        st.plotly_chart(fig_categories, use_container_width=True)
    
    # Detailed metrics visualization
    st.markdown("### 🎯 Detailed Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Club vs League comparison
        st.markdown("#### 🏟️ Club vs League Comparison")
        fig_comparison = create_club_league_comparison(stats_df)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Top/Bottom metrics
        st.markdown("#### ⭐ Strengths & Areas for Improvement")
        
        # Top 5 metrics
        top_metrics = stats_df.nlargest(5, 'Percentile')[['Metric', 'Percentile']]
        st.markdown("**🟢 Top Performing Metrics:**")
        for _, row in top_metrics.iterrows():
            st.markdown(f"• **{row['Metric']}**: {row['Percentile']:.1f}%")
        
        st.markdown("")
        
        # Bottom 5 metrics
        bottom_metrics = stats_df.nsmallest(5, 'Percentile')[['Metric', 'Percentile']]
        st.markdown("**🔴 Areas for Improvement:**")
        for _, row in bottom_metrics.iterrows():
            st.markdown(f"• **{row['Metric']}**: {row['Percentile']:.1f}%")

def create_category_comparison_chart(category_performance: dict) -> go.Figure:
    """Create horizontal bar chart comparing category performance"""
    
    categories = list(category_performance.keys())
    percentiles = [category_performance[cat]['percentile'] for cat in categories]
    
    # Color coding based on performance
    colors = []
    for percentile in percentiles:
        if percentile >= 75:
            colors.append('#28a745')  # Green
        elif percentile >= 50:
            colors.append('#ffc107')  # Yellow
        elif percentile >= 25:
            colors.append('#fd7e14')  # Orange
        else:
            colors.append('#dc3545')  # Red
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=categories,
        x=percentiles,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in percentiles],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Arial Bold')
    ))
    
    fig.update_layout(
        title="Category Performance Overview",
        xaxis_title="Average Percentile",
        yaxis_title="Category",
        height=300,
        xaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig

def create_club_league_comparison(stats_df: pd.DataFrame) -> go.Figure:
    """Create comparison chart showing club vs league percentiles"""
    
    # Get top 10 metrics for comparison
    top_metrics = stats_df.nlargest(10, 'Percentile')[['Metric', 'Club Rank Percentile', 'League Rank Percentile']]
    
    fig = go.Figure()
    
    # Club performance
    fig.add_trace(go.Bar(
        y=top_metrics['Metric'],
        x=top_metrics['Club Rank Percentile'],
        orientation='h',
        name='Club Rank',
        marker_color='rgba(55, 126, 184, 0.7)',
        text=[f"{p:.1f}%" for p in top_metrics['Club Rank Percentile']],
        textposition='inside'
    ))
    
    # League performance
    fig.add_trace(go.Bar(
        y=top_metrics['Metric'],
        x=top_metrics['League Rank Percentile'],
        orientation='h',
        name='League Rank',
        marker_color='rgba(228, 26, 28, 0.7)',
        text=[f"{p:.1f}%" for p in top_metrics['League Rank Percentile']],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Club vs League Performance (Top 10 Metrics)",
        xaxis_title="Percentile",
        yaxis_title="Metrics",
        height=400,
        barmode='group',
        xaxis=dict(range=[0, 100]),
        legend=dict(x=0.7, y=1)
    )
    
    return fig

def get_rank_color(rank_percentile: float) -> str:
    """Get color for league rank based on percentile"""
    if rank_percentile >= 75:
        return '#28a745'  # Green - Top 25%
    elif rank_percentile >= 25:
        return '#ffc107'  # Yellow - Middle 50%
    else:
        return '#dc3545'  # Red - Bottom 25%

def get_percentile_color(percentile: float) -> str:
    """Get color for percentile progress bar"""
    if percentile >= 80:
        return '#28a745'  # Green - Excellent
    elif percentile >= 60:
        return '#6bc547'  # Light Green - Good
    elif percentile >= 40:
        return '#ffc107'  # Yellow - Average
    elif percentile >= 20:
        return '#fd7e14'  # Orange - Below Average
    else:
        return '#dc3545'  # Red - Poor

def calculate_category_average(player_data: pd.Series, metrics: List[str]) -> float:
    """Calculate normalized average for a category of metrics"""
    
    valid_values = []
    for metric in metrics:
        if metric in player_data.index and not pd.isna(player_data[metric]):
            # Normalize to 0-10 scale (assuming max reasonable values)
            max_values = {
                'Goal': 20, 'Assist': 15, 'Shoot On Target': 30,
                'Passing': 50, 'Create Chance': 20, 'Cross': 30,
                'Tackle': 30, 'Intercept': 25, 'Clearance': 40
            }
            
            max_val = max_values.get(metric, 20)  # Default max
            normalized = min(player_data[metric] / max_val * 10, 10)
            valid_values.append(normalized)
    
    return sum(valid_values) / len(valid_values) if valid_values else 0

# Role Percentile Calculation Explanation
def show_role_calculation_explanation():
    """Show detailed explanation of how role percentiles are calculated"""
    st.markdown("---")
    st.subheader("📖 How Role Percentiles Are Calculated")
    
    with st.expander("🔍 Click to understand the methodology", expanded=False):
        st.markdown("""
        ### 📊 **Percentile Calculation Method**
        
        **Step 1: Individual Metric Percentiles**
        - Each player's statistic is compared against all other players in the league
        - **Formula**: `(Players with lower values / Total players) × 100`
        - **Example**: If a player has 10 goals and 80 players have fewer goals out of 200 total players, their percentile = (80/200) × 100 = 40th percentile
        
        ### ⚖️ **Role-Specific Weighted Scoring**
        
        **Step 2: Metric Weighting by Role**
        - Each role emphasizes different metrics with specific weights
        - **Example - Ball-Playing Defender**:
          - Passing: 35% weight
          - Ball Recovery: 25% weight  
          - Clearance: 25% weight
          - Intercept: 15% weight
        
        **Step 3: Weighted Average Calculation**
        - **Formula**: `Σ(Individual Percentile × Weight) / Total Weights`
        - **Example**: 
          - Passing (60th percentile × 0.35) + Ball Recovery (80th percentile × 0.25) + Clearance (70th percentile × 0.25) + Intercept (50th percentile × 0.15)
          - = (21 + 20 + 17.5 + 7.5) / 1.0 = **66th percentile overall**
        
        ### 🎯 **Role Comparison Context**
        
        **League-Wide Comparison**
        - All percentiles are calculated against the **entire league dataset**
        - Higher percentile = Better performance relative to all other players
        - **90th+ percentile**: Elite level (top 10%)
        - **75th+ percentile**: Very good (top 25%)
        - **50th+ percentile**: Above average (top 50%)
        - **25th+ percentile**: Below average (bottom 50%)
        - **Below 25th percentile**: Needs improvement (bottom 25%)
        
        ### 💡 **Key Insights**
        
        - **Position-Specific**: Each role uses metrics most relevant to that playing style
        - **Weighted Importance**: More critical metrics have higher influence on the final score
        - **Relative Performance**: Percentiles show how a player ranks among their peers, not absolute performance
        - **Dynamic Calculation**: Percentiles update as new data is added to the league
        
        *This methodology ensures fair, position-relevant comparisons across all players in the league.*
        """)

