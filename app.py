import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from utils.player_performance import (
    create_category_scores_visualization, 
    create_category_summary_table
)
from viz.player_comparison import show_player_comparison
from viz.profile_finder import show_profile_finder
from viz.scatter_analysis import show_scatter_analysis
from viz.player_report import show_player_report
from viz.player_screener import show_player_screener

# Global stats calculation functions
def calculate_overall_stats(df):
    """Calculate overall dashboard statistics"""
    stats = {
        'total_players': len(df),
        'avg_age': df['Age'].mean(),
        'total_teams': df['Team'].nunique(),
        'active_players': len(df[df['Appearances'] > 5]),
        'positions': df['Position'].value_counts().to_dict()
    }
    
    # Top scoring position
    position_goals = df.groupby('Position')['Goal'].sum()
    stats['top_scoring_position'] = position_goals.idxmax() if len(position_goals) > 0 else 'N/A'
    
    return stats

def calculate_attack_stats(df):
    """Calculate attack category statistics"""
    stats = {
        'total_goals': df['Goal'].sum(),
        'total_assists': df['Assist'].sum(),
        'total_shots_on_target': df['Shoot On Target'].sum(),
        'total_penalty_goals': df['Penalty Goal'].sum(),
        'total_shots_off_target': df['Shoot Off Target'].sum(),
        'total_create_chance': df['Create Chance'].sum()
    }
    
    # Calculate conversion rates
    total_shots = stats['total_shots_on_target'] + stats['total_shots_off_target']
    if total_shots > 0:
        stats['shot_conversion_rate'] = (stats['total_goals'] / total_shots * 100)
        stats['shots_on_target_rate'] = (stats['total_shots_on_target'] / total_shots * 100)
    else:
        stats['shot_conversion_rate'] = 0
        stats['shots_on_target_rate'] = 0
    
    # Average goals per appearance (for players with appearances)
    active_players = df[df['Appearances'] > 0]
    if len(active_players) > 0:
        total_appearances = active_players['Appearances'].sum()
        stats['goals_per_appearance'] = stats['total_goals'] / total_appearances if total_appearances > 0 else 0
    else:
        stats['goals_per_appearance'] = 0
    
    return stats

def calculate_defense_stats(df):
    """Calculate defense category statistics"""
    stats = {
        'total_tackles': df['Tackle'].sum(),
        'total_clearances': df['Clearance'].sum(),
        'total_interceptions': df['Intercept'].sum(),
        'total_blocks': df['Block'].sum(),
        'total_block_cross': df['Block Cross'].sum(),
        'total_ball_recovery': df['Ball Recovery'].sum(),
        'total_headers_won': df['Header Won'].sum()
    }
    
    # Calculate average defensive actions per player
    active_players = df[df['Appearances'] > 0]
    if len(active_players) > 0:
        stats['avg_defensive_actions'] = (
            stats['total_tackles'] + stats['total_clearances'] + 
            stats['total_interceptions'] + stats['total_blocks']
        ) / len(active_players)
        
        # Most defensive position
        defensive_actions_by_position = active_players.groupby('Position')[
            ['Tackle', 'Clearance', 'Intercept', 'Block']
        ].sum().sum(axis=1)
        stats['most_defensive_position'] = defensive_actions_by_position.idxmax() if len(defensive_actions_by_position) > 0 else 'N/A'
    else:
        stats['avg_defensive_actions'] = 0
        stats['most_defensive_position'] = 'N/A'
    
    return stats

def calculate_progression_stats(df):
    """Calculate progression category statistics"""
    stats = {
        'total_passes': df['Passing'].sum(),
        'total_crosses': df['Cross'].sum(),
        'total_dribble_success': df['Dribble Success'].sum(),
        'total_free_kicks': df['Free Kick'].sum()
    }
    
    # Most active playmaker (highest passing)
    if len(df) > 0:
        top_passer_idx = df['Passing'].idxmax()
        stats['most_active_playmaker'] = df.loc[top_passer_idx, 'Player Name']
        stats['top_passer_total'] = df.loc[top_passer_idx, 'Passing']
    else:
        stats['most_active_playmaker'] = 'N/A'
        stats['top_passer_total'] = 0
    
    # Average successful actions per player
    active_players = df[df['Appearances'] > 0]
    if len(active_players) > 0:
        stats['avg_progression_actions'] = (
            stats['total_passes'] + stats['total_crosses'] + 
            stats['total_dribble_success'] + stats['total_free_kicks']
        ) / len(active_players)
    else:
        stats['avg_progression_actions'] = 0
    
    return stats

def calculate_discipline_stats(df):
    """Calculate discipline category statistics"""
    stats = {
        'total_yellow_cards': df['Yellow Card'].sum(),
        'total_fouls_committed': df['Foul'].sum(),
        'total_own_goals': df['Own Goal'].sum(),
        'total_fouled': df['Fouled'].sum()
    }
    
    # Calculate foul rate (fouls per appearance)
    active_players = df[df['Appearances'] > 0]
    if len(active_players) > 0:
        total_appearances = active_players['Appearances'].sum()
        stats['foul_rate'] = stats['total_fouls_committed'] / total_appearances if total_appearances > 0 else 0
        
        # Most disciplined team (lowest average fouls)
        team_fouls = active_players.groupby('Team')['Foul'].sum() / active_players.groupby('Team')['Appearances'].sum()
        stats['most_disciplined_team'] = team_fouls.idxmin() if len(team_fouls) > 0 else 'N/A'
        stats['lowest_foul_rate'] = team_fouls.min() if len(team_fouls) > 0 else 0
    else:
        stats['foul_rate'] = 0
        stats['most_disciplined_team'] = 'N/A'
        stats['lowest_foul_rate'] = 0
    
    return stats

def calculate_team_based_stats(df):
    """Calculate comprehensive team-based statistics"""
    team_stats = {}
    
    try:
        # Group by team for calculations
        team_groups = df.groupby('Team')
        
        # 1. Top Goals Club
        team_goals = team_groups['Goal'].sum().sort_values(ascending=False)
        if len(team_goals) > 0:
            team_stats['top_goals_club'] = team_goals.index[0]
            team_stats['top_goals_total'] = int(team_goals.iloc[0])
        else:
            team_stats['top_goals_club'] = 'N/A'
            team_stats['top_goals_total'] = 0
        
        # 2. Top Defense Club (tackles + clearances + interceptions)
        defensive_actions = team_groups[['Tackle', 'Clearance', 'Intercept']].sum()
        team_defense_totals = defensive_actions.sum(axis=1).sort_values(ascending=False)
        if len(team_defense_totals) > 0:
            team_stats['top_defense_club'] = team_defense_totals.index[0]
            team_stats['top_defense_total'] = int(team_defense_totals.iloc[0])
        else:
            team_stats['top_defense_club'] = 'N/A'
            team_stats['top_defense_total'] = 0
        
        # 3. Top Progression Club (passing + crosses + dribbles)
        progression_actions = team_groups[['Passing', 'Cross', 'Dribble Success']].sum()
        team_progression_totals = progression_actions.sum(axis=1).sort_values(ascending=False)
        if len(team_progression_totals) > 0:
            team_stats['top_progression_club'] = team_progression_totals.index[0]
            team_stats['top_progression_total'] = int(team_progression_totals.iloc[0])
        else:
            team_stats['top_progression_club'] = 'N/A'
            team_stats['top_progression_total'] = 0
        
        # 4. Worst Discipline Club (highest foul/card rate per player)
        # Only consider players with appearances > 0
        active_players = df[df['Appearances'] > 0]
        if len(active_players) > 0:
            team_discipline = active_players.groupby('Team').agg({
                'Foul': 'sum',
                'Yellow Card': 'sum', 
                'Own Goal': 'sum',
                'Player Name': 'count'  # Number of players per team
            })
            
            # Calculate discipline rate (fouls + cards + own goals per player)
            team_discipline['discipline_rate'] = (
                team_discipline['Foul'] + 
                team_discipline['Yellow Card'] + 
                team_discipline['Own Goal']
            ) / team_discipline['Player Name']
            
            worst_discipline = team_discipline['discipline_rate'].sort_values(ascending=False)
            if len(worst_discipline) > 0:
                team_stats['worst_discipline_club'] = worst_discipline.index[0]
                team_stats['worst_discipline_rate'] = round(worst_discipline.iloc[0], 2)
            else:
                team_stats['worst_discipline_club'] = 'N/A'
                team_stats['worst_discipline_rate'] = 0
        else:
            team_stats['worst_discipline_club'] = 'N/A'
            team_stats['worst_discipline_rate'] = 0
        
        # 5. Youngest Average Club
        team_avg_ages = team_groups['Age'].mean().sort_values(ascending=True)
        if len(team_avg_ages) > 0:
            team_stats['youngest_club'] = team_avg_ages.index[0]
            team_stats['youngest_avg_age'] = round(team_avg_ages.iloc[0], 1)
        else:
            team_stats['youngest_club'] = 'N/A'
            team_stats['youngest_avg_age'] = 0
        
        # 6. Oldest Average Club
        team_avg_ages_desc = team_groups['Age'].mean().sort_values(ascending=False)
        if len(team_avg_ages_desc) > 0:
            team_stats['oldest_club'] = team_avg_ages_desc.index[0]
            team_stats['oldest_avg_age'] = round(team_avg_ages_desc.iloc[0], 1)
        else:
            team_stats['oldest_club'] = 'N/A'
            team_stats['oldest_avg_age'] = 0
        
        # 7. Top Clean Sheet Club (best goalkeeper performance - saves per appearance)
        # Consider only goalkeepers or teams with saves data
        teams_with_saves = team_groups.agg({
            'Saves': 'sum',
            'Appearances': 'sum'
        })
        
        # Calculate saves per appearance for teams with > 0 saves
        teams_with_saves = teams_with_saves[teams_with_saves['Saves'] > 0]
        if len(teams_with_saves) > 0:
            teams_with_saves['saves_per_appearance'] = teams_with_saves['Saves'] / teams_with_saves['Appearances']
            best_gk_team = teams_with_saves['saves_per_appearance'].sort_values(ascending=False)
            
            if len(best_gk_team) > 0:
                team_stats['top_cleansheet_club'] = best_gk_team.index[0]
                team_stats['top_cleansheet_rate'] = round(best_gk_team.iloc[0], 2)
            else:
                team_stats['top_cleansheet_club'] = 'N/A'
                team_stats['top_cleansheet_rate'] = 0
        else:
            team_stats['top_cleansheet_club'] = 'N/A'
            team_stats['top_cleansheet_rate'] = 0
        
    except Exception as e:
        # Return default values in case of error
        team_stats = {
            'top_goals_club': 'N/A', 'top_goals_total': 0,
            'top_defense_club': 'N/A', 'top_defense_total': 0,
            'top_progression_club': 'N/A', 'top_progression_total': 0,
            'worst_discipline_club': 'N/A', 'worst_discipline_rate': 0,
            'youngest_club': 'N/A', 'youngest_avg_age': 0,
            'oldest_club': 'N/A', 'oldest_avg_age': 0,
            'top_cleansheet_club': 'N/A', 'top_cleansheet_rate': 0
        }
    
    return team_stats

# Page config
st.set_page_config(
    page_title="Explore in-depth player insights and performance trends from the Indonesia Super League — all in one powerful dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the player statistics data with validation"""
    try:
        data_path = Path(__file__).parent / "data" / "players_statistics.csv"
        df = pd.read_csv(data_path)
        
        # Data validation and cleaning
        if df.empty:
            raise ValueError("The CSV file is empty")
        
        # Check for required columns
        required_columns = ['Name', 'Player Name', 'Team', 'Position', 'Age', 'Appearances']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        # Remove rows where essential columns are empty
        df = df.dropna(subset=['Player Name', 'Team', 'Position'])
        
        # Clean up Position column - remove any extra whitespace
        df['Position'] = df['Position'].str.strip()
        
        # Fill missing numeric values with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Ensure Age and Appearances are integers
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(int)
        df['Appearances'] = pd.to_numeric(df['Appearances'], errors='coerce').fillna(0).astype(int)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading player data: {str(e)}")

# Define player statistic categories
PLAYER_METRIC_CATEGORIES = {
    'Attack': ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
    'Defense': ['Block', 'Block Cross', 'Clearance', 'Tackle', 'Intercept', 'Ball Recovery', 'Header Won'],
    'Progression': ['Passing', 'Cross', 'Dribble Success', 'Free Kick'],
    'Discipline': ['Foul', 'Fouled', 'Yellow Card', 'Own Goal'],
    'Goalkeeper': ['Saves']
}

# Position mapping for better display
POSITION_DISPLAY = {
    'BELAKANG': 'Defense',
    'TENGAH': 'Midfield', 
    'DEPAN': 'Attack',
    'P. GAWANG': 'Goalkeeper'
}

# Define negative metrics (lower values are better)
NEGATIVE_METRICS = [
    'Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target'
]

# Main navigation
def main():
    # Header
    st.markdown('<h1 class="main-header">Indonesia Super League Player Performance Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        
        # Additional data validation
        if len(df) == 0:
            st.error("❌ No valid player data found after cleaning.")
            return
            
    except FileNotFoundError:
        st.error("❌ Data file not found! Please make sure players_statistics.csv exists in the data folder.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["📈 Player Performance", "🔄 Player Comparison", "🎯 Profile Finder", "📊 Scatter Analysis", "👤 Player Report", "🔍 Player Screener"],
        index=0
    )
    
    # Global Filters
    #st.sidebar.markdown("### 🔧 Global Filters")
    st.sidebar.markdown("---")
    
    # Team filter with error handling
    try:
        teams = sorted([team for team in df['Team'].unique() if pd.notna(team)])
        if not teams:
            st.error("❌ No valid teams found in data.")
            return
            
        selected_teams = st.sidebar.multiselect(
            "🏟️ Select Teams",
            options=teams,
            default=teams,
            help="Select one or more teams to filter players"
        )
    except Exception as e:
        st.error(f"❌ Error processing teams: {str(e)}")
        return
    
    # Position filter with error handling
    try:
        positions = sorted([pos for pos in df['Position'].unique() if pd.notna(pos)])
        if not positions:
            st.error("❌ No valid positions found in data.")
            return
            
        selected_positions = st.sidebar.multiselect(
            "🎯 Select Positions",
            options=positions,
            default=positions,
            help="Filter by player positions (BELAKANG=Defense, TENGAH=Midfield, DEPAN=Attack, P. GAWANG=Goalkeeper)"
        )
    except Exception as e:
        st.error(f"❌ Error processing positions: {str(e)}")
        return
    
    # Age range filter with error handling
    try:
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        if age_min == age_max:
            age_max = age_min + 1  # Ensure we have a valid range
            
        age_range = st.sidebar.slider(
            "👶 Age Range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            help="Filter players by age range"
        )
    except Exception as e:
        st.error(f"❌ Error processing age data: {str(e)}")
        return
    
    # Appearances range filter with error handling
    try:
        app_min, app_max = int(df['Appearances'].min()), int(df['Appearances'].max())
        if app_min == app_max:
            app_max = app_min + 1  # Ensure we have a valid range
            
        appearances_range = st.sidebar.slider(
            "🏃 Appearances Range",
            min_value=app_min,
            max_value=app_max,
            value=(app_min, app_max),
            help="Filter players by number of appearances"
        )
    except Exception as e:
        st.error(f"❌ Error processing appearances data: {str(e)}")
        return
    
    # Apply filters with error handling
    try:
        filtered_df = df[
            (df['Team'].isin(selected_teams)) &
            (df['Position'].isin(selected_positions)) &
            (df['Age'] >= age_range[0]) &
            (df['Age'] <= age_range[1]) &
            (df['Appearances'] >= appearances_range[0]) &
            (df['Appearances'] <= appearances_range[1])
        ]
    except Exception as e:
        st.error(f"❌ Error filtering data: {str(e)}")
        return
    
    # Display filtered data summary
    try:
        st.sidebar.markdown("### 📈 Filtered Data Summary")
        info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
        metric_count = len([col for col in df.columns if col not in info_columns])
        st.sidebar.info(f"**Players**: {len(filtered_df)} of {len(df)}\n**Teams**: {filtered_df['Team'].nunique()}\n**Metrics**: {metric_count}")
    except Exception as e:
        st.sidebar.error(f"Error calculating summary: {str(e)}")
    
    # Sidebar footer with app information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ App Information")
    
    # Get current date for last updated (you can modify this to actual data update date)
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    st.sidebar.markdown(f"""
    <div style='font-size: 0.8em; color: #666;'>
    📊 <strong>Data Source:</strong><br>
    <a href='https://ileague.id/' target='_blank' style='text-decoration: none;'>ileague.id</a>
    
    🗓️ <strong>Last Updated:</strong><br>
    {current_date}
    
    📱 <strong>Version:</strong><br>
    v1.2.0
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "📈 Player Performance":
        show_stats_dashboard(filtered_df)
    elif page == "🔄 Player Comparison":
        show_player_comparison(filtered_df)
    elif page == "🎯 Profile Finder":
        show_profile_finder(filtered_df)
    elif page == "📊 Scatter Analysis":
        show_scatter_analysis(filtered_df)
    elif page == "👤 Player Report":
        show_player_report(filtered_df)
    elif page == "🔍 Player Screener":
        show_player_screener(filtered_df)

def show_stats_dashboard(df):
    """Display the stats dashboard page"""
    st.header("📈 Player Performance Dashboard")
    
    if len(df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria.")
        return
    
    # Get all metric columns (excluding player info columns)
    info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    metric_columns = [col for col in df.columns if col not in info_columns]
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏅 Overall", "⚔️ Attack", "🛡️ Defense", "📈 Progression", "🟨 Discipline"])
    
    with tab1:
        show_overall_performance(df, metric_columns)
    
    with tab2:
        show_category_performance(df, "Attack", PLAYER_METRIC_CATEGORIES['Attack'])
    
    with tab3:
        show_category_performance(df, "Defense", PLAYER_METRIC_CATEGORIES['Defense'])
    
    with tab4:
        show_category_performance(df, "Progression", PLAYER_METRIC_CATEGORIES['Progression'])
    
    with tab5:
        show_category_performance(df, "Discipline", PLAYER_METRIC_CATEGORIES['Discipline'])

def show_overall_performance(df, metric_columns):
    """Show overall player performance across all metrics"""
    st.subheader("🏆 Top Performers - All Metrics")
    
    # Global Stats Dashboard
    st.markdown("### 📊 Overall Statistics Dashboard")
    try:
        overall_stats = calculate_overall_stats(df)
        team_stats = calculate_team_based_stats(df)
        
        # Row 1: Basic Stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Total Players", f"{overall_stats['total_players']:,}")
        
        with col2:
            st.metric("👥 Average Age", f"{overall_stats['avg_age']:.1f}")
        
        with col3:
            st.metric("🏟️ Teams", f"{overall_stats['total_teams']}")
        
        # Row 2: Team Performance Stats
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric("🏆 Top Goals Club", f"{team_stats['top_goals_club']}", 
                     f"{team_stats['top_goals_total']} goals")
        
        with col5:
            st.metric("🛡️ Top Defense Club", f"{team_stats['top_defense_club']}", 
                     f"{team_stats['top_defense_total']} actions")
        
        with col6:
            st.metric("📈 Top Progression Club", f"{team_stats['top_progression_club']}", 
                     f"{team_stats['top_progression_total']} actions")
        
        # Row 3: Additional Team Stats
        col7, col8, col9, col10 = st.columns(4)
        
        with col7:
            st.metric("🟨 Worst Discipline Club", f"{team_stats['worst_discipline_club']}", 
                     f"{team_stats['worst_discipline_rate']} rate")
        
        with col8:
            st.metric("👶 Youngest Club", f"{team_stats['youngest_club']}", 
                     f"{team_stats['youngest_avg_age']} avg age")
        
        with col9:
            st.metric("🧓 Oldest Club", f"{team_stats['oldest_club']}", 
                     f"{team_stats['oldest_avg_age']} avg age")
        
        with col10:
            st.metric("🥅 Top Clean Sheet Club", f"{team_stats['top_cleansheet_club']}", 
                     f"{team_stats['top_cleansheet_rate']} saves/app")
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error calculating overall stats: {str(e)}")
    
    # Calculate overall performance score
    # Normalize each metric to 0-1 scale, handling negative metrics appropriately
    normalized_df = df.copy()
    
    for col in metric_columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max == col_min:
                normalized_df[col] = 0.5
            else:
                if col in NEGATIVE_METRICS:
                    # For negative metrics, invert normalization (lower = better)
                    normalized_df[col] = 1 - ((df[col] - col_min) / (col_max - col_min))
                else:
                    # For positive metrics, normal normalization (higher = better)
                    normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
    
    # Calculate overall score as average of normalized metrics
    overall_score = normalized_df[metric_columns].mean(axis=1)
    df_with_score = df.copy()
    df_with_score['Overall_Score'] = overall_score
    
    # Top 10 players overall
    top_players = df_with_score.nlargest(10, 'Overall_Score')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display top players table
        display_df = top_players[['Player Name', 'Team', 'Position', 'Age', 'Appearances', 'Overall_Score']].copy()
        display_df['Overall_Score'] = display_df['Overall_Score'].round(3)
        display_df.index = range(1, len(display_df) + 1)
        
        st.dataframe(
            display_df,
            column_config={
                "Overall_Score": st.column_config.ProgressColumn(
                    "Overall Score",
                    help="Overall performance score (0-1 scale)",
                    min_value=0,
                    max_value=1,
                )
            },
            use_container_width=True
        )
    
    with col2:
        # Overall score distribution
        fig = px.histogram(
            df_with_score,
            x='Overall_Score',
            title="Overall Score Distribution",
            nbins=20,
            color_discrete_sequence=['#FF6B35']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Category Performance Section
    st.markdown("---")
    st.subheader("📊 Category Performance Analysis")
    st.markdown("*Performance breakdown by Attack, Defense, Progression, and Discipline*")
    
    try:
        # Generate category scores and visualizations
        category_scores_df, top_players_by_category, charts = create_category_scores_visualization(
            df, PLAYER_METRIC_CATEGORIES, NEGATIVE_METRICS
        )
        
        # Create summary table
        summary_table = create_category_summary_table(top_players_by_category)
        
        # Display category summary
        if not summary_table.empty:
            st.markdown("#### 🏆 Category Leaders")
            st.dataframe(summary_table, use_container_width=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Category Charts", "🎯 Radar Comparison", "📊 Score Distribution"])
        
        with tab1:
            st.markdown("#### Top Performers by Category")
            # Display category bar charts in a 2x2 grid
            col1, col2 = st.columns(2)
            
            categories = ['Attack', 'Defense', 'Progression', 'Discipline']
            for i, category in enumerate(categories):
                chart_key = f'{category}_bar'
                if chart_key in charts:
                    if i % 2 == 0:
                        with col1:
                            st.plotly_chart(charts[chart_key], use_container_width=True)
                    else:
                        with col2:
                            st.plotly_chart(charts[chart_key], use_container_width=True)
        
        with tab2:
            st.markdown("#### Category Comparison - Top 5 Players")
            if 'category_radar' in charts:
                st.plotly_chart(charts['category_radar'], use_container_width=True)
                st.info("💡 This radar chart shows how the top 5 overall players compare across all categories. Larger areas indicate better overall performance.")
            else:
                st.warning("Radar chart could not be generated.")
        
        with tab3:
            st.markdown("#### Category Score Distributions")
            if 'category_distribution' in charts:
                st.plotly_chart(charts['category_distribution'], use_container_width=True)
                st.info("💡 Box plots show the distribution of scores in each category. The box shows the middle 50% of players, with the line inside showing the median.")
            else:
                st.warning("Distribution chart could not be generated.")
        
        # Category-specific top players tables
        st.markdown("#### 📋 Detailed Category Rankings")
        
        # Create expandable sections for each category
        for category in ['Attack', 'Defense', 'Progression', 'Discipline']:
            if category in top_players_by_category:
                if category == 'Discipline':
                    expander_title = f"🔍 Worst 10 {category} Players"
                else:
                    expander_title = f"🔍 Top 10 {category} Players"
                    
                with st.expander(expander_title):
                    top_players = top_players_by_category[category]
                    score_col = f'{category}_Score'
                    
                    display_cols = ['Player Name', 'Team', 'Position', 'Age', score_col]
                    display_df = top_players[display_cols].copy()
                    display_df.index = range(1, len(display_df) + 1)
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            score_col: st.column_config.ProgressColumn(
                                f"{category} Score",
                                help=f"Normalized {category.lower()} performance score (0-1 scale)",
                                min_value=0,
                                max_value=1,
                            )
                        },
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"❌ Error generating category analysis: {str(e)}")
        st.info("Please try refreshing the page or adjusting your filters.")

def show_category_performance(df, category_name, metrics):
    """Show performance for a specific category"""
    st.subheader(f"🔍 {category_name} Performance")
    
    # Category-specific Global Stats Dashboard
    st.markdown(f"### 📊 {category_name} Statistics Dashboard")
    try:
        if category_name == 'Attack':
            stats = calculate_attack_stats(df)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("⚽ Total Goals", f"{int(stats['total_goals']):,}")
            with col2:
                st.metric("🎯 Total Assists", f"{int(stats['total_assists']):,}")
            with col3:
                st.metric("🔥 Shots On Target", f"{int(stats['total_shots_on_target']):,}")
            with col4:
                st.metric("🥅 Penalty Goals", f"{int(stats['total_penalty_goals']):,}")
            with col5:
                st.metric("📊 Shot Conversion", f"{stats['shot_conversion_rate']:.1f}%")
            
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("📈 Goals/Appearance", f"{stats['goals_per_appearance']:.2f}")
            with col7:
                st.metric("💡 Chances Created", f"{int(stats['total_create_chance']):,}")
            with col8:
                st.metric("🎯 Target Accuracy", f"{stats['shots_on_target_rate']:.1f}%")
        
        elif category_name == 'Defense':
            stats = calculate_defense_stats(df)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🦵 Total Tackles", f"{int(stats['total_tackles']):,}")
            with col2:
                st.metric("🧹 Total Clearances", f"{int(stats['total_clearances']):,}")
            with col3:
                st.metric("🛡️ Total Interceptions", f"{int(stats['total_interceptions']):,}")
            with col4:
                st.metric("🚫 Total Blocks", f"{int(stats['total_blocks']):,}")
            with col5:
                st.metric("📊 Avg Actions/Player", f"{stats['avg_defensive_actions']:.1f}")
            
            col6, col7 = st.columns(2)
            with col6:
                st.metric("🏃 Ball Recoveries", f"{int(stats['total_ball_recovery']):,}")
            with col7:
                st.metric("🏆 Most Defensive Pos", f"{POSITION_DISPLAY.get(stats['most_defensive_position'], stats['most_defensive_position'])}")
        
        elif category_name == 'Progression':
            stats = calculate_progression_stats(df)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("⚽ Total Passes", f"{int(stats['total_passes']):,}")
            with col2:
                st.metric("📐 Total Crosses", f"{int(stats['total_crosses']):,}")
            with col3:
                st.metric("🏃 Successful Dribbles", f"{int(stats['total_dribble_success']):,}")
            with col4:
                st.metric("⚡ Free Kicks", f"{int(stats['total_free_kicks']):,}")
            with col5:
                st.metric("📊 Avg Actions/Player", f"{stats['avg_progression_actions']:.1f}")
            
            col6, col7 = st.columns(2)
            with col6:
                st.metric("🎯 Top Playmaker", f"{stats['most_active_playmaker']}")
            with col7:
                st.metric("📈 Top Passer Total", f"{int(stats['top_passer_total']):,}")
        
        elif category_name == 'Discipline':
            stats = calculate_discipline_stats(df)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("🟨 Total Yellow Cards", f"{int(stats['total_yellow_cards']):,}")
            with col2:
                st.metric("🚫 Total Fouls", f"{int(stats['total_fouls_committed']):,}")
            with col3:
                st.metric("⚽ Own Goals", f"{int(stats['total_own_goals']):,}")
            with col4:
                st.metric("😵 Times Fouled", f"{int(stats['total_fouled']):,}")
            with col5:
                st.metric("📊 Foul Rate", f"{stats['foul_rate']:.2f}", help="Fouls per appearance")
            
            col6, col7 = st.columns(2)
            with col6:
                st.metric("🏆 Most Disciplined Team", f"{stats['most_disciplined_team']}")
            with col7:
                st.metric("📉 Lowest Foul Rate", f"{stats['lowest_foul_rate']:.2f}")
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error calculating {category_name.lower()} stats: {str(e)}")
    
    # Filter metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        st.warning(f"No metrics found for {category_name} category.")
        return
    
    # Create sub-tabs for each metric in the category
    if len(available_metrics) > 4:
        # Split into multiple rows if too many metrics
        metrics_per_row = 4
        rows = [available_metrics[i:i+metrics_per_row] for i in range(0, len(available_metrics), metrics_per_row)]
        
        for row_idx, row_metrics in enumerate(rows):
            tabs = st.tabs([f"📊 {metric}" for metric in row_metrics])
            for tab, metric in zip(tabs, row_metrics):
                with tab:
                    show_metric_performance(df, metric)
    else:
        tabs = st.tabs([f"📊 {metric}" for metric in available_metrics])
        for tab, metric in zip(tabs, available_metrics):
            with tab:
                show_metric_performance(df, metric)

def show_metric_performance(df, metric):
    """Show top performers for a specific metric"""
    try:
        if metric not in df.columns:
            st.warning(f"Metric '{metric}' not found in data.")
            return
        
        if len(df) == 0:
            st.warning("No players available with current filters.")
            return
        
        # Check if the metric has any valid data
        if df[metric].isna().all():
            st.warning(f"No valid data available for {metric}.")
            return
        
        # Determine if we should show top or bottom performers
        is_negative = metric in NEGATIVE_METRICS
        ascending = is_negative
        
        # Get top/bottom 10 performers
        try:
            top_performers = df.nsmallest(10, metric) if is_negative else df.nlargest(10, metric)
            if len(top_performers) == 0:
                st.warning(f"No performers found for {metric}.")
                return
        except Exception as e:
            st.error(f"Error getting top performers for {metric}: {str(e)}")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create bar chart
            title = f"{'Bottom' if is_negative else 'Top'} 10 Players: {metric}"
            color_scale = 'Reds_r' if is_negative else 'Blues'
            
            fig = px.bar(
                top_performers,
                x='Player Name',
                y=metric,
                hover_data=['Team', 'Position', 'Age', 'Appearances'],
                title=title,
                color=metric,
                color_continuous_scale=color_scale
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Player",
                yaxis_title=metric
            )
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics summary
            st.metric(
                f"Dataset Avg",
                f"{df[metric].mean():.1f}",
                help=f"Average {metric} across all filtered players"
            )
            
            st.metric(
                f"Dataset Max",
                f"{df[metric].max():.0f}",
                help=f"Highest {metric} in filtered dataset"
            )
            
            # Best performer info
            if is_negative:
                best_idx = df[metric].idxmin()
                best_value = df[metric].min()
                label = "Best (Lowest)"
            else:
                best_idx = df[metric].idxmax()
                best_value = df[metric].max()
                label = "Best (Highest)"
            
            best_player = df.loc[best_idx]
            st.metric(
                label,
                f"{best_player['Player Name']}",
                f"{best_value:.0f}",
                help=f"Best performer in {metric}"
            )
            
            # Show detailed top performers list
            st.markdown("#### 🏆 Top Performers")
            display_df = top_performers[['Player Name', 'Team', metric]].head(5)
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error displaying {metric} performance: {str(e)}")
        st.info("Please try refreshing the page or adjusting your filters.")

if __name__ == "__main__":
    main()