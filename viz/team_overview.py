import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
from typing import Dict, List, Tuple

def load_club_logos() -> Dict[str, str]:
    """Load club logos from JSON file"""
    try:
        with open('data/club_logo.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {club['club']: club['logoUrl'] for club in data['clubs']}
    except Exception as e:
        st.error(f"Error loading club logos: {e}")
        return {}

def load_club_stats() -> pd.DataFrame:
    """Load club statistics from CSV"""
    try:
        df = pd.read_csv('data/football_stats.csv')
        return df
    except Exception as e:
        st.error(f"Error loading club stats: {e}")
        return pd.DataFrame()

def load_player_stats() -> pd.DataFrame:
    """Load player statistics from CSV"""
    try:
        df = pd.read_csv('data/players_statistics.csv')
        return df
    except Exception as e:
        st.error(f"Error loading player stats: {e}")
        return pd.DataFrame()

def calculate_attacking_metrics(club_stats: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    """Calculate attacking metrics for each team"""

    # Aggregate player goals by team
    team_goals = player_stats.groupby('Team')['Goal'].sum().reset_index()
    team_goals.columns = ['Team', 'Total Goals']

    # Get top shooters per team
    player_stats['Total Shoots'] = player_stats['Shoot On Target'] + player_stats['Shoot Off Target']

    # Merge with club stats
    merged = club_stats.merge(team_goals, left_on='TEAM', right_on='Team', how='left')
    merged['Total Goals'] = merged['Total Goals'].fillna(0)

    # Calculate shot accuracy
    merged['Shot Accuracy'] = (merged['Tembakan ke Gawang'] / merged['Total Tembakan'] * 100).round(2)
    merged['Shot Accuracy'] = merged['Shot Accuracy'].fillna(0)

    # Calculate composite score
    # Formula: (Goals × 3) + (Shot Accuracy × 2) + (Total Shots × 0.1)
    merged['Score'] = (
        (merged['Total Goals'] * 3) +
        (merged['Shot Accuracy'] * 2) +
        (merged['Total Tembakan'] * 0.1)
    ).round(2)

    # Sort by score descending
    merged = merged.sort_values('Score', ascending=False).reset_index(drop=True)
    merged['Rank'] = range(1, len(merged) + 1)

    return merged

def get_top_shooters(player_stats: pd.DataFrame, team: str, top_n: int = 3) -> List[Dict]:
    """Get top shooters for a specific team"""
    team_players = player_stats[player_stats['Team'] == team].copy()
    team_players['Total Shoots'] = team_players['Shoot On Target'] + team_players['Shoot Off Target']

    # Sort by total shoots and get top N
    top_players = team_players.nlargest(top_n, 'Total Shoots')

    result = []
    for _, player in top_players.iterrows():
        if player['Total Shoots'] > 0:  # Only include players with shots
            result.append({
                'name': player['Player Name'],
                'image': player['Picture Url'],
                'shoots': int(player['Total Shoots']),
                'on_target': int(player['Shoot On Target']),
                'off_target': int(player['Shoot Off Target'])
            })

    return result

def get_score_color(score: float, min_score: float, max_score: float) -> str:
    """Get color based on score using gradient from dark blue to red"""
    if max_score == min_score:
        return '#4472C4'  # Default blue

    # Normalize score to 0-1
    normalized = (score - min_score) / (max_score - min_score)

    # Color gradient: Dark Blue → Light Blue → Yellow → Orange → Red
    if normalized >= 0.8:
        return '#002060'  # Dark blue (best)
    elif normalized >= 0.6:
        return '#4472C4'  # Blue
    elif normalized >= 0.4:
        return '#FFC000'  # Yellow
    elif normalized >= 0.2:
        return '#ED7D31'  # Orange
    else:
        return '#C00000'  # Red (worst)

def display_custom_table_layout(attacking_df: pd.DataFrame, club_logos: Dict[str, str], player_stats: pd.DataFrame):
    """Display custom table-like layout with images using native Streamlit components"""

    st.subheader("📊 Attacking Analysis Overview")
    st.caption("Teams ranked by attacking performance with top shooter")

    # Header row
    header_cols = st.columns([0.5, 2.5, 0.7, 0.7, 0.7, 1, 2.5, 1])
    headers = ["#", "Team", "Goal", "Shot", "SoT", "Acc%", "Player", "Score"]

    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    st.markdown("---")

    # Get min/max scores for color coding
    min_score = attacking_df['Score'].min()
    max_score = attacking_df['Score'].max()

    # Data rows
    for _, row in attacking_df.iterrows():
        cols = st.columns([0.5, 2.5, 0.7, 0.7, 0.7, 1, 2.5, 1])

        team_name = row['TEAM']
        logo_url = club_logos.get(team_name, '')

        # Column 1: Rank
        cols[0].markdown(f"<div style='text-align:center;padding-top:15px;font-weight:bold'>{row['Rank']}</div>", unsafe_allow_html=True)

        # Column 2: Team (logo + name)
        with cols[1]:
            team_col1, team_col2 = st.columns([1, 4])
            with team_col1:
                if logo_url:
                    try:
                        st.image(logo_url, width=35)
                    except:
                        st.write("🏅")
                else:
                    st.write("🏅")
            with team_col2:
                st.markdown(f"<div style='padding-top:12px'>{team_name}</div>", unsafe_allow_html=True)

        # Column 3-6: Stats
        cols[2].markdown(f"<div style='text-align:center;padding-top:15px'>{int(row['Total Goals'])}</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='text-align:center;padding-top:15px'>{int(row['Total Tembakan'])}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div style='text-align:center;padding-top:15px'>{int(row['Tembakan ke Gawang'])}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div style='text-align:center;padding-top:15px'>{row['Shot Accuracy']:.1f}%</div>", unsafe_allow_html=True)

        # Column 7: Top Player (image + name)
        with cols[6]:
            top_shooters = get_top_shooters(player_stats, team_name, top_n=1)
            if top_shooters:
                top_player = top_shooters[0]
                player_col1, player_col2 = st.columns([1, 4])

                with player_col1:
                    if top_player['image'] and str(top_player['image']).strip() and str(top_player['image']) != 'nan':
                        try:
                            st.image(top_player['image'], width=35)
                        except:
                            st.write("👤")
                    else:
                        st.write("👤")

                with player_col2:
                    # Truncate long names
                    player_name = top_player['name']
                    if len(player_name) > 20:
                        player_name = player_name[:17] + "..."
                    st.markdown(f"<div style='padding-top:8px;font-size:11px'>{player_name}<br><span style='color:gray;font-size:9px'>{top_player['shoots']} shots</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding-top:15px;color:gray'>No data</div>", unsafe_allow_html=True)

        # Column 8: Score with color
        score = row['Score']
        score_color = get_score_color(score, min_score, max_score)
        cols[7].markdown(
            f"<div style='background-color:{score_color};color:white;padding:10px;border-radius:5px;text-align:center;font-weight:bold;margin-top:5px'>{score:.1f}</div>",
            unsafe_allow_html=True
        )

        # Row divider
        st.markdown("<hr style='margin:5px 0;border:none;border-top:1px solid #e0e0e0'>", unsafe_allow_html=True)

def display_club_logos_grid(attacking_df: pd.DataFrame, club_logos: Dict[str, str]):
    """Display club logos in a grid layout"""
    st.subheader("🏆 Club Rankings")

    # Display in rows of 6 clubs
    clubs_per_row = 6
    num_rows = (len(attacking_df) + clubs_per_row - 1) // clubs_per_row

    for row_idx in range(num_rows):
        start_idx = row_idx * clubs_per_row
        end_idx = min(start_idx + clubs_per_row, len(attacking_df))

        cols = st.columns(clubs_per_row)

        for col_idx, idx in enumerate(range(start_idx, end_idx)):
            row = attacking_df.iloc[idx]
            team_name = row['TEAM']
            logo_url = club_logos.get(team_name, '')

            with cols[col_idx]:
                # Display logo
                if logo_url:
                    try:
                        st.image(logo_url, width=50)
                    except:
                        st.write("🏅")
                else:
                    st.write("🏅")

                # Display rank and team name
                st.caption(f"**#{row['Rank']} {team_name}**")
                st.caption(f"Score: {row['Score']:.1f}")

def create_styled_dataframe(attacking_df: pd.DataFrame) -> pd.DataFrame:
    """Create styled dataframe for display"""

    # Prepare display dataframe
    display_df = attacking_df[[
        'Rank', 'TEAM', 'Total Goals', 'Total Tembakan',
        'Tembakan ke Gawang', 'Shot Accuracy', 'Score'
    ]].copy()

    display_df.columns = [
        'Rank', 'Team', 'Goals', 'Total Shots',
        'Shots on Target', 'Accuracy %', 'Score'
    ]

    return display_df

def display_attacking_table(attacking_df: pd.DataFrame):
    """Display main attacking statistics table"""
    st.subheader("📊 Attacking Performance Table")

    display_df = create_styled_dataframe(attacking_df)

    # Apply color styling to score column
    def color_score(val):
        min_score = display_df['Score'].min()
        max_score = display_df['Score'].max()
        color = get_score_color(val, min_score, max_score)
        return f'background-color: {color}; color: white; font-weight: bold'

    # Style the dataframe
    styled_df = display_df.style.applymap(
        color_score,
        subset=['Score']
    ).format({
        'Accuracy %': '{:.1f}%',
        'Score': '{:.2f}',
        'Goals': '{:.0f}',
        'Total Shots': '{:.0f}',
        'Shots on Target': '{:.0f}'
    })

    # Display with streamlit
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=min(600, len(display_df) * 35 + 38)
    )

def display_top_shooters_section(attacking_df: pd.DataFrame, club_logos: Dict[str, str], player_stats: pd.DataFrame):
    """Display top shooters for each team with images"""
    st.subheader("🎯 Top Shooters by Team")

    for _, row in attacking_df.iterrows():
        team_name = row['TEAM']
        logo_url = club_logos.get(team_name, '')
        top_shooters = get_top_shooters(player_stats, team_name, top_n=3)

        with st.expander(f"#{row['Rank']} {team_name} - {int(row['Total Goals'])} Goals", expanded=False):
            # Team header with logo
            header_col1, header_col2 = st.columns([1, 4])

            with header_col1:
                if logo_url:
                    try:
                        st.image(logo_url, width=60)
                    except:
                        st.write("🏅")
                else:
                    st.write("🏅")

            with header_col2:
                st.markdown(f"**{team_name}**")
                st.caption(f"Score: {row['Score']:.2f} | Goals: {int(row['Total Goals'])} | Shots: {int(row['Total Tembakan'])} | Accuracy: {row['Shot Accuracy']:.1f}%")

            # Display top shooters
            if top_shooters:
                st.markdown("**Top 3 Shooters:**")

                for i, player in enumerate(top_shooters, 1):
                    player_col1, player_col2 = st.columns([1, 5])

                    with player_col1:
                        # Display player image
                        if player['image'] and str(player['image']).strip() and str(player['image']) != 'nan':
                            try:
                                st.image(player['image'], width=50)
                            except:
                                st.write("👤")
                        else:
                            st.write("👤")

                    with player_col2:
                        st.markdown(f"**{i}. {player['name']}**")
                        st.caption(
                            f"Total Shots: {player['shoots']} | "
                            f"On Target: {player['on_target']} | "
                            f"Off Target: {player['off_target']}"
                        )

                    if i < len(top_shooters):
                        st.markdown("---")
            else:
                st.info("No shooting data available for this team")

def render_team_overview(player_stats: pd.DataFrame = None):
    """Main function to render team overview visualization"""

    st.header("🏆 Team Overview - Attacking Analysis")
    st.markdown("*Comprehensive attacking performance analysis based on goals, shooting accuracy, and player contributions*")

    # Load data
    club_logos = load_club_logos()
    club_stats = load_club_stats()

    if player_stats is None:
        player_stats = load_player_stats()

    if club_stats.empty or player_stats.empty:
        st.error("Unable to load required data. Please ensure data files are available.")
        return

    # Calculate attacking metrics
    attacking_df = calculate_attacking_metrics(club_stats, player_stats)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Teams", len(attacking_df))
    with col2:
        total_goals = attacking_df['Total Goals'].sum()
        st.metric("Total Goals", int(total_goals))
    with col3:
        avg_accuracy = attacking_df['Shot Accuracy'].mean()
        st.metric("Avg Shot Accuracy", f"{avg_accuracy:.1f}%")
    with col4:
        total_shots = attacking_df['Total Tembakan'].sum()
        st.metric("Total Shots", int(total_shots))

    st.markdown("---")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview Table",
        "🏆 Club Rankings",
        "📈 Performance Table",
        "🎯 Top Shooters"
    ])

    with tab1:
        # Custom table-like layout with images
        display_custom_table_layout(attacking_df, club_logos, player_stats)

    with tab2:
        # Club logos grid
        display_club_logos_grid(attacking_df, club_logos)

    with tab3:
        # Styled dataframe table
        display_attacking_table(attacking_df)

    with tab4:
        # Top shooters section with expandable cards
        display_top_shooters_section(attacking_df, club_logos, player_stats)

    # Display methodology
    with st.expander("📊 Methodology & Score Calculation"):
        st.markdown("""
        ### Score Calculation Formula
        ```
        Score = (Goals × 3) + (Shot Accuracy × 2) + (Total Shots × 0.1)
        ```

        ### Components:
        - **Goals**: Total goals scored by all team players (highest weight)
        - **Shot Accuracy**: Percentage of shots on target (medium weight)
        - **Total Shots**: Total shooting attempts (lowest weight)

        ### Color Coding:
        - 🔵 **Dark Blue**: Top performers (80-100% of max score)
        - 🔵 **Blue**: Strong performance (60-80%)
        - 🟡 **Yellow**: Average performance (40-60%)
        - 🟠 **Orange**: Below average (20-40%)
        - 🔴 **Red**: Needs improvement (0-20%)

        ### Data Sources:
        - Club statistics from `football_stats.csv`
        - Player statistics from `players_statistics.csv`
        - Club logos from `club_logo.json`
        """)

    # Display detailed data table
    with st.expander("📋 View Detailed Data"):
        display_df = attacking_df[[
            'Rank', 'TEAM', 'Total Goals', 'Total Tembakan',
            'Tembakan ke Gawang', 'Shot Accuracy', 'Score'
        ]].copy()

        display_df.columns = [
            'Rank', 'Team', 'Goals', 'Total Shots',
            'Shots on Target', 'Accuracy %', 'Score'
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Goals": st.column_config.NumberColumn("Goals", width="small"),
                "Total Shots": st.column_config.NumberColumn("Total Shots", width="small"),
                "Shots on Target": st.column_config.NumberColumn("Shots on Target", width="small"),
                "Accuracy %": st.column_config.NumberColumn("Accuracy %", format="%.2f%%", width="small"),
                "Score": st.column_config.NumberColumn("Score", format="%.2f", width="small")
            }
        )

if __name__ == "__main__":
    render_team_overview()
