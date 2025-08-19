"""
AI Football Performance Analyst Page
Interactive interface for AI-powered football analysis using Mistral-7B-Instruct-v0.3
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import custom modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import PlayerDataLoader
from utils.hf_auth import get_hf_auth, quick_token_test
from ai.analyst import GemmaAnalyst
from ai.analysis_types import AnalysisType, AnalysisTypes

def render_authentication_sidebar():
    """Render Hugging Face authentication section in sidebar"""
    st.sidebar.markdown("### 🔑 Authentication")
    
    # Get current token sources
    hf_auth = get_hf_auth()
    token_sources = hf_auth.get_token_sources()
    
    # Show current token status
    if token_sources['active_token']:
        # Mask token for display
        masked_token = token_sources['active_token'][:8] + "..." + token_sources['active_token'][-4:]
        st.sidebar.success(f"✅ Token found: {masked_token}")
        st.sidebar.caption(f"Source: {token_sources['source']}")
        
        # Test token button
        if st.sidebar.button("🔍 Test Token"):
            with st.spinner("Testing token..."):
                test_result = quick_token_test(token_sources['active_token'])
                
                if test_result['success']:
                    st.sidebar.success("✅ Token is valid!")
                    if 'user_info' in test_result and test_result['user_info']:
                        user_name = test_result['user_info'].get('name', 'Unknown')
                        st.sidebar.info(f"👤 Logged in as: {user_name}")
                    
                    # Store validated token in session
                    st.session_state.hf_token = token_sources['active_token']
                    st.session_state.token_validated = True
                else:
                    st.sidebar.error(f"❌ Token test failed: {test_result.get('error', 'Unknown error')}")
                    st.session_state.token_validated = False
    
    else:
        st.sidebar.warning("⚠️ No token found")
        
        # Token input section
        with st.sidebar.expander("🔧 Configure Token", expanded=True):
            token_input = st.text_input(
                "Enter Hugging Face Token:",
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Token", disabled=not token_input):
                    if token_input:
                        # Validate and save token
                        validation = hf_auth.validate_token_format(token_input)
                        
                        if validation['valid']:
                            # Test connection
                            test_result = quick_token_test(token_input)
                            
                            if test_result['success']:
                                # Save token
                                save_result = hf_auth.save_token(token_input, method='file')
                                
                                if save_result['success']:
                                    st.success("✅ Token saved and validated!")
                                    st.session_state.hf_token = token_input
                                    st.session_state.token_validated = True
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to save: {save_result['error']}")
                            else:
                                st.error(f"❌ Token validation failed: {test_result.get('error')}")
                        else:
                            st.error(f"❌ Invalid format: {', '.join(validation['errors'])}")
            
            with col2:
                if st.button("🧪 Test Only", disabled=not token_input):
                    if token_input:
                        test_result = quick_token_test(token_input)
                        
                        if test_result['success']:
                            st.success("✅ Token is valid!")
                            if 'user_info' in test_result and test_result['user_info']:
                                user_name = test_result['user_info'].get('name', 'Unknown')
                                st.info(f"👤 User: {user_name}")
                            # Store in session but don't save to file
                            st.session_state.hf_token = token_input
                            st.session_state.token_validated = True
                        else:
                            st.error(f"❌ Test failed: {test_result.get('error')}")
            
            # Help section
            if st.button("❓ Setup Help"):
                st.session_state.show_token_help = not st.session_state.get('show_token_help', False)
            
            if st.session_state.get('show_token_help', False):
                instructions = hf_auth.get_setup_instructions()
                for instruction in instructions[:15]:  # Show first 15 lines
                    st.caption(instruction)
                
                if st.button("🔧 Troubleshooting"):
                    st.session_state.show_troubleshooting = not st.session_state.get('show_troubleshooting', False)
                
                if st.session_state.get('show_troubleshooting', False):
                    troubleshooting = hf_auth.get_troubleshooting_guide()
                    for tip in troubleshooting[:10]:  # Show first 10 lines
                        st.caption(tip)

def main():
    st.set_page_config(
        page_title="AI Football Analyst",
        page_icon="🤖⚽",
        layout="wide"
    )
    
    st.title("🤖 AI Football Performance Analyst")
    st.markdown("### Powered by Google Gemma-3-270M | Indonesia Super League Analysis")
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'ai_analyst' not in st.session_state:
        st.session_state.ai_analyst = None
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
    if 'token_validated' not in st.session_state:
        st.session_state.token_validated = False
    
    # Load data
    @st.cache_data
    def load_data():
        loader = PlayerDataLoader()
        return loader.load_data(), loader
    
    try:
        df, data_loader = load_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the data file exists at: `data/players_statistics.csv`")
        return
    
    # Initialize AI Analyst with protection
    if st.session_state.ai_analyst is None:
        with st.spinner("Initializing AI Analyst... This may take a few minutes on first run."):
            try:
                # Use token from session state if available
                token = st.session_state.get('hf_token')
                st.session_state.ai_analyst = GemmaAnalyst(token=token)
                
                # Check system status before attempting initialization
                system_status = st.session_state.ai_analyst.get_system_status()
                
                # Display system status
                if 'error' in system_status:
                    st.error(f"System check failed: {system_status['error']}")
                    st.info("AI Analyst will use fallback mode for basic analysis.")
                elif not system_status['memory_status'].get('sufficient', False):
                    st.warning("⚠️ Insufficient system memory detected")
                    
                    # Show memory details
                    memory_info = system_status['memory_status']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Available Memory", f"{memory_info.get('available_gb', 0):.1f} GB")
                    with col2:
                        st.metric("Required Memory", f"{memory_info.get('required_gb', 0):.1f} GB")
                    with col3:
                        st.metric("Total Memory", f"{memory_info.get('total_gb', 0):.1f} GB")
                    
                    # Show warnings and recommendations
                    if system_status.get('warnings'):
                        st.warning("System Warnings:")
                        for warning in system_status['warnings']:
                            st.caption(f"• {warning}")
                    
                    if system_status.get('recommendations'):
                        with st.expander("💡 Recommendations"):
                            for rec in system_status['recommendations']:
                                st.caption(f"• {rec}")
                    
                    st.info("AI Analyst will use statistical fallback mode instead of the full AI model.")
                    st.caption("💡 Gemma-3-270M only requires ~1GB RAM - much lighter than previous models!")
                
                # Attempt to initialize model (will fail gracefully if insufficient memory)
                init_success = st.session_state.ai_analyst.initialize()
                
                if init_success:
                    st.success("✅ AI model loaded successfully!")
                else:
                    st.warning("⚠️ AI model could not be loaded - using fallback mode")
                    st.info("You can still perform statistical analysis and get insights from your data.")
                
            except ImportError as e:
                st.error(f"Missing dependency: {str(e)}")
                st.info("Please run the setup script or install missing packages.")
                st.code("pip install transformers torch psutil")
                return
            except Exception as e:
                st.error(f"Error initializing AI Analyst: {str(e)}")
                if "token" in str(e).lower():
                    st.info("💡 Try setting up a Hugging Face token in the Authentication section")
                if "memory" in str(e).lower() or "cuda" in str(e).lower():
                    st.info("💡 Try closing other applications or using a system with more RAM")
                
                # Still allow fallback mode
                st.info("Continuing with statistical analysis mode...")
                try:
                    token = st.session_state.get('hf_token')
                    st.session_state.ai_analyst = GemmaAnalyst(token=token)
                except:
                    st.error("Could not initialize even fallback mode. Please check your setup.")
                    return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Analysis Configuration")
    
    # Authentication Section
    render_authentication_sidebar()
    
    st.sidebar.markdown("---")
    
    # Analysis type selection
    analysis_types_helper = AnalysisTypes()
    available_types = analysis_types_helper.get_available_analysis_types()
    
    analysis_type_names = [t['name'] for t in available_types]
    selected_type_name = st.sidebar.selectbox(
        "Select Analysis Type",
        analysis_type_names
    )
    
    # Get selected analysis type
    selected_type = None
    for t in available_types:
        if t['name'] == selected_type_name:
            selected_type = AnalysisType(t['type'])
            break
    
    # Show analysis description
    type_config = analysis_types_helper.get_analysis_config(selected_type)
    st.sidebar.info(f"**{selected_type_name}**\n\n{type_config['description']}")
    
    # Analysis complexity
    complexity = st.sidebar.selectbox(
        "Analysis Depth",
        ["Standard", "Detailed", "Professional"],
        index=1
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Analysis interface based on type
        if selected_type == AnalysisType.PLAYER_PERFORMANCE:
            render_player_analysis_interface(df, data_loader, st.session_state.ai_analyst, complexity)
        
        elif selected_type == AnalysisType.TEAM_ANALYSIS:
            render_team_analysis_interface(df, data_loader, st.session_state.ai_analyst, complexity)
        
        elif selected_type == AnalysisType.SCOUT_REPORT:
            render_scout_report_interface(df, data_loader, st.session_state.ai_analyst, complexity)
        
        elif selected_type == AnalysisType.PLAYER_COMPARISON:
            render_player_comparison_interface(df, data_loader, st.session_state.ai_analyst, complexity)
        
        elif selected_type == AnalysisType.TACTICAL_ANALYSIS:
            render_tactical_analysis_interface(df, data_loader, st.session_state.ai_analyst, complexity)
        
        elif selected_type == AnalysisType.CUSTOM_QUERY:
            render_custom_query_interface(df, data_loader, st.session_state.ai_analyst, complexity)
    
    with col2:
        # Analysis history and saved reports
        render_analysis_history()

def render_player_analysis_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render player performance analysis interface"""
    st.subheader("👤 Player Performance Analysis")
    
    # Player selection
    teams = data_loader.get_teams()
    selected_team = st.selectbox("Select Team", ["All Teams"] + teams)
    
    if selected_team == "All Teams":
        available_players = df['Player Name'].tolist()
    else:
        team_players = df[df['Team'] == selected_team]
        available_players = team_players['Player Name'].tolist()
    
    selected_player = st.selectbox("Select Player", available_players)
    
    if selected_player:
        player_data = df[df['Player Name'] == selected_player].iloc[0]
        
        # Show player info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Team", player_data['Team'])
        with col2:
            st.metric("Position", player_data['Position'])
        with col3:
            st.metric("Age", player_data['Age'])
        with col4:
            st.metric("Appearances", player_data['Appearances'])
        
        # Additional context
        context = st.text_area(
            "Additional Context (Optional)",
            placeholder="e.g., Recent form, injury concerns, tactical role..."
        )
        
        if st.button("🔍 Generate Player Analysis", type="primary"):
            with st.spinner("AI is analyzing the player's performance..."):
                try:
                    analysis = ai_analyst.analyze_player_performance(player_data, context)
                    
                    st.markdown("### 📊 AI Analysis Results")
                    st.markdown(analysis)
                    
                    # Save to history
                    save_to_history("Player Analysis", selected_player, analysis)
                    
                    # Show key metrics visualization
                    render_player_metrics_chart(player_data, data_loader)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try using a simpler analysis or check system resources.")

def render_team_analysis_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render team analysis interface"""
    st.subheader("🏆 Team Performance Analysis")
    
    teams = data_loader.get_teams()
    selected_team = st.selectbox("Select Team for Analysis", teams)
    
    if selected_team:
        team_data = df[df['Team'] == selected_team]
        
        # Show team overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(team_data))
        with col2:
            st.metric("Total Goals", team_data['Goal'].sum())
        with col3:
            st.metric("Total Assists", team_data['Assist'].sum())
        with col4:
            st.metric("Avg Age", f"{team_data['Age'].mean():.1f}")
        
        # Additional context
        context = st.text_area(
            "Analysis Context (Optional)",
            placeholder="e.g., Current league position, recent transfers, tactical changes..."
        )
        
        if st.button("🔍 Generate Team Analysis", type="primary"):
            with st.spinner("AI is analyzing the team's performance and tactics..."):
                try:
                    analysis = ai_analyst.analyze_team_performance(team_data, selected_team, context)
                    
                    st.markdown("### 📊 AI Team Analysis Results")
                    st.markdown(analysis)
                    
                    # Save to history
                    save_to_history("Team Analysis", selected_team, analysis)
                    
                    # Show team composition chart
                    render_team_composition_chart(team_data)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try using a simpler analysis or check system resources.")

def render_scout_report_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render scouting report interface"""
    st.subheader("🔍 Scout Report Generator")
    
    # Scouting criteria
    st.markdown("#### Define Scouting Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        positions = data_loader.get_positions()
        target_positions = st.multiselect("Target Positions", positions)
        
        age_range = st.slider(
            "Age Range",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(20, 28)
        )
    
    with col2:
        all_metrics = data_loader.get_all_metrics()
        key_metrics = st.multiselect(
            "Key Performance Metrics",
            all_metrics,
            default=['Goal', 'Assist', 'Passing', 'Tackle']
        )
        
        min_appearances = st.number_input(
            "Minimum Appearances",
            min_value=0,
            value=1
        )
    
    # Budget tier (optional)
    budget_tier = st.selectbox(
        "Budget Tier",
        ["Any", "Low Budget", "Medium Budget", "High Budget"]
    )
    
    # Additional context
    context = st.text_area(
        "Scouting Context (Optional)",
        placeholder="e.g., Team needs, playing style requirements, budget constraints..."
    )
    
    # Build criteria dictionary
    criteria = {
        'position': target_positions,
        'age_range': age_range,
        'key_metrics': key_metrics,
        'min_appearances': min_appearances,
        'budget_tier': budget_tier
    }
    
    if st.button("🔍 Generate Scout Report", type="primary"):
        with st.spinner("AI is generating your scout report..."):
            try:
                analysis = ai_analyst.generate_scout_report(df, criteria, context)
                
                st.markdown("### 📊 AI Scout Report")
                st.markdown(analysis)
                
                # Save to history
                save_to_history("Scout Report", f"Positions: {', '.join(target_positions)}", analysis)
                
                # Show top candidates chart
                render_scout_candidates_chart(df, criteria)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Try using a simpler analysis or check system resources.")

def render_player_comparison_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render player comparison interface"""
    st.subheader("⚔️ Player Comparison Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Player 1")
        team1 = st.selectbox("Team 1", data_loader.get_teams(), key="team1")
        team1_players = df[df['Team'] == team1]['Player Name'].tolist()
        player1 = st.selectbox("Player 1", team1_players, key="player1")
    
    with col2:
        st.markdown("#### Player 2") 
        team2 = st.selectbox("Team 2", data_loader.get_teams(), key="team2")
        team2_players = df[df['Team'] == team2]['Player Name'].tolist()
        player2 = st.selectbox("Player 2", team2_players, key="player2")
    
    # Metrics to compare
    all_metrics = data_loader.get_all_metrics()
    comparison_metrics = st.multiselect(
        "Metrics to Compare",
        all_metrics,
        default=['Goal', 'Assist', 'Passing', 'Tackle', 'Dribble Success']
    )
    
    # Additional context
    context = st.text_area(
        "Comparison Context (Optional)",
        placeholder="e.g., Transfer decision, tactical fit, position change..."
    )
    
    if player1 and player2 and comparison_metrics:
        player1_data = df[df['Player Name'] == player1].iloc[0]
        player2_data = df[df['Player Name'] == player2].iloc[0]
        
        if st.button("🔍 Compare Players", type="primary"):
            with st.spinner("AI is comparing the players..."):
                try:
                    analysis = ai_analyst.compare_players(player1_data, player2_data, comparison_metrics, context)
                    
                    st.markdown("### 📊 AI Player Comparison")
                    st.markdown(analysis)
                    
                    # Save to history
                    save_to_history("Player Comparison", f"{player1} vs {player2}", analysis)
                    
                    # Show comparison chart
                    render_player_comparison_chart(player1_data, player2_data, comparison_metrics)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try using a simpler analysis or check system resources.")

def render_tactical_analysis_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render tactical analysis interface"""
    st.subheader("🎯 Tactical Pattern Analysis")
    
    teams = data_loader.get_teams()
    selected_team = st.selectbox("Select Team for Tactical Analysis", teams)
    
    if selected_team:
        team_data = df[df['Team'] == selected_team]
        
        # Show tactical overview metrics
        st.markdown("#### Team Tactical Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Passes", team_data['Passing'].sum())
        with col2:
            st.metric("Total Crosses", team_data['Cross'].sum())
        with col3:
            st.metric("Successful Dribbles", team_data['Dribble Success'].sum())
        with col4:
            st.metric("Total Tackles", team_data['Tackle'].sum())
        
        # Additional context
        context = st.text_area(
            "Tactical Context (Optional)",
            placeholder="e.g., Recent formation changes, key player injuries, upcoming matches..."
        )
        
        if st.button("🔍 Generate Tactical Analysis", type="primary"):
            with st.spinner("AI is analyzing tactical patterns..."):
                try:
                    analysis = ai_analyst.identify_tactical_patterns(team_data, selected_team, context)
                    
                    st.markdown("### 📊 AI Tactical Analysis")
                    st.markdown(analysis)
                    
                    # Save to history
                    save_to_history("Tactical Analysis", selected_team, analysis)
                    
                    # Show tactical distribution chart
                    render_tactical_chart(team_data, selected_team)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try using a simpler analysis or check system resources.")

def render_custom_query_interface(df: pd.DataFrame, data_loader: PlayerDataLoader, ai_analyst: GemmaAnalyst, complexity: str):
    """Render custom query interface"""
    st.subheader("❓ Custom Football Analysis")
    
    # Query input
    custom_question = st.text_area(
        "Ask Your Football Question",
        placeholder="e.g., Who are the most undervalued players in the league? Which team has the best defensive record? What players should we target for our attacking needs?",
        height=100
    )
    
    # Data context selection
    st.markdown("#### Select Relevant Data Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Team selection for context
        include_teams = st.multiselect(
            "Include Specific Teams (Optional)",
            data_loader.get_teams()
        )
        
        # Position filter
        include_positions = st.multiselect(
            "Focus on Positions (Optional)",
            data_loader.get_positions()
        )
    
    with col2:
        # Metric categories
        metric_categories = data_loader.get_metric_categories()
        focus_categories = st.multiselect(
            "Focus on Metric Categories (Optional)",
            list(metric_categories.keys())
        )
        
        # Analysis scope
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["Individual Players", "Team Level", "League Wide", "Mixed"]
        )
    
    # Prepare data summary based on selections
    if custom_question:
        filtered_df = df.copy()
        
        if include_teams:
            filtered_df = filtered_df[filtered_df['Team'].isin(include_teams)]
        
        if include_positions:
            filtered_df = filtered_df[filtered_df['Position'].isin(include_positions)]
        
        # Create data summary
        data_summary = create_data_summary(filtered_df, focus_categories, metric_categories)
        
        if st.button("🔍 Get AI Analysis", type="primary"):
            with st.spinner("AI is analyzing your question..."):
                try:
                    analysis = ai_analyst.generate_analysis(
                        ai_analyst.prompt_templates.get_custom_analysis_prompt(
                            analysis_type="Custom Query",
                            data_summary=data_summary,
                            specific_question=custom_question,
                            context=f"Scope: {analysis_scope}"
                        )
                    )
                    
                    st.markdown("### 📊 AI Custom Analysis")
                    st.markdown(analysis)
                    
                    # Save to history
                    save_to_history("Custom Analysis", custom_question[:50] + "...", analysis)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try using a simpler analysis or check system resources.")

def render_analysis_history():
    """Render analysis history sidebar"""
    st.subheader("📚 Analysis History")
    
    if st.session_state.analysis_history:
        for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"{entry['type']} - {entry['subject'][:20]}..."):
                st.caption(f"Generated: {entry['timestamp']}")
                st.markdown(entry['analysis'][:200] + "...")
                
                if st.button(f"View Full Analysis", key=f"view_{i}"):
                    st.modal(entry['analysis'])
    else:
        st.info("No analysis history yet. Generate your first analysis!")
    
    # Clear history button
    if st.session_state.analysis_history and st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

def save_to_history(analysis_type: str, subject: str, analysis: str):
    """Save analysis to session history"""
    entry = {
        'type': analysis_type,
        'subject': subject,
        'analysis': analysis,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.analysis_history.append(entry)

def create_data_summary(df: pd.DataFrame, focus_categories: List[str], metric_categories: Dict[str, List[str]]) -> str:
    """Create data summary for custom analysis"""
    summary_lines = [f"Dataset Summary: {len(df)} players from {df['Team'].nunique()} teams"]
    
    if focus_categories:
        summary_lines.append(f"\nFocus Areas: {', '.join(focus_categories)}")
        
        for category in focus_categories:
            metrics = metric_categories.get(category, [])
            if metrics:
                summary_lines.append(f"\n{category.title()} Metrics:")
                for metric in metrics:
                    if metric in df.columns:
                        total = df[metric].sum()
                        avg = df[metric].mean()
                        summary_lines.append(f"  {metric}: Total={total}, Avg={avg:.1f}")
    
    # Top performers
    summary_lines.append(f"\nTop Scorers:")
    top_scorers = df.nlargest(3, 'Goal')[['Player Name', 'Team', 'Goal']]
    for _, player in top_scorers.iterrows():
        summary_lines.append(f"  {player['Player Name']} ({player['Team']}): {player['Goal']} goals")
    
    return '\n'.join(summary_lines)

# Visualization functions
def render_player_metrics_chart(player_data: pd.Series, data_loader: PlayerDataLoader):
    """Render player metrics visualization"""
    st.markdown("#### Player Performance Metrics")
    
    metric_categories = data_loader.get_metric_categories()
    
    # Create radar chart data
    categories = []
    values = []
    
    for category, metrics in metric_categories.items():
        category_total = 0
        category_count = 0
        
        for metric in metrics:
            if metric in player_data.index:
                category_total += player_data[metric]
                category_count += 1
        
        if category_count > 0:
            categories.append(category)
            values.append(category_total / category_count)
    
    if categories and values:
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player_data['Player Name']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )),
            showlegend=True,
            title=f"Performance Profile: {player_data['Player Name']}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_team_composition_chart(team_data: pd.DataFrame):
    """Render team composition visualization"""
    st.markdown("#### Team Composition")
    
    # Position distribution
    position_counts = team_data['Position'].value_counts()
    
    fig = px.pie(
        values=position_counts.values,
        names=position_counts.index,
        title="Team Position Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_scout_candidates_chart(df: pd.DataFrame, criteria: Dict[str, Any]):
    """Render scouting candidates visualization"""
    st.markdown("#### Top Scouting Candidates")
    
    # Apply filters
    filtered_df = df.copy()
    
    if criteria.get('position'):
        filtered_df = filtered_df[filtered_df['Position'].isin(criteria['position'])]
    
    if criteria.get('age_range'):
        min_age, max_age = criteria['age_range']
        filtered_df = filtered_df[(filtered_df['Age'] >= min_age) & (filtered_df['Age'] <= max_age)]
    
    # Calculate composite score
    key_metrics = criteria.get('key_metrics', ['Goal', 'Assist'])
    available_metrics = [m for m in key_metrics if m in filtered_df.columns]
    
    if available_metrics:
        filtered_df['scout_score'] = filtered_df[available_metrics].sum(axis=1)
        top_candidates = filtered_df.nlargest(10, 'scout_score')
        
        fig = px.bar(
            top_candidates,
            x='scout_score',
            y='Player Name',
            color='Team',
            orientation='h',
            title="Top Scouting Candidates by Performance Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_player_comparison_chart(player1_data: pd.Series, player2_data: pd.Series, metrics: List[str]):
    """Render player comparison visualization"""
    st.markdown("#### Player Comparison Chart")
    
    comparison_data = []
    for metric in metrics:
        if metric in player1_data.index and metric in player2_data.index:
            comparison_data.append({
                'Metric': metric,
                player1_data['Player Name']: player1_data[metric],
                player2_data['Player Name']: player2_data[metric]
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df.melt(id_vars=['Metric'], var_name='Player', value_name='Value'),
            x='Metric',
            y='Value',
            color='Player',
            barmode='group',
            title="Head-to-Head Performance Comparison"
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def render_tactical_chart(team_data: pd.DataFrame, team_name: str):
    """Render tactical analysis visualization"""
    st.markdown("#### Tactical Style Visualization")
    
    # Create tactical style metrics
    tactical_metrics = {
        'Attacking': team_data['Goal'].sum() + team_data['Assist'].sum() + team_data['Create Chance'].sum(),
        'Passing': team_data['Passing'].sum() + team_data['Cross'].sum(),
        'Defending': team_data['Tackle'].sum() + team_data['Block'].sum() + team_data['Clearance'].sum(),
        'Individual Skill': team_data['Dribble Success'].sum()
    }
    
    fig = px.bar(
        x=list(tactical_metrics.keys()),
        y=list(tactical_metrics.values()),
        title=f"Tactical Style Profile: {team_name}",
        labels={'x': 'Tactical Area', 'y': 'Total Actions'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()