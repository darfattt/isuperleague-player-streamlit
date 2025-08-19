import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class ProfileScorer:
    """
    Player profile scoring system with weighted metrics and percentile rankings
    """
    
    def __init__(self):
        # Define scoring categories with preset metrics and weights
        self.categories = {
            'Attacking Score': {
                'Goal': 0.35,
                'Assist': 0.25, 
                'Shoot On Target': 0.20,
                'Create Chance': 0.15,
                'Penalty Goal': 0.05
            },
            'Playmaking Score': {
                'Passing': 0.30,
                'Assist': 0.25,
                'Create Chance': 0.20,
                'Cross': 0.15,
                'Free Kick': 0.10
            },
            'Build up Score': {
                'Passing': 0.35,
                'Dribble Success': 0.25,
                'Ball Recovery': 0.20,
                'Cross': 0.20
            },
            'Defensive Score': {
                'Tackle': 0.25,
                'Clearance': 0.20,
                'Intercept': 0.20,
                'Block': 0.15,
                'Header Won': 0.10,
                'Ball Recovery': 0.10
            }
        }
        
        # Define negative metrics (lower is better)
        self.negative_metrics = ['Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target']
    
    def get_available_metrics(self, df: pd.DataFrame) -> List[str]:
        """Get all available metrics excluding info columns"""
        info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
        return [col for col in df.columns if col not in info_columns]
    
    def calculate_category_score(self, df: pd.DataFrame, category: str, weights: Dict[str, float], 
                                additional_metrics: Dict[str, float] = None, top_n_limit: int = 30) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Calculate weighted category score for all players, with percentiles based on top N performers
        
        Args:
            df: Player dataframe
            category: Category name
            weights: Dictionary of metric weights
            additional_metrics: Additional metrics with weights
            top_n_limit: Number of top players to use for percentile calculations (default: 30)
            
        Returns:
            Tuple of (DataFrame with calculated scores and percentiles, normalized weights used)
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # Combine preset and additional metrics
        all_weights = weights.copy()
        if additional_metrics:
            all_weights.update(additional_metrics)
        
        # Filter weights to only include metrics present in dataframe
        available_weights = {metric: weight for metric, weight in all_weights.items() 
                           if metric in df.columns}
        
        if not available_weights:
            st.warning(f"No valid metrics found for {category}")
            return pd.DataFrame()
        
        # Normalize weights to sum to 1
        total_weight = sum(available_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        else:
            normalized_weights = available_weights
        
        # Calculate normalized scores for each metric
        result_df = df.copy()
        weighted_scores = pd.Series(0.0, index=df.index)
        
        for metric, weight in normalized_weights.items():
            if metric in df.columns:
                col_values = df[metric]
                col_min = col_values.min()
                col_max = col_values.max()
                
                if col_max == col_min:
                    normalized_values = pd.Series(50.0, index=df.index)
                else:
                    if metric in self.negative_metrics:
                        # For negative metrics, invert normalization (lower = better)
                        normalized_values = 100 - ((col_values - col_min) / (col_max - col_min) * 100)
                    else:
                        # For positive metrics, normal normalization (higher = better)
                        normalized_values = (col_values - col_min) / (col_max - col_min) * 100
                
                # Add weighted contribution to total score
                weighted_scores += normalized_values * weight
        
        # Add scores to result dataframe
        score_column = f'{category.replace(" ", "_")}'
        result_df[score_column] = weighted_scores
        
        # Sort by score to get rankings
        result_df = result_df.sort_values(score_column, ascending=False).reset_index(drop=True)
        result_df['Rank'] = range(1, len(result_df) + 1)
        
        # Calculate percentile rank based on top N players only
        percentile_column = f'{score_column}_Percentile'
        top_n = min(top_n_limit, len(result_df))
        top_players_df = result_df.head(top_n).copy()
        
        # Calculate percentiles for top N players (spread 0-100% across top N)
        top_players_df[percentile_column] = top_players_df[score_column].rank(pct=True) * 100
        
        # For remaining players, assign percentiles below the lowest top N percentile
        if len(result_df) > top_n:
            min_top_percentile = top_players_df[percentile_column].min()
            remaining_players = result_df.iloc[top_n:].copy()
            # Assign percentiles from 0 to min_top_percentile for remaining players
            remaining_count = len(remaining_players)
            if remaining_count > 1:
                remaining_percentiles = np.linspace(0, min_top_percentile * 0.95, remaining_count)
                remaining_players[percentile_column] = remaining_percentiles
            else:
                remaining_players[percentile_column] = 0
            
            # Combine top N and remaining players
            result_df = pd.concat([top_players_df, remaining_players]).sort_values('Rank').reset_index(drop=True)
        else:
            # If we have fewer than top_n_limit players, use all players
            result_df[percentile_column] = result_df[score_column].rank(pct=True) * 100
        
        return result_df[[
            'Rank', 'Player Name', 'Team', 'Position', 'Age', 'Appearances',
            score_column, percentile_column
        ] + list(normalized_weights.keys())], normalized_weights
    
    def get_metric_contributions(self, df: pd.DataFrame, player_idx: int, category: str, 
                               weights: Dict[str, float], additional_metrics: Dict[str, float] = None, 
                               top_n_limit: int = 30) -> Dict[str, float]:
        """Get individual metric contributions to a player's total score"""
        all_weights = weights.copy()
        if additional_metrics:
            all_weights.update(additional_metrics)
        
        # Filter weights to only include metrics present in dataframe
        available_weights = {metric: weight for metric, weight in all_weights.items() 
                           if metric in df.columns}
        
        # Normalize weights
        total_weight = sum(available_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        else:
            normalized_weights = available_weights
        
        # Calculate scores for all players to determine top N subset
        weighted_scores = pd.Series(0.0, index=df.index)
        for metric, weight in normalized_weights.items():
            if metric in df.columns:
                col_values = df[metric]
                col_min = col_values.min()
                col_max = col_values.max()
                
                if col_max != col_min:
                    if metric in self.negative_metrics:
                        normalized_values = 100 - ((col_values - col_min) / (col_max - col_min) * 100)
                    else:
                        normalized_values = (col_values - col_min) / (col_max - col_min) * 100
                    weighted_scores += normalized_values * weight
        
        # Get top N players for metric percentile calculations
        df_with_scores = df.copy()
        df_with_scores['temp_score'] = weighted_scores
        top_n = min(top_n_limit, len(df_with_scores))
        top_players_df = df_with_scores.nlargest(top_n, 'temp_score')
        
        contributions = {}
        for metric, weight in normalized_weights.items():
            if metric in df.columns:
                # Use top N subset for min/max calculations
                col_values = top_players_df[metric]
                col_min = col_values.min()
                col_max = col_values.max()
                player_value = df.loc[player_idx, metric]
                
                if col_max == col_min:
                    normalized_value = 50.0
                else:
                    if metric in self.negative_metrics:
                        normalized_value = 100 - ((player_value - col_min) / (col_max - col_min) * 100)
                    else:
                        normalized_value = (player_value - col_min) / (col_max - col_min) * 100
                    
                    # Clamp normalized value to 0-100 range
                    normalized_value = max(0, min(100, normalized_value))
                
                contributions[metric] = {
                    'raw_value': player_value,
                    'normalized_score': normalized_value,
                    'weight': weight,
                    'weighted_contribution': normalized_value * weight
                }
        
        return contributions

def get_percentile_color(percentile_rank):
    """Return color code based on percentile range - optimized for top 30 distribution"""
    if percentile_rank >= 90:
        return '#1a5f27'  # Elite dark green
    elif percentile_rank >= 80:
        return '#1a9641'  # Excellent green
    elif percentile_rank >= 70:
        return '#4caf50'  # Very good green
    elif percentile_rank >= 60:
        return '#73c378'  # Good medium green  
    elif percentile_rank >= 50:
        return '#a4d65e'  # Above average light green
    elif percentile_rank >= 40:
        return '#f9d057'  # Average yellow
    elif percentile_rank >= 30:
        return '#ffa726'  # Below average orange
    elif percentile_rank >= 20:
        return '#fc8d59'  # Poor light orange
    elif percentile_rank >= 10:
        return '#e57373'  # Bad light red
    else:
        return '#d73027'  # Very bad red

def style_weighted_score(val, percentile_rank):
    """Style function for weighted score column"""
    color = get_percentile_color(percentile_rank)
    return f'background-color: {color}; color: white; font-weight: bold'

def show_profile_finder(filtered_df):
    """
    Main profile finder page with enhanced table display
    
    Features:
    - Weighted scoring across 4 categories (Attacking, Playmaking, Build-up, Defensive)
    - Interactive weight adjustment with real-time calculation
    - Enhanced results table with rank, weights, metrics, and color-coded percentiles
    - Individual player detailed analysis with metric breakdowns
    - Visual charts and distribution analysis
    """
    st.header("🎯 Player Profile Finder")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No players match the current filters. Please adjust your filter criteria in the sidebar.")
        return
    
    # Initialize profile scorer
    scorer = ProfileScorer()
    
    # Show current filter summary
    st.info(
        f"🔍 **Profile Analysis Ready**  \n"
        f"   **Available Players**: {len(filtered_df)} players from {filtered_df['Team'].nunique()} teams  \n"
        f"   **Metric Categories**: Attacking, Playmaking, Build-up, Defensive"
    )
    
    # Category selection
    st.subheader("📊 Select Scoring Category")
    category = st.selectbox(
        "Choose a category to analyze:",
        options=list(scorer.categories.keys()),
        help="Each category has preset metrics with default weights that you can customize"
    )
    
    # Display category metrics and allow weight adjustment
    st.subheader(f"⚖️ Adjust Weights for {category}")
    
    preset_metrics = scorer.categories[category]
    available_metrics = scorer.get_available_metrics(filtered_df)
    
    # Create two columns for weight adjustment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Preset Metrics")
        adjusted_weights = {}
        
        # Display sliders for preset metrics
        for metric, default_weight in preset_metrics.items():
            if metric in available_metrics:
                adjusted_weights[metric] = st.slider(
                    f"{metric}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.05,
                    help=f"Weight for {metric} (default: {default_weight})"
                )
            else:
                st.warning(f"⚠️ {metric} not available in current dataset")
    
    with col2:
        # Additional metrics section
        st.subheader("➕Other Metrics")
        with st.expander("Customize with additional metrics"):
            other_metrics = [m for m in available_metrics if m not in preset_metrics.keys()]
        
        if other_metrics:
            st.markdown("Select additional metrics to include in the scoring:")
            additional_weights = {}
            
            # Create columns for additional metrics
            cols = st.columns(3)
            for i, metric in enumerate(other_metrics):
                with cols[i % 3]:
                    include_metric = st.checkbox(f"{metric}", key=f"include_{metric}")
                    if include_metric:
                        weight = st.slider(
                            f"Weight for {metric}",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.05,
                            key=f"weight_{metric}"
                        )
                        additional_weights[metric] = weight
        else:
            st.info("All available metrics are already included in preset categories.")
            additional_weights = {}
    
    #st.markdown("#### Category Info")
        
    # Show weight distribution
    if adjusted_weights:
        weight_total = sum(adjusted_weights.values())
        st.info(f"**{category} : {weight_total:.2f}**\n\nThis category measures player performance in specific areas using weighted metrics. Adjust weights to emphasize different aspects of performance.")

        #st.metric("Total Weight", f"{weight_total:.2f}")
        #if weight_total > 0:
        #    st.success("✅ Weights will be normalized")
        #else:
        #    st.error("❌ Total weight cannot be zero")
    
    
    # Calculate scores button
    if st.button("🔄 Calculate Profile Scores", type="primary"):
        if sum(adjusted_weights.values()) == 0:
            st.error("❌ Please set at least one metric weight greater than 0")
            return
        
        with st.spinner("Calculating profile scores..."):
            # Calculate scores
            results_df, used_weights = scorer.calculate_category_score(
                filtered_df, 
                category, 
                adjusted_weights, 
                additional_weights
            )
            
            if results_df.empty:
                st.error("❌ Could not calculate scores. Please check your metric selection.")
                return
            
            # Display results
            st.subheader("📊 Profile Scoring Results")
            
            # Get score and percentile column names
            score_col = f'{category.replace(" ", "_")}'
            percentile_col = f'{score_col}_Percentile'
            
            # Results are already sorted by score in calculate_category_score method
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Results Table", "📊 Score Distribution", "🔍 Player Detail"])
            
            with tab1:
                st.markdown("#### Top Performers")
                
                # Enhanced table with pandas styling (CSS replaced by programmatic styling)
                
                # Prepare enhanced display dataframe - show only top 30 players
                display_df = results_df.head(30).copy()
                
                # Add weight information for each metric
                weight_info = []
                for _, row in display_df.iterrows():
                    weights_str = "; ".join([f"{metric}: {weight:.2f}" for metric, weight in used_weights.items()])
                    weight_info.append(weights_str)
                display_df['Weights Used'] = weight_info
                
                # Format scores and percentiles
                display_df[score_col] = display_df[score_col].round(2)
                display_df[percentile_col] = display_df[percentile_col].round(1)
                
                # Define column configuration - reorganized order
                base_columns = ['Rank', 'Player Name', 'Team', 'Position', 'Age']
                score_columns = [score_col, percentile_col]
                metric_columns = list(used_weights.keys())
                
                # Create column config
                column_config = {
                    'Rank': st.column_config.NumberColumn(
                        "#",
                        help="Player rank",
                        width="small"
                    ),
                    score_col: st.column_config.NumberColumn(
                        "Weighted Score",
                        help="Performance score based on weighted metrics (colored by percentile rank)",
                        format="%.1f",
                        width="medium"
                    ),
                    percentile_col: st.column_config.ProgressColumn(
                        "Percentile Rank",
                        help="Percentile ranking among all players",
                        min_value=0,
                        max_value=100,
                        format="%.0f%%",
                        width="medium"
                    )
                    # 'Weights Used': st.column_config.TextColumn(
                    #     "Weights",
                    #     help="Weights used for each metric",
                    #     width="large"
                    # )
                }
                
                # Add configuration for metric columns
                for metric in metric_columns:
                    column_config[metric] = st.column_config.NumberColumn(
                        metric,
                        help=f"Raw {metric} value",
                        format="%.1f",
                        width="small"
                    )
                
                # Create styled dataframe with enhanced color coding
                all_columns = base_columns + score_columns + metric_columns
                table_df = display_df[all_columns].copy()
                
                # Apply styling to weighted score column
                styled_df = table_df.style.apply(
                    lambda row: [
                        style_weighted_score(row[score_col], row[percentile_col])
                        if col == score_col else ''
                        for col in table_df.columns
                    ],
                    axis=1
                )
                
                # Display the enhanced styled table
                st.dataframe(
                    styled_df,
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Enhanced summary statistics
                st.markdown("##### 📊 Summary Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Top 30 Players", f"{len(display_df)}")
                with col2:
                    st.metric("Top Score", f"{display_df[score_col].max():.1f}")
                with col3:
                    st.metric("Average Score", f"{display_df[score_col].mean():.1f}")
                with col4:
                    st.metric("Median Score", f"{display_df[score_col].median():.1f}")
                with col5:
                    top_percentile_count = len(display_df[display_df[percentile_col] >= 80])
                    st.metric("Top 20%", f"{top_percentile_count} players")
                
                # Metrics breakdown
                st.markdown("##### ⚖️ Weights Used in Calculation")
                weights_df = pd.DataFrame([
                    {'Metric': metric, 'Weight': f"{weight:.2f}", 'Percentage': f"{weight*100:.1f}%"} 
                    for metric, weight in used_weights.items()
                ])
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
            
            with tab2:
                st.markdown("#### Score Distribution Analysis")
                
                # Score distribution histogram
                fig_hist = px.histogram(
                    results_df,
                    x=score_col,
                    nbins=20,
                    title=f"{category} Score Distribution",
                    color_discrete_sequence=['#3498db'],
                    labels={score_col: f"{category} Score", 'count': 'Number of Players'}
                )
                fig_hist.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top 10 vs Bottom 10 comparison
                if len(results_df) >= 20:
                    st.markdown("#### Top 10 vs Bottom 10 Comparison")
                    top_10 = results_df.head(10)
                    bottom_10 = results_df.tail(10)
                    
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        y=top_10['Player Name'][::-1],  # Reverse for better display
                        x=top_10[score_col][::-1],
                        orientation='h',
                        name='Top 10',
                        marker_color='#2ecc71'
                    ))
                    
                    fig_compare.add_trace(go.Bar(
                        y=bottom_10['Player Name'],
                        x=bottom_10[score_col],
                        orientation='h',
                        name='Bottom 10',
                        marker_color='#e74c3c'
                    ))
                    
                    fig_compare.update_layout(
                        title=f"{category} - Top vs Bottom Performers",
                        xaxis_title=f"{category} Score",
                        height=600,
                        barmode='group'
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
            
            with tab3:
                st.markdown("#### Individual Player Analysis")
                
                # Player selection for detailed view
                selected_player = st.selectbox(
                    "Select a player for detailed breakdown:",
                    options=results_df['Player Name'].tolist(),
                    key="player_detail_select"
                )
                
                if selected_player:
                    player_data = results_df[results_df['Player Name'] == selected_player].iloc[0]
                    player_idx = filtered_df[filtered_df['Player Name'] == selected_player].index[0]
                    
                    # Player overview
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Rank", f"#{int(player_data['Rank'])}")
                    with col2:
                        st.metric("Player", selected_player)
                    with col3:
                        st.metric("Team", player_data['Team'])
                    with col4:
                        st.metric("Score", f"{player_data[score_col]:.1f}")
                    with col5:
                        percentile_val = player_data[percentile_col]
                        if percentile_val >= 81:
                            delta_color = "normal"
                        elif percentile_val >= 61:
                            delta_color = "normal"
                        else:
                            delta_color = "inverse"
                        st.metric("Percentile", f"{percentile_val:.1f}%")
                    
                    # Metric breakdown
                    st.markdown("##### Metric Contributions")
                    contributions = scorer.get_metric_contributions(
                        filtered_df, player_idx, category, adjusted_weights, additional_weights, top_n_limit=30
                    )
                    
                    if contributions:
                        # Create enhanced breakdown dataframe
                        breakdown_data = []
                        for metric, data in contributions.items():
                            # Determine percentile rank for this metric using top 30 subset
                            # First get top 30 players by overall score for this metric context
                            score_col = f'{category.replace(" ", "_")}'
                            top_30_for_metric = results_df.head(30)
                            
                            # Get metric values from the original filtered_df for these top 30 players
                            top_30_names = top_30_for_metric['Player Name'].tolist()
                            top_30_metric_values = filtered_df[filtered_df['Player Name'].isin(top_30_names)][metric]
                            
                            player_percentile = (top_30_metric_values < data['raw_value']).sum() / len(top_30_metric_values) * 100
                            if metric in scorer.negative_metrics:
                                player_percentile = 100 - player_percentile
                            
                            breakdown_data.append({
                                'Metric': metric,
                                'Raw Value': f"{data['raw_value']:.1f}",
                                'Percentile': f"{player_percentile:.1f}%",
                                'Normalized Score': f"{data['normalized_score']:.1f}",
                                'Weight': f"{data['weight']:.2f}",
                                'Weighted Contribution': f"{data['weighted_contribution']:.1f}"
                            })
                        
                        breakdown_df = pd.DataFrame(breakdown_data)
                        
                        # Apply color coding to the breakdown table
                        def highlight_percentile(row):
                            percentile = float(row['Percentile'].replace('%', ''))
                            color = get_percentile_color(percentile)
                            return [f'background-color: {color}; color: white; font-weight: bold' if col == 'Percentile' else '' for col in row.index]
                        
                        styled_breakdown = breakdown_df.style.apply(highlight_percentile, axis=1)
                        st.dataframe(styled_breakdown, use_container_width=True, hide_index=True)
                        
                        # Enhanced contribution chart
                        fig_contrib = px.bar(
                            breakdown_df,
                            x='Metric',
                            y=[float(x) for x in breakdown_df['Weighted Contribution']],
                            title=f"Weighted Metric Contributions for {selected_player}",
                            color=[float(x.replace('%', '')) for x in breakdown_df['Percentile']],
                            color_continuous_scale=[
                                [0.0, '#d73027'],   # Very bad red
                                [0.1, '#e57373'],   # Bad light red
                                [0.2, '#fc8d59'],   # Poor light orange
                                [0.3, '#ffa726'],   # Below average orange
                                [0.4, '#f9d057'],   # Average yellow
                                [0.5, '#a4d65e'],   # Above average light green
                                [0.6, '#73c378'],   # Good medium green
                                [0.7, '#4caf50'],   # Very good green
                                [0.8, '#1a9641'],   # Excellent green
                                [1.0, '#1a5f27']    # Elite dark green
                            ],
                            labels={'color': 'Percentile', 'y': 'Weighted Contribution'}
                        )
                        fig_contrib.update_layout(
                            height=400, 
                            showlegend=True,
                            xaxis_title="Metrics",
                            yaxis_title="Weighted Contribution to Total Score"
                        )
                        fig_contrib.update_traces(
                            texttemplate='%{y:.1f}',
                            textposition='outside'
                        )
                        st.plotly_chart(fig_contrib, use_container_width=True)