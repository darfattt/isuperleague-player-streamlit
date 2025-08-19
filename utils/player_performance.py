import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

class PlayerVisualizer:
    """
    Visualization utilities for player analytics dashboard
    Contains various chart creation methods using Plotly for individual player data
    """
    
    def __init__(self):
        # Color schemes
        self.player_colors = [
            '#FF6B35', '#F7931E', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE',
            '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#AED6F1',
            '#F5B7B1', '#D5A6BD', '#AEB6BF', '#85929E', '#566573'
        ]
        
        self.category_colors = {
            'Attack': '#FF6B35',
            'Defense': '#4ECDC4', 
            'Progression': '#F7931E',
            'Discipline': '#E74C3C',
            'Goalkeeper': '#45B7D1'
        }
        
        # Color schemes for positive/negative metrics
        self.positive_color_scale = 'Blues'
        self.negative_color_scale = 'Reds_r'
        
        # Default layout settings
        self.default_layout = {
            'font': dict(family="Arial, sans-serif", size=12),
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'margin': dict(l=50, r=50, t=50, b=50)
        }
    
    def create_player_performance_bar(self, df, metric, n=10, is_negative_metric=False, title=None):
        """Create bar chart for top/bottom player performers"""
        if title is None:
            title = f"{'Bottom' if is_negative_metric else 'Top'} {n} Players: {metric}"
        
        # Get top/bottom performers
        if is_negative_metric:
            top_players = df.nsmallest(n, metric)
            color_scale = self.negative_color_scale
        else:
            top_players = df.nlargest(n, metric)
            color_scale = self.positive_color_scale
        
        # Create hover text with player info
        hover_text = []
        for _, player in top_players.iterrows():
            hover_text.append(
                f"<b>{player['Player Name']}</b><br>"
                f"Team: {player['Team']}<br>"
                f"Position: {player['Position']}<br>"
                f"Age: {player['Age']}<br>"
                f"Appearances: {player['Appearances']}<br>"
                f"{metric}: {player[metric]}"
            )
        
        fig = px.bar(
            top_players,
            x='Player Name',
            y=metric,
            title=title,
            color=metric,
            color_continuous_scale=color_scale,
            hover_name='Player Name'
        )
        
        # Add custom hover data
        fig.update_traces(
            hovertemplate=hover_text,
            hoverinfo='text'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Player",
            yaxis_title=metric,
            **self.default_layout
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_team_comparison_chart(self, df, metric, title=None):
        """Create team comparison chart showing average metric per team"""
        if title is None:
            title = f"Team Average: {metric}"
        
        # Calculate team averages
        team_stats = df.groupby('Team')[metric].agg(['mean', 'count']).reset_index()
        team_stats.columns = ['Team', f'{metric}_avg', 'player_count']
        team_stats = team_stats.sort_values(f'{metric}_avg', ascending=False)
        
        fig = px.bar(
            team_stats,
            x='Team',
            y=f'{metric}_avg',
            title=title,
            color=f'{metric}_avg',
            color_continuous_scale='viridis',
            hover_data=['player_count']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Team",
            yaxis_title=f"Average {metric}",
            **self.default_layout
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_position_comparison_chart(self, df, metric, title=None):
        """Create position comparison chart showing average metric per position"""
        if title is None:
            title = f"Position Average: {metric}"
        
        # Calculate position averages
        position_stats = df.groupby('Position')[metric].agg(['mean', 'count']).reset_index()
        position_stats.columns = ['Position', f'{metric}_avg', 'player_count']
        position_stats = position_stats.sort_values(f'{metric}_avg', ascending=False)
        
        fig = px.bar(
            position_stats,
            x='Position',
            y=f'{metric}_avg',
            title=title,
            color=f'{metric}_avg',
            color_continuous_scale='plasma',
            hover_data=['player_count']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Position",
            yaxis_title=f"Average {metric}",
            **self.default_layout
        )
        
        return fig
    
    def create_age_distribution_chart(self, df, metric=None, title=None):
        """Create age distribution chart, optionally colored by a metric"""
        if title is None:
            title = "Player Age Distribution"
        
        if metric and metric in df.columns:
            fig = px.histogram(
                df,
                x='Age',
                color=metric,
                title=f"{title} (colored by {metric})",
                nbins=15,
                color_continuous_scale='viridis'
            )
        else:
            fig = px.histogram(
                df,
                x='Age',
                title=title,
                nbins=15,
                color_discrete_sequence=[self.category_colors['Attack']]
            )
        
        # Add vertical line for mean age
        mean_age = df['Age'].mean()
        fig.add_vline(
            x=mean_age, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean Age: {mean_age:.1f}"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Age",
            yaxis_title="Number of Players",
            **self.default_layout
        )
        
        return fig
    
    def create_scatter_plot(self, df, x_metric, y_metric, color_by='Team', size_by=None, title=None):
        """Create scatter plot for correlation analysis between two metrics"""
        if title is None:
            title = f"{x_metric} vs {y_metric}"
        
        # Prepare size data
        size_data = df[size_by] if size_by and size_by in df.columns else None
        
        # Create hover text
        hover_text = []
        for _, player in df.iterrows():
            hover_text.append(
                f"<b>{player['Player Name']}</b><br>"
                f"Team: {player['Team']}<br>"
                f"Position: {player['Position']}<br>"
                f"{x_metric}: {player[x_metric]}<br>"
                f"{y_metric}: {player[y_metric]}"
            )
        
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            color=color_by,
            size=size_data,
            hover_name='Player Name',
            title=title,
            color_discrete_sequence=self.player_colors
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate=hover_text,
            hoverinfo='text'
        )
        
        # Calculate and display correlation
        correlation = df[x_metric].corr(df[y_metric])
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Correlation: {correlation:.3f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        )
        
        fig.update_layout(
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_radar_chart(self, df, player_names, metrics, title="Player Comparison Radar Chart"):
        """Create radar chart for player comparison"""
        # Filter to selected players
        player_data = df[df['Player Name'].isin(player_names)]
        
        if len(player_data) == 0:
            raise ValueError("No players found with the given names")
        
        fig = go.Figure()
        
        # Normalize metrics to 0-100 scale for better visualization
        normalized_data = player_data.copy()
        for metric in metrics:
            if metric in df.columns:
                col_min = df[metric].min()
                col_max = df[metric].max()
                if col_max > col_min:
                    normalized_data[metric] = ((player_data[metric] - col_min) / (col_max - col_min)) * 100
                else:
                    normalized_data[metric] = 50
        
        for i, (_, player_row) in enumerate(normalized_data.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=player_row[metrics].values,
                theta=metrics,
                fill='toself',
                name=player_row['Player Name'],
                line_color=self.player_colors[i % len(self.player_colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title=title,
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_performance_heatmap(self, df, metrics=None, title="Player Performance Heatmap"):
        """Create performance heatmap for players"""
        if metrics is None:
            # Get all numeric columns except info columns
            info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
            metrics = [col for col in df.columns if col not in info_columns]
        
        # Limit to top 20 players by total performance to avoid overcrowding
        df_subset = df.nlargest(20, 'Appearances')  # Use appearances as a proxy for active players
        
        # Normalize data for better visualization
        normalized_data = df_subset[metrics].copy()
        for col in metrics:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                normalized_data[col] = ((df_subset[col] - col_min) / (col_max - col_min)) * 100
            else:
                normalized_data[col] = 50
        
        fig = px.imshow(
            normalized_data.T,
            labels=dict(x="Player Index", y="Metrics", color="Normalized Score (0-100)"),
            x=df_subset['Player Name'],
            y=metrics,
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title=title
        )
        
        fig.update_layout(
            height=max(400, len(metrics) * 20),
            xaxis_title="Players",
            yaxis_title="Metrics",
            **self.default_layout
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_distribution_histogram(self, df, metric, bins=20, title=None):
        """Create histogram showing distribution of a metric"""
        if title is None:
            title = f"Distribution of {metric}"
        
        fig = px.histogram(
            df,
            x=metric,
            nbins=bins,
            title=title,
            color_discrete_sequence=[self.category_colors['Attack']]
        )
        
        # Add vertical lines for statistics
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.1f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="blue",
                     annotation_text=f"Median: {median_val:.1f}")
        
        fig.update_layout(
            height=400,
            xaxis_title=metric,
            yaxis_title="Number of Players",
            **self.default_layout
        )
        
        return fig
    
    def create_box_plot(self, df, metric, group_by='Position', title=None):
        """Create box plot showing metric distribution by group"""
        if title is None:
            title = f"{metric} Distribution by {group_by}"
        
        fig = px.box(
            df,
            x=group_by,
            y=metric,
            title=title,
            color=group_by,
            color_discrete_sequence=self.player_colors
        )
        
        fig.update_layout(
            height=400,
            **self.default_layout
        )
        
        return fig
    
    def create_top_performers_table(self, df, metric, n=10, is_negative_metric=False):
        """Create formatted table of top performers"""
        # Get top/bottom performers
        if is_negative_metric:
            top_players = df.nsmallest(n, metric)
            rank_label = "Bottom"
        else:
            top_players = df.nlargest(n, metric)
            rank_label = "Top"
        
        # Create display dataframe
        display_df = top_players[['Player Name', 'Team', 'Position', 'Age', 'Appearances', metric]].copy()
        display_df.index = range(1, len(display_df) + 1)
        
        return display_df, f"{rank_label} {n} Players: {metric}"
    
    def create_gauge_chart(self, value, title, max_value=None, min_value=0, color_scheme='green'):
        """Create gauge chart for single metric display"""
        if max_value is None:
            max_value = value * 2  # Default to 2x the value
        
        color_map = {
            'green': ['#ff4444', '#ffaa00', '#00aa00'],
            'blue': ['#ff4444', '#ffaa00', '#0088ff'],
            'red': ['#00aa00', '#ffaa00', '#ff4444']
        }
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': color_map[color_scheme][1]},
                'steps': [
                    {'range': [min_value, max_value*0.33], 'color': color_map[color_scheme][0]},
                    {'range': [max_value*0.33, max_value*0.67], 'color': color_map[color_scheme][1]},
                    {'range': [max_value*0.67, max_value], 'color': color_map[color_scheme][2]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300, **self.default_layout)
        
        return fig

# Utility functions
def get_player_color_palette(n_colors):
    """Get a color palette with n colors for players"""
    visualizer = PlayerVisualizer()
    return (visualizer.player_colors * (n_colors // len(visualizer.player_colors) + 1))[:n_colors]

def create_player_summary_table(df, metrics):
    """Create a summary table for player metrics"""
    summary = df[metrics].describe().round(2)
    return summary

def format_player_name_for_display(player_name, max_length=20):
    """Format player name for display (truncate if too long)"""
    if len(player_name) <= max_length:
        return player_name
    else:
        return player_name[:max_length-3] + "..."

def create_category_scores_visualization(df, player_metric_categories, negative_metrics):
    """
    Create category scores table and charts for Attack, Defense, Progression, Discipline
    
    Args:
        df: DataFrame with player data
        player_metric_categories: Dictionary with category -> metrics mapping
        negative_metrics: List of metrics where lower is better
    
    Returns:
        tuple: (category_scores_df, top_players_by_category, charts)
    """
    
    # Calculate category scores for each player
    category_scores = {}
    
    # Get relevant categories (exclude Goalkeeper for overall analysis)
    categories_to_analyze = ['Attack', 'Defense', 'Progression', 'Discipline']
    
    for category in categories_to_analyze:
        if category in player_metric_categories:
            metrics = player_metric_categories[category]
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
                # Calculate normalized scores for this category
                category_df = df[available_metrics].copy()
                normalized_category = pd.DataFrame(index=df.index)
                
                for metric in available_metrics:
                    col_min = df[metric].min()
                    col_max = df[metric].max()
                    
                    if col_max == col_min:
                        normalized_category[metric] = 0.5
                    else:
                        if metric in negative_metrics:
                            # For negative metrics, invert normalization (lower = better)
                            normalized_category[metric] = 1 - ((df[metric] - col_min) / (col_max - col_min))
                        else:
                            # For positive metrics, normal normalization (higher = better)
                            normalized_category[metric] = (df[metric] - col_min) / (col_max - col_min)
                
                # Calculate average score for this category
                category_scores[f'{category}_Score'] = normalized_category[available_metrics].mean(axis=1)
            else:
                # If no metrics available, set to 0
                category_scores[f'{category}_Score'] = pd.Series([0] * len(df), index=df.index)
    
    # Create comprehensive category scores dataframe
    category_scores_df = df[['Player Name', 'Team', 'Position', 'Age', 'Appearances']].copy()
    for category_score, scores in category_scores.items():
        category_scores_df[category_score] = scores.round(3)
    
    # Get top players by category (worst for discipline)
    top_players_by_category = {}
    for category in categories_to_analyze:
        score_col = f'{category}_Score'
        if score_col in category_scores_df.columns:
            if category == 'Discipline':
                # For discipline, we want the "worst" performers (lowest scores = worst discipline)
                top_players = category_scores_df.nsmallest(10, score_col)
            else:
                # For other categories, we want the best performers (highest scores)
                top_players = category_scores_df.nlargest(10, score_col)
            top_players_by_category[category] = top_players
    
    # Create visualizations
    charts = create_category_charts(category_scores_df, categories_to_analyze)
    
    return category_scores_df, top_players_by_category, charts

def create_category_charts(category_scores_df, categories):
    """Create charts for category performance visualization"""
    
    charts = {}
    
    # Category colors
    category_colors = {
        'Attack': '#FF6B35',
        'Defense': '#4ECDC4',
        'Progression': '#F7931E', 
        'Discipline': '#E74C3C'
    }
    
    # 1. Top performers bar chart for each category
    for category in categories:
        score_col = f'{category}_Score'
        if score_col in category_scores_df.columns:
            if category == 'Discipline':
                # For discipline, show worst performers (lowest scores)
                top_10 = category_scores_df.nsmallest(10, score_col)
                title = f"Worst 10 Players - {category}"
            else:
                # For other categories, show best performers (highest scores)
                top_10 = category_scores_df.nlargest(10, score_col)
                title = f"Top 10 Players - {category}"
            
            fig = px.bar(
                top_10,
                x='Player Name',
                y=score_col,
                title=title,
                color_discrete_sequence=[category_colors.get(category, '#FF6B35')],
                hover_data=['Team', 'Position', 'Age']
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Player",
                yaxis_title=f"{category} Score",
                font=dict(family="Arial, sans-serif", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_xaxes(tickangle=45)
            
            charts[f'{category}_bar'] = fig
    
    # 2. Category comparison radar chart for top overall performers
    if len(category_scores_df) > 0:
        # Calculate overall category performance for each player
        score_columns = [f'{cat}_Score' for cat in categories]
        category_scores_df['Category_Average'] = category_scores_df[score_columns].mean(axis=1)
        
        # Get top 5 players overall for radar chart
        top_overall = category_scores_df.nlargest(5, 'Category_Average')
        
        fig_radar = go.Figure()
        
        colors = ['#FF6B35', '#4ECDC4', '#F7931E', '#E74C3C', '#45B7D1']
        
        for i, (_, player) in enumerate(top_overall.iterrows()):
            values = []
            for category in categories:
                score_col = f'{category}_Score'
                values.append(player[score_col] * 100)  # Convert to 0-100 scale
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=player['Player Name'],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Top 5 Players - Category Comparison",
            height=500,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        charts['category_radar'] = fig_radar
    
    # 3. Category distribution comparison
    fig_dist = go.Figure()
    
    for i, category in enumerate(categories):
        score_col = f'{category}_Score'
        if score_col in category_scores_df.columns:
            fig_dist.add_trace(go.Box(
                y=category_scores_df[score_col],
                name=category,
                marker_color=category_colors.get(category, '#FF6B35'),
                boxpoints='outliers'
            ))
    
    fig_dist.update_layout(
        title="Category Score Distributions",
        yaxis_title="Category Score (0-1 scale)",
        height=400,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    charts['category_distribution'] = fig_dist
    
    return charts

def create_category_summary_table(top_players_by_category):
    """Create a summary table showing top performer in each category"""
    
    summary_data = []
    
    for category, top_players in top_players_by_category.items():
        if len(top_players) > 0:
            best_player = top_players.iloc[0]
            score_col = f'{category}_Score'
            
            if category == 'Discipline':
                label = 'Worst Player'
            else:
                label = 'Top Player'
                
            summary_data.append({
                'Category': category,
                label: best_player['Player Name'],
                'Team': best_player['Team'],
                'Score': f"{best_player[score_col]:.3f}",
                'Position': best_player['Position']
            })
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()