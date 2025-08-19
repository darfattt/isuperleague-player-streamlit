"""
AI utilities and helper functions for football analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st
import logging
from datetime import datetime
import json

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging for AI operations"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class FootballDataProcessor:
    """
    Utility class for processing football data for AI analysis
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
        # Indonesian position translations
        self.position_translations = {
            'BELAKANG': 'Defender',
            'TENGAH': 'Midfielder',
            'DEPAN': 'Forward',
            'PENJAGA GAWANG': 'Goalkeeper'
        }
        
        # Metric importance weights by position
        self.position_weights = {
            'BELAKANG': {  # Defender
                'Block': 1.5, 'Clearance': 1.5, 'Tackle': 1.5, 'Header Won': 1.3,
                'Passing': 1.2, 'Ball Recovery': 1.3, 'Intercept': 1.4,
                'Goal': 0.5, 'Assist': 0.7, 'Dribble Success': 0.6
            },
            'TENGAH': {  # Midfielder
                'Passing': 1.5, 'Assist': 1.4, 'Create Chance': 1.4, 'Tackle': 1.3,
                'Ball Recovery': 1.2, 'Dribble Success': 1.3, 'Intercept': 1.2,
                'Goal': 1.1, 'Block': 0.8, 'Clearance': 0.8
            },
            'DEPAN': {  # Forward
                'Goal': 1.5, 'Assist': 1.4, 'Shoot On Target': 1.4, 'Create Chance': 1.3,
                'Dribble Success': 1.3, 'Shoot Off Target': 0.6, 'Fouled': 1.2,
                'Tackle': 0.5, 'Block': 0.3, 'Clearance': 0.3
            },
            'PENJAGA GAWANG': {  # Goalkeeper
                'Saves': 1.5, 'Clearance': 1.3, 'Passing': 1.2,
                'Goal': 0.1, 'Dribble Success': 0.2, 'Tackle': 0.3
            }
        }
        
        # Negative metrics (lower is better)
        self.negative_metrics = {
            'Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target'
        }
    
    def normalize_player_data(self, player_data: pd.Series, position: str = None) -> Dict[str, float]:
        """
        Normalize player data with position-specific weights
        
        Args:
            player_data: Player statistics as pandas Series
            position: Player position (optional, will use from data if available)
            
        Returns:
            Dictionary of normalized metrics
        """
        if position is None:
            position = player_data.get('Position', 'TENGAH')
        
        normalized = {}
        position_weights = self.position_weights.get(position, {})
        
        # Get numeric columns only
        for metric, value in player_data.items():
            if isinstance(value, (int, float)) and metric not in ['Age', 'Appearances']:
                # Apply position weight if available
                weight = position_weights.get(metric, 1.0)
                
                # Handle negative metrics
                if metric in self.negative_metrics:
                    # For negative metrics, lower values are better
                    normalized[metric] = max(0, 10 - value) * weight
                else:
                    # For positive metrics, higher values are better
                    normalized[metric] = value * weight
        
        return normalized
    
    def calculate_performance_score(self, player_data: pd.Series, position: str = None) -> float:
        """
        Calculate overall performance score for a player
        
        Args:
            player_data: Player statistics
            position: Player position
            
        Returns:
            Performance score (0-100 scale)
        """
        normalized_data = self.normalize_player_data(player_data, position)
        
        if not normalized_data:
            return 0.0
        
        # Calculate weighted average
        total_score = sum(normalized_data.values())
        max_possible = len(normalized_data) * 15  # Assuming max weighted value is 15
        
        # Scale to 0-100
        score = min(100, (total_score / max_possible) * 100)
        return round(score, 1)
    
    def identify_player_strengths(self, player_data: pd.Series, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify player's strongest metrics
        
        Args:
            player_data: Player statistics
            top_n: Number of top strengths to return
            
        Returns:
            List of (metric, value) tuples
        """
        position = player_data.get('Position', 'TENGAH')
        normalized_data = self.normalize_player_data(player_data, position)
        
        # Sort by normalized values
        strengths = sorted(normalized_data.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out zero values and return top N
        non_zero_strengths = [(metric, value) for metric, value in strengths if value > 0]
        return non_zero_strengths[:top_n]
    
    def identify_player_weaknesses(self, player_data: pd.Series, league_averages: pd.Series, top_n: int = 3) -> List[Tuple[str, float, float]]:
        """
        Identify player's weaknesses compared to league average
        
        Args:
            player_data: Player statistics
            league_averages: League average statistics
            top_n: Number of weaknesses to return
            
        Returns:
            List of (metric, player_value, league_average) tuples
        """
        weaknesses = []
        position = player_data.get('Position', 'TENGAH')
        position_weights = self.position_weights.get(position, {})
        
        for metric in league_averages.index:
            if metric in player_data.index and isinstance(player_data[metric], (int, float)):
                player_val = player_data[metric]
                league_avg = league_averages[metric]
                
                # Consider metric importance for position
                weight = position_weights.get(metric, 1.0)
                
                if weight > 1.0:  # Important metric for this position
                    if metric in self.negative_metrics:
                        # For negative metrics, player value should be lower
                        if player_val > league_avg * 1.2:
                            weaknesses.append((metric, player_val, league_avg))
                    else:
                        # For positive metrics, player value should be higher
                        if player_val < league_avg * 0.8:
                            weaknesses.append((metric, player_val, league_avg))
        
        # Sort by gap from league average
        weaknesses.sort(key=lambda x: abs(x[1] - x[2]) / max(x[2], 1), reverse=True)
        return weaknesses[:top_n]
    
    def compare_players_detailed(self, player1: pd.Series, player2: pd.Series, metrics: List[str]) -> Dict[str, Any]:
        """
        Detailed comparison of two players
        
        Args:
            player1: First player's data
            player2: Second player's data
            metrics: Metrics to compare
            
        Returns:
            Detailed comparison dictionary
        """
        comparison = {
            'basic_info': {
                'player1': {
                    'name': player1.get('Player Name', 'Unknown'),
                    'team': player1.get('Team', 'Unknown'),
                    'position': player1.get('Position', 'Unknown'),
                    'age': player1.get('Age', 0)
                },
                'player2': {
                    'name': player2.get('Player Name', 'Unknown'),
                    'team': player2.get('Team', 'Unknown'),
                    'position': player2.get('Position', 'Unknown'),
                    'age': player2.get('Age', 0)
                }
            },
            'metric_comparison': {},
            'performance_scores': {},
            'advantages': {'player1': [], 'player2': []},
            'summary': {}
        }
        
        # Performance scores
        comparison['performance_scores']['player1'] = self.calculate_performance_score(player1)
        comparison['performance_scores']['player2'] = self.calculate_performance_score(player2)
        
        # Metric by metric comparison
        for metric in metrics:
            if metric in player1.index and metric in player2.index:
                val1 = player1[metric]
                val2 = player2[metric]
                
                comparison['metric_comparison'][metric] = {
                    'player1': val1,
                    'player2': val2,
                    'difference': val1 - val2,
                    'percentage_diff': ((val1 - val2) / max(val2, 1)) * 100 if val2 != 0 else 0
                }
                
                # Determine advantage
                if metric in self.negative_metrics:
                    if val1 < val2:
                        comparison['advantages']['player1'].append(metric)
                    elif val2 < val1:
                        comparison['advantages']['player2'].append(metric)
                else:
                    if val1 > val2:
                        comparison['advantages']['player1'].append(metric)
                    elif val2 > val1:
                        comparison['advantages']['player2'].append(metric)
        
        # Summary
        p1_advantages = len(comparison['advantages']['player1'])
        p2_advantages = len(comparison['advantages']['player2'])
        total_metrics = len(metrics)
        
        comparison['summary'] = {
            'player1_advantage_count': p1_advantages,
            'player2_advantage_count': p2_advantages,
            'tied_metrics': total_metrics - p1_advantages - p2_advantages,
            'overall_leader': 'player1' if p1_advantages > p2_advantages else 'player2' if p2_advantages > p1_advantages else 'tied'
        }
        
        return comparison
    
    def analyze_team_balance(self, team_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze team balance and composition
        
        Args:
            team_data: Team players' data
            
        Returns:
            Team balance analysis
        """
        analysis = {
            'position_distribution': {},
            'age_analysis': {},
            'performance_distribution': {},
            'key_players': {},
            'balance_score': 0
        }
        
        # Position distribution
        position_counts = team_data['Position'].value_counts()
        analysis['position_distribution'] = position_counts.to_dict()
        
        # Age analysis
        analysis['age_analysis'] = {
            'average_age': team_data['Age'].mean(),
            'age_range': (team_data['Age'].min(), team_data['Age'].max()),
            'young_players': len(team_data[team_data['Age'] < 23]),
            'experienced_players': len(team_data[team_data['Age'] > 30])
        }
        
        # Performance distribution
        performance_scores = [self.calculate_performance_score(row) for _, row in team_data.iterrows()]
        analysis['performance_distribution'] = {
            'average_performance': np.mean(performance_scores),
            'performance_std': np.std(performance_scores),
            'top_performers': len([s for s in performance_scores if s > 70]),
            'underperformers': len([s for s in performance_scores if s < 30])
        }
        
        # Key players by position
        for position in team_data['Position'].unique():
            pos_players = team_data[team_data['Position'] == position]
            if len(pos_players) > 0:
                pos_scores = [self.calculate_performance_score(row) for _, row in pos_players.iterrows()]
                best_idx = np.argmax(pos_scores)
                best_player = pos_players.iloc[best_idx]
                
                analysis['key_players'][position] = {
                    'name': best_player['Player Name'],
                    'performance_score': pos_scores[best_idx]
                }
        
        # Balance score (0-100)
        balance_factors = []
        
        # Position balance (ideal: 4-4-2 formation)
        ideal_distribution = {'BELAKANG': 4, 'TENGAH': 4, 'DEPAN': 2, 'PENJAGA GAWANG': 1}
        position_balance = 0
        for pos, ideal_count in ideal_distribution.items():
            actual_count = position_counts.get(pos, 0)
            # Penalize deviations from ideal
            deviation = abs(actual_count - ideal_count)
            position_balance += max(0, 10 - deviation * 2)
        
        balance_factors.append(position_balance / 4)  # Average across positions
        
        # Age balance (prefer mix of young and experienced)
        age_balance = 50
        if analysis['age_analysis']['young_players'] == 0:
            age_balance -= 15  # No young players
        if analysis['age_analysis']['experienced_players'] == 0:
            age_balance -= 10  # No experienced players
        
        balance_factors.append(age_balance)
        
        # Performance balance (prefer consistent performance)
        perf_std = analysis['performance_distribution']['performance_std']
        perf_balance = max(0, 100 - perf_std)  # Lower std deviation = better balance
        balance_factors.append(perf_balance)
        
        analysis['balance_score'] = np.mean(balance_factors)
        
        return analysis
    
    def generate_improvement_suggestions(self, player_data: pd.Series, league_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions for a player
        
        Args:
            player_data: Player's statistics
            league_data: League-wide player data for comparison
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        position = player_data.get('Position', 'TENGAH')
        
        # Calculate league averages for the same position
        same_position_players = league_data[league_data['Position'] == position]
        if len(same_position_players) == 0:
            same_position_players = league_data  # Fallback to all players
        
        league_averages = same_position_players.select_dtypes(include=[np.number]).mean()
        
        # Find weaknesses
        weaknesses = self.identify_player_weaknesses(player_data, league_averages)
        
        for metric, player_val, league_avg in weaknesses:
            gap_percentage = ((league_avg - player_val) / max(league_avg, 1)) * 100
            
            suggestion = {
                'area': metric,
                'current_value': player_val,
                'league_average': league_avg,
                'gap_percentage': gap_percentage,
                'priority': 'High' if gap_percentage > 50 else 'Medium' if gap_percentage > 25 else 'Low',
                'suggestion': self._get_improvement_advice(metric, position)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_improvement_advice(self, metric: str, position: str) -> str:
        """Get specific improvement advice for a metric"""
        advice_map = {
            'Goal': "Focus on finishing drills, work on shot placement and power. Practice different shooting techniques.",
            'Assist': "Improve vision and passing accuracy. Work on creating chances and final ball delivery.",
            'Passing': "Practice passing accuracy under pressure. Work on short and long passing techniques.",
            'Tackle': "Improve defensive positioning and timing. Practice different tackling techniques safely.",
            'Block': "Work on defensive positioning and anticipation. Practice blocking techniques and body positioning.",
            'Dribble Success': "Practice ball control drills and 1v1 situations. Improve agility and close control.",
            'Cross': "Focus on crossing accuracy from different positions. Practice both low and high crosses.",
            'Shoot On Target': "Work on shot accuracy and composure in front of goal. Practice finishing from various angles.",
            'Create Chance': "Improve creativity and vision. Practice through balls and key passes in training.",
            'Clearance': "Work on defensive positioning and clearance techniques. Practice heading and kicking clearances."
        }
        
        return advice_map.get(metric, f"Focus on improving {metric} through targeted training and match practice.")

# Caching utilities for Streamlit
@st.cache_data
def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and process football data with caching"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic processing
        processor = FootballDataProcessor()
        
        # Calculate league statistics
        league_stats = {
            'total_players': len(df),
            'total_teams': df['Team'].nunique(),
            'avg_age': df['Age'].mean(),
            'position_distribution': df['Position'].value_counts().to_dict(),
            'top_scorers': df.nlargest(10, 'Goal')[['Player Name', 'Team', 'Goal']].to_dict('records'),
            'league_averages': df.select_dtypes(include=[np.number]).mean().to_dict()
        }
        
        return df, league_stats
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}

@st.cache_data
def calculate_player_rankings(df: pd.DataFrame, metric: str, position: str = None) -> pd.DataFrame:
    """Calculate and cache player rankings for a specific metric"""
    if position:
        filtered_df = df[df['Position'] == position]
    else:
        filtered_df = df
    
    if metric not in filtered_df.columns:
        return pd.DataFrame()
    
    rankings = filtered_df[['Player Name', 'Team', 'Position', metric]].sort_values(
        metric, ascending=False
    ).reset_index(drop=True)
    
    rankings.index = rankings.index + 1  # Start ranking from 1
    return rankings

def export_analysis_report(analysis_data: Dict[str, Any], filename: str = None) -> str:
    """
    Export analysis report to JSON format
    
    Args:
        analysis_data: Analysis results to export
        filename: Optional filename for export
        
    Returns:
        JSON string of the analysis
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"football_analysis_{timestamp}.json"
    
    # Prepare export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': analysis_data.get('type', 'Unknown'),
        'subject': analysis_data.get('subject', 'Unknown'),
        'results': analysis_data.get('results', ''),
        'metadata': {
            'generated_by': 'AI Football Analyst',
            'model': 'Mistral-7B-Instruct-v0.3',
            'league': 'Indonesia Super League'
        }
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

# Error handling utilities
def handle_ai_error(error: Exception, context: str = "") -> str:
    """
    Handle AI-related errors gracefully
    
    Args:
        error: The exception that occurred
        context: Additional context about the error
        
    Returns:
        User-friendly error message
    """
    error_messages = {
        'OutOfMemoryError': "Insufficient memory to run AI model. Try using a smaller model or reducing batch size.",
        'ModelNotFoundError': "AI model not found. Please check your model configuration.",
        'TokenizerError': "Error with text processing. Please try rephrasing your input.",
        'ConnectionError': "Network connection issue. Please check your internet connection.",
        'TimeoutError': "AI analysis is taking too long. Please try again with a simpler query."
    }
    
    error_type = type(error).__name__
    base_message = error_messages.get(error_type, f"An error occurred during AI analysis: {str(error)}")
    
    if context:
        return f"{base_message}\n\nContext: {context}"
    
    return base_message