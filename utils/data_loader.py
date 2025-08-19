import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

class PlayerDataLoader:
    """
    Data loader class for player statistics
    Handles data loading, cleaning, and preprocessing for individual player data
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data" / "players_statistics.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.metric_categories = {
            'Attack': ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
            'Defense': ['Block', 'Block Cross', 'Clearance', 'Tackle', 'Intercept', 'Ball Recovery', 'Header Won'],
            'Progression': ['Passing', 'Cross', 'Dribble Success', 'Free Kick'],
            'Discipline': ['Foul', 'Fouled', 'Yellow Card', 'Own Goal'],
            'Goalkeeper': ['Saves']
        }
        
        # Define info columns (non-metric columns)
        self.info_columns = ['Name', 'Player Name', 'Team', 'Country', 'Age', 'Position', 'Picture Url']
    
    @st.cache_data
    def load_data(_self):
        """Load and cache the player statistics data"""
        try:
            df = pd.read_csv(_self.data_path)
            return _self._preprocess_data(df)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {_self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _preprocess_data(self, df):
        """Preprocess the loaded data"""
        # Clean string columns
        for col in ['Name', 'Player Name', 'Team', 'Country', 'Position']:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Ensure Age and Appearances are integers
        if 'Age' in df.columns:
            df['Age'] = df['Age'].astype(int)
        if 'Appearances' in df.columns:
            df['Appearances'] = df['Appearances'].astype(int)
        
        # Remove any duplicate players (by Name and Team combination)
        df = df.drop_duplicates(subset=['Name', 'Team'], keep='first')
        
        # Sort players by Team then by Player Name
        df = df.sort_values(['Team', 'Player Name']).reset_index(drop=True)
        
        return df
    
    def get_metric_categories(self):
        """Return the metric categories dictionary"""
        return self.metric_categories
    
    def get_all_metrics(self):
        """Get all available metrics (excluding info columns)"""
        if self.df is None:
            self.df = self.load_data()
        return [col for col in self.df.columns if col not in self.info_columns]
    
    def get_info_columns(self):
        """Get list of info columns"""
        return self.info_columns
    
    def get_teams(self):
        """Get list of all team names"""
        if self.df is None:
            self.df = self.load_data()
        return sorted(self.df['Team'].unique())
    
    def get_positions(self):
        """Get list of all positions"""
        if self.df is None:
            self.df = self.load_data()
        return sorted(self.df['Position'].unique())
    
    def get_players_by_team(self, team_name):
        """Get all players for a specific team"""
        if self.df is None:
            self.df = self.load_data()
        
        return self.df[self.df['Team'] == team_name]
    
    def get_players_by_position(self, position):
        """Get all players for a specific position"""
        if self.df is None:
            self.df = self.load_data()
        
        return self.df[self.df['Position'] == position]
    
    def filter_players(self, teams=None, positions=None, age_range=None, appearances_range=None):
        """Filter players based on various criteria"""
        if self.df is None:
            self.df = self.load_data()
        
        filtered_df = self.df.copy()
        
        if teams:
            filtered_df = filtered_df[filtered_df['Team'].isin(teams)]
        
        if positions:
            filtered_df = filtered_df[filtered_df['Position'].isin(positions)]
        
        if age_range:
            filtered_df = filtered_df[
                (filtered_df['Age'] >= age_range[0]) & 
                (filtered_df['Age'] <= age_range[1])
            ]
        
        if appearances_range:
            filtered_df = filtered_df[
                (filtered_df['Appearances'] >= appearances_range[0]) & 
                (filtered_df['Appearances'] <= appearances_range[1])
            ]
        
        return filtered_df
    
    def get_category_metrics(self, category):
        """Get metrics for a specific category"""
        return self.metric_categories.get(category, [])
    
    def get_top_performers(self, metric, n=10, ascending=False):
        """Get top N performers for a specific metric"""
        if self.df is None:
            self.df = self.load_data()
        
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        if ascending:
            return self.df.nsmallest(n, metric)
        else:
            return self.df.nlargest(n, metric)
    
    def get_player_rankings(self, metric, ascending=False):
        """Get player rankings for a specific metric"""
        if self.df is None:
            self.df = self.load_data()
        
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        rankings = self.df[self.info_columns + [metric]].sort_values(metric, ascending=ascending).reset_index(drop=True)
        rankings.index = rankings.index + 1  # Start ranking from 1
        return rankings
    
    def get_correlation_matrix(self, metrics=None):
        """Get correlation matrix for specified metrics or all numeric metrics"""
        if self.df is None:
            self.df = self.load_data()
        
        if metrics is None:
            metrics = self.get_all_metrics()
        
        # Filter to only include metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        return self.df[available_metrics].corr()
    
    def get_summary_stats(self, metrics=None):
        """Get summary statistics for specified metrics"""
        if self.df is None:
            self.df = self.load_data()
        
        if metrics is None:
            metrics = self.get_all_metrics()
        
        # Filter to only include metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        return self.df[available_metrics].describe()
    
    def search_players(self, query):
        """Search for players by name"""
        if self.df is None:
            self.df = self.load_data()
        
        query = query.lower()
        matching_players = self.df[
            self.df['Player Name'].str.lower().str.contains(query, na=False) |
            self.df['Name'].str.lower().str.contains(query, na=False)
        ]
        return matching_players
    
    def get_team_stats_summary(self):
        """Get summary statistics by team"""
        if self.df is None:
            self.df = self.load_data()
        
        metrics = self.get_all_metrics()
        team_stats = self.df.groupby('Team')[metrics].agg(['mean', 'sum', 'count']).round(2)
        return team_stats
    
    def get_position_stats_summary(self):
        """Get summary statistics by position"""
        if self.df is None:
            self.df = self.load_data()
        
        metrics = self.get_all_metrics()
        position_stats = self.df.groupby('Position')[metrics].agg(['mean', 'sum', 'count']).round(2)
        return position_stats
    
    def calculate_normalized_scores(self, negative_metrics=None):
        """Calculate normalized scores for all metrics (0-1 scale)"""
        if self.df is None:
            self.df = self.load_data()
        
        if negative_metrics is None:
            negative_metrics = ['Own Goal', 'Yellow Card', 'Foul', 'Shoot Off Target']
        
        metrics = self.get_all_metrics()
        normalized_df = self.df.copy()
        
        for metric in metrics:
            if metric in self.df.columns:
                col_min = self.df[metric].min()
                col_max = self.df[metric].max()
                
                if col_max == col_min:
                    normalized_df[f'{metric}_normalized'] = 0.5
                else:
                    if metric in negative_metrics:
                        # For negative metrics, invert normalization (lower = better)
                        normalized_df[f'{metric}_normalized'] = 1 - ((self.df[metric] - col_min) / (col_max - col_min))
                    else:
                        # For positive metrics, normal normalization (higher = better)
                        normalized_df[f'{metric}_normalized'] = (self.df[metric] - col_min) / (col_max - col_min)
        
        return normalized_df
    
    def validate_data(self):
        """Validate the loaded data for completeness and consistency"""
        if self.df is None:
            self.df = self.load_data()
        
        validation_results = {
            'total_players': len(self.df),
            'total_teams': self.df['Team'].nunique(),
            'total_positions': self.df['Position'].nunique(),
            'total_metrics': len(self.get_all_metrics()),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_players': self.df.duplicated(subset=['Name', 'Team']).sum(),
            'age_range': (self.df['Age'].min(), self.df['Age'].max()),
            'appearances_range': (self.df['Appearances'].min(), self.df['Appearances'].max()),
            'categories_coverage': {}
        }
        
        # Check category coverage
        for category, metrics in self.metric_categories.items():
            available = sum(1 for m in metrics if m in self.df.columns)
            validation_results['categories_coverage'][category] = {
                'available_metrics': available,
                'total_metrics': len(metrics),
                'coverage_percentage': (available / len(metrics)) * 100 if len(metrics) > 0 else 0
            }
        
        return validation_results

# Utility functions for easy access
@st.cache_data
def load_player_data():
    """Simple function to load player data with caching"""
    loader = PlayerDataLoader()
    return loader.load_data()

def get_player_metric_categories():
    """Get metric categories for players"""
    loader = PlayerDataLoader()
    return loader.get_metric_categories()

def filter_player_data(df, teams=None, positions=None, age_range=None, appearances_range=None):
    """Filter player data based on criteria"""
    filtered_df = df.copy()
    
    if teams:
        filtered_df = filtered_df[filtered_df['Team'].isin(teams)]
    
    if positions:
        filtered_df = filtered_df[filtered_df['Position'].isin(positions)]
    
    if age_range:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= age_range[0]) & 
            (filtered_df['Age'] <= age_range[1])
        ]
    
    if appearances_range:
        filtered_df = filtered_df[
            (filtered_df['Appearances'] >= appearances_range[0]) & 
            (filtered_df['Appearances'] <= appearances_range[1])
        ]
    
    return filtered_df