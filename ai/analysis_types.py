"""
Analysis types and categorization for football performance analysis
"""

from typing import Dict, List, Any
from enum import Enum

class AnalysisType(Enum):
    """Types of football analysis available"""
    PLAYER_PERFORMANCE = "player_performance"
    TEAM_ANALYSIS = "team_analysis"
    SCOUT_REPORT = "scout_report"
    PLAYER_COMPARISON = "player_comparison"
    TACTICAL_ANALYSIS = "tactical_analysis"
    CUSTOM_QUERY = "custom_query"

class AnalysisTypes:
    """
    Analysis type configurations and metadata
    """
    
    def __init__(self):
        self.analysis_configs = {
            AnalysisType.PLAYER_PERFORMANCE: {
                'name': 'Player Performance Analysis',
                'description': 'Comprehensive analysis of individual player performance, strengths, weaknesses, and recommendations',
                'required_data': ['player_stats'],
                'optional_data': ['team_context', 'position_averages'],
                'key_metrics': ['Goal', 'Assist', 'Passing', 'Tackle', 'Block', 'Dribble Success'],
                'output_sections': [
                    'Strengths & Strong Areas',
                    'Areas for Improvement', 
                    'Playing Style Assessment',
                    'Market Value & Potential',
                    'Recommendations'
                ]
            },
            
            AnalysisType.TEAM_ANALYSIS: {
                'name': 'Team Performance Analysis',
                'description': 'Analysis of team tactics, player distribution, strengths, and strategic recommendations',
                'required_data': ['team_roster', 'team_stats'],
                'optional_data': ['league_context', 'opponent_data'],
                'key_metrics': ['team_goals', 'team_assists', 'team_passing', 'team_tackles', 'position_balance'],
                'output_sections': [
                    'Team Strengths',
                    'Tactical Style & Formation',
                    'Areas of Concern',
                    'Key Player Dependencies',
                    'Transfer & Development Recommendations',
                    'Competitive Analysis'
                ]
            },
            
            AnalysisType.SCOUT_REPORT: {
                'name': 'Scouting Report',
                'description': 'Professional scouting report identifying talent based on specific criteria',
                'required_data': ['player_pool', 'scouting_criteria'],
                'optional_data': ['market_values', 'injury_history'],
                'key_metrics': ['performance_consistency', 'potential_rating', 'value_score'],
                'output_sections': [
                    'Primary Recommendations',
                    'Player Profiles',
                    'Risk Assessment',
                    'Hidden Gems',
                    'Tactical Fit Analysis',
                    'Market Intelligence',
                    'Watching Recommendations'
                ]
            },
            
            AnalysisType.PLAYER_COMPARISON: {
                'name': 'Player Comparison',
                'description': 'Head-to-head comparison of two players across multiple dimensions',
                'required_data': ['player1_stats', 'player2_stats'],
                'optional_data': ['positional_context', 'team_systems'],
                'key_metrics': ['comparative_metrics', 'performance_gaps', 'value_comparison'],
                'output_sections': [
                    'Overall Performance Comparison',
                    'Strengths vs Strengths',
                    'Weakness Analysis',
                    'Playing Style Differences',
                    'Value Proposition',
                    'Team Fit Analysis',
                    'Future Projection',
                    'Recommendation'
                ]
            },
            
            AnalysisType.TACTICAL_ANALYSIS: {
                'name': 'Tactical Pattern Analysis',
                'description': 'Deep dive into team tactics, playing patterns, and strategic approach',
                'required_data': ['team_stats', 'positional_data'],
                'optional_data': ['match_data', 'opponent_analysis'],
                'key_metrics': ['possession_style', 'attacking_patterns', 'defensive_organization'],
                'output_sections': [
                    'Formation and System Analysis',
                    'Playing Style Identification',
                    'Phase Analysis',
                    'Key Tactical Roles',
                    'Strengths and Vulnerabilities',
                    'Game Management',
                    'Strategic Recommendations',
                    'Opponent Preparation'
                ]
            },
            
            AnalysisType.CUSTOM_QUERY: {
                'name': 'Custom Analysis',
                'description': 'Flexible analysis based on specific questions or requirements',
                'required_data': ['relevant_data', 'specific_question'],
                'optional_data': ['context', 'constraints'],
                'key_metrics': ['question_dependent'],
                'output_sections': [
                    'Direct Answer',
                    'Supporting Evidence',
                    'Detailed Analysis',
                    'Recommendations',
                    'Additional Insights'
                ]
            }
        }
        
        # Metric categories for different analysis types
        self.metric_categories = {
            'attacking': ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
            'defending': ['Block', 'Block Cross', 'Clearance', 'Tackle', 'Intercept', 'Ball Recovery', 'Header Won'],
            'passing': ['Passing', 'Cross', 'Free Kick'],
            'dribbling': ['Dribble Success'],
            'discipline': ['Foul', 'Fouled', 'Yellow Card', 'Own Goal'],
            'goalkeeping': ['Saves']
        }
        
        # Position-specific key metrics
        self.position_key_metrics = {
            'PENJAGA GAWANG': ['Saves', 'Clearance', 'Passing'],  # Goalkeeper
            'BELAKANG': ['Block', 'Clearance', 'Tackle', 'Passing', 'Header Won'],  # Defender
            'TENGAH': ['Passing', 'Assist', 'Tackle', 'Create Chance', 'Dribble Success'],  # Midfielder
            'DEPAN': ['Goal', 'Assist', 'Shoot On Target', 'Create Chance', 'Dribble Success']  # Forward
        }
        
        # Analysis complexity levels
        self.complexity_levels = {
            'basic': {
                'max_tokens': 300,
                'focus': 'key_insights',
                'sections': 3
            },
            'standard': {
                'max_tokens': 512,
                'focus': 'comprehensive',
                'sections': 5
            },
            'detailed': {
                'max_tokens': 800,
                'focus': 'in_depth',
                'sections': 8
            },
            'professional': {
                'max_tokens': 1200,
                'focus': 'expert_analysis',
                'sections': 10
            }
        }
    
    def get_analysis_config(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get configuration for a specific analysis type"""
        return self.analysis_configs.get(analysis_type, {})
    
    def get_available_analysis_types(self) -> List[Dict[str, str]]:
        """Get list of available analysis types with descriptions"""
        return [
            {
                'type': analysis_type.value,
                'name': config['name'],
                'description': config['description']
            }
            for analysis_type, config in self.analysis_configs.items()
        ]
    
    def get_position_metrics(self, position: str) -> List[str]:
        """Get key metrics for a specific position"""
        return self.position_key_metrics.get(position.upper(), [])
    
    def get_metric_category(self, metric: str) -> str:
        """Get the category for a specific metric"""
        for category, metrics in self.metric_categories.items():
            if metric in metrics:
                return category
        return 'general'
    
    def get_metrics_by_category(self, category: str) -> List[str]:
        """Get all metrics in a specific category"""
        return self.metric_categories.get(category.lower(), [])
    
    def get_complexity_config(self, level: str) -> Dict[str, Any]:
        """Get configuration for analysis complexity level"""
        return self.complexity_levels.get(level.lower(), self.complexity_levels['standard'])
    
    def validate_analysis_request(self, analysis_type: AnalysisType, provided_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if provided data meets requirements for analysis type"""
        config = self.get_analysis_config(analysis_type)
        validation_result = {
            'valid': True,
            'missing_required': [],
            'missing_optional': [],
            'suggestions': []
        }
        
        # Check required data
        required_data = config.get('required_data', [])
        for req_data in required_data:
            if req_data not in provided_data or provided_data[req_data] is None:
                validation_result['missing_required'].append(req_data)
                validation_result['valid'] = False
        
        # Check optional data
        optional_data = config.get('optional_data', [])
        for opt_data in optional_data:
            if opt_data not in provided_data or provided_data[opt_data] is None:
                validation_result['missing_optional'].append(opt_data)
        
        # Add suggestions based on missing data
        if validation_result['missing_required']:
            validation_result['suggestions'].append(
                f"Required data missing: {', '.join(validation_result['missing_required'])}"
            )
        
        if validation_result['missing_optional']:
            validation_result['suggestions'].append(
                f"Optional data that could improve analysis: {', '.join(validation_result['missing_optional'])}"
            )
        
        return validation_result
    
    def get_recommended_metrics(self, analysis_type: AnalysisType, position: str = None) -> List[str]:
        """Get recommended metrics for a specific analysis type and position"""
        config = self.get_analysis_config(analysis_type)
        base_metrics = config.get('key_metrics', [])
        
        if position and position.upper() in self.position_key_metrics:
            position_metrics = self.get_position_metrics(position)
            # Combine and deduplicate
            combined_metrics = list(set(base_metrics + position_metrics))
            return combined_metrics
        
        return base_metrics