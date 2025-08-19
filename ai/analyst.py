"""
Core AI Football Analyst using Mistral-7B-Instruct-v0.3
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

# Import dependencies with fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    torch = None

from .prompts import PromptTemplates
from .analysis_types import AnalysisTypes

class GemmaAnalyst:
    """
    AI Football Performance Analyst using Google Gemma-3-270M
    """
    
    def __init__(self, model_name: str = "google/gemma-3-270m", cache_dir: Optional[str] = None, token: Optional[str] = None, use_quantization: bool = True):
        """
        Initialize the Gemma Analyst
        
        Args:
            model_name: Hugging Face model identifier
            cache_dir: Optional cache directory for model files
            token: Hugging Face authentication token (optional, can also be set via environment)
            use_quantization: Whether to use quantized model for lower memory usage
        """
        # Check dependencies
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required but not installed. Please run: pip install pandas")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required but not installed. Please run: pip install transformers torch")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.prompt_templates = PromptTemplates()
        self.analysis_types = AnalysisTypes()
        self.is_initialized = False
        
        # Setup authentication token
        self.token = self._setup_authentication_token(token)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_authentication_token(self, provided_token: Optional[str]) -> Optional[str]:
        """
        Setup Hugging Face authentication token from various sources
        
        Args:
            provided_token: Token provided directly to the constructor
            
        Returns:
            Authentication token or None
        """
        import os
        
        # Priority order: provided_token > environment variable > HF config file
        if provided_token:
            self._validate_token_format(provided_token)
            return provided_token
        
        # Check environment variable
        env_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if env_token:
            self._validate_token_format(env_token)
            return env_token
        
        # Check HF config file
        try:
            from huggingface_hub import HfFolder
            config_token = HfFolder.get_token()
            if config_token:
                self._validate_token_format(config_token)
                return config_token
        except ImportError:
            pass
        except Exception:
            pass
        
        # No token found - this is okay for public models
        return None
    
    def _validate_token_format(self, token: str) -> bool:
        """
        Basic validation of Hugging Face token format
        
        Args:
            token: Token to validate
            
        Returns:
            True if token format appears valid
            
        Raises:
            ValueError: If token format is invalid
        """
        if not token or not isinstance(token, str):
            raise ValueError("Token must be a non-empty string")
        
        # Basic format validation - HF tokens typically start with 'hf_'
        if not (token.startswith('hf_') or len(token) >= 20):
            raise ValueError("Token format appears invalid. Hugging Face tokens typically start with 'hf_' and are at least 20 characters long")
        
        return True
    
    def test_token_authentication(self) -> Dict[str, Any]:
        """
        Test if the current token can authenticate with Hugging Face
        
        Returns:
            Dictionary with authentication test results
        """
        result = {
            'has_token': bool(self.token),
            'token_valid': False,
            'user_info': None,
            'error': None
        }
        
        if not self.token:
            result['error'] = "No authentication token available"
            return result
        
        try:
            from huggingface_hub import whoami
            
            # Test token by getting user info
            user_info = whoami(token=self.token)
            result['token_valid'] = True
            result['user_info'] = {
                'name': user_info.get('name', 'Unknown'),
                'type': user_info.get('type', 'unknown')
            }
            
        except ImportError:
            result['error'] = "huggingface_hub not available for token testing"
        except Exception as e:
            result['error'] = f"Token authentication failed: {str(e)}"
        
        return result
        
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check available system resources before model loading"""
        import psutil
        import os
        
        # Get memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        # Estimate model memory requirements
        if self.use_quantization:
            model_size_gb = 0.18  # Gemma-3-270M INT4 quantized size
            required_memory_gb = model_size_gb * 2.5  # Add overhead for safety
        else:
            model_size_gb = 0.55  # Gemma-3-270M full precision size
            required_memory_gb = model_size_gb * 2.0  # Add overhead for safety
        
        resource_check = {
            'total_memory_gb': total_gb,
            'available_memory_gb': available_gb,
            'required_memory_gb': required_memory_gb,
            'memory_sufficient': available_gb >= required_memory_gb,
            'warnings': []
        }
        
        if available_gb < required_memory_gb:
            resource_check['warnings'].append(f"Insufficient memory: {available_gb:.1f}GB available, {required_memory_gb:.1f}GB required")
        
        if available_gb < 2.0:
            resource_check['warnings'].append("Less than 2GB available memory - model may not load or be very slow")
        
        return resource_check

    def _load_model(self):
        """Load and cache the Mistral model and tokenizer with memory management"""
        try:
            # Check system resources first
            resource_check = self._check_system_resources()
            
            if not resource_check['memory_sufficient']:
                self.logger.warning(f"System may not have sufficient memory for model loading")
                for warning in resource_check['warnings']:
                    self.logger.warning(warning)
            
            self.logger.info(f"Loading {self.model_name}...")
            self.logger.info(f"Available memory: {resource_check['available_memory_gb']:.1f}GB")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            # Load tokenizer first (lighter operation)
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=self.token
            )
            
            # Load model with memory-efficient settings
            if self.use_quantization:
                self.logger.info("Loading model with INT4 quantization...")
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs = {
                        "cache_dir": self.cache_dir,
                        "trust_remote_code": True,
                        "token": self.token,
                        "low_cpu_mem_usage": True,
                        "quantization_config": quantization_config,
                        "torch_dtype": torch.float16,
                    }
                except ImportError:
                    self.logger.warning("BitsAndBytesConfig not available, using standard loading")
                    self.use_quantization = False
            
            if not self.use_quantization:
                self.logger.info("Loading model with standard optimization...")
                model_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,
                    "token": self.token,
                    "low_cpu_mem_usage": True,
                }
                
                # Optimize for device
                if device == "cuda":
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    })
                else:
                    model_kwargs.update({
                        "torch_dtype": torch.float32,
                        "device_map": None
                    })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline optimized for Gemma
            self.logger.info("Creating inference pipeline...")
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "return_full_text": False,
                "max_new_tokens": 256,  # Reduced for Gemma's smaller context
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            if device == "cuda":
                pipeline_kwargs["torch_dtype"] = torch.float16
                pipeline_kwargs["device_map"] = "auto"
            else:
                pipeline_kwargs["torch_dtype"] = torch.float32
            
            self.pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )
            
            self.is_initialized = True
            self.logger.info("Model loaded successfully!")
            return True
            
        except ImportError as e:
            self.logger.error(f"Missing dependency: {str(e)}")
            self.is_initialized = False
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error(f"Out of memory error: {str(e)}")
                self.logger.error("Try closing other applications or using a system with more RAM")
            else:
                self.logger.error(f"Runtime error loading model: {str(e)}")
            self.is_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.is_initialized = False
            return False
    
    def initialize(self) -> bool:
        """Initialize the model if not already done"""
        if not self.is_initialized:
            try:
                # Pre-check system resources
                resource_check = self._check_system_resources()
                
                # If insufficient memory, fail gracefully
                if not resource_check['memory_sufficient']:
                    self.logger.error("Insufficient system memory for model initialization")
                    self.logger.error(f"Available: {resource_check['available_memory_gb']:.1f}GB, Required: {resource_check['required_memory_gb']:.1f}GB")
                    return False
                
                return self._load_model()
                
            except Exception as e:
                self.logger.error(f"Error during initialization: {str(e)}")
                self.is_initialized = False
                return False
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system resource status for UI display"""
        try:
            resource_check = self._check_system_resources()
            
            status = {
                'memory_status': {
                    'total_gb': resource_check['total_memory_gb'],
                    'available_gb': resource_check['available_memory_gb'],
                    'required_gb': resource_check['required_memory_gb'],
                    'sufficient': resource_check['memory_sufficient']
                },
                'model_status': {
                    'initialized': self.is_initialized,
                    'model_name': self.model_name
                },
                'warnings': resource_check['warnings'],
                'recommendations': []
            }
            
            # Add recommendations based on status
            if not resource_check['memory_sufficient']:
                status['recommendations'].extend([
                    "Close other applications to free up memory",
                    "Consider using a system with more RAM (4GB+ recommended)",
                    "Try restarting your computer to free up memory"
                ])
            
            if resource_check['available_memory_gb'] < 4:
                status['recommendations'].append("Consider upgrading to 8GB+ RAM for optimal performance")
            
            return status
            
        except Exception as e:
            return {
                'error': f"Could not check system status: {str(e)}",
                'memory_status': {},
                'model_status': {'initialized': False},
                'warnings': ["Unable to check system resources"],
                'recommendations': ["Ensure psutil is installed: pip install psutil"]
            }
    
    def generate_fallback_analysis(self, analysis_type: str, data_summary: str, context: str = "") -> str:
        """
        Generate basic statistical analysis when AI model is unavailable
        
        Args:
            analysis_type: Type of analysis requested
            data_summary: Summary of relevant data
            context: Additional context
            
        Returns:
            Basic statistical analysis
        """
        analysis_parts = [
            f"# {analysis_type} - Statistical Analysis",
            "",
            "**Note: This is a basic statistical analysis. AI model unavailable.**",
            "",
            "## Data Summary",
            data_summary,
            ""
        ]
        
        if context:
            analysis_parts.extend([
                "## Additional Context",
                context,
                ""
            ])
        
        analysis_parts.extend([
            "## Analysis Notes",
            "- This analysis is based on statistical data only",
            "- For detailed AI-powered insights, ensure sufficient system memory",
            "- Consider running AI analysis on a system with 16GB+ RAM",
            "",
            "## Recommendations",
            "- Review the statistical data above",
            "- Compare metrics with league averages",
            "- Consider tactical and contextual factors",
            "- Use domain expertise to interpret the numbers"
        ])
        
        return "\n".join(analysis_parts)

    def generate_analysis(self, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate AI analysis using the loaded model with fallback support
        
        Args:
            prompt: The input prompt for analysis
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            
        Returns:
            Generated analysis text
        """
        # Try to initialize the model
        init_success = self.initialize()
        
        if not init_success:
            # Return fallback analysis
            self.logger.warning("AI model unavailable, using fallback analysis")
            return self.generate_fallback_analysis(
                "AI Analysis", 
                "AI model could not be loaded due to insufficient system resources.",
                "Please refer to statistical data and use domain expertise."
            )
        
        try:
            # Format prompt for Gemma (standard instruction format)
            formatted_prompt = f"As a football analyst, {prompt}\n\nAnalysis:"
            
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the input prompt from the output
            if "Analysis:" in generated_text:
                analysis = generated_text.split("Analysis:", 1)[1].strip()
            else:
                analysis = generated_text.replace(formatted_prompt, "").strip()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating analysis: {str(e)}")
            # Return fallback on error
            return self.generate_fallback_analysis(
                "Analysis Error",
                f"Error occurred during AI analysis: {str(e)}",
                "Falling back to basic statistical analysis."
            )
    
    def analyze_player_performance(self, player_data: pd.Series, context: str = "") -> str:
        """
        Analyze individual player performance
        
        Args:
            player_data: Player statistics as pandas Series
            context: Additional context for analysis
            
        Returns:
            Performance analysis text
        """
        # Prepare player statistics summary
        stats_summary = self._prepare_player_stats(player_data)
        
        # Create prompt
        prompt = self.prompt_templates.get_player_analysis_prompt(
            player_name=player_data.get('Player Name', 'Unknown'),
            team=player_data.get('Team', 'Unknown'),
            position=player_data.get('Position', 'Unknown'),
            stats_summary=stats_summary,
            context=context
        )
        
        return self.generate_analysis(prompt)
    
    def analyze_team_performance(self, team_data: pd.DataFrame, team_name: str, context: str = "") -> str:
        """
        Analyze team performance
        
        Args:
            team_data: Team players' statistics as DataFrame
            team_name: Name of the team
            context: Additional context for analysis
            
        Returns:
            Team analysis text
        """
        # Prepare team statistics summary
        team_summary = self._prepare_team_stats(team_data, team_name)
        
        # Create prompt
        prompt = self.prompt_templates.get_team_analysis_prompt(
            team_name=team_name,
            team_summary=team_summary,
            context=context
        )
        
        return self.generate_analysis(prompt)
    
    def generate_scout_report(self, players_data: pd.DataFrame, criteria: Dict[str, Any], context: str = "") -> str:
        """
        Generate scout report based on criteria
        
        Args:
            players_data: All players' statistics
            criteria: Scouting criteria (position, age, metrics, etc.)
            context: Additional context
            
        Returns:
            Scout report text
        """
        # Filter and rank players based on criteria
        filtered_players = self._filter_players_by_criteria(players_data, criteria)
        top_players = self._rank_players(filtered_players, criteria.get('key_metrics', []))
        
        # Prepare summary for top players
        scouts_summary = self._prepare_scout_summary(top_players)
        
        # Create prompt
        prompt = self.prompt_templates.get_scout_report_prompt(
            criteria=criteria,
            scouts_summary=scouts_summary,
            context=context
        )
        
        return self.generate_analysis(prompt)
    
    def compare_players(self, player1_data: pd.Series, player2_data: pd.Series, metrics: List[str], context: str = "") -> str:
        """
        Compare two players across specified metrics
        
        Args:
            player1_data: First player's statistics
            player2_data: Second player's statistics
            metrics: List of metrics to compare
            context: Additional context
            
        Returns:
            Player comparison analysis
        """
        # Prepare comparison data
        comparison_data = self._prepare_player_comparison(player1_data, player2_data, metrics)
        
        # Create prompt
        prompt = self.prompt_templates.get_player_comparison_prompt(
            player1_name=player1_data.get('Player Name', 'Player 1'),
            player2_name=player2_data.get('Player Name', 'Player 2'),
            comparison_data=comparison_data,
            context=context
        )
        
        return self.generate_analysis(prompt)
    
    def identify_tactical_patterns(self, team_data: pd.DataFrame, team_name: str, context: str = "") -> str:
        """
        Identify tactical patterns and playing style
        
        Args:
            team_data: Team players' statistics
            team_name: Name of the team
            context: Additional context
            
        Returns:
            Tactical analysis text
        """
        # Analyze team's tactical patterns
        tactical_summary = self._prepare_tactical_analysis(team_data, team_name)
        
        # Create prompt
        prompt = self.prompt_templates.get_tactical_analysis_prompt(
            team_name=team_name,
            tactical_summary=tactical_summary,
            context=context
        )
        
        return self.generate_analysis(prompt)
    
    def _prepare_player_stats(self, player_data: pd.Series) -> str:
        """Prepare player statistics summary for prompt"""
        stats_lines = []
        
        # Basic info
        stats_lines.append(f"Player: {player_data.get('Player Name', 'N/A')}")
        stats_lines.append(f"Team: {player_data.get('Team', 'N/A')}")
        stats_lines.append(f"Position: {player_data.get('Position', 'N/A')}")
        stats_lines.append(f"Age: {player_data.get('Age', 'N/A')}")
        stats_lines.append(f"Appearances: {player_data.get('Appearances', 0)}")
        
        # Performance metrics by category
        metric_categories = {
            'Attack': ['Goal', 'Assist', 'Shoot On Target', 'Shoot Off Target', 'Penalty Goal', 'Create Chance'],
            'Defense': ['Block', 'Block Cross', 'Clearance', 'Tackle', 'Intercept', 'Ball Recovery', 'Header Won'],
            'Progression': ['Passing', 'Cross', 'Dribble Success', 'Free Kick'],
            'Discipline': ['Foul', 'Fouled', 'Yellow Card', 'Own Goal']
        }
        
        for category, metrics in metric_categories.items():
            category_stats = []
            for metric in metrics:
                if metric in player_data.index:
                    value = player_data.get(metric, 0)
                    if value > 0:
                        category_stats.append(f"{metric}: {value}")
            
            if category_stats:
                stats_lines.append(f"\n{category} Stats:")
                stats_lines.extend([f"  {stat}" for stat in category_stats])
        
        return "\n".join(stats_lines)
    
    def _prepare_team_stats(self, team_data: pd.DataFrame, team_name: str) -> str:
        """Prepare team statistics summary for prompt"""
        stats_lines = [f"Team: {team_name}"]
        stats_lines.append(f"Total Players: {len(team_data)}")
        
        # Position distribution
        position_counts = team_data['Position'].value_counts()
        stats_lines.append("\nPosition Distribution:")
        for pos, count in position_counts.items():
            stats_lines.append(f"  {pos}: {count}")
        
        # Key metrics aggregation
        numeric_cols = team_data.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['Age', 'Appearances']]
        
        stats_lines.append(f"\nTeam Totals:")
        for metric in metric_cols:
            total = team_data[metric].sum()
            if total > 0:
                stats_lines.append(f"  {metric}: {total}")
        
        # Top performers
        stats_lines.append(f"\nTop Scorers:")
        top_scorers = team_data.nlargest(3, 'Goal')[['Player Name', 'Goal']]
        for _, player in top_scorers.iterrows():
            if player['Goal'] > 0:
                stats_lines.append(f"  {player['Player Name']}: {player['Goal']} goals")
        
        return "\n".join(stats_lines)
    
    def _prepare_tactical_analysis(self, team_data: pd.DataFrame, team_name: str) -> str:
        """Prepare tactical analysis data"""
        analysis_lines = [f"Tactical Analysis for {team_name}:"]
        
        # Calculate team style indicators
        total_passes = team_data['Passing'].sum()
        total_crosses = team_data['Cross'].sum()
        total_dribbles = team_data['Dribble Success'].sum()
        total_tackles = team_data['Tackle'].sum()
        total_blocks = team_data['Block'].sum()
        
        analysis_lines.append(f"\nPlaying Style Indicators:")
        analysis_lines.append(f"  Total Passes: {total_passes}")
        analysis_lines.append(f"  Total Crosses: {total_crosses}")
        analysis_lines.append(f"  Total Successful Dribbles: {total_dribbles}")
        analysis_lines.append(f"  Total Tackles: {total_tackles}")
        analysis_lines.append(f"  Total Blocks: {total_blocks}")
        
        # Key players by position
        positions = team_data['Position'].unique()
        analysis_lines.append(f"\nKey Players by Position:")
        for position in positions:
            pos_players = team_data[team_data['Position'] == position]
            if len(pos_players) > 0:
                # Find most active player in position
                pos_players['activity_score'] = pos_players[['Goal', 'Assist', 'Passing', 'Tackle', 'Block']].sum(axis=1)
                top_player = pos_players.loc[pos_players['activity_score'].idxmax()]
                analysis_lines.append(f"  {position}: {top_player['Player Name']} (Activity Score: {top_player['activity_score']})")
        
        return "\n".join(analysis_lines)
    
    def _filter_players_by_criteria(self, players_data: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Filter players based on scouting criteria"""
        filtered = players_data.copy()
        
        # Filter by position
        if 'position' in criteria and criteria['position']:
            filtered = filtered[filtered['Position'].isin(criteria['position'])]
        
        # Filter by age range
        if 'age_range' in criteria and criteria['age_range']:
            min_age, max_age = criteria['age_range']
            filtered = filtered[(filtered['Age'] >= min_age) & (filtered['Age'] <= max_age)]
        
        # Filter by minimum appearances
        if 'min_appearances' in criteria:
            filtered = filtered[filtered['Appearances'] >= criteria['min_appearances']]
        
        return filtered
    
    def _rank_players(self, players_data: pd.DataFrame, key_metrics: List[str]) -> pd.DataFrame:
        """Rank players based on key metrics"""
        if not key_metrics:
            key_metrics = ['Goal', 'Assist', 'Passing', 'Tackle']
        
        # Calculate composite score
        available_metrics = [m for m in key_metrics if m in players_data.columns]
        if available_metrics:
            players_data['composite_score'] = players_data[available_metrics].sum(axis=1)
            return players_data.nlargest(10, 'composite_score')
        
        return players_data.head(10)
    
    def _prepare_scout_summary(self, top_players: pd.DataFrame) -> str:
        """Prepare scout summary for top players"""
        summary_lines = ["Top Scouting Targets:"]
        
        for idx, (_, player) in enumerate(top_players.head(5).iterrows(), 1):
            summary_lines.append(f"\n{idx}. {player['Player Name']} ({player['Team']})")
            summary_lines.append(f"   Position: {player['Position']}, Age: {player['Age']}")
            summary_lines.append(f"   Key Stats: Goals: {player.get('Goal', 0)}, Assists: {player.get('Assist', 0)}")
            if 'composite_score' in player.index:
                summary_lines.append(f"   Overall Score: {player['composite_score']:.1f}")
        
        return "\n".join(summary_lines)
    
    def _prepare_player_comparison(self, player1: pd.Series, player2: pd.Series, metrics: List[str]) -> str:
        """Prepare player comparison data"""
        comparison_lines = ["Player Comparison:"]
        
        comparison_lines.append(f"\n{player1.get('Player Name', 'Player 1')} vs {player2.get('Player Name', 'Player 2')}")
        comparison_lines.append(f"Teams: {player1.get('Team', 'N/A')} vs {player2.get('Team', 'N/A')}")
        comparison_lines.append(f"Ages: {player1.get('Age', 'N/A')} vs {player2.get('Age', 'N/A')}")
        comparison_lines.append(f"Positions: {player1.get('Position', 'N/A')} vs {player2.get('Position', 'N/A')}")
        
        comparison_lines.append(f"\nMetric Comparison:")
        for metric in metrics:
            if metric in player1.index and metric in player2.index:
                val1 = player1.get(metric, 0)
                val2 = player2.get(metric, 0)
                comparison_lines.append(f"  {metric}: {val1} vs {val2}")
        
        return "\n".join(comparison_lines)