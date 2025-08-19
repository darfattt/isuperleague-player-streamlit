"""
Football-specific prompt templates for AI analysis
Specialized prompts for Indonesia Super League football analysis
"""

from typing import Dict, Any, List

class PromptTemplates:
    """
    Collection of prompt templates for football analysis
    """
    
    def __init__(self):
        # Indonesian football terminology
        self.indonesian_terms = {
            'BELAKANG': 'Defender',
            'TENGAH': 'Midfielder', 
            'DEPAN': 'Forward',
            'PENJAGA GAWANG': 'Goalkeeper'
        }
    
    def get_player_analysis_prompt(self, player_name: str, team: str, position: str, stats_summary: str, context: str = "") -> str:
        """Generate prompt for individual player performance analysis"""
        position_translated = self.indonesian_terms.get(position.upper(), position)
        
        prompt = f"""You are an expert football analyst specializing in the Indonesia Super League. Analyze the following player's performance and provide detailed insights.

Player Information:
{stats_summary}

Please provide a comprehensive analysis covering:

1. **Strengths & Strong Areas**: 
   - Identify the player's best performing metrics and skills
   - Highlight what makes this player valuable to their team
   - Compare performance relative to their position ({position_translated})

2. **Areas for Improvement**:
   - Point out weaknesses or underperforming areas
   - Suggest specific skills or metrics that need development
   - Consider position-specific requirements

3. **Playing Style Assessment**:
   - Describe the player's likely playing style based on their statistics
   - How do they contribute to team tactics?
   - What role do they play in their team's formation?

4. **Market Value & Potential**:
   - Is this player undervalued or overvalued based on performance?
   - Future potential and development trajectory
   - Suitability for different tactical systems

5. **Recommendations**:
   - For the player: specific areas to focus on for improvement
   - For coaches: how to best utilize this player
   - For scouts: is this player worth targeting?

Context: {context if context else 'Standard performance analysis for Indonesia Super League.'}

Provide your analysis in a clear, structured format with specific examples from the statistics. Use professional football terminology and consider the unique characteristics of Indonesian football."""

        return prompt
    
    def get_team_analysis_prompt(self, team_name: str, team_summary: str, context: str = "") -> str:
        """Generate prompt for team performance analysis"""
        prompt = f"""You are an expert football analyst specializing in the Indonesia Super League. Analyze the following team's performance and tactical approach.

Team Information:
{team_summary}

Please provide a comprehensive team analysis covering:

1. **Team Strengths**:
   - What are this team's main strengths based on player statistics?
   - Which areas of the pitch do they dominate?
   - Key players who drive team performance

2. **Tactical Style & Formation**:
   - Based on player positions and statistics, what tactical style does this team likely employ?
   - Are they more attacking or defensive minded?
   - How do they build attacks and defend?

3. **Areas of Concern**:
   - Where are the team's weaknesses or gaps?
   - Which positions need strengthening?
   - Potential vulnerabilities opponents could exploit

4. **Key Player Dependencies**:
   - Which players are crucial to the team's success?
   - How balanced is the squad across positions?
   - Risk assessment if key players are unavailable

5. **Transfer & Development Recommendations**:
   - What type of players should they target in transfers?
   - Which current players have potential for development?
   - Strategic priorities for squad building

6. **Competitive Analysis**:
   - How does this team likely perform against different tactical approaches?
   - Strengths and weaknesses in the context of Indonesia Super League competition

Context: {context if context else 'Tactical and performance analysis for Indonesia Super League competition.'}

Provide detailed analysis with specific insights based on the statistics. Consider the competitive level and characteristics of Indonesian football."""

        return prompt
    
    def get_scout_report_prompt(self, criteria: Dict[str, Any], scouts_summary: str, context: str = "") -> str:
        """Generate prompt for scouting report"""
        criteria_text = self._format_scouting_criteria(criteria)
        
        prompt = f"""You are a professional football scout specializing in the Indonesia Super League. Provide a detailed scouting report based on the following criteria and player data.

Scouting Criteria:
{criteria_text}

Top Candidates Identified:
{scouts_summary}

Please provide a comprehensive scouting report covering:

1. **Primary Recommendations**:
   - Rank the top 3-5 players based on the criteria
   - Detailed assessment of each recommended player
   - Why they meet your scouting requirements

2. **Player Profiles**:
   For each recommended player, provide:
   - Current performance level and consistency
   - Potential for development and improvement
   - Tactical fit and versatility
   - Age profile and career stage

3. **Risk Assessment**:
   - Injury concerns or performance consistency issues
   - Adaptation potential to new teams/systems
   - Value for money considerations

4. **Hidden Gems**:
   - Undervalued players who might be overlooked
   - Young players with high potential
   - Players who could exceed expectations

5. **Tactical Fit Analysis**:
   - How would these players fit different tactical systems?
   - Compatibility with various playing styles
   - Position flexibility and utility

6. **Market Intelligence**:
   - Estimated market values and transfer feasibility
   - Contract situations and availability
   - Competition from other clubs

7. **Watching Recommendations**:
   - Priority players to monitor in upcoming matches
   - Specific aspects to observe during live scouting
   - Alternative options if primary targets aren't available

Context: {context if context else 'Professional scouting report for Indonesia Super League talent identification.'}

Provide specific, actionable recommendations with clear reasoning based on statistical analysis and football expertise."""

        return prompt
    
    def get_player_comparison_prompt(self, player1_name: str, player2_name: str, comparison_data: str, context: str = "") -> str:
        """Generate prompt for player comparison analysis"""
        prompt = f"""You are an expert football analyst specializing in player comparisons. Provide a detailed comparative analysis of these two Indonesia Super League players.

Comparison Data:
{comparison_data}

Please provide a comprehensive comparison covering:

1. **Overall Performance Comparison**:
   - Who is performing better overall and why?
   - Key metrics where each player excels
   - Statistical significance of differences

2. **Strengths vs Strengths**:
   - Compare each player's strongest attributes
   - Which player has more impactful strengths?
   - How do their strengths complement different tactical needs?

3. **Weakness Analysis**:
   - Identify each player's main weaknesses
   - Which player has more concerning gaps in their game?
   - Areas where both players need improvement

4. **Playing Style Differences**:
   - How do their playing styles differ?
   - Which player is more versatile or specialized?
   - Tactical system preferences for each player

5. **Value Proposition**:
   - Which player offers better value considering age, performance, and potential?
   - Transfer market implications
   - Long-term vs immediate impact assessment

6. **Team Fit Analysis**:
   - Which player would fit better in different team systems?
   - Consider tactical flexibility and role adaptability
   - Team chemistry and leadership qualities

7. **Future Projection**:
   - Which player has better development potential?
   - Career trajectory predictions
   - Sustainability of current performance levels

8. **Recommendation**:
   - For different scenarios (team needs, budget, tactical system), which player would you recommend?
   - Clear reasoning for your recommendation
   - Alternative scenarios where the choice might change

Context: {context if context else 'Professional player comparison for tactical and transfer decision-making in Indonesia Super League.'}

Provide a balanced, objective analysis with clear conclusions and reasoning based on the statistical evidence and football expertise."""

        return prompt
    
    def get_tactical_analysis_prompt(self, team_name: str, tactical_summary: str, context: str = "") -> str:
        """Generate prompt for tactical pattern analysis"""
        prompt = f"""You are a tactical expert specializing in football analysis. Analyze the tactical patterns and playing style of this Indonesia Super League team.

Tactical Data:
{tactical_summary}

Please provide a detailed tactical analysis covering:

1. **Formation and System Analysis**:
   - What formation does this team likely use based on player positions and statistics?
   - How rigid or fluid is their tactical system?
   - Adaptability to different match situations

2. **Playing Style Identification**:
   - Possession-based, direct, counter-attacking, or mixed approach?
   - Tempo and rhythm preferences
   - Risk vs security in their play

3. **Phase Analysis**:
   
   **Build-up Play**:
   - How do they construct attacks from defense?
   - Key players in the build-up phase
   - Patterns of ball progression

   **Final Third**:
   - Goal creation methods (crosses, through balls, individual skill)
   - Set piece effectiveness
   - Finishing patterns and shot selection

   **Defensive Organization**:
   - Pressing intensity and triggers
   - Defensive line height and compactness
   - Individual vs collective defending

4. **Key Tactical Roles**:
   - Identify players with specific tactical responsibilities
   - Creative hubs and playmakers
   - Defensive anchors and work horses

5. **Strengths and Vulnerabilities**:
   - What tactical approaches work best for this team?
   - Where are they most vulnerable tactically?
   - How opponents might target their weaknesses

6. **Game Management**:
   - Ability to control games and see out results
   - Response to different match states (leading, trailing, level)
   - Substitution patterns and tactical flexibility

7. **Strategic Recommendations**:
   - Tactical tweaks that could improve performance
   - Player deployment optimizations
   - Training focus areas based on tactical needs

8. **Opponent Preparation**:
   - How should opponents prepare to face this team?
   - Key battles and matchups to focus on
   - Tactical countermeasures

Context: {context if context else 'Professional tactical analysis for Indonesia Super League competitive intelligence and coaching development.'}

Provide specific tactical insights with clear reasoning based on the statistical patterns and football tactical principles."""

        return prompt
    
    def get_custom_analysis_prompt(self, analysis_type: str, data_summary: str, specific_question: str, context: str = "") -> str:
        """Generate prompt for custom analysis questions"""
        prompt = f"""You are an expert football analyst specializing in the Indonesia Super League. Answer the following specific question with detailed analysis.

Analysis Type: {analysis_type}
Data Summary: {data_summary}

Specific Question: {specific_question}

Please provide a comprehensive answer that:
1. Directly addresses the question asked
2. Uses the provided statistical data as evidence
3. Applies professional football knowledge and expertise
4. Considers the specific context of Indonesian football
5. Provides actionable insights and recommendations
6. Uses clear, structured presentation of findings

Context: {context if context else 'Custom football analysis for Indonesia Super League.'}

Ensure your response is detailed, well-reasoned, and based on the available statistical evidence."""

        return prompt
    
    def _format_scouting_criteria(self, criteria: Dict[str, Any]) -> str:
        """Format scouting criteria for prompt inclusion"""
        criteria_lines = []
        
        if 'position' in criteria and criteria['position']:
            positions = ', '.join(criteria['position'])
            criteria_lines.append(f"Target Positions: {positions}")
        
        if 'age_range' in criteria and criteria['age_range']:
            min_age, max_age = criteria['age_range']
            criteria_lines.append(f"Age Range: {min_age}-{max_age} years")
        
        if 'key_metrics' in criteria and criteria['key_metrics']:
            metrics = ', '.join(criteria['key_metrics'])
            criteria_lines.append(f"Key Performance Metrics: {metrics}")
        
        if 'min_appearances' in criteria:
            criteria_lines.append(f"Minimum Appearances: {criteria['min_appearances']}")
        
        if 'team_exclusions' in criteria and criteria['team_exclusions']:
            exclusions = ', '.join(criteria['team_exclusions'])
            criteria_lines.append(f"Excluded Teams: {exclusions}")
        
        if 'budget_tier' in criteria:
            criteria_lines.append(f"Budget Tier: {criteria['budget_tier']}")
        
        return '\n'.join(criteria_lines) if criteria_lines else "General scouting without specific criteria"