# Indonesia Super League Player Analytics Dashboard

A Streamlit web application for analyzing individual player statistics from the Indonesia Super League.

## Features

### 📈 Player Performance Dashboard
- **Top 10 Players**: View top performers for each statistic
- **Negative Metrics Handling**: Properly handles metrics where lower values are better (Own Goal, Yellow Card, Foul, Shoot Off Target)
- **Category Organization**: Metrics organized into logical categories:
  - **Attack**: Goal, Assist, Shoot On Target, Shoot Off Target, Penalty Goal, Create Chance
  - **Defense**: Block, Block Cross, Clearance, Tackle, Intercept, Ball Recovery, Header Won
  - **Progression**: Passing, Cross, Dribble Success, Free Kick
  - **Discipline**: Foul, Fouled, Yellow Card, Own Goal
  - **Goalkeeper**: Saves

### 🔄 NEW: Player Comparison
- **Multi-Player Analysis**: Compare 2-4 players side-by-side with advanced filtering
- **Smart Player Search**: Filter by team, position, age, and appearances to find relevant players
- **Radar Chart Comparison**: Multi-dimensional visualization of all 22 player metrics
- **Performance Bar Charts**: Category-based horizontal bar charts with percentile rankings
- **Detailed Statistics Table**: Comprehensive metric comparison with best performer highlights
- **Filtered Percentiles**: Rankings calculated against filtered player pool (not entire dataset)
- **Dark Mode Compatible**: Professional sports analytics styling with enhanced readability

### 📊 NEW: Category Performance Analysis (Overall Tab)
- **Category Scores**: Normalized performance scores for Attack, Defense, Progression, Discipline
- **Category Leaders**: Summary table showing top performer in each category (worst for discipline)
- **Interactive Charts**: Bar charts for top 10 players per category
- **Radar Comparison**: Multi-dimensional comparison of top 5 overall players
- **Score Distribution**: Box plots showing performance spread across categories
- **Detailed Rankings**: Expandable tables with top 10 players per category

### 📈 NEW: Global Statistics Dashboard (All Tabs)
- **Overall Tab**: Total players, average age, teams, active players, top scoring position
- **Attack Tab**: Total goals, assists, shot conversion rates, goals per appearance, target accuracy
- **Defense Tab**: Total tackles, clearances, interceptions, defensive actions per player
- **Progression Tab**: Total passes, crosses, dribbles, most active playmaker
- **Discipline Tab**: Fouls, yellow cards, foul rates, most disciplined team

### 🎨 NEW: Enhanced UI Features
- **Dark Mode Support**: All charts now work perfectly in both light and dark themes
- **Discipline Logic**: Shows "worst" performers for discipline metrics (proper negative handling)
- **Responsive Stats**: All statistics update based on current filter selection

### 🔧 Global Filters
- **Team Name**: Filter by one or multiple teams
- **Player Position**: Filter by position (BELAKANG, TENGAH, DEPAN)
- **Age Range**: Slider to filter players by age
- **Appearances Range**: Slider to filter by number of match appearances

## Data Source

The app reads data from `data/players_statistics.csv` which contains comprehensive player statistics including:

- Player demographics (Name, Team, Country, Age, Position)
- Match statistics (Appearances, Goals, Assists, etc.)
- Performance metrics across 22 different categories

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Alternative Port
```bash
streamlit run app.py --server.port 8502
```

## Project Structure

```
isuperleage-player-streamlit/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── players_statistics.csv  # Player statistics data
└── utils/
    ├── data_loader.py        # Data loading and filtering utilities
    └── visualization.py      # Plotly visualization components
```

## Key Features

### Smart Metric Handling
- **Positive Metrics**: Higher values = better performance (Goals, Assists, etc.)
- **Negative Metrics**: Lower values = better performance (Own Goals, Yellow Cards, etc.)
- **Proper Ranking**: Shows "Top 10" for positive metrics, "Bottom 10" for negative metrics

### Interactive Visualizations
- Bar charts for top/bottom performers
- Team and position comparisons
- Age distribution analysis
- Performance score calculations

### Comprehensive Filtering
- Multi-team selection
- Position-based filtering
- Age range filtering
- Appearances threshold filtering

## Data Pipeline Integration

This app is part of the larger Indonesia Super League scraping pipeline:

1. **Data Collection**: `team_players_scraper.py` → `data/teams_info.json`
2. **Statistics Scraping**: `player_scraper.py` → `player_stats.csv`
3. **Data Combination**: `combine_player_data.py` → `data/players_statistics.csv`
4. **Visualization**: This Streamlit app reads the final combined dataset

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **numpy**: Numerical computing

## Related Apps

- **Club Statistics Dashboard**: `../isuperleague-club-streamlit/` - Team-level analytics
- **Data Scrapers**: Root directory contains the scraping pipeline for data collection