#!/usr/bin/env python3
"""
MLBDataLoader - A comprehensive class for loading MLB historical data for machine learning.

Features:
- Load historical game results and team statistics
- Calculate rolling averages (3, 5, 10 games + season-to-date)
- Include advanced sabermetrics
- Multiple output formats (wide and separate DataFrames)
- Local caching with API rate limiting
- Comprehensive validation and error handling
"""

import pandas as pd
import numpy as np
import statsapi
import time
import os
import pickle
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional, Tuple
import logging

class MLBDataLoader:
    """
    MLB data loader for machine learning projects.
    
    Loads historical game results and team statistics with advanced features,
    caching, and comprehensive validation.
    """
    
    def __init__(self, 
                 cache_dir: str = './mlb_cache', 
                 api_delay: float = 0.5,
                 debug: bool = False):
        """
        Initialize the MLB data loader.
        
        Args:
            cache_dir: Directory to store cached data
            api_delay: Delay between API calls in seconds
            debug: If True, fail loudly on errors; if False, handle gracefully
        """
        self.cache_dir = cache_dir
        self.api_delay = api_delay
        self.debug = debug
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Team mapping cache
        self._team_mapping = None
        
        # Validation criteria
        self.validation_criteria = {
            'min_games_for_rolling': 3,
            'required_game_columns': ['game_id', 'home_team', 'away_team', 'home_score', 'away_score'],
            'required_team_stat_columns': ['team_id', 'team_name', 'season'],
            'max_missing_percentage': 0.1  # Allow up to 10% missing data
        }
    
    def _handle_error(self, error_msg: str, exception: Exception = None):
        """Handle errors based on debug mode."""
        if self.debug:
            self.logger.error(error_msg)
            if exception:
                raise exception
            else:
                raise ValueError(error_msg)
        else:
            self.logger.warning(error_msg)
            warnings.warn(error_msg)
    
    def _api_call_with_delay(self, func, *args, **kwargs):
        """Make API call with rate limiting."""
        try:
            time.sleep(self.api_delay)
            return func(*args, **kwargs)
        except Exception as e:
            self._handle_error(f"API call failed: {str(e)}", e)
            return None
    
    def _get_cache_key(self, method_name: str, **params) -> str:
        """Generate cache key for method and parameters."""
        param_str = str(sorted(params.items()))
        hash_obj = hashlib.md5(f"{method_name}_{param_str}".encode())
        return hash_obj.hexdigest()
    
    def _save_to_cache(self, data, cache_key: str):
        """Save data to cache file."""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            self._handle_error(f"Failed to save cache: {str(e)}", e)
    
    def _load_from_cache(self, cache_key: str):
        """Load data from cache file."""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.debug(f"Loaded data from cache: {cache_key}")
                return data
        except Exception as e:
            self._handle_error(f"Failed to load cache: {str(e)}", e)
        return None
    
    def get_team_mapping(self) -> Dict[str, int]:
        """
        Get mapping of team names to team IDs.
        
        Returns:
            Dictionary mapping team names to IDs
        """
        if self._team_mapping is not None:
            return self._team_mapping
        
        cache_key = self._get_cache_key('team_mapping')
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            self._team_mapping = cached_data
            return self._team_mapping
        
        try:
            teams_data = self._api_call_with_delay(
                statsapi.get, 'teams', {'sportId': 1, 'season': 2024}
            )
            
            if teams_data and 'teams' in teams_data:
                self._team_mapping = {
                    team['name']: team['id'] 
                    for team in teams_data['teams']
                }
                self._save_to_cache(self._team_mapping, cache_key)
                self.logger.info(f"Loaded {len(self._team_mapping)} teams")
                return self._team_mapping
            else:
                self._handle_error("No team data received from API")
                return {}
                
        except Exception as e:
            self._handle_error(f"Failed to load team mapping: {str(e)}", e)
            return {}
    
    def get_available_seasons(self) -> List[int]:
        """
        Get list of available seasons from the API.
        
        Returns:
            List of available season years
        """
        cache_key = self._get_cache_key('available_seasons')
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Try to get seasons by checking team data across years
        seasons = []
        current_year = datetime.now().year
        
        for year in range(2010, current_year):
            try:
                teams_data = self._api_call_with_delay(
                    statsapi.get, 'teams', {'sportId': 1, 'season': year}
                )
                if teams_data and 'teams' in teams_data and teams_data['teams']:
                    seasons.append(year)
            except:
                continue
        
        self._save_to_cache(seasons, cache_key)
        self.logger.info(f"Found {len(seasons)} available seasons: {min(seasons)}-{max(seasons)}")
        return seasons
    
    def load_game_results(self, 
                         start_season: int, 
                         end_season: int,
                         teams: Optional[List[str]] = None,
                         exclude_doubleheaders: bool = True) -> pd.DataFrame:
        """
        Load game results for specified seasons.
        
        Args:
            start_season: First season to load
            end_season: Last season to load (inclusive)
            teams: List of team names to filter (None for all teams)
            exclude_doubleheaders: Whether to exclude doubleheader games
            
        Returns:
            DataFrame with game results
        """
        cache_key = self._get_cache_key(
            'game_results', 
            start_season=start_season, 
            end_season=end_season,
            teams=teams,
            exclude_doubleheaders=exclude_doubleheaders
        )
        
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        all_games = []
        team_ids = None
        
        # Get team IDs if specific teams requested
        if teams:
            team_mapping = self.get_team_mapping()
            team_ids = [team_mapping.get(team) for team in teams if team in team_mapping]
            if not team_ids:
                self._handle_error("No valid team IDs found for specified teams")
                return pd.DataFrame()
        
        # Load games for each season
        for season in range(start_season, end_season + 1):
            self.logger.info(f"Loading games for season {season}")
            
            try:
                # Get full season schedule
                season_games = self._api_call_with_delay(
                    statsapi.schedule,
                    start_date=f'{season}-03-01',
                    end_date=f'{season}-11-30'
                )
                
                if not season_games:
                    self.logger.warning(f"No games found for season {season}")
                    continue
                
                for game in season_games:
                    # Skip if specific teams requested and this game doesn't involve them
                    if team_ids:
                        home_id = game.get('home_id')
                        away_id = game.get('away_id')
                        if home_id not in team_ids and away_id not in team_ids:
                            continue
                    
                    # Skip postponed/suspended games
                    if game.get('status') in ['Postponed', 'Suspended', 'Cancelled']:
                        continue
                    
                    # Skip doubleheaders if requested
                    if exclude_doubleheaders and game.get('doubleheader', 'N') != 'N':
                        continue
                    
                    # Skip games without final scores
                    home_score = game.get('home_score')
                    away_score = game.get('away_score')
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    # Add game data
                    game_data = {
                        'game_id': game.get('game_id'),
                        'date': pd.to_datetime(game.get('game_date')),
                        'season': season,
                        'home_team': game.get('home_name'),
                        'away_team': game.get('away_name'),
                        'home_team_id': game.get('home_id'),
                        'away_team_id': game.get('away_id'),
                        'home_score': int(home_score),
                        'away_score': int(away_score),
                        'venue': game.get('venue_name', ''),
                        'game_type': game.get('game_type', 'R')  # R = Regular season
                    }
                    
                    all_games.append(game_data)
                
            except Exception as e:
                self._handle_error(f"Failed to load games for season {season}: {str(e)}")
        
        if not all_games:
            self._handle_error("No games loaded")
            return pd.DataFrame()
        
        # Create DataFrame
        games_df = pd.DataFrame(all_games)
        games_df = games_df.sort_values(['date', 'game_id']).reset_index(drop=True)
        
        # Save to cache
        self._save_to_cache(games_df, cache_key)
        
        self.logger.info(f"Loaded {len(games_df)} games from {start_season}-{end_season}")
        return games_df
    
    def load_team_season_stats(self, 
                              seasons: List[int],
                              stat_groups: List[str] = ['hitting', 'pitching']) -> pd.DataFrame:
        """
        Load team season statistics.
        
        Args:
            seasons: List of seasons to load
            stat_groups: Types of stats to load ('hitting', 'pitching', 'fielding')
            
        Returns:
            DataFrame with team season statistics
        """
        cache_key = self._get_cache_key(
            'team_season_stats',
            seasons=tuple(seasons),
            stat_groups=tuple(stat_groups)
        )
        
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        team_mapping = self.get_team_mapping()
        if not team_mapping:
            return pd.DataFrame()
        
        all_stats = []
        
        for season in seasons:
            self.logger.info(f"Loading team stats for season {season}")
            
            # Try to get standings data which includes basic team performance stats
            try:
                standings_data = self._api_call_with_delay(
                    statsapi.standings_data, 
                    leagueId="103,104",  # AL and NL
                    season=season
                )
                
                if standings_data:
                    # Create a mapping from standings data
                    for team_standing in standings_data:
                        team_season_stats = {
                            'team_id': team_standing.get('team_id'),
                            'team_name': team_standing.get('name'),
                            'season': season,
                            'wins': team_standing.get('w'),
                            'losses': team_standing.get('l'),
                            'win_percentage': team_standing.get('w_percent'),
                            'games_back': team_standing.get('gb'),
                            'runs_scored': team_standing.get('runs_scored'),
                            'runs_allowed': team_standing.get('runs_allowed'),
                            'run_differential': team_standing.get('run_diff')
                        }
                        all_stats.append(team_season_stats)
                    
                    self.logger.info(f"Loaded standings-based stats for {len(standings_data)} teams in {season}")
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Failed to load standings data for {season}: {str(e)}")
            
            # Fallback: Try individual team stats (may not work for all seasons)
            for team_name, team_id in team_mapping.items():
                team_season_stats = {'team_id': team_id, 'team_name': team_name, 'season': season}
                
                # Try alternative endpoint for team stats
                try:
                    # Try team leaders endpoint which might have more reliable data
                    hitting_leaders = self._api_call_with_delay(
                        statsapi.team_leader_data,
                        team_id,
                        'homeRuns',
                        season=season
                    )
                    
                    if hitting_leaders:
                        team_season_stats['team_hr_leader'] = hitting_leaders[0].get('value', 0) if hitting_leaders else 0
                        
                except Exception as e:
                    self.logger.debug(f"Could not load leader data for {team_name} {season}: {str(e)}")
                
                # Add basic placeholder stats if no data available
                if len(team_season_stats) == 3:  # Only has team_id, team_name, season
                    team_season_stats.update({
                        'wins': None,
                        'losses': None,
                        'win_percentage': None,
                        'runs_scored': None,
                        'runs_allowed': None
                    })
                
                all_stats.append(team_season_stats)
        
        if not all_stats:
            self._handle_error("No team statistics loaded")
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(all_stats)
        self._save_to_cache(stats_df, cache_key)
        
        self.logger.info(f"Loaded team stats for {len(seasons)} seasons, {len(stats_df)} team records")
        return stats_df
    
    def calculate_rolling_stats(self, 
                               games_df: pd.DataFrame,
                               windows: List[Union[int, str]] = [3, 5, 10, 'season']) -> pd.DataFrame:
        """
        Calculate rolling statistics for teams.
        
        Args:
            games_df: DataFrame with game results
            windows: List of rolling window sizes (integers or 'season')
            
        Returns:
            DataFrame with rolling statistics added
        """
        if games_df.empty:
            return games_df
        
        # Validate required columns
        required_cols = ['date', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
        missing_cols = [col for col in required_cols if col not in games_df.columns]
        if missing_cols:
            self._handle_error(f"Missing required columns for rolling stats: {missing_cols}")
            return games_df
        
        df = games_df.copy()
        df = df.sort_values(['date', 'game_id']).reset_index(drop=True)
        
        # Get all unique teams
        home_teams = set(df['home_team_id'].dropna())
        away_teams = set(df['away_team_id'].dropna())
        all_teams = home_teams.union(away_teams)
        
        # Calculate rolling stats for each team
        for team_id in all_teams:
            team_games = []
            
            # Get all games for this team (home and away)
            home_games = df[df['home_team_id'] == team_id][['date', 'home_score', 'away_score']].copy()
            home_games['team_score'] = home_games['home_score']
            home_games['opp_score'] = home_games['away_score']
            
            away_games = df[df['away_team_id'] == team_id][['date', 'home_score', 'away_score']].copy()
            away_games['team_score'] = away_games['away_score'] 
            away_games['opp_score'] = away_games['home_score']
            
            team_games = pd.concat([home_games, away_games])[['date', 'team_score', 'opp_score']]
            team_games = team_games.sort_values('date').reset_index(drop=True)
            
            if len(team_games) == 0:
                continue
            
            # Calculate rolling windows
            for window in windows:
                if window == 'season':
                    # Season-to-date averages
                    team_games[f'rolling_season_runs_scored'] = team_games['team_score'].expanding().mean()
                    team_games[f'rolling_season_runs_allowed'] = team_games['opp_score'].expanding().mean()
                else:
                    # Fixed window rolling averages
                    if isinstance(window, int) and window > 0:
                        team_games[f'rolling_{window}_runs_scored'] = (
                            team_games['team_score'].rolling(window=window, min_periods=1).mean()
                        )
                        team_games[f'rolling_{window}_runs_allowed'] = (
                            team_games['opp_score'].rolling(window=window, min_periods=1).mean()
                        )
            
            # Merge rolling stats back to main DataFrame
            for idx, row in team_games.iterrows():
                game_date = row['date']
                
                # Find matching games in main DataFrame (both home and away)
                home_mask = (df['home_team_id'] == team_id) & (df['date'] == game_date)
                away_mask = (df['away_team_id'] == team_id) & (df['date'] == game_date)
                
                for window in windows:
                    window_str = 'season' if window == 'season' else str(window)
                    
                    runs_scored_col = f'rolling_{window_str}_runs_scored'
                    runs_allowed_col = f'rolling_{window_str}_runs_allowed'
                    
                    if runs_scored_col in team_games.columns:
                        # Add to home games
                        df.loc[home_mask, f'home_{runs_scored_col}'] = row[runs_scored_col]
                        df.loc[home_mask, f'home_{runs_allowed_col}'] = row[runs_allowed_col]
                        
                        # Add to away games  
                        df.loc[away_mask, f'away_{runs_scored_col}'] = row[runs_scored_col]
                        df.loc[away_mask, f'away_{runs_allowed_col}'] = row[runs_allowed_col]
        
        self.logger.info(f"Calculated rolling statistics for {len(windows)} windows")
        return df
    
    def create_target_variables(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for machine learning.
        
        Args:
            games_df: DataFrame with game results
            
        Returns:
            DataFrame with target variables added
        """
        if games_df.empty:
            return games_df
        
        df = games_df.copy()
        
        # Validate required columns
        if 'home_score' not in df.columns or 'away_score' not in df.columns:
            self._handle_error("Missing score columns for target variable creation")
            return df
        
        # Create target variables
        df['total_runs'] = df['home_score'] + df['away_score']
        df['home_runs'] = df['home_score']
        df['away_runs'] = df['away_score'] 
        df['run_differential'] = df['home_score'] - df['away_score']  # Positive = home team won
        
        self.logger.info("Created target variables: total_runs, home_runs, away_runs, run_differential")
        return df
    
    def validate_data(self, 
                     df: pd.DataFrame, 
                     data_type: str = 'games') -> Dict[str, Union[bool, str, int]]:
        """
        Validate data quality and completeness.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('games', 'team_stats', 'features')
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'missing_data_percentage': 0.0
        }
        
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame is empty")
            return validation_result
        
        # Check for required columns based on data type
        required_columns = []
        if data_type == 'games':
            required_columns = self.validation_criteria['required_game_columns']
        elif data_type == 'team_stats':
            required_columns = self.validation_criteria['required_team_stat_columns']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for missing data
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        validation_result['missing_data_percentage'] = missing_percentage
        
        if missing_percentage > self.validation_criteria['max_missing_percentage'] * 100:
            validation_result['warnings'].append(
                f"High missing data percentage: {missing_percentage:.2f}%"
            )
        
        # Data type specific validations
        if data_type == 'games':
            # Check for null scores
            if 'home_score' in df.columns and 'away_score' in df.columns:
                null_scores = df[df['home_score'].isnull() | df['away_score'].isnull()]
                if len(null_scores) > 0:
                    validation_result['warnings'].append(f"{len(null_scores)} games with missing scores")
            
            # Check for reasonable score ranges
            if 'home_score' in df.columns:
                max_score = df['home_score'].max()
                if max_score > 30:  # Unreasonably high score
                    validation_result['warnings'].append(f"Unusually high score detected: {max_score}")
        
        # Log validation results
        if validation_result['errors']:
            self.logger.error(f"Validation failed: {validation_result['errors']}")
        if validation_result['warnings']:
            self.logger.warning(f"Validation warnings: {validation_result['warnings']}")
        
        return validation_result
    
    def create_ml_dataset_wide(self, 
                              start_season: int,
                              end_season: int,
                              features: List[str] = ['rolling', 'season_stats'],
                              teams: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create wide-format ML dataset (one row per game).
        
        Args:
            start_season: First season to include
            end_season: Last season to include 
            features: List of feature types to include
            teams: Specific teams to include (None for all)
            
        Returns:
            Wide-format DataFrame ready for ML
        """
        self.logger.info(f"Creating wide ML dataset for seasons {start_season}-{end_season}")
        
        # Load game results
        games_df = self.load_game_results(start_season, end_season, teams)
        if games_df.empty:
            return pd.DataFrame()
        
        # Create target variables
        games_df = self.create_target_variables(games_df)
        
        # Add rolling statistics if requested
        if 'rolling' in features:
            games_df = self.calculate_rolling_stats(games_df)
        
        # Add season statistics if requested
        if 'season_stats' in features:
            seasons = list(range(start_season, end_season + 1))
            team_stats = self.load_team_season_stats(seasons)
            
            if not team_stats.empty:
                # Merge home team stats
                games_df = games_df.merge(
                    team_stats,
                    left_on=['home_team_id', 'season'],
                    right_on=['team_id', 'season'],
                    how='left',
                    suffixes=('', '_home_team_stats')
                )
                
                # Merge away team stats  
                games_df = games_df.merge(
                    team_stats,
                    left_on=['away_team_id', 'season'], 
                    right_on=['team_id', 'season'],
                    how='left',
                    suffixes=('', '_away_team_stats')
                )
        
        # Validate final dataset
        validation = self.validate_data(games_df, 'features')
        if not validation['is_valid']:
            self._handle_error(f"Dataset validation failed: {validation['errors']}")
        
        self.logger.info(f"Created wide dataset with {len(games_df)} games and {len(games_df.columns)} features")
        return games_df
    
    def create_ml_dataset_separate(self, 
                                  start_season: int,
                                  end_season: int,
                                  teams: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Create separate DataFrames for different data types.
        
        Args:
            start_season: First season to include
            end_season: Last season to include
            teams: Specific teams to include (None for all)
            
        Returns:
            Dictionary with separate DataFrames
        """
        self.logger.info(f"Creating separate ML datasets for seasons {start_season}-{end_season}")
        
        result = {}
        
        # Load game results
        games_df = self.load_game_results(start_season, end_season, teams)
        if not games_df.empty:
            result['games'] = self.create_target_variables(games_df)
            result['rolling_stats'] = self.calculate_rolling_stats(games_df.copy())
        
        # Load team statistics
        seasons = list(range(start_season, end_season + 1))
        team_stats = self.load_team_season_stats(seasons)
        if not team_stats.empty:
            result['team_stats'] = team_stats
        
        # Validate each dataset
        for name, df in result.items():
            validation = self.validate_data(df)
            if not validation['is_valid']:
                self.logger.warning(f"Validation issues with {name}: {validation['errors']}")
        
        self.logger.info(f"Created {len(result)} separate datasets")
        return result
