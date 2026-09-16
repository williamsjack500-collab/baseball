from pybaseball import statcast
from pybaseball import statcast_sprint_speed
import pandas as pd
import numpy as np
import os

PERTINENT_COLUMNS = [
    # Game & At-Bat Scenario
    'game_date', 'game_type', 'game_pk', 'at_bat_number', 'pitch_number', 
    'inning', 'inning_topbot', 'outs_when_up', 'balls', 'strikes',
    'on_3b', 'on_2b', 'on_1b', 'home_score', 'away_score', 'post_home_score', 'post_away_score',
    
    # Player Identifiers & Matchup
    'pitcher', 'batter', 'p_throws', 'stand',
    
    # Pitch Characteristics & Location
    'pitch_type', 'pitch_name', 'release_speed', 'release_spin_rate', 
    'spin_axis', 'release_pos_x', 'release_pos_z', 'release_extension', 
    'plate_x', 'plate_z', 'zone', 'pfx_x', 'pfx_z', 
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot',
    
    # Pitch Outcome
    'events', 'description', 'type',
    
    # Hit & Batted Ball Data
    'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type', 
    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
    'hc_x', 'hc_y',
    
    # Defensive Positioning Data
    'if_fielding_alignment', 'of_fielding_alignment'
]

def add_sprint_speed(df: pd.DataFrame, year: int, min_opp: int = 1) -> pd.DataFrame:

    # 1. Fetch the sprint speed leaderboard for the given season
    speed_df = statcast_sprint_speed(year, min_opp=min_opp)
    
    # 2. Select relevant columns and align player_id with the 'batter' column
    speed_df = speed_df[['player_id', 'sprint_speed']].rename(
        columns={'player_id': 'batter'}
    )
    
    # 3. Left merge back onto the main pitch-by-pitch Statcast DataFrame
    df_merged = df.merge(speed_df, on='batter', how='left')
    
    return df_merged

def add_spray_angle(df: pd.DataFrame) -> pd.DataFrame:
    # Create the column populated with NaN by default
    df['spray_angle'] = np.nan
    
    # Identify rows where both coordinates are present
    valid_mask = df['hc_x'].notna() & df['hc_y'].notna()
    
    if valid_mask.any():
        # Translate coordinates relative to home plate
        dx = df.loc[valid_mask, 'hc_x'] - 125
        dy = 198.27 - df.loc[valid_mask, 'hc_y']
        
        # Calculate spray angle in degrees for valid rows only
        df.loc[valid_mask, 'spray_angle'] = np.degrees(np.arctan2(dx, dy))
        
    return df

def fetch_and_save_season_statcast(year: int, output_dir: str = 'Data', columns: list = PERTINENT_COLUMNS) -> None:
    """
    Pulls Statcast pitch data for an entire year and exports separate Parquet files
    for Spring Training, Regular Season, and Postseason games.
    
    Parameters:
        year (int): The season year (e.g., 2024 or 2025).
        output_dir (str): Directory where Parquet files will be saved.
        columns (list): List of columns to retain.
        
    Returns:
        dict: A dictionary containing the split DataFrames for each category.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    df = statcast(start_dt=start_date, end_dt=end_date)
    
    if df.empty:
        print(f"No data returned for year {year}.")
        return {}

    # Filter for desired columns available in dataset
    available_cols = [col for col in PERTINENT_COLUMNS if col in df.columns]
    df = df[available_cols]

    # Add column for batter sprint speed
    df = add_sprint_speed(df, year)

    #Add columns for spray angle
    df = add_spray_angle(df)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define Game Type Groups
    # 'S' = Spring Training, 'E' = Exhibition
    # 'R' = Regular Season
    # 'F' (Wild Card), 'D' (Division Series), 'L' (LCS), 'W' (World Series) = Postseason
    postseason_types = ['F', 'D', 'L', 'W']
    
    splits = {
        f"{year}_spring_training.parquet": df[df['game_type'].isin(['S', 'E'])],
        f"{year}_regular_season.parquet": df[df['game_type'] == 'R'],
        f"{year}_postseason.parquet": df[df['game_type'].isin(postseason_types)]
    }

    for filename, split_df in splits.items():
        file_path = os.path.join(output_dir, filename)
        split_df.to_parquet(file_path, index=False)
        category = filename.replace(f"{year}_", "").replace(".parquet", "").replace("_", " ").title()
        print(f"Saved {len(split_df):,} rows to '{file_path}' ({category}).")
# Usage:
# fetch_and_save_season_statcast(2025, output_dir='/baseball/Data')

def read_parquet_to_df(file_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the Parquet file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from '{file_path}'.")
        return df
    except Exception as e:
        print(f"Error reading Parquet file at '{file_path}': {e}")
        raise e