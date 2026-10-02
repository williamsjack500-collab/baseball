import pandas as pd
import requests
import json
import re
import os

def save_park_dimensions_to_parquet(season: int, output_dir: str = 'Data/ballpark_dims'):
    """
    Scrapes Statcast park dimensions for a given season, extracts venue identifiers 
    and the 5 core fence measurements, and saves the result as a Parquet file.
    
    Parameters:
        season (int): The MLB season year (e.g., 2024, 2025, 2026).
        output_dir (str): Directory where the Parquet file should be saved.
        
    Returns:
        str: File path of the saved Parquet file.
    """
    url = f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=dimensions&year={season}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    match = re.search(r'var data = (\[.*?\]);', response.text)

    if not match:
        raise ValueError(f"Could not locate park dimension data for season {season}.")

    raw_data = json.loads(match.group(1))
    
    # Target venue identifiers and the 5 outfield distance markers
    cols = [
        'venue_id', 'season_id', 'team_id', 'name_display_club', 
        'distance_lf_line', 'distance_lf_gap', 'distance_cf', 
        'distance_rf_gap', 'distance_rf_line'
    ]
    
    # Filter to available columns
    park_dims = pd.DataFrame(raw_data)
    selected_cols = [c for c in cols if c in park_dims.columns]
    park_dims = park_dims[selected_cols]

    # Convert dimension values to numeric floats
    dim_cols = [c for c in ['distance_lf_line', 'distance_lf_gap', 'distance_cf', 'distance_rf_gap', 'distance_rf_line'] if c in park_dims.columns]
    park_dims[dim_cols] = park_dims[dim_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Parquet
    filepath = os.path.join(output_dir, f"park_dims_{season}.parquet")
    park_dims.to_parquet(filepath, index=False)
    
    print(f"Saved {len(park_dims)} venue dimension records to '{filepath}'")
    return