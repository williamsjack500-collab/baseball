import pandas as pd

def prepare_batted_ball_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the Statcast DataFrame for balls in play (type == 'X')
    and creates the binary 'is_hit' target column (1 for hits, 0 for outs).
    
    Parameters:
        df (pd.DataFrame): Raw Statcast DataFrame.
        
    Returns:
        pd.DataFrame: Processed DataFrame containing only balls in play with 'is_hit'.
    """
    hit_events = ['single', 'double', 'triple', 'home_run']
    
    # Filter for balls in play
    batted_balls = df[df['type'] == 'X'].copy()
    
    # Assign binary hit outcome target
    batted_balls['is_hit'] = batted_balls['events'].isin(hit_events).astype(int)
    
    return batted_balls