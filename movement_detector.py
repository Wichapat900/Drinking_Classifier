import pandas as pd
import numpy as np

def detect_segments(df: pd.DataFrame, hz_est: int, threshold: float = 0.15, min_window: int = 60) -> list:
    """
    Splits the data based on large time gaps, then trims silence from the edges
    to perfectly isolate the active drinking 'wiggles'.
    """
    segments = []
    
    # 1. Break the data into continuous blocks based on time gaps.
    # If there's a gap of more than 2 seconds between rows, it's a new recording block.
    dt = df['seconds'].diff().fillna(0)
    break_points = df.index[dt > 2.0].tolist()
    
    blocks = []
    start_b = 0
    for bp in break_points:
        blocks.append((start_b, bp - 1))
        start_b = bp
    blocks.append((start_b, len(df) - 1))
    
    # 2. Within each block, find the actual movement (trimming the silence at start/end)
    rolling_win = max(10, hz_est // 2)
    
    for b_start, b_end in blocks:
        if (b_end - b_start) < min_window:
            continue  # Skip blocks that are too short to be a real drink
            
        block_df = df.iloc[b_start:b_end+1]
        
        # Calculate motion variance for this specific block
        motion_var = (block_df['x'].rolling(rolling_win).std().fillna(0) +
                      block_df['y'].rolling(rolling_win).std().fillna(0) +
                      block_df['z'].rolling(rolling_win).std().fillna(0))
        
        is_moving = (motion_var > threshold).to_numpy()
        
        # Find all rows where motion exceeded our threshold
        moving_indices = np.where(is_moving)[0]
        
        if len(moving_indices) > 0:
            # seg_start: The first index where it moved
            seg_start = b_start + moving_indices[0]
            # Back up slightly to capture the very beginning of the motion
            seg_start = max(b_start, seg_start - (rolling_win // 2)) 
            
            # seg_end: The last index where it moved
            seg_end = b_start + moving_indices[-1]
            
            # Ensure the segment is long enough for the ML model
            if (seg_end - seg_start) >= min_window:
                segments.append((seg_start, seg_end))
                
    return segments