import pandas as pd

def detect_segments(df: pd.DataFrame, hz_est: int, threshold: float = 0.5, min_window: int = 60) -> list:
    """
    Scans accelerometer data and returns a list of (start_idx, end_idx) 
    tuples representing segments where motion was detected.
    """
    # Calculate moving standard deviation to find areas of motion
    rolling_win = max(10, hz_est // 2)
    motion_var = (df['x'].rolling(rolling_win).std().fillna(0) +
                  df['y'].rolling(rolling_win).std().fillna(0) +
                  df['z'].rolling(rolling_win).std().fillna(0))
    
    # Convert to boolean numpy array for super-fast looping
    is_moving = (motion_var > threshold).to_numpy()
    
    segments = []
    in_segment = False
    start_idx = 0
    silence_counter = 0
    max_silence = hz_est  # Allow up to 1 second of silence before cutting the segment
    
    for i in range(len(is_moving)):
        if is_moving[i]:
            if not in_segment:
                in_segment = True
                start_idx = i
            silence_counter = 0 # reset silence countdown
        else:
            if in_segment:
                silence_counter += 1
                # If it's been quiet for too long, cut the clip!
                if silence_counter > max_silence:
                    end_idx = i - max_silence
                    
                    # Only keep the segment if it's long enough for the model
                    if (end_idx - start_idx) >= min_window:
                        segments.append((start_idx, end_idx))
                    
                    in_segment = False

    # Edge case: file ends while still in an active segment
    if in_segment and (len(is_moving) - start_idx) >= min_window:
        segments.append((start_idx, len(is_moving) - 1))
        
    return segments