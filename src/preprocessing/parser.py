"""
CAN Frame Parsing and Low-Level Feature Transformation Module
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def parse_hex_id(id_series: pd.Series) -> pd.Series:
    """
    Parse hexadecimal CAN ID strings into integer values (0 to 2047 for 11-bit standard CAN).
    
    Args:
        id_series: Series of hex strings (e.g. '0316', '018f').
        
    Returns:
        Series of integer CAN IDs.
    """
    def _hex_to_int(val):
        if pd.isna(val):
            return 0
        val_str = str(val).strip()
        try:
            return int(val_str, 16)
        except (ValueError, TypeError):
            return 0

    return id_series.apply(_hex_to_int).astype(int)


def parse_hex_payload(
    df: pd.DataFrame,
    byte_cols: Optional[list] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Parse 8 hexadecimal payload byte columns into a numpy array of uint8 or normalized float.
    
    Args:
        df: DataFrame containing the byte columns.
        byte_cols: List of column names (defaults to DATA[0]..DATA[7]).
        normalize: If True, scale values to [0.0, 1.0] by dividing by 255.0.
        
    Returns:
        2D numpy array of shape (N, 8).
    """
    if byte_cols is None:
        byte_cols = [f"DATA[{i}]" for i in range(8)]
        
    parsed_bytes = np.zeros((len(df), len(byte_cols)), dtype=np.float32 if normalize else np.uint8)
    
    for i, col in enumerate(byte_cols):
        if col in df.columns:
            def _to_byte(v):
                if pd.isna(v):
                    return 0
                v_str = str(v).strip()
                try:
                    return int(v_str, 16) & 0xFF
                except (ValueError, TypeError):
                    return 0
                    
            vals = df[col].apply(_to_byte).to_numpy()
            if normalize:
                parsed_bytes[:, i] = vals / 255.0
            else:
                parsed_bytes[:, i] = vals.astype(np.uint8)
        else:
            parsed_bytes[:, i] = 0
            
    return parsed_bytes


def compute_inter_arrival_times(
    timestamps: Union[pd.Series, np.ndarray],
    max_delta: float = 1.0,
    normalize_log: bool = True,
    scale_to_unit_interval: bool = True,
) -> np.ndarray:
    """
    Compute inter-arrival time (delta-t) between consecutive CAN frames.
    
    Scientific Fixes:
      1. Leakage-Free Initialization (Fix #4):
         At sequence index 0, no prior frame exists. The delta is deterministically
         initialized to 0.0. This eliminates the previous forward-looking leakage where
         np.median(diffs) across all 100,000 frames (including future validation/test frames)
         was imputed into frame 0.
         
      2. Domain-Based [0, 1] Normalization (Fix #3):
         log1p-transformed delta-t is normalized to [0.0, 1.0] using the fixed theoretical
         domain upper bound: log10(1.0 + max_delta * 1000.0) ≈ log10(1001.0) ≈ 3.000434.
         This matches the scale of the other 10 CAN features without relying on empirical sample
         statistics from train or test sets, preventing feature scale dominance in later
         distance-based Byzantine aggregation algorithms (e.g. Krum, Trimmed Mean).
    
    Args:
        timestamps: Series or array of float timestamps in seconds.
        max_delta: Maximum delta clipping threshold in seconds (default 1.0s).
        normalize_log: If True, apply log1p transform: log10(1 + delta_t * 1000).
        scale_to_unit_interval: If True, scale log1p values to [0.0, 1.0] using fixed domain bound.
        
    Returns:
        1D numpy array of inter-arrival times.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    deltas = np.zeros_like(ts, dtype=np.float32)
    
    if len(ts) > 1:
        diffs = np.diff(ts)
        # Handle non-monotonic timestamps or recording gaps
        diffs = np.clip(diffs, a_min=0.0, a_max=max_delta)
        deltas[1:] = diffs
        # FIX #4: Deterministic, leak-free initial state (0.0 ms)
        deltas[0] = 0.0
    elif len(ts) == 1:
        deltas[0] = 0.0
        
    if normalize_log:
        # Scale ms delta using log1p
        deltas = np.log10(1.0 + deltas * 1000.0).astype(np.float32)
        
        # FIX #3: Fixed theoretical domain normalization into [0.0, 1.0]
        if scale_to_unit_interval:
            max_log_bound = np.float32(np.log10(1.0 + max_delta * 1000.0))
            deltas = np.clip(deltas / max_log_bound, 0.0, 1.0).astype(np.float32)
            
    return deltas
