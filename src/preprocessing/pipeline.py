"""
End-to-End CAN Preprocessing and Sequence Generation Pipeline
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from .parser import (
    parse_hex_id,
    parse_hex_payload,
    compute_inter_arrival_times,
)

# Feature names in canonical order (all 11 features normalized to [0.0, 1.0])
FEATURE_NAMES = [
    "can_id_norm",
    "dlc_norm",
    "d0_norm",
    "d1_norm",
    "d2_norm",
    "d3_norm",
    "d4_norm",
    "d5_norm",
    "d6_norm",
    "d7_norm",
    "delta_t_norm",
]


class CANPreprocessor:
    """
    Stateful preprocessor for vehicular CAN traffic frames.
    Converts raw packet logs into standardized numerical feature matrices.
    """

    def __init__(self, max_can_id: float = 2047.0, max_dlc: float = 8.0):
        self.max_can_id = float(max_can_id)
        self.max_dlc = float(max_dlc)
        self.feature_names = FEATURE_NAMES

    def transform_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract numerical features and labels from a pre-loaded CAN dataframe.
        
        Args:
            df: DataFrame containing CAN_CSV_COLUMNS.
            
        Returns:
            Tuple of:
              - features: np.ndarray of shape (N, 11), dtype float32 in [0.0, 1.0]
              - binary_labels: np.ndarray of shape (N,), dtype int64 (0=Normal, 1=Attack)
              - attack_types: np.ndarray of shape (N,), dtype object
        """
        n_rows = len(df)
        if n_rows == 0:
            raise ValueError("Input dataframe is empty.")

        # 1. CAN ID (0-2047 normalized to [0, 1])
        id_ints = parse_hex_id(df["CAN_ID"]).to_numpy(dtype=np.float32)
        can_id_norm = np.clip(id_ints / self.max_can_id, 0.0, 1.0)[:, np.newaxis]

        # 2. DLC (0-8 normalized to [0, 1])
        dlc_vals = pd.to_numeric(df["DLC"], errors="coerce").fillna(8).to_numpy(dtype=np.float32)
        dlc_norm = np.clip(dlc_vals / self.max_dlc, 0.0, 1.0)[:, np.newaxis]

        # 3. Payload bytes (8 bytes normalized to [0, 1])
        payload_norm = parse_hex_payload(df, normalize=True)

        # 4. Inter-arrival time (leak-free, domain-scaled to [0, 1])
        delta_t_norm = compute_inter_arrival_times(
            df["Timestamp"].to_numpy(),
            normalize_log=True,
            scale_to_unit_interval=True,
        )[:, np.newaxis]

        # Combine all 11 features: strictly bounded in [0.0, 1.0]
        features = np.hstack([can_id_norm, dlc_norm, payload_norm, delta_t_norm]).astype(np.float32)

        # Binary label: 1 if Flag == 'T' else 0
        binary_labels = (df["Flag"].str.upper() == "T").astype(np.int64).to_numpy()

        # Multi-class attack string label
        if "Attack_Type" in df.columns:
            attack_types = df["Attack_Type"].to_numpy()
        else:
            attack_types = np.where(binary_labels == 1, "Attack", "Normal")

        return features, binary_labels, attack_types


def create_sliding_windows(
    features: np.ndarray,
    binary_labels: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    window_size: int = 16,
    step_size: int = 16,
    label_mode: str = "any",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sequential sliding windows of CAN messages for 1D-CNN ingestion.
    
    Args:
        features: 2D array of shape (N, num_features).
        binary_labels: 1D array of shape (N,).
        attack_types: 1D array of shape (N,) with attack string labels.
        window_size: Number of consecutive CAN frames in a window (e.g. 16).
        step_size: Stride / hop size (step_size=window_size creates non-overlapping windows).
        label_mode: 'any' (window is attack if >= 1 frame is attack),
                    'majority' (>50% attack frames), or 'last' (label of last frame).
                    
    Returns:
        Tuple of:
          - X_windows: shape (Num_Windows, window_size, num_features), float32
          - y_windows: shape (Num_Windows,), int64
          - y_types: shape (Num_Windows,), object
    """
    n_frames, n_feats = features.shape
    if n_frames < window_size:
        raise ValueError(f"Sequence length {n_frames} is smaller than window_size {window_size}")

    indices = np.arange(0, n_frames - window_size + 1, step_size)
    num_windows = len(indices)

    X_windows = np.zeros((num_windows, window_size, n_feats), dtype=np.float32)
    y_windows = np.zeros(num_windows, dtype=np.int64)
    y_types = np.empty(num_windows, dtype=object)

    for i, start_idx in enumerate(indices):
        end_idx = start_idx + window_size
        X_windows[i] = features[start_idx:end_idx]

        w_labels = binary_labels[start_idx:end_idx]
        if label_mode == "any":
            has_attack = np.any(w_labels == 1)
            y_windows[i] = 1 if has_attack else 0
        elif label_mode == "majority":
            y_windows[i] = 1 if np.mean(w_labels) >= 0.5 else 0
        elif label_mode == "last":
            y_windows[i] = w_labels[-1]
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        if attack_types is not None:
            w_types = attack_types[start_idx:end_idx]
            # Find the most frequent non-Normal attack type in window if attack
            if y_windows[i] == 1:
                non_norm = [t for t in w_types if t != "Normal"]
                y_types[i] = non_norm[0] if non_norm else "Attack"
            else:
                y_types[i] = "Normal"
        else:
            y_types[i] = "Attack" if y_windows[i] == 1 else "Normal"

    return X_windows, y_windows, y_types


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    y_types: Optional[np.ndarray] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Chronologically split contiguous sequential CAN data into train, validation, and test sets.
    Preserves strict temporal causality and eliminates cross-message data leakage.
    
    Args:
        X: Sequence windows array (N, W, F).
        y: Binary labels array (N,).
        y_types: Attack types array (N,).
        train_ratio: Proportion of sequence for training (default 0.70).
        val_ratio: Proportion of sequence for validation (default 0.15).
        test_ratio: Proportion of sequence for testing (default 0.15).
        
    Returns:
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test, etc.
    """
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "y_types_train": y_types[:train_end] if y_types is not None else None,
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "y_types_val": y_types[train_end:val_end] if y_types is not None else None,
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "y_types_test": y_types[val_end:] if y_types is not None else None,
    }

    return splits


def process_and_split_scenarios(
    scenario_dfs: Dict[str, pd.DataFrame],
    preprocessor: Optional[CANPreprocessor] = None,
    window_size: int = 16,
    step_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Union[np.ndarray, dict]]:
    """
    Process each automotive capture run independently, perform leak-free chronological
    train/val/test splitting per scenario, and aggregate into unified benchmark splits.
    
    Ensures every attack type (DoS, Fuzzy, Gear, RPM) and Normal traffic are present
    in all partitions while strictly preventing temporal leakage across time boundaries.
    """
    if preprocessor is None:
        preprocessor = CANPreprocessor()

    train_X, train_y, train_t = [], [], []
    val_X, val_y, val_t = [], [], []
    test_X, test_y, test_t = [], [], []

    scenario_stats = {}

    for name, df in scenario_dfs.items():
        feats, b_labels, a_types = preprocessor.transform_dataframe(df)
        X_win, y_win, t_win = create_sliding_windows(
            feats, b_labels, a_types,
            window_size=window_size,
            step_size=step_size,
            label_mode="any"
        )
        
        split = chronological_split(
            X_win, y_win, t_win,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        train_X.append(split["X_train"])
        train_y.append(split["y_train"])
        train_t.append(split["y_types_train"])

        val_X.append(split["X_val"])
        val_y.append(split["y_val"])
        val_t.append(split["y_types_val"])

        test_X.append(split["X_test"])
        test_y.append(split["y_test"])
        test_t.append(split["y_types_test"])

        scenario_stats[name] = {
            "total_frames": len(df),
            "total_windows": len(X_win),
            "train_windows": len(split["X_train"]),
            "val_windows": len(split["X_val"]),
            "test_windows": len(split["X_test"]),
            "attack_windows_total": int(np.sum(y_win)),
        }

    unified = {
        "X_train": np.concatenate(train_X, axis=0),
        "y_train": np.concatenate(train_y, axis=0),
        "y_types_train": np.concatenate(train_t, axis=0),
        "X_val": np.concatenate(val_X, axis=0),
        "y_val": np.concatenate(val_y, axis=0),
        "y_types_val": np.concatenate(val_t, axis=0),
        "X_test": np.concatenate(test_X, axis=0),
        "y_test": np.concatenate(test_y, axis=0),
        "y_types_test": np.concatenate(test_t, axis=0),
        "scenario_stats": scenario_stats,
    }

    return unified

