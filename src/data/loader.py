"""
Automotive CAN Dataset Loader Module
Supports HCRL Car-Hacking Dataset CSV format and normal run TXT format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import pandas as pd
import numpy as np

# Standard HCRL column names
CAN_CSV_COLUMNS = [
    "Timestamp",
    "CAN_ID",
    "DLC",
    "DATA[0]",
    "DATA[1]",
    "DATA[2]",
    "DATA[3]",
    "DATA[4]",
    "DATA[5]",
    "DATA[6]",
    "DATA[7]",
    "Flag",
]

DATA_BYTE_COLS = [f"DATA[{i}]" for i in range(8)]


def discover_raw_datasets(raw_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Discover all available raw CAN dataset files in the raw data directory.
    
    Args:
        raw_dir: Path to directory containing raw data.
        
    Returns:
        Dictionary mapping dataset identifier to Path.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
        
    discovered = {}
    
    # Check for known HCRL CSV files
    for csv_file in raw_path.glob("*.csv"):
        name_lower = csv_file.stem.lower()
        if "dos" in name_lower:
            discovered["DoS"] = csv_file
        elif "fuzzy" in name_lower:
            discovered["Fuzzy"] = csv_file
        elif "rpm" in name_lower:
            discovered["RPM"] = csv_file
        elif "gear" in name_lower:
            discovered["Gear"] = csv_file
        else:
            discovered[csv_file.stem] = csv_file
            
    # Check for normal run TXT file
    for txt_file in raw_path.glob("*.txt"):
        if "normal" in txt_file.stem.lower():
            discovered["Normal"] = txt_file
            
    return discovered


def load_csv_dataset(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    attack_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load an HCRL CAN dataset CSV file.
    
    Args:
        filepath: Path to the CSV file.
        nrows: Optional limit on number of rows to read.
        attack_name: Name of attack (e.g. 'DoS', 'Fuzzy') to attach as metadata.
        
    Returns:
        DataFrame with standardized columns:
        Timestamp, CAN_ID, DLC, DATA[0]..DATA[7], Flag, Attack_Type
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read CSV without header using defined schema
    df = pd.read_csv(
        path,
        header=None,
        names=CAN_CSV_COLUMNS,
        nrows=nrows,
        dtype=str,
        low_memory=False,
    )
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            
    # Convert Timestamp to float
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    
    # Convert DLC to integer
    df["DLC"] = pd.to_numeric(df["DLC"], errors="coerce").fillna(8).astype(int)
    
    # Clean Flag column (typically 'R' for Regular / Normal, 'T' for Target / Attack)
    df["Flag"] = df["Flag"].str.upper()
    
    # Set Attack_Type column
    attack_label = attack_name if attack_name else path.stem.replace("_dataset", "")
    df["Attack_Type"] = np.where(df["Flag"] == "T", attack_label, "Normal")
    
    return df


def load_txt_normal(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parse HCRL normal_run_data.txt format.
    Format example:
    Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
    
    Args:
        filepath: Path to the TXT file.
        nrows: Optional limit on number of lines to read.
        
    Returns:
        DataFrame with standardized columns matching CSV schema.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Normal run file not found: {path}")

    records = []
    line_count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if nrows is not None and line_count >= nrows:
                break
            line = line.strip()
            if not line:
                continue
                
            # Regex parse standard normal log line
            # Timestamp: <ts>  ID: <id>  <extra>  DLC: <dlc>  <b0> <b1> <b2> ...
            match = re.match(
                r"Timestamp:\s+([\d\.]+)\s+ID:\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]*\s*DLC:\s+(\d+)\s+(.*)",
                line,
            )
            if match:
                ts, can_id, dlc_str, data_str = match.groups()
                bytes_list = data_str.strip().split()
                # Pad to 8 bytes if fewer
                while len(bytes_list) < 8:
                    bytes_list.append("00")
                bytes_list = bytes_list[:8]
                
                record = [float(ts), can_id.lower(), int(dlc_str)] + bytes_list + ["R", "Normal"]
                records.append(record)
                line_count += 1
            else:
                # Fallback simple split if regex misses
                parts = line.split()
                if len(parts) >= 6 and "Timestamp:" in parts:
                    ts_idx = parts.index("Timestamp:") + 1
                    id_idx = parts.index("ID:") + 1
                    dlc_idx = parts.index("DLC:") + 1
                    ts = float(parts[ts_idx])
                    can_id = parts[id_idx].lower()
                    dlc = int(parts[dlc_idx])
                    data_bytes = parts[dlc_idx + 1: dlc_idx + 1 + 8]
                    while len(data_bytes) < 8:
                        data_bytes.append("00")
                    data_bytes = data_bytes[:8]
                    records.append([ts, can_id, dlc] + data_bytes + ["R", "Normal"])
                    line_count += 1

    columns = CAN_CSV_COLUMNS + ["Attack_Type"]
    df = pd.DataFrame(records, columns=columns)
    return df
