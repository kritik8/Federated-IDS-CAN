"""
Automotive CAN Dataset Loader Module
Supports HCRL Car-Hacking Dataset CSV format (with dynamic variable DLC parsing)
and normal run TXT format.
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
    pad_byte: str = "00",
    return_stats: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Load an HCRL CAN dataset CSV file using dynamic, variable-DLC parsing.
    
    In CAN networks, frames can carry fewer than 8 data bytes (e.g. DLC=2, DLC=5).
    This function splits each line dynamically by comma, ensuring that:
      1. Timestamp, CAN_ID, and DLC are correctly identified.
      2. The payload bytes are parsed strictly according to the stated DLC.
      3. Missing payload positions up to 8 bytes are padded with `pad_byte` (default '00').
      4. The Flag ('R' or 'T') is always extracted as the final field and never consumed
         into payload data.
      5. Malformed rows are identified, logged in stats, and safely skipped without corruption.
      
    Args:
        filepath: Path to the CSV file.
        nrows: Optional limit on number of valid rows to read.
        attack_name: Name of attack (e.g. 'DoS', 'Fuzzy') to attach as metadata.
        pad_byte: Hex string used to pad missing payload bytes up to 8 bytes (default '00').
        return_stats: If True, returns tuple of (DataFrame, stats_dict).
        verbose: If True, prints parsing validation statistics.
        
    Returns:
        DataFrame with standardized columns:
        Timestamp, CAN_ID, DLC, DATA[0]..DATA[7], Flag, Attack_Type
        (or tuple of (DataFrame, stats_dict) if return_stats=True).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    total_rows = 0
    valid_rows = 0
    malformed_rows = 0
    dlc_dist: Dict[int, int] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if nrows is not None and valid_rows >= nrows:
                break
                
            line_str = line.strip()
            if not line_str:
                continue
                
            total_rows += 1
            tokens = [t.strip() for t in line_str.split(",")]
            
            # Minimum structure: Timestamp, CAN_ID, DLC, Flag (at least 4 tokens)
            if len(tokens) < 4:
                malformed_rows += 1
                continue
                
            try:
                # 1. Timestamp (float)
                ts = float(tokens[0])
                
                # 2. CAN ID (hex string, lowercased)
                can_id = tokens[1].lower()
                
                # 3. DLC (integer)
                dlc = int(tokens[2])
                
                # 4. Flag (final field: 'R' for Regular/Normal, 'T' for Target/Attack)
                flag = tokens[-1].upper()
                if flag not in ("R", "T"):
                    # Malformed flag
                    malformed_rows += 1
                    continue
                    
                # 5. Payload bytes: strictly tokens between DLC and Flag
                payload = tokens[3:-1]
                
                # Validate payload length vs stated DLC
                if len(payload) != dlc:
                    if len(payload) > dlc:
                        payload = payload[:dlc]
                    else:
                        malformed_rows += 1
                        continue

                # 6. Pad missing payload positions up to 8 bytes with pad_byte ('00')
                if len(payload) < 8:
                    payload = payload + [pad_byte] * (8 - len(payload))
                else:
                    payload = payload[:8]

                records.append([ts, can_id, dlc] + payload + [flag])
                valid_rows += 1
                dlc_dist[dlc] = dlc_dist.get(dlc, 0) + 1
                
            except (ValueError, TypeError, IndexError):
                malformed_rows += 1
                continue

    columns = CAN_CSV_COLUMNS.copy()
    df = pd.DataFrame(records, columns=columns)
    
    # Type standardization
    df["Timestamp"] = df["Timestamp"].astype(np.float64)
    df["CAN_ID"] = df["CAN_ID"].astype(str)
    df["DLC"] = df["DLC"].astype(int)
    for col in DATA_BYTE_COLS:
        df[col] = df[col].astype(str)
    df["Flag"] = df["Flag"].astype(str)
    
    # Set Attack_Type column
    attack_label = attack_name if attack_name else path.stem.replace("_dataset", "")
    df["Attack_Type"] = np.where(df["Flag"] == "T", attack_label, "Normal")

    stats = {
        "file": str(path),
        "total_rows_read": total_rows,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "dlc_distribution": dict(sorted(dlc_dist.items())),
        "rows_dlc_lt_8": sum(cnt for d, cnt in dlc_dist.items() if d < 8),
        "rows_dlc_eq_8": dlc_dist.get(8, 0),
    }
    
    df.attrs["parsing_stats"] = stats

    if verbose:
        print(f"=== CSV Ingestion Summary: {path.name} ===")
        print(f"  Total rows read: {total_rows:,}")
        print(f"  Valid rows:      {valid_rows:,}")
        print(f"  Malformed rows:  {malformed_rows:,}")
        print(f"  DLC distribution: {stats['dlc_distribution']}")
        print(f"  Rows with DLC < 8: {stats['rows_dlc_lt_8']:,}")
        print(f"  Rows with DLC = 8: {stats['rows_dlc_eq_8']:,}")

    if return_stats:
        return df, stats
    return df


def validate_csv_file(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
) -> Dict:
    """
    Quick validation helper to inspect CSV structure and variable DLC statistics.
    """
    _, stats = load_csv_dataset(filepath, nrows=nrows, return_stats=True, verbose=False)
    return stats


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
    
    # Type standardization
    df["Timestamp"] = df["Timestamp"].astype(np.float64)
    df["CAN_ID"] = df["CAN_ID"].astype(str)
    df["DLC"] = df["DLC"].astype(int)
    for col in DATA_BYTE_COLS:
        df[col] = df[col].astype(str)
    df["Flag"] = df["Flag"].astype(str)
    df["Attack_Type"] = df["Attack_Type"].astype(str)
    
    return df
