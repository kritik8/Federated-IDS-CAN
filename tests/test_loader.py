"""
Unit tests for automotive CAN CSV loader and variable DLC parsing.
"""

import tempfile
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

from src.data.loader import load_csv_dataset, validate_csv_file, CAN_CSV_COLUMNS


class TestVariableDLCLoader(unittest.TestCase):
    """Test suite for dynamic variable DLC CSV parsing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "synthetic_can_test.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_synthetic_csv(self, lines):
        with open(self.csv_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def test_variable_dlc_parsing_8_5_2(self):
        """Verify DLC=8, DLC=5, and DLC=2 parse without column shifting or corruption."""
        lines = [
            # DLC = 8: 12 tokens
            "1478195721.000100,0316,8,05,28,84,66,6d,00,00,a2,R",
            # DLC = 5: 9 tokens
            "1478195721.000200,02b0,5,ff,7f,00,05,49,R",
            # DLC = 2: 6 tokens
            "1478195721.000300,05f0,2,01,00,T",
        ]
        self.create_synthetic_csv(lines)

        df, stats = load_csv_dataset(self.csv_path, return_stats=True)

        self.assertEqual(len(df), 3)
        self.assertEqual(stats["valid_rows"], 3)
        self.assertEqual(stats["malformed_rows"], 0)
        self.assertEqual(stats["rows_dlc_lt_8"], 2)
        self.assertEqual(stats["rows_dlc_eq_8"], 1)

        # Row 0: DLC = 8
        row0 = df.iloc[0]
        self.assertAlmostEqual(row0["Timestamp"], 1478195721.000100, places=5)
        self.assertEqual(row0["CAN_ID"], "0316")
        self.assertEqual(row0["DLC"], 8)
        self.assertEqual(row0["DATA[0]"], "05")
        self.assertEqual(row0["DATA[4]"], "6d")
        self.assertEqual(row0["DATA[7]"], "a2")
        self.assertEqual(row0["Flag"], "R")
        self.assertEqual(row0["Attack_Type"], "Normal")

        # Row 1: DLC = 5 -> padded with 3 '00's, Flag is 'R'
        row1 = df.iloc[1]
        self.assertAlmostEqual(row1["Timestamp"], 1478195721.000200, places=5)
        self.assertEqual(row1["CAN_ID"], "02b0")
        self.assertEqual(row1["DLC"], 5)
        self.assertEqual(row1["DATA[0]"], "ff")
        self.assertEqual(row1["DATA[1]"], "7f")
        self.assertEqual(row1["DATA[2]"], "00")
        self.assertEqual(row1["DATA[3]"], "05")
        self.assertEqual(row1["DATA[4]"], "49")
        self.assertEqual(row1["DATA[5]"], "00")  # Padded
        self.assertEqual(row1["DATA[6]"], "00")  # Padded
        self.assertEqual(row1["DATA[7]"], "00")  # Padded
        self.assertEqual(row1["Flag"], "R")      # Flag correctly preserved
        self.assertEqual(row1["Attack_Type"], "Normal")

        # Row 2: DLC = 2 -> padded with 6 '00's, Flag is 'T' (Attack)
        row2 = df.iloc[2]
        self.assertAlmostEqual(row2["Timestamp"], 1478195721.000300, places=5)
        self.assertEqual(row2["CAN_ID"], "05f0")
        self.assertEqual(row2["DLC"], 2)
        self.assertEqual(row2["DATA[0]"], "01")
        self.assertEqual(row2["DATA[1]"], "00")
        for i in range(2, 8):
            self.assertEqual(row2[f"DATA[{i}]"], "00")
        self.assertEqual(row2["Flag"], "T")
        # Attack type assigned
        self.assertNotEqual(row2["Attack_Type"], "Normal")

    def test_flag_never_consumed_by_payload(self):
        """Verify that Flag 'R' or 'T' is never mistaken for a payload byte."""
        lines = [
            "1478195721.100000,018f,3,aa,bb,cc,R",
            "1478195721.200000,018f,3,aa,bb,cc,T",
        ]
        self.create_synthetic_csv(lines)
        df = load_csv_dataset(self.csv_path)

        self.assertEqual(df.iloc[0]["DATA[3]"], "00")
        self.assertEqual(df.iloc[0]["Flag"], "R")
        self.assertEqual(df.iloc[1]["DATA[3]"], "00")
        self.assertEqual(df.iloc[1]["Flag"], "T")

    def test_malformed_rows_handling(self):
        """Verify that corrupted/truncated lines are caught and recorded in statistics."""
        lines = [
            "1478195721.100000,0316,8,05,28,84,66,6d,00,00,a2,R",  # Valid
            "corrupted line with no commas",                         # Malformed
            "1478195721.200000,02b0",                                # Too few fields
            "not_a_ts,02b0,2,01,00,R",                              # Invalid timestamp
            "1478195721.300000,02b0,5,01,02,R",                     # Stated DLC=5 but only 2 payload tokens
            "1478195721.400000,02b0,2,01,00,INVALID_FLAG",           # Invalid flag
            "1478195721.500000,05f0,2,01,00,T",                     # Valid
        ]
        self.create_synthetic_csv(lines)
        df, stats = load_csv_dataset(self.csv_path, return_stats=True)

        self.assertEqual(stats["total_rows_read"], 7)
        self.assertEqual(stats["valid_rows"], 2)
        self.assertEqual(stats["malformed_rows"], 5)
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
