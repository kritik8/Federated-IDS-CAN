"""
Models package initialization
"""

from .cnn1d import CAN1DCNN, count_parameters

__all__ = [
    "CAN1DCNN",
    "count_parameters",
]
