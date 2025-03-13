"""Enumerations for network simulation.

This module defines enumerations used throughout the network simulator.
"""

from enum import Enum


class TrafficPattern(Enum):
    """Enum for different traffic patterns.

    Attributes:
        CONSTANT: Constant bit rate traffic.
        VARIABLE: Variable bit rate traffic.
        BURSTY: Bursty traffic with periods of high activity.
        MIXED: Combination of different traffic patterns.
    """

    CONSTANT = 1
    VARIABLE = 2
    BURSTY = 3
    MIXED = 4
