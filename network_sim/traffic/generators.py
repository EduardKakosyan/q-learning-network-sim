"""Traffic generators for network simulation.

This module provides functions for generating different types of network traffic,
including constant, variable, Poisson, and Pareto traffic patterns.
"""

import random
import numpy as np
from typing import Callable


def constant_traffic(rate: float) -> Callable[[], float]:
    """Generate constant bit rate traffic.

    Args:
        rate: Rate of packet generation in packets per second.

    Returns:
        Function that returns constant interval between packets.
    """
    return lambda: 1 / rate


def variable_traffic(min_rate: float, max_rate: float) -> Callable[[], float]:
    """Generate variable bit rate traffic.

    Args:
        min_rate: Minimum rate of packet generation in packets per second.
        max_rate: Maximum rate of packet generation in packets per second.

    Returns:
        Function that returns variable interval between packets.
    """
    return lambda: 1 / random.uniform(min_rate, max_rate)


def poisson_traffic(rate: float) -> Callable[[], float]:
    """Generate Poisson traffic.

    Args:
        rate: Average rate of packet generation in packets per second.

    Returns:
        Function that returns exponentially distributed interval between packets.
    """
    return lambda: np.random.exponential(1 / rate)


def pareto_traffic(rate: float, alpha: float = 1.5) -> Callable[[], float]:
    """Generate Pareto (heavy-tailed) traffic.

    Args:
        rate: Average rate of packet generation in packets per second.
        alpha: Shape parameter for Pareto distribution (default: 1.5).

    Returns:
        Function that returns Pareto distributed interval between packets.
    """
    scale = (alpha - 1) / (alpha * rate)
    return lambda: np.random.pareto(alpha) * scale


def constant_size(size: int) -> Callable[[], int]:
    """Generate constant size packets.

    Args:
        size: Size of packets in bytes.

    Returns:
        Function that returns constant packet size.
    """
    return lambda: size


def variable_size(min_size: int, max_size: int) -> Callable[[], int]:
    """Generate variable size packets.

    Args:
        min_size: Minimum size of packets in bytes.
        max_size: Maximum size of packets in bytes.

    Returns:
        Function that returns random packet size between min_size and max_size.
    """
    return lambda: random.randint(min_size, max_size)


def bimodal_size(
    small_size: int, large_size: int, small_prob: float = 0.7
) -> Callable[[], int]:
    """Generate bimodal packet sizes (e.g., small and large packets).

    Args:
        small_size: Size of small packets in bytes.
        large_size: Size of large packets in bytes.
        small_prob: Probability of generating a small packet (default: 0.7).

    Returns:
        Function that returns either small_size or large_size with probability small_prob.
    """
    return lambda: small_size if random.random() < small_prob else large_size
