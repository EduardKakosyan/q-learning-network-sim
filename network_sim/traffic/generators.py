"""Traffic generators for network simulation.

This module provides functions for generating different types of network traffic,
including constant, variable, Poisson, and Pareto traffic patterns.
"""

import random
import numpy as np
from typing import Callable, Union


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


def bursty_traffic(
    burst_size: int, packet_interval: Union[float, Callable[[], float]]
) -> Callable[[], float]:
    """Generate bursty network traffic pattern.

    Creates traffic that comes in bursts of packets followed by quiet periods.
    Each burst contains a fixed number of packets sent at regular intervals,
    followed by a longer gap before the next burst starts.

    Args:
        burst_size: Number of packets to send in each burst
        packet_interval: Time between packets within a burst, either as:
            - Fixed float value in seconds
            - Callable that returns variable intervals

    Returns:
        Function that returns the time interval until the next packet should be sent:
        - Returns 0 for first packet in burst (send immediately)
        - Returns packet_interval for subsequent packets in burst
        - Returns packet_interval * burst_size for gap between bursts
    """
    get_interval = (
        packet_interval if callable(packet_interval) else lambda: packet_interval
    )

    in_burst = False
    packets = 0

    def next_packet_delay() -> float:
        nonlocal in_burst, packets
        interval = get_interval()

        if not in_burst:
            # start new burst
            in_burst = True
            packets = 1
            return 0

        if packets < burst_size:
            # continue burst
            packets += 1
            return interval
        else:
            # end burst
            in_burst = False
            packets = 0
            return interval * burst_size

    return next_packet_delay


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
