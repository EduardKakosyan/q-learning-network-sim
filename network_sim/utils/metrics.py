"""Metrics utilities for network simulation.

This module provides functions for calculating and analyzing network simulation metrics,
including throughput, delay, packet loss, and link utilization.
"""

import os
import json
import csv
from typing import Dict, Any, List, Optional

from network_sim.core.simulator import NetworkSimulator


def save_metrics_to_json(
    metrics: Dict[str, Any],
    output_dir: str,
    filename: str,
) -> None:
    """Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics to save.
        output_dir: Output directory.
        filename: The name for the file.
    """
    if not metrics:
        raise ValueError("Metrics dictionary is empty. Provide a non-empty dictionary of metrics.")

    # Convert non-serializable types
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == "link_utilization":
            # Convert tuple keys to strings
            serializable_metrics[key] = {
                f"{src}->{dst}": util for (src, dst), util in value.items()
            }
        elif key == "queue_lengths":
            # Convert defaultdict to dict
            serializable_metrics[key] = dict(value)
        else:
            serializable_metrics[key] = value

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{filename}.json")
    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def save_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    router_types: List[str],
    output_dir: str,
    filename: str,
) -> None:
    """Save comparison of metrics from different routers to a CSV file.

    Args:
        metrics_list: List of metrics dictionaries from different simulations.
        router_types: List of router types corresponding to metrics_list.
        output_dir: Output directory.
        filename: The filename for the file.
    """
    if not metrics_list or not router_types:
        raise ValueError("Metrics list or router types are empty.")
    if len(metrics_list) != len(router_types):
        raise ValueError(f"Length of metrics_list and router_types must match. Found lengths: {len(metrics_list)} and {len(router_types)}.")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["Router", "Throughput", "Average Delay", "Packet Loss Rate"])

        # Write data
        for i, metrics in enumerate(metrics_list):
            writer.writerow(
                [
                    router_types[i],
                    metrics.get("throughput", 0),
                    metrics.get("average_delay", 0),
                    metrics.get("packet_loss_rate", 0),
                ]
            )


def compare_routers(
    simulators: List[NetworkSimulator], output_dir: str
) -> Dict[str, List[float]]:
    """Compare metrics from different routers.

    Args:
        simulators: List of NetworkSimulator instances with different routers.
        output_dir: Directory to save output files.

    Returns:
        Dictionary of metric comparisons.
    """
    if not simulators:
        raise ValueError("Simulators list is empty.")

    metrics_list = [sim.metrics for sim in simulators]
    router_types = [sim.router_type for sim in simulators]

    # Save metrics to files
    save_metrics_to_csv(
        metrics_list, router_types, output_dir, "metrics_comparison.csv"
    )

    # Create comparison dictionary
    comparison = {
        "router_types": router_types,
        "throughput": [m.get("throughput", 0) for m in metrics_list],
        "average_delay": [m.get("average_delay", 0) for m in metrics_list],
        "packet_loss_rate": [m.get("packet_loss_rate", 0) for m in metrics_list],
    }

    return comparison


def calculate_fairness_index(
    simulator: NetworkSimulator, flow_throughputs: Optional[Dict[str, float]] = None
) -> float:
    """Calculate Jain's fairness index for flow throughputs.

    Args:
        simulator: NetworkSimulator instance.
        flow_throughputs: Dictionary mapping flow IDs to throughputs.
            If None, calculates from completed packets.

    Returns:
        Fairness index between 0 and 1 (1 is perfectly fair).
    """
    if flow_throughputs is None:
        # Calculate throughput for each flow
        flow_bytes = {}
        flow_start_times = {}
        flow_end_times = {}

        for packet in simulator.completed_packets:
            flow_id = packet.flow_id

            if flow_id not in flow_bytes:
                flow_bytes[flow_id] = 0
                flow_start_times[flow_id] = packet.creation_time
                flow_end_times[flow_id] = packet.arrival_time
            else:
                flow_bytes[flow_id] += packet.size
                flow_start_times[flow_id] = min(
                    flow_start_times[flow_id], packet.creation_time
                )
                flow_end_times[flow_id] = max(
                    flow_end_times[flow_id], packet.arrival_time
                )

        # Calculate throughput for each flow
        flow_throughputs = {}
        for flow_id, bytes_sent in flow_bytes.items():
            duration = flow_end_times[flow_id] - flow_start_times[flow_id]
            if duration > 0:
                flow_throughputs[flow_id] = bytes_sent / duration

    # Calculate Jain's fairness index
    if not flow_throughputs:
        return 0.0

    throughputs = list(flow_throughputs.values())
    n = len(throughputs)

    if n == 0:
        return 0.0

    sum_throughput = sum(throughputs)
    sum_squared = sum(x**2 for x in throughputs)

    if sum_squared == 0:
        return 0.0

    return (sum_throughput**2) / (n * sum_squared)
