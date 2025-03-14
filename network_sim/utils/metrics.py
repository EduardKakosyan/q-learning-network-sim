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
    metrics: Dict[str, Any], filename: str = "results/metrics.json"
) -> None:
    """Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics to save.
        filename: Output filename.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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

    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=2)


def save_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    scheduler_types: List[str],
    filename: str = "results/metrics_comparison.csv",
) -> None:
    """Save comparison of metrics from different schedulers to a CSV file.

    Args:
        metrics_list: List of metrics dictionaries from different simulations.
        scheduler_types: List of scheduler types corresponding to metrics_list.
        filename: Output filename.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            ["Scheduler", "Throughput", "Average Delay", "Packet Loss Rate"]
        )

        # Write data
        for i, metrics in enumerate(metrics_list):
            writer.writerow(
                [
                    scheduler_types[i],
                    metrics["throughput"],
                    metrics["average_delay"],
                    metrics["packet_loss_rate"],
                ]
            )


def compare_schedulers(
    simulators: List[NetworkSimulator], output_dir: str = "results"
) -> Dict[str, List[float]]:
    """Compare metrics from different schedulers.

    Args:
        simulators: List of NetworkSimulator instances with different schedulers.
        output_dir: Directory to save output files.

    Returns:
        Dictionary of metric comparisons.
    """
    metrics_list = [sim.metrics for sim in simulators]
    scheduler_types = [sim.scheduler.name for sim in simulators]

    # Save metrics to files
    save_metrics_to_csv(
        metrics_list, scheduler_types, f"{output_dir}/metrics_comparison.csv"
    )

    for i, sim in enumerate(simulators):
        save_metrics_to_json(
            sim.metrics, f"{output_dir}/{sim.scheduler.name.lower()}_metrics.json"
        )

    # Create comparison dictionary
    comparison = {
        "scheduler_types": scheduler_types,
        "throughput": [m["throughput"] for m in metrics_list],
        "average_delay": [m["average_delay"] for m in metrics_list],
        "packet_loss_rate": [m["packet_loss_rate"] for m in metrics_list],
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
