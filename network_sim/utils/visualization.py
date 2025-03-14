"""Visualization utilities for network simulation.

This module provides functions for visualizing network simulation results,
including network topology, traffic patterns, and performance metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pydot # This import is just to show that it is being used by the project.

from network_sim.core.simulator import NetworkSimulator


def save_network_visualization(
    simulator: NetworkSimulator, filename: str, figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Save network topology visualization to a file.

    Args:
        simulator: NetworkSimulator instance.
        filename: Output filename.
        figsize: Figure size as (width, height) in inches.
    """
    fig = plt.figure(figsize=figsize)

    graph = simulator.graph
    pos = nx.nx_pydot.graphviz_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")

    edge_capacities = [
        graph[u][v]["capacity"] / 1000000 for u, v in graph.edges()
    ]
    max_capacity = max(edge_capacities) if edge_capacities else 1
    edge_widths = [cap / max_capacity * 3 for cap in edge_capacities]

    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        alpha=0.7,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
    )

    nx.draw_networkx_labels(graph, pos, font_size=12)

    edge_labels = {
        (u, v): f"{graph[u][v]['capacity']/1000000:.1f}Mbps\n{graph[u][v]['delay']*1000:.1f}ms"
        for u, v in graph.edges()
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8
    )

    plt.title(f"Network Topology (Router: {simulator.router_type})")
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)


def plot_metrics(
    metrics_list: List[Dict[str, Any]],
    router_types: List[str],
    output_dir: str = "results",
) -> None:
    """Plot and save performance metrics for different routers.

    Args:
        metrics_list: List of metrics dictionaries from different simulations.
        router_types: List of router types corresponding to metrics_list.
        output_dir: Directory to save output plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Throughput comparison
    plt.figure(figsize=(10, 6))
    throughputs = [m["throughput"] for m in metrics_list]
    plt.bar(router_types, throughputs)
    plt.title("Throughput Comparison")
    plt.ylabel("Throughput (bytes/second)")
    plt.xlabel("Router Type")
    plt.savefig(f"{output_dir}/throughput_comparison.png")
    plt.close()

    # Delay comparison
    plt.figure(figsize=(10, 6))
    delays = [m["average_delay"] for m in metrics_list]
    plt.bar(router_types, delays)
    plt.title("Average Delay Comparison")
    plt.ylabel("Average Delay (seconds)")
    plt.xlabel("Router Type")
    plt.savefig(f"{output_dir}/delay_comparison.png")
    plt.close()

    # Packet loss rate comparison
    plt.figure(figsize=(10, 6))
    loss_rates = [m["packet_loss_rate"] for m in metrics_list]
    plt.bar(router_types, loss_rates)
    plt.title("Packet Loss Rate Comparison")
    plt.ylabel("Packet Loss Rate")
    plt.xlabel("Router Type")
    plt.savefig(f"{output_dir}/loss_rate_comparison.png")
    plt.close()


def plot_link_utilization(
    simulator: NetworkSimulator, output_file: str = "results/link_utilization.png"
) -> None:
    """Plot and save link utilization.

    Args:
        simulator: NetworkSimulator instance.
        output_file: Output filename.
    """
    link_utilization = simulator.metrics["link_utilization"]
    if not link_utilization:
        return

    links = [f"{src}->{dst}" for (src, dst) in link_utilization.keys()]
    utilizations = list(link_utilization.values())

    plt.figure(figsize=(12, 6))
    plt.bar(links, utilizations)
    plt.title(f"Link Utilization (Router: {simulator.router_type})")
    plt.ylabel("Utilization")
    plt.xlabel("Link")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()


def plot_packet_journey(
    simulator: NetworkSimulator,
    packet_ids: Optional[List[int]] = None,
    max_packets: int = 10,
    output_file: str = "results/packet_journey.png",
) -> None:
    """Plot and save packet journey through the network.

    Args:
        simulator: NetworkSimulator instance.
        packet_ids: List of packet IDs to plot (if None, selects random completed packets).
        max_packets: Maximum number of packets to plot.
        output_file: Output filename.
    """
    completed_packets = simulator.completed_packets
    if not completed_packets:
        return

    if packet_ids is None:
        # Select random packets if not specified
        if len(completed_packets) > max_packets:
            packets_to_plot = np.random.choice(
                completed_packets, max_packets, replace=False
            )
        else:
            packets_to_plot = completed_packets
    else:
        # Filter packets by ID
        packets_to_plot = [p for p in completed_packets if p.id in packet_ids]

    plt.figure(figsize=(12, 8))

    for i, packet in enumerate(packets_to_plot):
        nodes = [hop[0] for hop in packet.hops]
        times = [hop[1] for hop in packet.hops]

        # Add source node
        nodes.insert(0, packet.source)
        times.insert(0, packet.creation_time)

        plt.plot(times, nodes, "o-", label=f"Packet {packet.id}")

        # Add delay annotations
        for j in range(1, len(nodes)):
            delay = times[j] - times[j - 1]
            plt.annotate(
                f"{delay:.3f}s",
                xy=((times[j] + times[j - 1]) / 2, (nodes[j] + nodes[j - 1]) / 2),
                fontsize=8,
            )

    plt.title("Packet Journey Through Network")
    plt.xlabel("Simulation Time (seconds)")
    plt.ylabel("Node ID")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
