"""Visualization utilities for network simulation.

This module provides functions for visualizing network simulation results,
including network topology, traffic patterns, and performance metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from network_sim.core.simulator import NetworkSimulator


def save_network_visualization(
    simulator: NetworkSimulator,
    filename: str | None = None,
    figsize: Tuple[int, int] = (10, 8),
    block = True,
) -> None:
    """Save network topology visualization to a file.

    Args:
        simulator: NetworkSimulator instance.
        filename: Output filename, or None to show it immediately.
        figsize: Figure size as (width, height) in inches.
    """
    fig = plt.figure(figsize=figsize)

    graph = simulator.graph
    pos = nx.nx_pydot.graphviz_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="gray",
        arrows=False,
    )


    for source, destination in simulator.generators:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(source, destination)],
            width=2,
            alpha=0.4,
            edge_color="blue",
            style='dashed',
            connectionstyle='arc3,rad=0.2',
            arrows=True,
            arrowsize=30,
        )

    nx.draw_networkx_labels(graph, pos, font_size=16)

    edge_labels = {(u, v): f"{graph[u][v]['delay']*1000:.1f}ms" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12, rotate=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.axis("off")
    plt.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    else:
        plt.show(block=block)
        if not block:
            plt.pause(0.001)


def plot_metrics(
    metrics_list: List[Dict[str, Any]],
    output_dir: str | None = None,
    show = True,
) -> None:
    """Plot and save performance metrics for different routers.

    Args:
        metrics_list: List of metrics dictionaries from different simulations.
        router_types: List of router types corresponding to metrics_list.
        output_dir: Directory to save output plots.
    """
    router_types = [metrics["router_type"] for metrics in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    throughputs = [m["throughput"] for m in metrics_list]
    delays = [m["average_delay"] for m in metrics_list]
    loss_rates = [m["packet_loss_rate"] for m in metrics_list]

    x = np.arange(len(router_types))

    # Throughput subplot
    axes[0].bar(x, throughputs, width=0.4)
    axes[0].set_ylabel('Throughput (packets)')
    axes[0].set_title('Throughput Comparison')
    axes[0].set_xlabel('Router Type')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(router_types)

    # Average Delay subplot
    axes[1].bar(x, delays, width=0.4, color='orange')
    axes[1].set_ylabel('Average Delay (seconds)')
    axes[1].set_title('Average Delay Comparison')
    axes[1].set_xlabel('Router Type')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(router_types)

    # Packet Loss Rate subplot
    axes[2].bar(x, loss_rates, width=0.4, color='green')
    axes[2].set_ylabel('Packet Loss Rate')
    axes[2].set_title('Packet Loss Rate Comparison')
    axes[2].set_xlabel('Router Type')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(router_types)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()
    elif show:
        plt.show()


def plot_link_utilizations(
    metrics_list: List[Dict[str, Any]],
    output_dir: str | None = None,
    filename: str | None = None,
    show = True,
) -> None:
    """Plot and save link utilization.

    Args:
        simulator: NetworkSimulator instance.
        output_file: Output filename.
        filename: The filename for the file.
    """
    num_metrics = len(metrics_list)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics), sharex=True)

    for i, metrics in enumerate(metrics_list):
        link_utilization = metrics["link_utilization"]
        if not link_utilization:
            continue

        links = [f"{src}->{dst}" for (src, dst) in link_utilization.keys()]
        utilizations = list(link_utilization.values())

        ax = axes[i] if num_metrics > 1 else axes
        ax.bar(links, utilizations)
        ax.set_title(f"Link Utilization (Router: {metrics['router_type']})")
        ax.set_ylabel("Utilization")
        if i == num_metrics - 1:
            ax.set_xlabel("Link")
            ax.set_xticks(range(len(links)))
            ax.set_xticklabels(links, rotation=45)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()
    elif show:
        plt.show()


def plot_packet_journeys(
    simulator_list: List[NetworkSimulator],
    packet_ids: Optional[List[int]] = None,
    max_packets: int = 10,
    output_dir: str | None = None,
    show = True,
) -> None:
    """Plot and save packet journey through the network.

    Args:
        simulator: NetworkSimulator instance.
        packet_ids: List of packet IDs to plot (if None, selects evenly spaced packets).
        max_packets: Maximum number of packets to plot.
        output_dir: Output directory.
    """
    num_simulators = len(simulator_list)
    fig, axes = plt.subplots(num_simulators, 1, figsize=(12, 5 * num_simulators), sharex=True)

    for i, simulator in enumerate(simulator_list):
        completed_packets = simulator.completed_packets
        if not completed_packets:
            continue

        if packet_ids is None:
            # Select packets spaced out by creation time if not specified
            if len(completed_packets) > max_packets:
                sorted_packets = sorted(completed_packets, key=lambda p: p.creation_time)
                step = len(sorted_packets) // max_packets
                packets_to_plot = [sorted_packets[i * step] for i in range(max_packets)]
            else:
                packets_to_plot = completed_packets
        else:
            # Filter packets by ID
            packets_to_plot = [p for p in completed_packets if p.id in packet_ids]

        ax = axes[i] if num_simulators > 1 else axes
        for packet in packets_to_plot:
            nodes = [hop[0] for hop in packet.hops]
            times = [hop[1] for hop in packet.hops]

            # Add source node
            nodes.insert(0, packet.source)
            times.insert(0, packet.creation_time)

            ax.plot(times, nodes, "o-", label=f"Packet {packet.id}")

        ax.set_title(f"Packet Journey Through Network (Router: {simulator.router_type})")
        ax.set_ylabel("Node ID")
        ax.grid(True, linestyle="--", alpha=0.7)
        if i == num_simulators - 1:
            ax.set_xlabel("Simulation Time (seconds)")

    plt.tight_layout(rect=[0, 0.05, 1, 0.98], h_pad=3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "packet_journey.png"))
        plt.close()
    elif show:
        plt.show()


def plot_buffer_usages(
    simulator_list: List[NetworkSimulator],
    output_dir: str | None = None,
    filename: str | None = None,
    show = True,
) -> None:
    """Plot and save buffer size at each node over time.

    Args:
        simulator: NetworkSimulator instance.
        output_dir: Output directory.
        filename: The filename for the file.
    """
    num_simulators = len(simulator_list)
    fig, axes = plt.subplots(num_simulators, 1, figsize=(12, 5 * num_simulators), sharex=True)

    for i, simulator in enumerate(simulator_list):
        ax = axes[i] if num_simulators > 1 else axes
        for node_id, node in simulator.nodes.items():
            times, usages = zip(*node.buffer_usage_history)
            ax.plot(times, usages, label=f"Node {node_id}")

        ax.set_title(f"Buffer Usage at Each Node Over Time (Router: {simulator.router_type})")
        ax.set_ylabel("Buffer Usage")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        if i == num_simulators - 1:
            ax.set_xlabel("Simulation Time (seconds)")

    plt.tight_layout(rect=[0, 0.07, 1, 0.98], h_pad=3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()
    elif show:
        plt.show()
