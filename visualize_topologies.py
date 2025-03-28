#!/usr/bin/env python3
"""Visualize different network topologies used in the Q-learning routing experiments."""

from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random


def generate_bottleneck_graph(
    n: int, bottleneck_factor: float = 0.5
) -> List[Tuple[int, int]]:
    """Generate a graph with bottleneck links.

    Args:
        n: Number of nodes in the graph.
        bottleneck_factor: Fraction of cross-links between clusters.

    Returns:
        A list of edges representing the graph.
    """
    edges: List[Tuple[int, int]] = []
    nodes: List[int] = list(range(1, n + 1))

    # Create two clusters
    cluster1: List[int] = nodes[: n // 2]
    cluster2: List[int] = nodes[n // 2 :]

    # Connect nodes within clusters densely
    for i in cluster1:
        for j in cluster1:
            if i < j:
                edges.append((i, j))
    for i in cluster2:
        for j in cluster2:
            if i < j:
                edges.append((i, j))

    # Add bottleneck connections between clusters
    num_cross_links: int = max(
        1, int(min(len(cluster1), len(cluster2)) * bottleneck_factor)
    )
    cross_pairs: List[Tuple[int, int]] = [(i, j) for i in cluster1 for j in cluster2]
    random.shuffle(cross_pairs)
    edges.extend(cross_pairs[:num_cross_links])

    return edges


def generate_ring_graph(n: int, cross_links: int) -> List[Tuple[int, int]]:
    """Generate a ring topology with cross connections.

    Args:
        n: Number of nodes in the ring.
        cross_links: Number of additional cross-links.

    Returns:
        A list of edges representing the ring graph.
    """
    edges: List[Tuple[int, int]] = []
    nodes: List[int] = list(range(1, n + 1))

    # Create ring
    for i in range(n):
        edges.append((nodes[i], nodes[(i + 1) % n]))

    # Add cross links
    possible_crosses: List[Tuple[int, int]] = [
        (i, j)
        for i in nodes
        for j in nodes
        if i < j and (i, j) not in edges and abs(i - j) > 1
    ]
    random.shuffle(possible_crosses)
    edges.extend(possible_crosses[:cross_links])

    return edges


def generate_scale_free_graph(n: int, m: int = 2) -> List[Tuple[int, int]]:
    """Generate a scale-free network using preferential attachment.

    Args:
        n: Total number of nodes in the network.
        m: Number of edges to attach from a new node to existing nodes.

    Returns:
        A list of edges representing the scale-free graph.
    """
    edges: List[Tuple[int, int]] = []
    nodes: List[int] = list(range(1, n + 1))

    # Start with a small complete graph
    for i in range(1, m + 2):
        for j in range(i + 1, m + 2):
            edges.append((nodes[i - 1], nodes[j - 1]))

    # Add remaining nodes with preferential attachment
    for i in range(m + 2, n + 1):
        # Calculate degree distribution
        degree_count: defaultdict[int, int] = defaultdict(int)
        for e in edges:
            degree_count[e[0]] += 1
            degree_count[e[1]] += 1

        # Add m edges to existing nodes based on their degree
        existing_nodes: List[int] = list(range(1, i))
        weights: List[float] = [
            degree_count.get(node, 0) + 1 for node in existing_nodes
        ]
        total_weight: float = sum(weights)
        weights = [w / total_weight for w in weights]

        for _ in range(m):
            target: int = np.random.choice(existing_nodes, p=weights)
            edges.append((i, target))

    return edges


def visualize_topology(
    edges: List[Tuple[int, int]], title: str, output_dir: str, filename: str
) -> None:
    """Visualize a network topology using networkx and matplotlib.

    Args:
        edges: List of edges in the graph.
        title: Title for the plot.
        output_dir: Directory to save the plot.
        filename: Name of the file to save the plot as.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes with size based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 200 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title(title, pad=20)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{filename}.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def main() -> None:
    """Generate and visualize different network topologies."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    output_dir: str = "results"

    os.makedirs(output_dir, exist_ok=True)

    # Generate and visualize bottleneck topology
    bottleneck_edges: List[Tuple[int, int]] = generate_bottleneck_graph(10, 0.3)
    visualize_topology(
        bottleneck_edges,
        "Bottleneck Network Topology\n(10 nodes, 0.3 bottleneck factor)",
        output_dir,
        "bottleneck_topology",
    )

    # Generate and visualize ring topology
    ring_edges: List[Tuple[int, int]] = generate_ring_graph(12, 4)
    visualize_topology(
        ring_edges,
        "Ring Network with Cross Connections\n(12 nodes, 4 cross links)",
        output_dir,
        "ring_topology",
    )

    # Generate and visualize scale-free topology
    scale_free_edges: List[Tuple[int, int]] = generate_scale_free_graph(15, 2)
    visualize_topology(
        scale_free_edges,
        "Scale-Free Network\n(15 nodes, m=2)",
        output_dir,
        "scale_free_topology",
    )

    print(f"Topology visualizations have been saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    main()
