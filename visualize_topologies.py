#!/usr/bin/env python3
"""Visualize different network topologies used in the Q-learning routing experiments."""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
from collections import defaultdict
import numpy as np


def generate_bottleneck_graph(
    n: int, bottleneck_factor: float = 0.5
) -> List[Tuple[int, int]]:
    """Generate a graph with bottleneck links."""
    edges = []
    nodes = list(range(1, n + 1))

    # Create two clusters
    cluster1 = nodes[: n // 2]
    cluster2 = nodes[n // 2 :]

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
    num_cross_links = max(1, int(min(len(cluster1), len(cluster2)) * bottleneck_factor))
    cross_pairs = [(i, j) for i in cluster1 for j in cluster2]
    random.shuffle(cross_pairs)
    edges.extend(cross_pairs[:num_cross_links])

    return edges


def generate_ring_graph(n: int, cross_links: int) -> List[Tuple[int, int]]:
    """Generate a ring topology with cross connections."""
    edges = []
    nodes = list(range(1, n + 1))

    # Create ring
    for i in range(n):
        edges.append((nodes[i], nodes[(i + 1) % n]))

    # Add cross links
    possible_crosses = [
        (i, j)
        for i in nodes
        for j in nodes
        if i < j and (i, j) not in edges and abs(i - j) > 1
    ]
    random.shuffle(possible_crosses)
    edges.extend(possible_crosses[:cross_links])

    return edges


def generate_scale_free_graph(n: int, m: int = 2) -> List[Tuple[int, int]]:
    """Generate a scale-free network using preferential attachment."""
    edges = []
    nodes = list(range(1, n + 1))

    # Start with a small complete graph
    for i in range(1, m + 2):
        for j in range(i + 1, m + 2):
            edges.append((nodes[i - 1], nodes[j - 1]))

    # Add remaining nodes with preferential attachment
    for i in range(m + 2, n + 1):
        # Calculate degree distribution
        degree_count = defaultdict(int)
        for e in edges:
            degree_count[e[0]] += 1
            degree_count[e[1]] += 1

        # Add m edges to existing nodes based on their degree
        existing_nodes = list(range(1, i))
        weights = [degree_count.get(node, 0) + 1 for node in existing_nodes]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        for _ in range(m):
            target = np.random.choice(existing_nodes, p=weights)
            edges.append((i, target))

    return edges


def visualize_topology(edges: List[Tuple[int, int]], title: str, filename: str):
    """Visualize a network topology using networkx and matplotlib."""
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
    plt.savefig(f"results/{filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """Generate and visualize different network topologies."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create results directory if it doesn't exist
    import os

    os.makedirs("results", exist_ok=True)

    # Generate and visualize bottleneck topology
    bottleneck_edges = generate_bottleneck_graph(10, 0.3)
    visualize_topology(
        bottleneck_edges,
        "Bottleneck Network Topology\n(10 nodes, 0.3 bottleneck factor)",
        "bottleneck_topology",
    )

    # Generate and visualize ring topology
    ring_edges = generate_ring_graph(12, 4)
    visualize_topology(
        ring_edges,
        "Ring Network with Cross Connections\n(12 nodes, 4 cross links)",
        "ring_topology",
    )

    # Generate and visualize scale-free topology
    scale_free_edges = generate_scale_free_graph(15, 2)
    visualize_topology(
        scale_free_edges, "Scale-Free Network\n(15 nodes, m=2)", "scale_free_topology"
    )

    print("Topology visualizations have been saved to the 'results' directory.")


if __name__ == "__main__":
    main()
