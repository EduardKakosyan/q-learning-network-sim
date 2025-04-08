#!/usr/bin/env python3
"""Example network simulation using the network_sim package.

This script demonstrates how to use the network_sim package to create and run
a simple network simulation with different routers.
"""

from collections import Counter
from pprint import pprint
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import os
import random
import simpy

from network_sim.core.simulator import NetworkSimulator
from network_sim.core.routing_algorithms import router_factory
from network_sim.traffic.generators import bimodal_size, bursty_traffic, poisson_traffic
from network_sim.utils.visualization import (
    save_network_visualization,
    plot_metrics,
    plot_link_utilizations,
)
from network_sim.utils.metrics import (
    save_metrics_to_json,
    compare_routers,
    calculate_fairness_index,
)


def simulator_creator(
    num_nodes: int,
    excess_edges: int,
    num_generators: int,
    router_time_scale: float = 1.0,
    ql_params: Dict[str, Any] = {},
    seed: int = 42,
    output_dir: str | None = None,
    log: bool = False,
    show: bool = True,
    block: bool = True,
) -> Callable[[str, int], NetworkSimulator]:
    """Create a network simulator with specified parameters.

    Args:
        num_nodes: Number of nodes in the network.
        excess_edges: Number of additional edges beyond a minimum spanning tree.
        num_generators: Number of traffic generators.
        router_time_scale: Time scale for router operations.
        ql_params: Parameters for Q-learning routers.
        seed: Random seed for reproducibility.
        output_dir: Directory to save visualizations.
        log: Whether to log generator information.
        show: Whether to display the network visualization.
        block: Whether to block execution until visualization is closed.

    Returns:
        A function that instantiates a NetworkSimulator with a given router type and packet scale.
    """
    random.seed(seed)
    np.random.seed(seed)

    def generate_random_graph(n: int, excess_edges: int) -> List[Tuple[int, int]]:
        """Generate a random graph with a minimum spanning tree and excess edges."""
        edges: List[Tuple[int, int]] = []
        nodes: List[int] = list(range(1, n + 1))
        random.shuffle(nodes)

        # Create a minimum spanning tree
        for i in range(n - 1):
            edges.append((nodes[i], nodes[i + 1]))

        # Add excess edges
        possible_edges: List[Tuple[int, int]] = [
            (i, j)
            for i in nodes
            for j in nodes
            if i < j and (i, j) not in edges and (j, i) not in edges
        ]
        random.shuffle(possible_edges)
        edges.extend(possible_edges[:excess_edges])

        return edges

    edges = generate_random_graph(num_nodes, excess_edges)

    link_delays: List[float] = [random.uniform(0.001, 0.05) for _ in range(len(edges))]

    possible_node_pairs: List[Tuple[int, int]] = []
    # Generate all possible directed edges between non-neighbor nodes
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i != j and (i, j) not in edges and (j, i) not in edges:
                possible_node_pairs.append((i, j))

    node_pairs: List[Tuple[int, int]] = []
    while len(possible_node_pairs) > 0:
        source, destination = random.sample(possible_node_pairs, 1)[0]
        node_pairs.append((source, destination))
        if len(node_pairs) >= num_generators:
            break
        possible_node_pairs = [
            pair
            for pair in possible_node_pairs
            if pair[0] != source and pair[1] != destination
        ]
    else:
        if len(node_pairs) != num_generators:
            raise ValueError(
                f"Cannot generate {num_generators} node pairs. Max is {len(node_pairs)}"
            )

    if log:
        print("Generators:")
        print(node_pairs)

    def instantiate_simulator(
        router_type: str, packet_scale: int = 5
    ) -> NetworkSimulator:
        """Instantiate a network simulator with a specific router type and packet scale."""
        env = simpy.Environment()
        simulator = NetworkSimulator(env, router_type)

        def create_router(node: int) -> Any:
            return router_factory(router_type, node, simulator=simulator, seed=seed, **ql_params)

        for node in range(1, num_nodes + 1):
            simulator.add_node(
                node,
                router_func=create_router,
                buffer_size=1e4,
                time_scale=router_time_scale,
            )

        for edge, delay in zip(edges, link_delays):
            simulator.add_link(edge[0], edge[1], 1e5, delay)

        simulator.compute_shortest_paths()

        for source, destination in node_pairs:
            simulator.packet_generator(
                source=source,
                destination=destination,
                packet_size=bimodal_size(10 * packet_scale, 100 * packet_scale, 0.8),
                interval=bursty_traffic(5, poisson_traffic(100)),
            )

        return simulator

    if show:
        simulator = instantiate_simulator("Dijkstra")
        save_network_visualization(
            simulator,
            os.path.join(output_dir, "topology.png") if output_dir else None,
            block=block,
        )

    return instantiate_simulator


def run_simulation(
    simulator_creator: Callable[[str], NetworkSimulator],
    router_type: str,
    duration: float = 30.0,
) -> NetworkSimulator:
    """Run a network simulation with the specified router.

    Args:
        simulator_creator: Function to create a simulator.
        router_type: Type of router to use.
        duration: Simulation duration in seconds.

    Returns:
        NetworkSimulator instance after simulation.
    """

    simulator = simulator_creator(router_type)

    simulator.run(duration, updates=True)

    return simulator


def main() -> None:
    """Run simulations with different routers and compare results."""

    output_dir: str = "results"

    # Topology parameters
    num_nodes: int = 8
    excess_edges: int = 10
    num_generators: int = 5

    # Simulation parameters
    routers: List[str] = ["Dijkstra", "LCF", "QL"]
    router_time_scale: float = 0.0
    duration: float = 10.0

    # The best Q Learning parameters:
    ql_params: Dict[str, float] = {
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "exploration_rate": 0.2,
        "bins": 4,
        "bin_base": 20,
    }

    simulator_func = simulator_creator(
        num_nodes,
        excess_edges,
        num_generators,
        router_time_scale,
        ql_params,
        output_dir=output_dir,
        log=True,
    )

    simulators: List[NetworkSimulator] = []
    metrics_list: List[Dict[str, Any]] = []
    for router in routers:
        print(f"Running simulation with {router} router...")
        simulator = run_simulation(simulator_func, router, duration)
        simulators.append(simulator)
        metrics_list.append(simulator.metrics)

        save_metrics_to_json(simulator.metrics, output_dir, f"{router.lower()}_metrics")

        delay: float = simulator.metrics["average_delay"]
        packet_loss: float = simulator.metrics["packet_loss_rate"]
        throughput: float = simulator.metrics["throughput"]
        fairness: float = calculate_fairness_index(simulator)
        print(f"  Average Delay:  {delay:.2f} s")
        print(f"  Packet loss:    {packet_loss * 100:.2f}%")
        print(f"  Throughput:     {int(throughput)} packets")
        print(f"  Fairness index: {fairness:.4f}")
        if simulator.dropped_packets:
            counter = Counter(
                [
                    f"{packet.current_node}: {reason}"
                    for packet, reason in simulator.dropped_packets
                ]
            )
            print("  Number of packets:", len(simulator.packets))
            pprint(counter.items())

    compare_routers(simulators, output_dir)
    plot_link_utilizations(metrics_list, output_dir, "link_utilizations")
    plot_metrics(metrics_list, output_dir)

    print(f"\nSimulation complete. Results saved to '{output_dir}' directory.")


if __name__ == "__main__":
    main()
