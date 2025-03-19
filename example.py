#!/usr/bin/env python3
"""Example network simulation using the network_sim package.

This script demonstrates how to use the network_sim package to create and run
a simple network simulation with different routers.
"""

from collections import Counter
import os
import simpy

from network_sim.core.simulator import NetworkSimulator
from network_sim.core.routing_algorithms import router_factory
from network_sim.traffic.generators import constant_traffic, constant_size
from network_sim.utils.visualization import (
    save_network_visualization,
    plot_metrics,
    plot_link_utilization,
)
from network_sim.utils.metrics import (
    save_metrics_to_json,
    compare_routers,
    calculate_fairness_index,
)


def run_simulation(router_type, duration=30.0):
    """Run a network simulation with the specified router.

    Args:
        router_type: Type of router to use.
        duration: Simulation duration in seconds.

    Returns:
        NetworkSimulator instance after simulation.
    """
    env = simpy.Environment()
    simulator = NetworkSimulator(env, router_type)

    def create_router(node):
        return router_factory(router_type, node, simulator=simulator)
    
    def create_q_router(node):
        return router_factory("QL", node, simulator)

    simulator.add_node(1, router_func=create_router, buffer_size=1e6)
    simulator.add_node(2, buffer_size=1e6)
    simulator.add_node(3, buffer_size=1e6)
    simulator.add_node(4, buffer_size=1e6)
    simulator.add_node(5, buffer_size=1e6)

    simulator.add_link(1, 2, 1e7, 0.01)
    simulator.add_link(1, 3, 1e7, 0.01)
    simulator.add_link(1, 4, 1e7, 0.01)
    simulator.add_link(2, 5, 1e7, 0.01)
    simulator.add_link(3, 5, 1e7, 0.01)
    simulator.add_link(4, 5, 1e7, 0.01)

    simulator.compute_shortest_paths()

    simulator.packet_generator(
        source=1,
        destination=5,
        packet_size=constant_size(1000),
        interval=constant_traffic(2000),
    )

    simulator.run(duration)

    return simulator


def main():
    """Run simulations with different routers and compare results."""
    os.makedirs("results", exist_ok=True)

    routers = ["Dijkstra", "LCF", "QL"]
    simulators = []
    metrics_list = []

    for router in routers:
        print(f"Running simulation with {router} router...")
        simulator = run_simulation(router)
        save_network_visualization(simulator, "results/topology.png")
        simulators.append(simulator)
        metrics_list.append(simulator.metrics)

        save_metrics_to_json(simulator.metrics, f"results/{router.lower()}_metrics.png")
        plot_link_utilization(
            simulator, f"results/{router.lower()}_link_utilization.png"
        )

        fairness = calculate_fairness_index(simulator)
        print(f"Fairness index: {fairness:.4f}")
        print("Number of packets:", len(simulator.packets))
        if simulator.dropped_packets:
            counter = Counter([f"{packet.current_node}: {reason}" for packet, reason in simulator.dropped_packets])
            print(counter)

    compare_routers(simulators)

    plot_metrics(metrics_list, routers)

    print("\nSimulation complete. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
