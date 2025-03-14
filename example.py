#!/usr/bin/env python3
"""Example network simulation using the network_sim package.

This script demonstrates how to use the network_sim package to create and run
a simple network simulation with different routers.
"""

from collections import Counter
import os
import simpy

from network_sim.core.simulator import NetworkSimulator
from network_sim.core.routing_algorithms import *
from network_sim.traffic.generators import *
from network_sim.utils.visualization import *
from network_sim.utils.metrics import *


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

    router_func = lambda: router_factory(router_type)
    simulator.add_node(1)
    simulator.add_node(2)
    simulator.add_node(3, router=router_func())
    simulator.add_node(4)

    simulator.add_link(1, 3, float("inf"), 0.01)
    simulator.add_link(2, 3, float("inf"), 0.01)
    simulator.add_link(3, 4, float("inf"), 0.01)

    simulator.compute_shortest_paths()

    simulator.packet_generator(
        source=1,
        destination=4,
        packet_size=constant_size(1000),
        interval=constant_traffic(10),
    )

    simulator.packet_generator(
        source=2,
        destination=4,
        packet_size=constant_size(1000),
        interval=constant_traffic(10),
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

        plot_link_utilization(
            simulator, f"results/{router.lower()}_link_utilization.png"
        )

        fairness = calculate_fairness_index(simulator)
        print(f"  Fairness index: {fairness:.4f}")
        print("Number of packets:", len(simulator.packets))
        if simulator.dropped_packets:
            counter = Counter([reason for _, reason in simulator.dropped_packets])
            print(counter)

    compare_routers(simulators)

    plot_metrics(metrics_list, routers)

    print("\nSimulation complete. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
