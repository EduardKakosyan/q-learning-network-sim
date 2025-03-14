#!/usr/bin/env python3
"""Example network simulation using the network_sim package.

This script demonstrates how to use the network_sim package to create and run
a simple network simulation with different schedulers.
"""

import os
import simpy

from network_sim.core.simulator import NetworkSimulator
from network_sim.core.scheduling_algorithms import scheduling_algorithm_factory
from network_sim.core.enums import TrafficPattern
from network_sim.traffic.generators import (
    constant_traffic,
    poisson_traffic,
    constant_size,
    variable_size,
)
from network_sim.utils.visualization import (
    save_network_visualization,
    plot_metrics,
    plot_link_utilization,
)
from network_sim.utils.metrics import (
    compare_schedulers,
    calculate_fairness_index,
)


def create_dumbbell_topology(simulator: NetworkSimulator):
    """Create a dumbbell network topology.

    Args:
        simulator: NetworkSimulator instance.

    Returns:
        Tuple of (source_nodes, destination_nodes).
    """
    for i in range(1, 7):
        simulator.add_node(i, processing_delay=0.001)

    simulator.add_link(1, 3, 10e6, 0.01, 64000)
    simulator.add_link(2, 3, 10e6, 0.01, 64000)

    simulator.add_link(3, 4, 5e6, 0.02, 32000)

    simulator.add_link(4, 5, 10e6, 0.01, 64000)
    simulator.add_link(4, 6, 10e6, 0.01, 64000)

    simulator.compute_shortest_paths()

    return ([1, 2], [5, 6])


def run_simulation(scheduler_type, duration=10.0):
    """Run a network simulation with the specified scheduler.

    Args:
        scheduler_type: Type of scheduler to use.
        duration: Simulation duration in seconds.

    Returns:
        NetworkSimulator instance after simulation.
    """
    env = simpy.Environment()
    scheduler = scheduling_algorithm_factory(scheduler_type, env)
    simulator = NetworkSimulator(env, scheduler)

    source_nodes, dest_nodes = create_dumbbell_topology(simulator)

    simulator.packet_generator(
        source=source_nodes[0],
        destination=dest_nodes[0],
        packet_size=constant_size(1000),
        interval=constant_traffic(100),
        pattern=TrafficPattern.CONSTANT,
    )

    simulator.packet_generator(
        source=source_nodes[1],
        destination=dest_nodes[1],
        packet_size=variable_size(500, 1500),
        interval=poisson_traffic(80),
        pattern=TrafficPattern.VARIABLE,
    )

    simulator.run(duration)

    return simulator


def main():
    """Run simulations with different schedulers and compare results."""
    os.makedirs("results", exist_ok=True)

    schedulers = ["FIFO", "RR", "QL"]
    simulators = []
    metrics_list = []

    for scheduler in schedulers:
        print(f"Running simulation with {scheduler} scheduler...")
        simulator = run_simulation(scheduler)
        save_network_visualization(simulator, "results/topology.png")
        simulators.append(simulator)
        metrics_list.append(simulator.metrics)

        plot_link_utilization(
            simulator, f"results/{scheduler.lower()}_link_utilization.png"
        )

        fairness = calculate_fairness_index(simulator)
        print(f"  Fairness index: {fairness:.4f}")

    compare_schedulers(simulators)

    plot_metrics(metrics_list, schedulers)

    print("\nSimulation complete. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
