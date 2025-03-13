#!/usr/bin/env python3
"""Example network simulation using the network_sim package.

This script demonstrates how to use the network_sim package to create and run
a simple network simulation with different schedulers.
"""

import os
import simpy

from network_sim.core.simulator import NetworkSimulator
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
    plot_packet_journey,
)
from network_sim.utils.metrics import (
    save_metrics_to_json,
    compare_schedulers,
    calculate_fairness_index,
)


def create_dumbbell_topology(simulator):
    """Create a dumbbell network topology.

    Args:
        simulator: NetworkSimulator instance.

    Returns:
        Tuple of (source_nodes, destination_nodes).
    """
    # Create nodes
    for i in range(1, 7):
        simulator.add_node(i, processing_delay=0.001)

    # Create links
    # Left side
    simulator.add_link(1, 3, 10e6, 0.01, 64000)  # 10 Mbps, 10ms delay, 64KB buffer
    simulator.add_link(2, 3, 10e6, 0.01, 64000)

    # Bottleneck link
    simulator.add_link(3, 4, 5e6, 0.02, 32000)  # 5 Mbps, 20ms delay, 32KB buffer

    # Right side
    simulator.add_link(4, 5, 10e6, 0.01, 64000)
    simulator.add_link(4, 6, 10e6, 0.01, 64000)

    # Compute routing tables
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
    # Create simulator
    env = simpy.Environment()
    simulator = NetworkSimulator(env=env, scheduler_type=scheduler_type)

    # Create network topology
    source_nodes, dest_nodes = create_dumbbell_topology(simulator)

    # Create traffic flows
    # Flow 1: Constant bit rate
    simulator.packet_generator(
        source=source_nodes[0],
        destination=dest_nodes[0],
        packet_size=constant_size(1000),  # 1000 bytes
        interval=constant_traffic(100),  # 100 packets per second
        pattern=TrafficPattern.CONSTANT,
    )

    # Flow 2: Poisson traffic
    simulator.packet_generator(
        source=source_nodes[1],
        destination=dest_nodes[1],
        packet_size=variable_size(500, 1500),  # 500-1500 bytes
        interval=poisson_traffic(80),  # 80 packets per second average
        pattern=TrafficPattern.VARIABLE,
    )

    # Run simulation
    simulator.run(duration)

    return simulator


def main():
    """Run simulations with different schedulers and compare results."""
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run simulations with different schedulers
    schedulers = ["FIFO", "RR"]
    simulators = []
    metrics_list = []

    for scheduler in schedulers:
        print(f"Running simulation with {scheduler} scheduler...")
        simulator = run_simulation(scheduler)
        simulators.append(simulator)
        metrics_list.append(simulator.metrics)

        # Save individual results
        save_network_visualization(
            simulator, f"results/{scheduler.lower()}_topology.png"
        )
        plot_link_utilization(
            simulator, f"results/{scheduler.lower()}_link_utilization.png"
        )
        plot_packet_journey(
            simulator, output_file=f"results/{scheduler.lower()}_packet_journey.png"
        )
        save_metrics_to_json(
            simulator.metrics, f"results/{scheduler.lower()}_metrics.json"
        )

        # Calculate and print fairness index
        fairness = calculate_fairness_index(simulator)
        print(f"  Fairness index: {fairness:.4f}")

    # Compare schedulers
    comparison = compare_schedulers(simulators)

    # Plot comparison metrics
    plot_metrics(metrics_list, schedulers)

    print("\nSimulation complete. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()
