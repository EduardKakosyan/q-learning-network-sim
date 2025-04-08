#!/usr/bin/env python3
"""Demo script to compare different routing algorithms in a simulated network."""

import argparse
from collections import Counter
from pprint import pprint
import random
import time
import matplotlib.pyplot as plt
from typing import Dict, List
from example import simulator_creator
from network_sim.core.simulator import NetworkSimulator
from network_sim.utils.metrics import calculate_fairness_index
from network_sim.utils.visualization import (
    plot_buffer_usages,
    plot_link_utilizations,
    plot_metrics,
    plot_packet_journeys,
)


def main() -> None:
    """Run the specific simulations for the demo to compare results."""
    parser = argparse.ArgumentParser(
        description="Run the specific simulations for the demo to compare results."
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Skip every iteration except the last item from both for loops",
    )
    args = parser.parse_args()

    # Topology parameters
    num_nodes: int = 8
    num_generators: int = 4

    # Simulation parameters
    routers: List[str] = ["Dijkstra", "OSPF", "QL"]
    router_time_scale: float = 100.0
    duration: float = 10.0

    # The best Q Learning parameters:
    ql_params: Dict[str, float] = {
        "learning_rate": 0.5,
        "discount_factor": 0.99,
        "exploration_rate": 0.1,
        "bins": 4,
        "bin_base": 10,
    }

    seed: int = random.randint(0, 2**32 - 1)
    print("Seed:", seed)

    excess_edges_list: List[int] = [4, 15]

    # Calculate the time scale for this computer.
    start_time = time.perf_counter()
    sum = 0
    for i in range(10000000):
        sum += i
    test_time = time.perf_counter() - start_time
    reference_test_time = 0.3002118999526526
    relative_speedup = reference_test_time / test_time
    router_time_scale *= relative_speedup

    if args.last:
        excess_edges_list = [excess_edges_list[-1]]

    for i, excess_edges in enumerate(excess_edges_list):
        print(f"\nCreating topology with excess_edges={excess_edges}")
        simulator_func = simulator_creator(
            num_nodes,
            excess_edges,
            num_generators,
            router_time_scale,
            ql_params,
            seed=seed,
            block=False,
        )
        graph_fig_manager = plt.get_current_fig_manager()
        input("Press enter to continue")

        # Set packet scale list based on the index of excess_edges
        if i == 0 and not args.last:
            packet_scale_list = [1, 3]
        else:
            packet_scale_list = [3]

        for packet_scale in packet_scale_list:
            simulator_list: List[NetworkSimulator] = []
            metrics_list: List[dict] = []
            print(f"\nRunning simulations with packet_scale={packet_scale}")
            for router in routers:
                print(f"\nRunning simulation with {router} router...")
                simulator: NetworkSimulator = simulator_func(router, packet_scale)
                simulator.run(duration, updates=True)

                simulator_list.append(simulator)
                metrics_list.append(simulator.metrics)

                delay: float = simulator.metrics["average_delay"]
                packet_loss: float = simulator.metrics["packet_loss_rate"]
                throughput: float = simulator.metrics["throughput"]
                fairness: float = calculate_fairness_index(simulator)
                print(f"  Average Delay:  {delay:.2f} s")
                print(f"  Packet loss:    {packet_loss * 100:.2f}%")
                print(f"  Throughput:     {int(throughput)} packets")
                print(f"  Fairness index: {fairness:.4f}")
                print(f"  Average Routing Delays: {simulator.metrics['average_routing_delays']}")
                if simulator.dropped_packets:
                    counter = Counter(
                        [
                            f"{packet.current_node}: {reason}"
                            for packet, reason in simulator.dropped_packets
                        ]
                    )
                    print("  Number of packets:", len(simulator.packets))
                    pprint(dict({k: v for k, v in counter.items()}))

            plot_buffer_usages(simulator_list, show=False)
            fig_manager_1 = plt.get_current_fig_manager()
            plot_link_utilizations(metrics_list, show=False)
            fig_manager_2 = plt.get_current_fig_manager()
            plot_packet_journeys(simulator_list, show=False)
            fig_manager_3 = plt.get_current_fig_manager()
            plot_metrics(metrics_list, show=False)
            fig_manager_4 = plt.get_current_fig_manager()
            # Arrange the figures in a 2x2 grid
            fig_manager_list = [
                fig_manager_1,
                fig_manager_2,
                fig_manager_3,
                fig_manager_4,
            ]
            screen_width, screen_height = fig_manager_list[
                0
            ].canvas.manager.window.wm_maxsize()
            fig_width, fig_height = screen_width // 2, screen_height // 2

            for i, fig_manager in enumerate(fig_manager_list):
                x = (i % 2) * fig_width
                y = (i // 2) * fig_height
                fig_manager.window.wm_geometry(f"{fig_width}x{fig_height}+{x}+{y}")

            plt.show(block=False)
            input("Press enter to continue")
            for fig_manager in fig_manager_list:
                plt.close(fig_manager.canvas.figure)

        plt.close(graph_fig_manager.canvas.figure)


if __name__ == "__main__":
    main()
