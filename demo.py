from collections import Counter
from pprint import pprint
import random
import matplotlib.pyplot as plt

from example import simulator_creator
from network_sim.core.simulator import NetworkSimulator
from network_sim.utils.metrics import calculate_fairness_index
from network_sim.utils.visualization import plot_buffer_usages, plot_link_utilizations, plot_metrics, plot_packet_journeys


def main():
    """Run the specific simulations for the demo to compare results."""

    # Topology parameters
    num_nodes = 8
    num_generators = 5

    # Simulation parameters
    packet_scales = [1, 2, 3]
    routers = ["Dijkstra", "LCF", "QL"]
    router_time_scale = 0.0
    duration = 10.0

    # The best Q Learning parameters:
    ql_params = {
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "exploration_rate": 0.0,
        "exploration_rate": 0.5,
        "bins": 4,
        "bin_base": 20,
    }

    seed = random.randint(0, 2**32 - 1)
    print("Seed:", seed)

    for excess_edges in [4, 15]:
        simulator_func = simulator_creator(
            num_nodes,
            excess_edges,
            num_generators,
            router_time_scale,
            ql_params,
            seed=seed,
            block=False
        )
        graph_fig_manager = plt.get_current_fig_manager()
        for packet_scale in packet_scales:
            simulator_list = []
            metrics_list = []
            print(f"\nRunning simulations with excess_edges={excess_edges} and packet_scale={packet_scale}")
            for router in routers:
                print(f"\nRunning simulation with {router} router...")
                simulator: NetworkSimulator = simulator_func(router, packet_scale)
                simulator.run(duration, updates=True)
                
                simulator_list.append(simulator)
                metrics_list.append(simulator.metrics)

                delay = simulator.metrics["average_delay"]
                packet_loss = simulator.metrics["packet_loss_rate"]
                throughput = simulator.metrics["throughput"]
                fairness = calculate_fairness_index(simulator)
                print(f"  Average Delay:  {delay:.2f} s")
                print(f"  Packet loss:    {packet_loss * 100:.2f}%")
                print(f"  Throughput:     {int(throughput)} packets")
                print(f"  Fairness index: {fairness:.4f}")
                if simulator.dropped_packets:
                    counter = Counter([f"{packet.current_node}: {reason}" for packet, reason in simulator.dropped_packets])
                    print("  Number of packets:", len(simulator.packets))
                    pprint(dict({k: v for k, v in counter.items()}))

            plot_link_utilizations(metrics_list, show=False)
            fig_manager_1 = plt.get_current_fig_manager()
            plot_packet_journeys(simulator_list, show=False)
            fig_manager_2 = plt.get_current_fig_manager()
            plot_buffer_usages(simulator_list, show=False)
            fig_manager_3 = plt.get_current_fig_manager()
            plot_metrics(metrics_list, show=False)
            fig_manager_4 = plt.get_current_fig_manager()
            # Arrange the figures in a 2x2 grid
            fig_manager_list = [fig_manager_1, fig_manager_2, fig_manager_3, fig_manager_4]
            screen_width, screen_height = fig_manager_list[0].canvas.manager.window.wm_maxsize()
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
