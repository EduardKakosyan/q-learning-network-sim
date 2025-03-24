from collections import Counter
from example import simulator_creator
from network_sim.core.simulator import NetworkSimulator
from network_sim.utils.metrics import calculate_fairness_index
from network_sim.utils.visualization import plot_link_utilization, plot_metrics
from pprint import pprint


def main():
    """Run the specific simulations for the demo to compare results."""

    # Topology parameters
    num_nodes = 8
    excess_edges = 10
    num_generators = 5

    # Simulation parameters
    link_scales = [10, 5, 1]
    routers = ["Dijkstra", "LCF", "QL"]
    router_time_scale = 0.0
    duration = 10.0

    # The best Q Learning parameters:
    ql_params = {
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "exploration_rate": 0.2,
        "exploration_rate": 0.5,
        "bins": 4,
        "bin_base": 20,
    }

    simulator_func = simulator_creator(
        num_nodes,
        excess_edges,
        num_generators,
        router_time_scale,
        ql_params,
    )

    for link_scale in link_scales:
        metrics_list = []
        print(f"Running simulations with link_scale={link_scale}")
        for router in routers:
            print(f"Running simulation with {router} router...")
            simulator: NetworkSimulator = simulator_func(router, link_scale)
            simulator.run(duration, updates=True)
            
            metrics_list.append(simulator.metrics)

            plot_link_utilization(simulator)

            delay = simulator.metrics["average_delay"]
            packet_loss = simulator.metrics["packet_loss_rate"]
            throughput = simulator.metrics["throughput"]
            fairness = calculate_fairness_index(simulator)
            print(f"Average Delay:  {delay:.2f}")
            print(f"Packet loss:    {packet_loss * 100:.2f}%")
            print(f"Throughput:     {int(throughput)}")
            print(f"Fairness index: {fairness:.4f}")
            if simulator.dropped_packets:
                counter = Counter([f"{packet.current_node}: {reason}" for packet, reason in simulator.dropped_packets])
                print("Number of packets:", len(simulator.packets))
                pprint(counter)

        plot_metrics(metrics_list, routers)


if __name__ == "__main__":
    main()
