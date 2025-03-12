import matplotlib.pyplot as plt
import numpy as np
from network_simulator import (
    NetworkSimulator,
    TrafficPattern,
    constant_traffic,
    variable_traffic,
    poisson_traffic,
    pareto_traffic,
    constant_size,
    variable_size,
    bimodal_size,
)


def create_dumbbell_topology(sim, num_hosts=3, link_capacity=10e6, buffer_size=64000):
    """
    Create a dumbbell topology with multiple hosts on each side

    Args:
        sim: NetworkSimulator instance
        num_hosts: Number of hosts on each side
        link_capacity: Capacity of links in bits per second
        buffer_size: Buffer size in bytes

    Returns:
        Lists of source and destination node IDs
    """
    # Create router nodes
    sim.add_node("R1", processing_delay=0.0001, buffer_size=buffer_size)
    sim.add_node("R2", processing_delay=0.0001, buffer_size=buffer_size)

    # Create source and destination hosts
    sources = []
    destinations = []

    for i in range(num_hosts):
        src_id = f"S{i+1}"
        dst_id = f"D{i+1}"

        sim.add_node(src_id, processing_delay=0.0001)
        sim.add_node(dst_id, processing_delay=0.0001)

        # Connect sources to R1
        sim.add_link(src_id, "R1", link_capacity, 0.001, buffer_size)

        # Connect R2 to destinations
        sim.add_link("R2", dst_id, link_capacity, 0.001, buffer_size)

        sources.append(src_id)
        destinations.append(dst_id)

    # Connect R1 to R2 (bottleneck link)
    bottleneck_capacity = link_capacity * 0.5  # Make it a bottleneck
    sim.add_link("R1", "R2", bottleneck_capacity, 0.005, buffer_size)

    # Compute routing tables
    sim.compute_shortest_paths()

    return sources, destinations


def create_mesh_topology(sim, size=3, link_capacity=10e6, buffer_size=64000):
    """
    Create a mesh topology with size x size nodes

    Args:
        sim: NetworkSimulator instance
        size: Grid size (size x size nodes)
        link_capacity: Capacity of links in bits per second
        buffer_size: Buffer size in bytes

    Returns:
        List of all node IDs
    """
    # Create nodes
    nodes = []
    for i in range(size):
        for j in range(size):
            node_id = f"N{i}{j}"
            sim.add_node(node_id, processing_delay=0.0001, buffer_size=buffer_size)
            nodes.append(node_id)

    # Create links (connect each node to its neighbors)
    for i in range(size):
        for j in range(size):
            node_id = f"N{i}{j}"

            # Connect to right neighbor
            if j < size - 1:
                right_id = f"N{i}{j+1}"
                sim.add_link(node_id, right_id, link_capacity, 0.001, buffer_size)
                sim.add_link(right_id, node_id, link_capacity, 0.001, buffer_size)

            # Connect to bottom neighbor
            if i < size - 1:
                bottom_id = f"N{i+1}{j}"
                sim.add_link(node_id, bottom_id, link_capacity, 0.001, buffer_size)
                sim.add_link(bottom_id, node_id, link_capacity, 0.001, buffer_size)

    # Compute routing tables
    sim.compute_shortest_paths()

    return nodes


def run_simulation(
    topology_type="dumbbell", traffic_pattern=TrafficPattern.CONSTANT, duration=10
):
    """
    Run a network simulation with specified topology and traffic pattern

    Args:
        topology_type: Type of topology ('dumbbell' or 'mesh')
        traffic_pattern: Traffic pattern to use
        duration: Simulation duration in seconds

    Returns:
        Simulation metrics
    """
    # Create simulator
    sim = NetworkSimulator(seed=42)

    # Create topology
    if topology_type == "dumbbell":
        sources, destinations = create_dumbbell_topology(sim)

        # Create traffic flows
        for i in range(len(sources)):
            source = sources[i]
            destination = destinations[i]

            # Set up traffic generators based on pattern
            if traffic_pattern == TrafficPattern.CONSTANT:
                # Constant bit rate traffic
                interval_gen = constant_traffic(100)  # 100 packets per second
                size_gen = constant_size(1500)  # 1500 bytes per packet

            elif traffic_pattern == TrafficPattern.VARIABLE:
                # Variable bit rate traffic
                interval_gen = variable_traffic(50, 150)  # 50-150 packets per second
                size_gen = variable_size(500, 1500)  # 500-1500 bytes per packet

            elif traffic_pattern == TrafficPattern.BURSTY:
                # Bursty traffic
                interval_gen = poisson_traffic(20)  # Average 20 bursts per second
                size_gen = constant_size(1000)  # 1000 bytes per packet
                sim.env.process(
                    sim.packet_generator(
                        source,
                        destination,
                        size_gen,
                        interval_gen,
                        pattern=TrafficPattern.BURSTY,
                        burst_size=5,
                        burst_interval=0.001,
                    )
                )
                continue

            else:  # MIXED
                # Mixed traffic (Pareto distribution)
                interval_gen = pareto_traffic(
                    100, alpha=1.5
                )  # Heavy-tailed distribution
                size_gen = bimodal_size(
                    200, 1500, small_prob=0.7
                )  # Mix of small and large packets

            # Start packet generator process
            sim.env.process(
                sim.packet_generator(
                    source, destination, size_gen, interval_gen, jitter=0.1
                )
            )

    elif topology_type == "mesh":
        nodes = create_mesh_topology(sim)

        # Create traffic flows between random pairs of nodes
        num_flows = 5
        np.random.seed(42)

        for _ in range(num_flows):
            # Select random source and destination
            source, destination = np.random.choice(nodes, 2, replace=False)

            # Set up traffic generators based on pattern
            if traffic_pattern == TrafficPattern.CONSTANT:
                interval_gen = constant_traffic(50)
                size_gen = constant_size(1000)
            elif traffic_pattern == TrafficPattern.VARIABLE:
                interval_gen = variable_traffic(30, 70)
                size_gen = variable_size(500, 1500)
            elif traffic_pattern == TrafficPattern.BURSTY:
                interval_gen = poisson_traffic(10)
                size_gen = constant_size(1000)
                sim.env.process(
                    sim.packet_generator(
                        source,
                        destination,
                        size_gen,
                        interval_gen,
                        pattern=TrafficPattern.BURSTY,
                        burst_size=3,
                        burst_interval=0.001,
                    )
                )
                continue
            else:  # MIXED
                interval_gen = pareto_traffic(50)
                size_gen = bimodal_size(200, 1500)

            # Start packet generator process
            sim.env.process(
                sim.packet_generator(
                    source, destination, size_gen, interval_gen, jitter=0.1
                )
            )

    # Visualize network topology
    plt.figure(figsize=(10, 8))
    sim.visualize_network()
    plt.savefig(f"{topology_type}_topology.png")
    plt.close()

    # Run simulation
    print(
        f"Running simulation with {topology_type} topology and {traffic_pattern.name} traffic..."
    )
    metrics = sim.run(duration)

    # Print metrics
    print("\nSimulation Results:")
    print(f"Throughput: {metrics['throughput']/1000:.2f} KB/s")
    print(f"Average Delay: {metrics['average_delay']*1000:.2f} ms")
    print(f"Packet Loss Rate: {metrics['packet_loss_rate']*100:.2f}%")

    # Print link utilization for bottleneck links
    print("\nLink Utilization:")
    for (src, dst), util in metrics["link_utilization"].items():
        print(f"{src} -> {dst}: {util*100:.2f}%")

    return metrics


def compare_traffic_patterns():
    """Compare different traffic patterns on a dumbbell topology"""
    patterns = [
        TrafficPattern.CONSTANT,
        TrafficPattern.VARIABLE,
        TrafficPattern.BURSTY,
        TrafficPattern.MIXED,
    ]

    results = {}

    for pattern in patterns:
        results[pattern.name] = run_simulation(
            topology_type="dumbbell", traffic_pattern=pattern, duration=30
        )

    # Plot comparison
    plt.figure(figsize=(15, 5))

    # Throughput comparison
    plt.subplot(1, 3, 1)
    throughputs = [results[p.name]["throughput"] / 1000 for p in patterns]
    plt.bar([p.name for p in patterns], throughputs)
    plt.title("Throughput Comparison")
    plt.ylabel("Throughput (KB/s)")
    plt.xticks(rotation=45)

    # Delay comparison
    plt.subplot(1, 3, 2)
    delays = [results[p.name]["average_delay"] * 1000 for p in patterns]
    plt.bar([p.name for p in patterns], delays)
    plt.title("Average Delay Comparison")
    plt.ylabel("Delay (ms)")
    plt.xticks(rotation=45)

    # Packet loss comparison
    plt.subplot(1, 3, 3)
    losses = [results[p.name]["packet_loss_rate"] * 100 for p in patterns]
    plt.bar([p.name for p in patterns], losses)
    plt.title("Packet Loss Comparison")
    plt.ylabel("Loss Rate (%)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("traffic_pattern_comparison.png")
    plt.close()


def compare_topologies():
    """Compare different topologies with constant traffic"""
    topologies = ["dumbbell", "mesh"]

    results = {}

    for topology in topologies:
        results[topology] = run_simulation(
            topology_type=topology, traffic_pattern=TrafficPattern.CONSTANT, duration=30
        )

    # Plot comparison
    plt.figure(figsize=(15, 5))

    # Throughput comparison
    plt.subplot(1, 3, 1)
    throughputs = [results[t]["throughput"] / 1000 for t in topologies]
    plt.bar(topologies, throughputs)
    plt.title("Throughput Comparison")
    plt.ylabel("Throughput (KB/s)")

    # Delay comparison
    plt.subplot(1, 3, 2)
    delays = [results[t]["average_delay"] * 1000 for t in topologies]
    plt.bar(topologies, delays)
    plt.title("Average Delay Comparison")
    plt.ylabel("Delay (ms)")

    # Packet loss comparison
    plt.subplot(1, 3, 3)
    losses = [results[t]["packet_loss_rate"] * 100 for t in topologies]
    plt.bar(topologies, losses)
    plt.title("Packet Loss Comparison")
    plt.ylabel("Loss Rate (%)")

    plt.tight_layout()
    plt.savefig("topology_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Run a single simulation
    run_simulation(
        topology_type="dumbbell", traffic_pattern=TrafficPattern.CONSTANT, duration=10
    )

    # Compare different traffic patterns
    # compare_traffic_patterns()

    # Compare different topologies
    # compare_topologies()
