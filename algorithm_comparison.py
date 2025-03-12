import simpy
import matplotlib.pyplot as plt
import numpy as np
from network_simulator import (
    NetworkSimulator,
    TrafficPattern,
    constant_traffic,
    variable_traffic,
    poisson_traffic,
    constant_size,
    variable_size,
)
from scheduling_algorithms import FIFOScheduler, RoundRobinScheduler, QLearningScheduler


def create_bottleneck_topology(sim, num_flows=3, link_capacity=10e6, buffer_size=64000):
    """
    Create a topology with a bottleneck link

    Args:
        sim: NetworkSimulator instance
        num_flows: Number of flows
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

    for i in range(num_flows):
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
    bottleneck_capacity = link_capacity * 0.3  # Make it a significant bottleneck
    sim.add_link("R1", "R2", bottleneck_capacity, 0.005, buffer_size)

    # Compute routing tables
    sim.compute_shortest_paths()

    return sources, destinations


def run_simulation_with_scheduler(
    scheduler_type, traffic_pattern=TrafficPattern.MIXED, duration=30
):
    """
    Run a simulation with a specific scheduler

    Args:
        scheduler_type: Type of scheduler to use ("FIFO", "RR", or "QL")
        traffic_pattern: Traffic pattern to use
        duration: Simulation duration in seconds

    Returns:
        Simulation metrics
    """
    # Create simulator with the specified scheduler
    sim = NetworkSimulator(seed=42, scheduler_type=scheduler_type)

    # Create topology
    sources, destinations = create_bottleneck_topology(sim, num_flows=5)

    print(f"Running simulation with {scheduler_type} scheduler...")

    # Create traffic flows with different characteristics
    for i, (source, destination) in enumerate(zip(sources, destinations)):
        # Vary traffic patterns for different flows
        if i % 3 == 0:
            # High priority, constant bit rate traffic
            interval_gen = constant_traffic(100)  # 100 packets per second
            size_gen = constant_size(1000)  # 1000 bytes per packet
            priority = 2
        elif i % 3 == 1:
            # Medium priority, variable bit rate traffic
            interval_gen = variable_traffic(50, 150)  # 50-150 packets per second
            size_gen = variable_size(500, 1500)  # 500-1500 bytes per packet
            priority = 1
        else:
            # Low priority, bursty traffic
            interval_gen = poisson_traffic(20)  # Average 20 bursts per second
            size_gen = constant_size(2000)  # 2000 bytes per packet
            priority = 0

            # For bursty traffic, use the bursty pattern
            if (
                traffic_pattern == TrafficPattern.BURSTY
                or traffic_pattern == TrafficPattern.MIXED
            ):
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

        # Start packet generator process
        sim.env.process(
            sim.packet_generator(
                source, destination, size_gen, interval_gen, jitter=0.1
            )
        )

    # Visualize network topology
    plt.figure(figsize=(10, 8))
    sim.visualize_network()
    plt.savefig(f"{scheduler_type}_topology.png")
    plt.close()

    # Run simulation
    metrics = sim.run(duration)

    # Print metrics
    print(f"\nResults with {scheduler_type} scheduler:")
    print(f"Throughput: {metrics['throughput']/1000:.2f} KB/s")
    print(f"Average Delay: {metrics['average_delay']*1000:.2f} ms")
    print(f"Packet Loss Rate: {metrics['packet_loss_rate']*100:.2f}%")

    # Print link utilization for bottleneck link
    print("\nLink Utilization:")
    for (src, dst), util in metrics["link_utilization"].items():
        if src == "R1" and dst == "R2":  # Bottleneck link
            print(f"Bottleneck Link {src} -> {dst}: {util*100:.2f}%")
        else:
            print(f"{src} -> {dst}: {util*100:.2f}%")

    return metrics


def compare_schedulers():
    """Compare different scheduling algorithms"""
    schedulers = ["FIFO", "RR"]  # We're implementing FIFO and RR for now

    results = {}

    for scheduler_type in schedulers:
        # Run simulation with this scheduler
        results[scheduler_type] = run_simulation_with_scheduler(
            scheduler_type, traffic_pattern=TrafficPattern.MIXED, duration=30
        )

    # Plot comparison
    plt.figure(figsize=(15, 5))

    # Throughput comparison
    plt.subplot(1, 3, 1)
    throughputs = [results[s]["throughput"] / 1000 for s in schedulers]
    plt.bar(schedulers, throughputs)
    plt.title("Throughput Comparison")
    plt.ylabel("Throughput (KB/s)")
    plt.xticks(rotation=45)

    # Delay comparison
    plt.subplot(1, 3, 2)
    delays = [results[s]["average_delay"] * 1000 for s in schedulers]
    plt.bar(schedulers, delays)
    plt.title("Average Delay Comparison")
    plt.ylabel("Delay (ms)")
    plt.xticks(rotation=45)

    # Packet loss comparison
    plt.subplot(1, 3, 3)
    losses = [results[s]["packet_loss_rate"] * 100 for s in schedulers]
    plt.bar(schedulers, losses)
    plt.title("Packet Loss Comparison")
    plt.ylabel("Loss Rate (%)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("scheduler_comparison.png")
    plt.show()

    return results


def run_fairness_test():
    """
    Run a test to compare fairness between FIFO and Round Robin

    This test creates a scenario with uneven traffic to highlight
    the fairness properties of Round Robin compared to FIFO.
    """
    print("\n=== Running Fairness Test ===")

    # Define traffic patterns for fairness test
    # We'll have one aggressive flow and several normal flows

    results = {}

    for scheduler_type in ["FIFO", "RR"]:
        # Create simulator with the specified scheduler
        sim = NetworkSimulator(seed=42, scheduler_type=scheduler_type)

        # Create topology with 3 flows
        sources, destinations = create_bottleneck_topology(sim, num_flows=3)

        print(f"Running fairness test with {scheduler_type} scheduler...")

        # Create one aggressive flow (high rate, large packets)
        aggressive_source = sources[0]
        aggressive_dest = destinations[0]

        # Aggressive flow: 200 packets/sec, 2000 bytes each
        sim.env.process(
            sim.packet_generator(
                aggressive_source,
                aggressive_dest,
                constant_size(2000),
                constant_traffic(200),
                jitter=0.05,
            )
        )

        # Create two normal flows
        for i in range(1, 3):
            # Normal flow: 50 packets/sec, 1000 bytes each
            sim.env.process(
                sim.packet_generator(
                    sources[i],
                    destinations[i],
                    constant_size(1000),
                    constant_traffic(50),
                    jitter=0.05,
                )
            )

        # Run simulation
        sim.run(30)

        # Calculate per-flow statistics
        flow_stats = {}

        for packet in sim.completed_packets:
            flow_id = packet.flow_id
            if flow_id not in flow_stats:
                flow_stats[flow_id] = {"packets": 0, "bytes": 0, "delays": []}

            flow_stats[flow_id]["packets"] += 1
            flow_stats[flow_id]["bytes"] += packet.size
            flow_stats[flow_id]["delays"].append(packet.get_total_delay())

        # Calculate average throughput and delay per flow
        print(f"\nPer-flow statistics with {scheduler_type}:")
        for flow_id, stats in flow_stats.items():
            avg_delay = (
                sum(stats["delays"]) / len(stats["delays"]) if stats["delays"] else 0
            )
            throughput = stats["bytes"] / 30  # bytes per second over 30 seconds

            print(
                f"Flow {flow_id}: {stats['packets']} packets, {throughput/1000:.2f} KB/s, {avg_delay*1000:.2f} ms"
            )

            # Add to results
            if scheduler_type not in results:
                results[scheduler_type] = {}

            results[scheduler_type][flow_id] = {
                "packets": stats["packets"],
                "throughput": throughput,
                "avg_delay": avg_delay,
            }

    # Calculate fairness index for each scheduler
    for scheduler_type in results:
        throughputs = [
            stats["throughput"] for flow_id, stats in results[scheduler_type].items()
        ]
        fairness_index = calculate_jains_fairness_index(throughputs)
        print(f"\n{scheduler_type} Jain's Fairness Index: {fairness_index:.4f}")

    # Plot fairness comparison
    plt.figure(figsize=(12, 6))

    # Plot throughput per flow for each scheduler
    plt.subplot(1, 2, 1)

    # Get flow IDs and organize data
    all_flow_ids = set()
    for scheduler in results:
        all_flow_ids.update(results[scheduler].keys())

    all_flow_ids = sorted(list(all_flow_ids))

    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(all_flow_ids))

    # Plot FIFO bars
    fifo_throughputs = [
        results["FIFO"].get(flow_id, {"throughput": 0})["throughput"] / 1000
        for flow_id in all_flow_ids
    ]
    plt.bar(index, fifo_throughputs, bar_width, label="FIFO", color="skyblue")

    # Plot RR bars
    rr_throughputs = [
        results["RR"].get(flow_id, {"throughput": 0})["throughput"] / 1000
        for flow_id in all_flow_ids
    ]
    plt.bar(
        index + bar_width,
        rr_throughputs,
        bar_width,
        label="Round Robin",
        color="lightgreen",
    )

    plt.xlabel("Flow")
    plt.ylabel("Throughput (KB/s)")
    plt.title("Throughput per Flow")
    plt.xticks(index + bar_width / 2, all_flow_ids)
    plt.legend()

    # Plot delay per flow for each scheduler
    plt.subplot(1, 2, 2)

    # Plot FIFO bars
    fifo_delays = [
        results["FIFO"].get(flow_id, {"avg_delay": 0})["avg_delay"] * 1000
        for flow_id in all_flow_ids
    ]
    plt.bar(index, fifo_delays, bar_width, label="FIFO", color="skyblue")

    # Plot RR bars
    rr_delays = [
        results["RR"].get(flow_id, {"avg_delay": 0})["avg_delay"] * 1000
        for flow_id in all_flow_ids
    ]
    plt.bar(
        index + bar_width, rr_delays, bar_width, label="Round Robin", color="lightgreen"
    )

    plt.xlabel("Flow")
    plt.ylabel("Average Delay (ms)")
    plt.title("Delay per Flow")
    plt.xticks(index + bar_width / 2, all_flow_ids)
    plt.legend()

    plt.tight_layout()
    plt.savefig("fairness_comparison.png")
    plt.show()

    return results


def calculate_jains_fairness_index(values):
    """
    Calculate Jain's fairness index

    Args:
        values: List of values to calculate fairness for

    Returns:
        Fairness index between 0 and 1 (1 is perfectly fair)
    """
    if not values:
        return 0

    n = len(values)
    sum_values = sum(values)
    sum_squared = sum(x**2 for x in values)

    if sum_squared == 0:
        return 0

    return (sum_values**2) / (n * sum_squared)


if __name__ == "__main__":
    # Compare FIFO and RR schedulers
    compare_results = compare_schedulers()

    # Run fairness test to highlight differences
    fairness_results = run_fairness_test()
