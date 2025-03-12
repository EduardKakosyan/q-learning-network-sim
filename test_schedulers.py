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


def create_test_topology(sim, num_flows=3, link_capacity=10e6, buffer_size=64000):
    """
    Create a simple test topology with a bottleneck link

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


def test_fifo_scheduler():
    """Test the FIFO scheduler with a simple scenario"""
    print("\n=== Testing FIFO Scheduler ===")

    # Create simulator with FIFO scheduler
    sim = NetworkSimulator(seed=42, scheduler_type="FIFO")

    # Create topology
    sources, destinations = create_test_topology(sim, num_flows=3)

    # Create traffic flows
    # Flow 1: High priority, constant bit rate
    sim.env.process(
        sim.packet_generator(
            sources[0],
            destinations[0],
            constant_size(1000),  # 1000 bytes per packet
            constant_traffic(100),  # 100 packets per second
            jitter=0.05,
        )
    )

    # Flow 2: Medium priority, variable bit rate
    sim.env.process(
        sim.packet_generator(
            sources[1],
            destinations[1],
            variable_size(500, 1500),  # 500-1500 bytes per packet
            variable_traffic(50, 150),  # 50-150 packets per second
            jitter=0.1,
        )
    )

    # Flow 3: Low priority, bursty traffic
    sim.env.process(
        sim.packet_generator(
            sources[2],
            destinations[2],
            constant_size(2000),  # 2000 bytes per packet
            poisson_traffic(20),  # Average 20 bursts per second
            pattern=TrafficPattern.BURSTY,
            burst_size=5,
            burst_interval=0.001,
        )
    )

    # Visualize network topology
    plt.figure(figsize=(10, 8))
    sim.visualize_network()
    plt.savefig("fifo_topology.png")
    plt.close()

    # Run simulation
    print("Running simulation with FIFO scheduler...")
    metrics = sim.run(30)

    # Print metrics
    print("\nFIFO Scheduler Results:")
    print(f"Throughput: {metrics['throughput']/1000:.2f} KB/s")
    print(f"Average Delay: {metrics['average_delay']*1000:.2f} ms")
    print(f"Packet Loss Rate: {metrics['packet_loss_rate']*100:.2f}%")

    # Calculate per-flow statistics
    flow_stats = {}

    for packet in sim.completed_packets:
        flow_id = packet.flow_id
        if flow_id not in flow_stats:
            flow_stats[flow_id] = {"packets": 0, "bytes": 0, "delays": []}

        flow_stats[flow_id]["packets"] += 1
        flow_stats[flow_id]["bytes"] += packet.size
        flow_stats[flow_id]["delays"].append(packet.get_total_delay())

    # Print per-flow statistics
    print("\nPer-flow statistics with FIFO:")
    for flow_id, stats in flow_stats.items():
        avg_delay = (
            sum(stats["delays"]) / len(stats["delays"]) if stats["delays"] else 0
        )
        throughput = stats["bytes"] / 30  # bytes per second over 30 seconds

        print(
            f"Flow {flow_id}: {stats['packets']} packets, {throughput/1000:.2f} KB/s, {avg_delay*1000:.2f} ms"
        )

    return metrics, flow_stats


def test_rr_scheduler():
    """Test the Round Robin scheduler with a simple scenario"""
    print("\n=== Testing Round Robin Scheduler ===")

    # Create simulator with RR scheduler
    sim = NetworkSimulator(seed=42, scheduler_type="RR")

    # Create topology
    sources, destinations = create_test_topology(sim, num_flows=3)

    # Create traffic flows (same as FIFO test)
    # Flow 1: High priority, constant bit rate
    sim.env.process(
        sim.packet_generator(
            sources[0],
            destinations[0],
            constant_size(1000),  # 1000 bytes per packet
            constant_traffic(100),  # 100 packets per second
            jitter=0.05,
        )
    )

    # Flow 2: Medium priority, variable bit rate
    sim.env.process(
        sim.packet_generator(
            sources[1],
            destinations[1],
            variable_size(500, 1500),  # 500-1500 bytes per packet
            variable_traffic(50, 150),  # 50-150 packets per second
            jitter=0.1,
        )
    )

    # Flow 3: Low priority, bursty traffic
    sim.env.process(
        sim.packet_generator(
            sources[2],
            destinations[2],
            constant_size(2000),  # 2000 bytes per packet
            poisson_traffic(20),  # Average 20 bursts per second
            pattern=TrafficPattern.BURSTY,
            burst_size=5,
            burst_interval=0.001,
        )
    )

    # Visualize network topology
    plt.figure(figsize=(10, 8))
    sim.visualize_network()
    plt.savefig("rr_topology.png")
    plt.close()

    # Run simulation
    print("Running simulation with Round Robin scheduler...")
    metrics = sim.run(30)

    # Print metrics
    print("\nRound Robin Scheduler Results:")
    print(f"Throughput: {metrics['throughput']/1000:.2f} KB/s")
    print(f"Average Delay: {metrics['average_delay']*1000:.2f} ms")
    print(f"Packet Loss Rate: {metrics['packet_loss_rate']*100:.2f}%")

    # Calculate per-flow statistics
    flow_stats = {}

    for packet in sim.completed_packets:
        flow_id = packet.flow_id
        if flow_id not in flow_stats:
            flow_stats[flow_id] = {"packets": 0, "bytes": 0, "delays": []}

        flow_stats[flow_id]["packets"] += 1
        flow_stats[flow_id]["bytes"] += packet.size
        flow_stats[flow_id]["delays"].append(packet.get_total_delay())

    # Print per-flow statistics
    print("\nPer-flow statistics with Round Robin:")
    for flow_id, stats in flow_stats.items():
        avg_delay = (
            sum(stats["delays"]) / len(stats["delays"]) if stats["delays"] else 0
        )
        throughput = stats["bytes"] / 30  # bytes per second over 30 seconds

        print(
            f"Flow {flow_id}: {stats['packets']} packets, {throughput/1000:.2f} KB/s, {avg_delay*1000:.2f} ms"
        )

    return metrics, flow_stats


def compare_schedulers():
    """Compare FIFO and RR schedulers"""
    # Run tests
    fifo_metrics, fifo_flow_stats = test_fifo_scheduler()
    rr_metrics, rr_flow_stats = test_rr_scheduler()

    # Combine flow stats
    all_flow_ids = set(fifo_flow_stats.keys()) | set(rr_flow_stats.keys())
    all_flow_ids = sorted(list(all_flow_ids))

    # Plot comparison
    plt.figure(figsize=(15, 10))

    # Throughput comparison
    plt.subplot(2, 2, 1)
    schedulers = ["FIFO", "Round Robin"]
    throughputs = [fifo_metrics["throughput"] / 1000, rr_metrics["throughput"] / 1000]
    plt.bar(schedulers, throughputs, color=["skyblue", "lightgreen"])
    plt.title("Overall Throughput Comparison")
    plt.ylabel("Throughput (KB/s)")

    # Delay comparison
    plt.subplot(2, 2, 2)
    delays = [fifo_metrics["average_delay"] * 1000, rr_metrics["average_delay"] * 1000]
    plt.bar(schedulers, delays, color=["skyblue", "lightgreen"])
    plt.title("Overall Delay Comparison")
    plt.ylabel("Delay (ms)")

    # Per-flow throughput comparison
    plt.subplot(2, 2, 3)

    # Set up bar positions
    bar_width = 0.35
    index = np.arange(len(all_flow_ids))

    # Plot FIFO bars
    fifo_throughputs = [
        fifo_flow_stats.get(flow_id, {"bytes": 0})["bytes"] / 30 / 1000
        for flow_id in all_flow_ids
    ]
    plt.bar(index, fifo_throughputs, bar_width, label="FIFO", color="skyblue")

    # Plot RR bars
    rr_throughputs = [
        rr_flow_stats.get(flow_id, {"bytes": 0})["bytes"] / 30 / 1000
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
    plt.title("Per-Flow Throughput")
    plt.xticks(index + bar_width / 2, all_flow_ids)
    plt.legend()

    # Per-flow delay comparison
    plt.subplot(2, 2, 4)

    # Calculate average delays
    fifo_delays = []
    rr_delays = []

    for flow_id in all_flow_ids:
        if flow_id in fifo_flow_stats and fifo_flow_stats[flow_id]["delays"]:
            fifo_delay = (
                sum(fifo_flow_stats[flow_id]["delays"])
                / len(fifo_flow_stats[flow_id]["delays"])
                * 1000
            )
        else:
            fifo_delay = 0

        if flow_id in rr_flow_stats and rr_flow_stats[flow_id]["delays"]:
            rr_delay = (
                sum(rr_flow_stats[flow_id]["delays"])
                / len(rr_flow_stats[flow_id]["delays"])
                * 1000
            )
        else:
            rr_delay = 0

        fifo_delays.append(fifo_delay)
        rr_delays.append(rr_delay)

    # Plot FIFO bars
    plt.bar(index, fifo_delays, bar_width, label="FIFO", color="skyblue")

    # Plot RR bars
    plt.bar(
        index + bar_width, rr_delays, bar_width, label="Round Robin", color="lightgreen"
    )

    plt.xlabel("Flow")
    plt.ylabel("Average Delay (ms)")
    plt.title("Per-Flow Delay")
    plt.xticks(index + bar_width / 2, all_flow_ids)
    plt.legend()

    plt.tight_layout()
    plt.savefig("scheduler_test_comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_schedulers()
