"""Network simulator class for network simulation.

This module defines the NetworkSimulator class, which provides the main
simulation environment for network simulation.
"""

import simpy
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Callable, Optional, Any, Union

from network_sim.core.packet import Packet
from network_sim.core.link import Link
from network_sim.core.node import Node
from network_sim.core.enums import TrafficPattern


class NetworkSimulator:
    """Network simulation environment.

    Attributes:
        env: SimPy environment.
        graph: NetworkX directed graph representing the network.
        nodes: Node objects keyed by node ID.
        links: Link objects keyed by (source, target) tuple.
        packets: All packets in the simulation.
        active_packets: Packets currently in transit.
        completed_packets: Packets that reached their destination.
        dropped_packets: Packets that were dropped.
        scheduler_type: Type of scheduler to use.
        metrics: Performance metrics for the simulation.
    """

    def __init__(
        self,
        env: Optional[simpy.Environment] = None,
        seed: int = 42,
        scheduler_type: str = "FIFO",
    ):
        """Initialize the network simulator.

        Args:
            env: SimPy environment (creates one if None).
            seed: Random seed for reproducibility.
            scheduler_type: Type of scheduler to use ("FIFO", "RR", or "QL").
        """
        self.env = env if env is not None else simpy.Environment()
        self.graph = nx.DiGraph()
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[Tuple[int, int], Link] = {}
        self.packets: List[Packet] = []
        self.active_packets: Set[int] = set()
        self.completed_packets: List[Packet] = []
        self.dropped_packets: List[Tuple[Packet, str]] = []
        self.scheduler_type = scheduler_type

        random.seed(seed)
        np.random.seed(seed)

        self.metrics: Dict[str, Any] = {
            "throughput": 0,
            "average_delay": 0,
            "packet_loss_rate": 0,
            "link_utilization": defaultdict(float),
            "queue_lengths": defaultdict(list),
        }

    def add_node(
        self,
        node_id: int,
        processing_delay: float = 0,
        buffer_size: float = float("inf"),
    ) -> Node:
        """Add a node to the network.

        Args:
            node_id: Unique identifier for the node.
            processing_delay: Time to process a packet in seconds.
            buffer_size: Maximum buffer size in bytes.

        Returns:
            The created Node object.
        """
        node = Node(self.env, node_id, processing_delay, buffer_size)
        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        return node

    def add_link(
        self,
        source: int,
        target: int,
        capacity: float,
        propagation_delay: float,
        buffer_size: float = float("inf"),
    ) -> Link:
        """Add a link between nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            capacity: Link capacity in bits per second.
            propagation_delay: Propagation delay in seconds.
            buffer_size: Maximum buffer size in bytes.

        Returns:
            The created Link object.
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Nodes {source} and/or {target} do not exist")

        link = Link(self.env, source, target, capacity, propagation_delay, buffer_size)
        self.links[(source, target)] = link
        self.nodes[source].add_link(link)
        self.graph.add_edge(
            source,
            target,
            capacity=capacity,
            delay=propagation_delay,
            buffer=buffer_size,
        )
        return link

    def compute_shortest_paths(self) -> None:
        """Compute shortest paths and set routing tables for all nodes."""
        shortest_paths = nx.all_pairs_dijkstra_path(self.graph, weight="delay")

        for source, paths in shortest_paths:
            routing_table = {}
            for destination, path in paths.items():
                if source != destination and len(path) > 1:
                    routing_table[destination] = path[1]
            self.nodes[source].set_routing_table(routing_table)

    def create_packet(
        self, source: int, destination: int, size: int, priority: int = 0
    ) -> Packet:
        """Create a new packet.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            size: Size of packet in bytes.
            priority: Priority level (higher means more important).

        Returns:
            The created Packet object.
        """
        packet = Packet(source, destination, size, priority, self.env.now)
        self.packets.append(packet)
        self.active_packets.add(packet.id)
        return packet

    def packet_generator(
        self,
        source: int,
        destination: int,
        packet_size: Union[int, Callable[[], int]],
        interval: Union[float, Callable[[], float]],
        jitter: float = 0,
        pattern: TrafficPattern = TrafficPattern.CONSTANT,
        burst_size: int = 1,
        burst_interval: Optional[float] = None,
    ) -> simpy.events.Process:
        """Generate packets according to specified pattern.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            packet_size: Size of packets in bytes (can be callable for variable sizes).
            interval: Time between packets/bursts in seconds (can be callable).
            jitter: Random variation in interval (fraction of interval).
            pattern: Traffic pattern (CONSTANT, VARIABLE, BURSTY, MIXED).
            burst_size: Number of packets in a burst (for BURSTY pattern).
            burst_interval: Time between packets in a burst (for BURSTY pattern).

        Returns:
            SimPy process for the packet generator.
        """

        def generator_process():
            while True:
                current_interval = interval() if callable(interval) else interval

                if jitter > 0:
                    current_interval *= 1 + random.uniform(-jitter, jitter)

                if pattern == TrafficPattern.BURSTY:
                    for i in range(burst_size):
                        size = packet_size() if callable(packet_size) else packet_size
                        packet = self.create_packet(source, destination, size)

                        self.env.process(self.process_packet(packet))

                        if i < burst_size - 1 and burst_interval is not None:
                            yield self.env.timeout(burst_interval)
                else:
                    size = packet_size() if callable(packet_size) else packet_size
                    packet = self.create_packet(source, destination, size)

                    self.env.process(self.process_packet(packet))

                yield self.env.timeout(current_interval)

        return self.env.process(generator_process())

    def process_packet(self, packet: Packet) -> simpy.events.Process:
        """Process a packet's journey through the network.

        Args:
            packet: The Packet object to process.

        Returns:
            SimPy process for the packet journey.
        """

        # Define the generator function directly
        def packet_journey():
            current_node = self.nodes[packet.current_node]

            if packet.current_node == packet.destination:
                yield self.env.timeout(current_node.processing_delay)
                self.packet_arrived(packet)
                return

            next_hop = current_node.get_next_hop(packet.destination)
            if next_hop is None:
                self.packet_dropped(packet, "No route to destination")
                yield self.env.timeout(0)
                return

            link = current_node.links.get(next_hop)
            if link is None:
                self.packet_dropped(packet, "No link to next hop")
                yield self.env.timeout(0)
                return

            if not link.can_queue_packet(packet):
                self.packet_dropped(packet, "Buffer overflow")
                link.packets_dropped += 1
                yield self.env.timeout(0)
                return

            queuing_start = self.env.now
            link.buffer_usage += packet.size
            link.add_packet_to_queue(packet, self.scheduler_type)

            with link.resource.request() as request:
                yield request

                next_packet = link.get_next_packet(self.scheduler_type)
                if next_packet and next_packet.id == packet.id:
                    queuing_delay = self.env.now - queuing_start
                    packet.record_queuing_delay(queuing_delay)

                    transmission_delay = link.calculate_transmission_delay(packet.size)
                    yield self.env.timeout(transmission_delay)
                    yield self.env.timeout(link.propagation_delay)

                    link.buffer_usage -= packet.size
                    link.packets_sent += 1
                    link.bytes_sent += packet.size

                    packet.record_hop(next_hop, self.env.now)

                    # Create a new process for the next hop
                    next_process = self.process_packet(packet)
                    self.env.process(next_process)
                else:
                    self.packet_dropped(packet, "Scheduling error")
                    yield self.env.timeout(0)

        # Return the generator function directly
        return packet_journey()

    def packet_arrived(self, packet: Packet) -> None:
        """Handle packet arrival at destination.

        Args:
            packet: The packet that arrived.
        """
        packet.arrival_time = self.env.now
        self.active_packets.remove(packet.id)
        self.completed_packets.append(packet)
        self.nodes[packet.destination].packets_received += 1

    def packet_dropped(self, packet: Packet, reason: str) -> None:
        """Handle packet drop.

        Args:
            packet: The packet that was dropped.
            reason: Reason for dropping the packet.
        """
        packet.dropped = True
        if packet.id in self.active_packets:
            self.active_packets.remove(packet.id)
        self.dropped_packets.append((packet, reason))
        self.nodes[packet.current_node].packets_dropped += 1

    def calculate_metrics(
        self, start_time: float = 0, end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate performance metrics.

        Args:
            start_time: Start time for metric calculation.
            end_time: End time for metric calculation (defaults to current time).

        Returns:
            Dictionary of calculated metrics.
        """
        if end_time is None:
            end_time = self.env.now

        simulation_time = end_time - start_time

        if simulation_time <= 0:
            return self.metrics

        total_bytes = sum(
            p.size
            for p in self.completed_packets
            if start_time <= p.arrival_time <= end_time
        )
        throughput = total_bytes / simulation_time

        delays = [
            p.get_total_delay()
            for p in self.completed_packets
            if start_time <= p.arrival_time <= end_time
        ]
        average_delay = sum(delays) / len(delays) if delays else 0

        total_packets = len(self.completed_packets) + len(self.dropped_packets)
        packet_loss_rate = (
            len(self.dropped_packets) / total_packets if total_packets > 0 else 0
        )

        link_utilization = {}
        for (source, target), link in self.links.items():
            bits_sent = link.bytes_sent * 8
            max_bits = link.capacity * simulation_time
            utilization = bits_sent / max_bits if max_bits > 0 else 0
            link_utilization[(source, target)] = utilization

        self.metrics["throughput"] = throughput
        self.metrics["average_delay"] = average_delay
        self.metrics["packet_loss_rate"] = packet_loss_rate
        self.metrics["link_utilization"] = link_utilization
        self.metrics["scheduler_type"] = self.scheduler_type

        return self.metrics

    def run(self, duration: float) -> Dict[str, Any]:
        """Run the simulation for a specified duration.

        Args:
            duration: Simulation duration in seconds.

        Returns:
            Dictionary of calculated metrics.
        """
        self.env.run(until=duration)

        self.calculate_metrics()

        return self.metrics

    def set_scheduler(self, scheduler_type: str) -> None:
        """Set the scheduler type.

        Args:
            scheduler_type: Type of scheduler to use ("FIFO", "RR", or "QL").
        """
        self.scheduler_type = scheduler_type

    def visualize_network(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Visualize the network topology.

        Args:
            figsize: Figure size as (width, height) in inches.

        Returns:
            Matplotlib figure object.
        """
        fig = plt.figure(figsize=figsize)

        pos = nx.spring_layout(self.graph)

        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color="lightblue")

        edge_capacities = [
            self.graph[u][v]["capacity"] / 1000000 for u, v in self.graph.edges()
        ]
        max_capacity = max(edge_capacities) if edge_capacities else 1
        edge_widths = [cap / max_capacity * 3 for cap in edge_capacities]

        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=edge_widths,
            alpha=0.7,
            edge_color="gray",
            arrows=True,
            arrowsize=15,
        )

        nx.draw_networkx_labels(self.graph, pos, font_size=12)

        edge_labels = {
            (
                u,
                v,
            ): f"{self.graph[u][v]['capacity']/1000000:.1f}Mbps\n{self.graph[u][v]['delay']*1000:.1f}ms"
            for u, v in self.graph.edges()
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=8
        )

        plt.title(f"Network Topology (Scheduler: {self.scheduler_type})")
        plt.axis("off")
        plt.tight_layout()
        return fig
