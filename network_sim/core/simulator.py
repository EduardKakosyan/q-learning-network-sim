"""Network simulator class for network simulation.

This module defines the NetworkSimulator class, which provides the main
simulation environment for network simulation.
"""

import simpy
import networkx as nx
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Callable, Optional, Any

from network_sim.core.packet import Packet
from network_sim.core.link import Link
from network_sim.core.node import Node
from network_sim.core.scheduling_algorithms import SchedulingAlgorithm


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
        metrics: Performance metrics for the simulation.
    """

    def __init__(
        self,
        env: simpy.Environment,
        scheduler_type: str,
        seed: int = 42,
    ):
        """Initialize the network simulator.

        Args:
            env: SimPy environment.
            scheduler_type: The scheduler type that is used.
            seed: Random seed for reproducibility.
        """
        self.env = env
        self.scheduler_type = scheduler_type
        self.graph = nx.DiGraph()
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[Tuple[int, int], Link] = {}
        self.packets: List[Packet] = []
        self.active_packets: Set[int] = set()
        self.completed_packets: List[Packet] = []
        self.dropped_packets: List[Tuple[Packet, str]] = []

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
        scheduler: Optional[SchedulingAlgorithm] = None,
        buffer_size: float = float("inf"),
    ) -> Node:
        """Add a node to the network.

        Args:
            node_id: Unique identifier for the node.
            buffer_size: Maximum buffer size in bytes.

        Returns:
            The created Node object.
        """
        node = Node(self.env, node_id, scheduler, buffer_size)
        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        return node

    def add_link(
        self,
        source: int,
        target: int,
        capacity: float,
        propagation_delay: float
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

        link = Link(self.env, source, target, capacity, propagation_delay)
        self.links[(source, target)] = link
        self.nodes[source].add_link(link)
        self.graph.add_edge(
            source,
            target,
            capacity=capacity,
            delay=propagation_delay,
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

    def create_packet(self, source: int, destination: int, size: int) -> Packet:
        """Create a new packet.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            size: Size of packet in bytes.

        Returns:
            The created Packet object.
        """
        packet = Packet(source, destination, size, self.env.now)
        self.packets.append(packet)
        self.active_packets.add(packet.id)
        return packet

    def packet_generator(
        self,
        source: int,
        destination: int,
        packet_size: Callable[[], int],
        interval: Callable[[], float],
        jitter: float = 0,
        burst_size: int = 1,
        burst_interval: Optional[float] = None,
    ) -> simpy.events.Process:
        """Generate packets according to specified pattern.

        Args:
            source: Source node ID.
            destination: Destination node ID.
            packet_size: Size of packets in bytes.
            interval: Time between packets/bursts in seconds.
            jitter: Random variation in interval (fraction of interval).            

        Returns:
            SimPy process for the packet generator.
        """

        def generator_process():
            while True:
                current_interval = interval()

                if jitter > 0:
                    current_interval *= 1 + random.uniform(-jitter, jitter)

                size = packet_size()
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
        
        def packet_journey(packet: Packet):
            current_node = self.nodes[packet.current_node]
            
            if not current_node.can_queue_packet(packet):
                self.packet_dropped(packet, "Buffer overflow")
                yield self.env.timeout(0)
                return

            current_node.add_packet_to_queue(packet)

            queuing_start = self.env.now
            with current_node.resource.request() as node_resource:
                yield node_resource
                queuing_delay = self.env.now - queuing_start
                packet.record_queuing_delay(queuing_delay)

                if packet.current_node == packet.destination:
                    self.packet_arrived(packet)
                    return

                next_hop = current_node.get_next_hop(packet.destination)
                if next_hop is None:
                    self.packet_dropped(packet, "No route to destination")
                    return

                link = current_node.links.get(next_hop)
                if link is None:
                    self.packet_dropped(packet, "No link to next hop")
                    return

                packet, scheduling_delay = current_node.get_next_packet(link)
                if packet == None:
                    return
                yield self.env.timeout(scheduling_delay)

            queuing_start = self.env.now
            with link.resource.request() as link_resource:
                yield link_resource
                queuing_delay = self.env.now - queuing_start
                packet.record_queuing_delay(queuing_delay)

                transmission_delay = link.calculate_transmission_delay(packet.size)
                yield self.env.timeout(transmission_delay)
        
            yield self.env.timeout(link.propagation_delay)
            link.bytes_sent += packet.size

            packet.record_hop(next_hop, self.env.now)
            self.env.process(self.process_packet(packet))
        
        return packet_journey(packet)

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
