import simpy
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from enum import Enum


class TrafficPattern(Enum):
    """Enum for different traffic patterns"""

    CONSTANT = 1
    VARIABLE = 2
    BURSTY = 3
    MIXED = 4


class Packet:
    """Represents a network packet"""

    id_counter = 0

    def __init__(self, source, destination, size, priority=0, creation_time=0):
        """
        Initialize a packet

        Args:
            source: Source node ID
            destination: Destination node ID
            size: Size of packet in bytes
            priority: Priority level (higher means more important)
            creation_time: Time when packet was created
        """
        Packet.id_counter += 1
        self.id = Packet.id_counter
        self.source = source
        self.destination = destination
        self.size = size
        self.priority = priority
        self.creation_time = creation_time
        self.current_node = source
        self.hops = []
        self.arrival_time = None
        self.dropped = False
        self.queuing_delays = []
        # Add flow identifier (source-destination pair)
        self.flow_id = f"{source}-{destination}"

    def __repr__(self):
        return f"Packet(id={self.id}, src={self.source}, dst={self.destination}, size={self.size}B)"

    def record_hop(self, node, time):
        """Record a hop in the packet's journey"""
        self.hops.append((node, time))
        self.current_node = node

    def record_queuing_delay(self, delay):
        """Record queuing delay at a node"""
        self.queuing_delays.append(delay)

    def get_total_delay(self):
        """Calculate total delay if packet has arrived"""
        if self.arrival_time is None:
            return None
        return self.arrival_time - self.creation_time

    def get_hop_count(self):
        """Get number of hops taken"""
        return len(self.hops)


class Link:
    """Represents a network link between nodes"""

    def __init__(
        self, env, source, target, capacity, propagation_delay, buffer_size=float("inf")
    ):
        """
        Initialize a network link

        Args:
            env: SimPy environment
            source: Source node ID
            target: Target node ID
            capacity: Link capacity in bits per second
            propagation_delay: Propagation delay in seconds
            buffer_size: Maximum buffer size in bytes (default: infinite)
        """
        self.env = env
        self.source = source
        self.target = target
        self.capacity = capacity  # bits per second
        self.propagation_delay = propagation_delay  # seconds
        self.buffer_size = buffer_size  # bytes
        self.buffer_usage = 0  # current buffer usage in bytes
        self.busy = False  # is the link currently transmitting
        self.queue = deque()  # packet queue
        self.packets_dropped = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.resource = simpy.Resource(
            env, capacity=1
        )  # link can only transmit one packet at a time

        # For Round Robin: separate queues for different flows
        self.flow_queues = defaultdict(deque)
        self.active_flows = []  # List of active flow IDs
        self.current_flow_index = 0  # Index of the current flow being served

    def can_queue_packet(self, packet):
        """Check if there's enough buffer space for the packet"""
        return self.buffer_usage + packet.size <= self.buffer_size

    def calculate_transmission_delay(self, packet_size):
        """Calculate transmission delay based on packet size and link capacity"""
        # Convert packet size from bytes to bits and divide by capacity (bits/second)
        return (packet_size * 8) / self.capacity

    def get_total_delay(self, packet):
        """Calculate total delay for a packet (transmission + propagation)"""
        transmission_delay = self.calculate_transmission_delay(packet.size)
        return transmission_delay + self.propagation_delay

    def add_packet_to_queue(self, packet, scheduler_type):
        """
        Add a packet to the appropriate queue based on scheduler type

        Args:
            packet: The packet to add
            scheduler_type: Type of scheduler being used
        """
        if scheduler_type == "FIFO":
            # For FIFO, just add to the main queue
            self.queue.append(packet)
        elif scheduler_type == "RR":
            # For Round Robin, add to the flow-specific queue
            flow_id = packet.flow_id
            self.flow_queues[flow_id].append(packet)

            # Add flow to active flows list if not already there
            if flow_id not in self.active_flows:
                self.active_flows.append(flow_id)
        else:
            # Default to FIFO for other schedulers
            self.queue.append(packet)

    def get_next_packet(self, scheduler_type):
        """
        Get the next packet to transmit based on scheduler type

        Args:
            scheduler_type: Type of scheduler being used

        Returns:
            The next packet to transmit or None if no packets are available
        """
        if scheduler_type == "FIFO":
            # For FIFO, just get the first packet in the queue
            return self.queue.popleft() if self.queue else None

        elif scheduler_type == "RR":
            # For Round Robin, cycle through flows
            if not self.active_flows:
                return None

            # Try to find a non-empty flow queue
            attempts = 0
            while attempts < len(self.active_flows):
                # Get the current flow
                flow_id = self.active_flows[self.current_flow_index]

                # Move to the next flow for the next time
                self.current_flow_index = (self.current_flow_index + 1) % len(
                    self.active_flows
                )

                # If this flow has packets, return the first one
                if self.flow_queues[flow_id]:
                    packet = self.flow_queues[flow_id].popleft()

                    # If the flow queue is now empty, remove it from active flows
                    if not self.flow_queues[flow_id]:
                        self.active_flows.remove(flow_id)
                        # Adjust the current index if necessary
                        if self.active_flows and self.current_flow_index >= len(
                            self.active_flows
                        ):
                            self.current_flow_index = 0

                    return packet

                attempts += 1

            return None
        else:
            # Default to FIFO for other schedulers
            return self.queue.popleft() if self.queue else None

    def __repr__(self):
        return f"Link({self.source}->{self.target}, {self.capacity/1000000:.1f}Mbps, {self.propagation_delay*1000:.1f}ms)"


class Node:
    """Represents a network node (router, switch, host)"""

    def __init__(self, env, node_id, processing_delay=0, buffer_size=float("inf")):
        """
        Initialize a network node

        Args:
            env: SimPy environment
            node_id: Unique identifier for the node
            processing_delay: Time to process a packet in seconds
            buffer_size: Maximum buffer size in bytes (default: infinite)
        """
        self.env = env
        self.id = node_id
        self.processing_delay = processing_delay
        self.buffer_size = buffer_size
        self.links = {}  # outgoing links keyed by destination
        self.routing_table = {}  # next hop for each destination
        self.packets_received = 0
        self.packets_dropped = 0
        self.packets_forwarded = 0
        self.is_destination = False  # whether this node is a destination

    def add_link(self, link):
        """Add an outgoing link from this node"""
        if link.source == self.id:
            self.links[link.target] = link

    def set_routing_table(self, routing_table):
        """Set the routing table for this node"""
        self.routing_table = routing_table

    def get_next_hop(self, destination):
        """Get next hop for a destination from routing table"""
        return self.routing_table.get(destination)

    def __repr__(self):
        return f"Node({self.id})"


class NetworkSimulator:
    """Network simulation environment"""

    def __init__(self, env=None, seed=42, scheduler_type="FIFO"):
        """
        Initialize the network simulator

        Args:
            env: SimPy environment (creates one if None)
            seed: Random seed for reproducibility
            scheduler_type: Type of scheduler to use ("FIFO", "RR", or "QL")
        """
        self.env = env if env is not None else simpy.Environment()
        self.graph = nx.DiGraph()  # directed graph representing the network
        self.nodes = {}  # node objects keyed by node ID
        self.links = {}  # link objects keyed by (source, target) tuple
        self.packets = []  # all packets in the simulation
        self.active_packets = set()  # packets currently in transit
        self.completed_packets = []  # packets that reached their destination
        self.dropped_packets = []  # packets that were dropped
        self.scheduler_type = scheduler_type  # type of scheduler to use

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Performance metrics
        self.metrics = {
            "throughput": 0,
            "average_delay": 0,
            "packet_loss_rate": 0,
            "link_utilization": defaultdict(float),
            "queue_lengths": defaultdict(list),
        }

    def add_node(self, node_id, processing_delay=0, buffer_size=float("inf")):
        """
        Add a node to the network

        Args:
            node_id: Unique identifier for the node
            processing_delay: Time to process a packet in seconds
            buffer_size: Maximum buffer size in bytes

        Returns:
            The created Node object
        """
        node = Node(self.env, node_id, processing_delay, buffer_size)
        self.nodes[node_id] = node
        self.graph.add_node(node_id)
        return node

    def add_link(
        self, source, target, capacity, propagation_delay, buffer_size=float("inf")
    ):
        """
        Add a link between nodes

        Args:
            source: Source node ID
            target: Target node ID
            capacity: Link capacity in bits per second
            propagation_delay: Propagation delay in seconds
            buffer_size: Maximum buffer size in bytes

        Returns:
            The created Link object
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

    def compute_shortest_paths(self):
        """Compute shortest paths and set routing tables for all nodes"""
        # Use NetworkX to compute shortest paths based on propagation delay
        shortest_paths = nx.all_pairs_dijkstra_path(self.graph, weight="delay")

        # Set routing tables for each node
        for source, paths in shortest_paths:
            routing_table = {}
            for destination, path in paths.items():
                if source != destination and len(path) > 1:
                    # Next hop is the second node in the path
                    routing_table[destination] = path[1]
            self.nodes[source].set_routing_table(routing_table)

    def create_packet(self, source, destination, size, priority=0):
        """
        Create a new packet

        Args:
            source: Source node ID
            destination: Destination node ID
            size: Size of packet in bytes
            priority: Priority level (higher means more important)

        Returns:
            The created Packet object
        """
        packet = Packet(source, destination, size, priority, self.env.now)
        self.packets.append(packet)
        self.active_packets.add(packet.id)
        return packet

    def packet_generator(
        self,
        source,
        destination,
        packet_size,
        interval,
        jitter=0,
        pattern=TrafficPattern.CONSTANT,
        burst_size=1,
        burst_interval=None,
    ):
        """
        Generate packets according to specified pattern

        Args:
            source: Source node ID
            destination: Destination node ID
            packet_size: Size of packets in bytes (can be callable for variable sizes)
            interval: Time between packets/bursts in seconds (can be callable)
            jitter: Random variation in interval (fraction of interval)
            pattern: Traffic pattern (CONSTANT, VARIABLE, BURSTY, MIXED)
            burst_size: Number of packets in a burst (for BURSTY pattern)
            burst_interval: Time between packets in a burst (for BURSTY pattern)
        """
        while True:
            # Determine current interval based on pattern
            current_interval = interval() if callable(interval) else interval

            # Add jitter if specified
            if jitter > 0:
                current_interval *= 1 + random.uniform(-jitter, jitter)

            if pattern == TrafficPattern.BURSTY:
                # Generate a burst of packets
                for i in range(burst_size):
                    # Create packet
                    size = packet_size() if callable(packet_size) else packet_size
                    packet = self.create_packet(source, destination, size)

                    # Start packet journey
                    self.env.process(self.process_packet(packet))

                    # Wait between packets in burst if specified
                    if i < burst_size - 1 and burst_interval is not None:
                        yield self.env.timeout(burst_interval)
            else:
                # Create a single packet
                size = packet_size() if callable(packet_size) else packet_size
                packet = self.create_packet(source, destination, size)

                # Start packet journey
                self.env.process(self.process_packet(packet))

            # Wait until next packet/burst
            yield self.env.timeout(current_interval)

    def process_packet(self, packet):
        """
        Process a packet's journey through the network

        Args:
            packet: The Packet object to process
        """
        current_node = self.nodes[packet.current_node]

        # If we're already at the destination
        if packet.current_node == packet.destination:
            yield self.env.timeout(current_node.processing_delay)
            self.packet_arrived(packet)
            return

        # Get next hop from routing table
        next_hop = current_node.get_next_hop(packet.destination)

        # If no route to destination
        if next_hop is None:
            self.packet_dropped(packet, "No route to destination")
            return

        # Get the link to the next hop
        link = current_node.links.get(next_hop)

        # If no link to next hop
        if link is None:
            self.packet_dropped(packet, "No link to next hop")
            return

        # Check if link has enough buffer space
        if not link.can_queue_packet(packet):
            self.packet_dropped(packet, "Buffer overflow")
            link.packets_dropped += 1
            return

        # Add packet to link's buffer using the appropriate scheduler
        queuing_start = self.env.now
        link.buffer_usage += packet.size
        link.add_packet_to_queue(packet, self.scheduler_type)

        # Request access to the link
        with link.resource.request() as request:
            yield request

            # Get the next packet to transmit based on the scheduler
            next_packet = link.get_next_packet(self.scheduler_type)

            # If this is our packet, process it
            if next_packet and next_packet.id == packet.id:
                # Record queuing delay
                queuing_delay = self.env.now - queuing_start
                packet.record_queuing_delay(queuing_delay)

                # Calculate transmission delay
                transmission_delay = link.calculate_transmission_delay(packet.size)

                # Transmit the packet (this takes time)
                yield self.env.timeout(transmission_delay)

                # Propagation delay
                yield self.env.timeout(link.propagation_delay)

                # Update link statistics
                link.buffer_usage -= packet.size
                link.packets_sent += 1
                link.bytes_sent += packet.size

                # Packet has arrived at next hop
                packet.record_hop(next_hop, self.env.now)

                # Continue packet journey from next hop
                self.env.process(self.process_packet(packet))
            else:
                # This shouldn't happen in normal operation
                # If it does, it means our packet was somehow lost in the queue
                self.packet_dropped(packet, "Scheduling error")

    def packet_arrived(self, packet):
        """Handle packet arrival at destination"""
        packet.arrival_time = self.env.now
        self.active_packets.remove(packet.id)
        self.completed_packets.append(packet)
        self.nodes[packet.destination].packets_received += 1

    def packet_dropped(self, packet, reason):
        """Handle packet drop"""
        packet.dropped = True
        if packet.id in self.active_packets:
            self.active_packets.remove(packet.id)
        self.dropped_packets.append((packet, reason))
        self.nodes[packet.current_node].packets_dropped += 1

    def calculate_metrics(self, start_time=0, end_time=None):
        """
        Calculate performance metrics

        Args:
            start_time: Start time for metric calculation
            end_time: End time for metric calculation (defaults to current time)

        Returns:
            Dictionary of calculated metrics
        """
        if end_time is None:
            end_time = self.env.now

        simulation_time = end_time - start_time

        # Skip if simulation time is zero
        if simulation_time <= 0:
            return self.metrics

        # Calculate throughput (bytes per second)
        total_bytes = sum(
            p.size
            for p in self.completed_packets
            if start_time <= p.arrival_time <= end_time
        )
        throughput = total_bytes / simulation_time

        # Calculate average delay
        delays = [
            p.get_total_delay()
            for p in self.completed_packets
            if start_time <= p.arrival_time <= end_time
        ]
        average_delay = sum(delays) / len(delays) if delays else 0

        # Calculate packet loss rate
        total_packets = len(self.completed_packets) + len(self.dropped_packets)
        packet_loss_rate = (
            len(self.dropped_packets) / total_packets if total_packets > 0 else 0
        )

        # Calculate link utilization
        link_utilization = {}
        for (source, target), link in self.links.items():
            # Utilization = bits sent / (capacity * time)
            bits_sent = link.bytes_sent * 8
            max_bits = link.capacity * simulation_time
            utilization = bits_sent / max_bits if max_bits > 0 else 0
            link_utilization[(source, target)] = utilization

        # Update metrics
        self.metrics["throughput"] = throughput
        self.metrics["average_delay"] = average_delay
        self.metrics["packet_loss_rate"] = packet_loss_rate
        self.metrics["link_utilization"] = link_utilization
        self.metrics["scheduler_type"] = self.scheduler_type

        return self.metrics

    def run(self, duration):
        """
        Run the simulation for a specified duration

        Args:
            duration: Simulation duration in seconds
        """
        self.env.run(until=duration)

        # Calculate final metrics
        self.calculate_metrics()

        return self.metrics

    def set_scheduler(self, scheduler_type):
        """
        Set the scheduler type

        Args:
            scheduler_type: Type of scheduler to use ("FIFO", "RR", or "QL")
        """
        self.scheduler_type = scheduler_type

    def visualize_network(self, figsize=(10, 8)):
        """Visualize the network topology"""
        plt.figure(figsize=figsize)

        # Create position layout
        pos = nx.spring_layout(self.graph)

        # Draw nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color="lightblue")

        # Draw edges with varying width based on capacity
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

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12)

        # Draw edge labels (capacity and delay)
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
        return plt


# Example traffic pattern generators
def constant_traffic(rate):
    """Generate constant bit rate traffic"""
    return lambda: 1 / rate


def variable_traffic(min_rate, max_rate):
    """Generate variable bit rate traffic"""
    return lambda: 1 / random.uniform(min_rate, max_rate)


def poisson_traffic(rate):
    """Generate Poisson traffic"""
    return lambda: np.random.exponential(1 / rate)


def pareto_traffic(rate, alpha=1.5):
    """Generate Pareto (heavy-tailed) traffic"""
    scale = (alpha - 1) / (alpha * rate)
    return lambda: np.random.pareto(alpha) * scale


# Example packet size generators
def constant_size(size):
    """Generate constant size packets"""
    return lambda: size


def variable_size(min_size, max_size):
    """Generate variable size packets"""
    return lambda: random.randint(min_size, max_size)


def bimodal_size(small_size, large_size, small_prob=0.7):
    """Generate bimodal packet sizes (e.g., small and large packets)"""
    return lambda: small_size if random.random() < small_prob else large_size
