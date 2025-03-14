"""Node class for network simulation.

This module defines the Node class, which represents a network node
(router, switch, host) in the simulated network.
"""

from collections import deque
import time
import simpy
from typing import Deque, Dict, Optional, Tuple

from network_sim.core.link import Link
from network_sim.core.packet import Packet
from network_sim.core.scheduling_algorithms import SchedulingAlgorithm, FIFOScheduler


class Node:
    """Represents a network node (router, switch, host).

    Attributes:
        env: SimPy environment.
        id: Unique identifier for the node.
        buffer_size: Maximum buffer size in bytes.
        links: Outgoing links keyed by destination.
        routing_table: Next hop for each destination.
        packets_received: Number of packets received by this node.
        packets_dropped: Number of packets dropped by this node.
        resource: SimPy resource for node scheduling control.
    """

    def __init__(
        self,
        env: simpy.Environment,
        node_id: int,
        scheduler: Optional[SchedulingAlgorithm] = None,
        buffer_size: float = float("inf"),
    ):
        """Initialize a network node.

        Args:
            env: SimPy environment.
            node_id: Unique identifier for the node.
            scheduler: Scheduling algorithm for packet processing.
            buffer_size: Maximum buffer size in bytes (default: infinite).
        """
        self.env = env
        self.id = node_id
        self.scheduler = scheduler if scheduler else FIFOScheduler()
        self.buffer_usage = 0
        self.buffer_size = buffer_size
        self.links: Dict[int, Link] = {}
        self.routing_table: Dict[int, int] = {}
        self.packets_received = 0
        self.packets_dropped = 0
        self.queue: Deque[Packet] = deque()
        self.resource = simpy.Resource(env, capacity=1)

    def add_link(self, link: Link) -> None:
        """Add an outgoing link from this node.

        Args:
            link: The link to add.
        """
        if link.source == self.id:
            self.links[link.target] = link

    def set_routing_table(self, routing_table: Dict[int, int]) -> None:
        """Set the routing table for this node.

        Args:
            routing_table: Dictionary mapping destinations to next hops.
        """
        self.routing_table = routing_table

    def get_next_hop(self, destination: int) -> Optional[int]:
        """Get next hop for a destination from routing table.

        Args:
            destination: Destination node ID.

        Returns:
            Next hop node ID or None if no route exists.
        """
        return self.routing_table.get(destination)
    
    def can_queue_packet(self, packet: Packet) -> bool:
        """Check if there's enough buffer space for the packet.

        Args:
            packet: The packet to check.

        Returns:
            True if there's enough buffer space, False otherwise.
        """
        return self.buffer_usage + packet.size <= self.buffer_size

    def add_packet_to_queue(self, packet: Packet) -> None:
        """Add a packet to the appropriate queue based on scheduler type.

        Args:
            packet: The packet to add.
            scheduler: Scheduler object.
        """
        self.queue.append(packet)
        self.buffer_usage += packet.size

    def get_next_packet(self, link: Link) -> Tuple[Optional[Packet], float]:
        """Get the next packet to transmit based on scheduler type.

        Args:
            scheduler: Scheduler object.

        Returns:
            The next packet to transmit or None if no packets are available, and
            the scheduling delay.
        """
        start_time = time.perf_counter()
        packet = self.scheduler.select_next_packet(self)
        end_time = time.perf_counter()
        scheduling_delay = end_time - start_time

        if packet:
            self.queue.remove(packet)
            self.buffer_usage -= packet.size
        
        return packet, scheduling_delay

    def __repr__(self) -> str:
        """Return string representation of the node.

        Returns:
            String representation of the node.
        """
        return f"Node({self.id})"
