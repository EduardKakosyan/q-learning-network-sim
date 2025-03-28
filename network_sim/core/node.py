"""Node class for network simulation.

This module defines the Node class, which represents a network node
(router, switch, host) in the simulated network.
"""

from collections import deque
import time
import simpy
from typing import Any, Callable, Deque, Dict, Generator, List, Tuple

from network_sim.core.link import Link
from network_sim.core.packet import Packet
from network_sim.core.routing_algorithms import Router


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
        router_func: Callable[[], Router],
        buffer_size: float = float("inf"),
        time_scale: float = 1.0,
    ) -> None:
        """Initialize a network node.

        Args:
            env: SimPy environment.
            node_id: Unique identifier for the node.
            router_func: Function to create a router for packet processing.
            buffer_size: Maximum buffer size in bytes (default: infinite).
            time_scale: Time scale for scheduling operations.
        """
        self.env = env
        self.id = node_id
        self.router: Router = router_func(self)
        if self.router is None:
            raise ValueError("router_func did not return a router.")
        self.buffer_used = 0
        self.buffer_size = buffer_size
        self.time_scale = time_scale
        self.links: Dict[int, Link] = {}
        self.routing_table: Dict[int, int] = {}
        self.packets_arrived = 0
        self.packets_dropped = 0
        self.queue: Deque[Packet] = deque()
        self.resource = simpy.Resource(env, capacity=1)
        self.neighbours: Dict[int, Node] = {}
        self.buffer_usage_history: List[Tuple[float, float]] = []

        def update() -> Generator[Any, Any, Any]:
            while True:
                yield self.env.timeout(0.1)
                self.buffer_usage_history.append((self.env.now, self.buffer_usage()))

        self.env.process(update())

    def add_link(self, link: Link, node: "Node") -> None:
        """Add an outgoing link from this node.

        Args:
            link: The link to add.
            node: The destination node for the link.
        """
        if link.source != self.id or link.destination == self.id:
            raise ValueError("Link source or destination is incorrect for this node. Verify the link's configuration.")
        self.links[link.destination] = link
        self.neighbours[link.destination] = node

    def set_routing_table(self, routing_table: Dict[int, int]) -> None:
        """Set the routing table for this node.

        Args:
            routing_table: Dictionary mapping destinations to next hops.
        """
        self.routing_table = routing_table

    def can_queue_packet(self, packet: Packet) -> bool:
        """Check if there's enough buffer space for the packet.

        Args:
            packet: The packet to check.

        Returns:
            True if there's enough buffer space, False otherwise.
        """
        return self.buffer_used + packet.size <= self.buffer_size

    def add_packet_to_queue(self, packet: Packet) -> None:
        """Add a packet to the appropriate queue based on router type.

        Args:
            packet: The packet to add.
        """
        self.queue.append(packet)
        self.buffer_used += packet.size

    def buffer_usage(self) -> float:
        """Calculate the buffer usage as a fraction of the total buffer size.

        Returns:
            The buffer usage as a fraction of the total buffer size.
        """
        if self.buffer_size == float("inf"):
            return self.buffer_used
        return self.buffer_used / self.buffer_size

    def packet_arrived(self, packet: Packet) -> None:
        """Handle packet arrival at the node.

        Args:
            packet: The packet that has arrived.
        """
        self.packets_arrived += 1
        self.buffer_used -= packet.size

    def packet_dropped(self, packet: Packet) -> None:
        """Handle packet drop at the node.

        Args:
            packet: The packet that was dropped.
        """
        self.packets_dropped += 1
        self.buffer_used -= packet.size

    def route_packet(self, packet: Packet) -> Tuple[int, float]:
        """Get the next packet to transmit based on router type.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID and the scheduling delay.
        """
        start_time = time.perf_counter()
        hop = self.router.route_packet(packet)
        end_time = time.perf_counter()
        scheduling_delay = (end_time - start_time) * self.time_scale

        self.queue.remove(packet)
        self.buffer_used -= packet.size

        return hop, scheduling_delay

    def __repr__(self) -> str:
        """Return string representation of the node.

        Returns:
            String representation of the node.
        """
        return f"Node({self.id})"
