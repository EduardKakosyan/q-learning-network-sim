"""Node class for network simulation.

This module defines the Node class, which represents a network node
(router, switch, host) in the simulated network.
"""

import simpy
from typing import Dict, Optional

from network_sim.core.link import Link


class Node:
    """Represents a network node (router, switch, host).

    Attributes:
        env: SimPy environment.
        id: Unique identifier for the node.
        processing_delay: Time to process a packet in seconds.
        buffer_size: Maximum buffer size in bytes.
        links: Outgoing links keyed by destination.
        routing_table: Next hop for each destination.
        packets_received: Number of packets received by this node.
        packets_dropped: Number of packets dropped by this node.
        packets_forwarded: Number of packets forwarded by this node.
        is_destination: Whether this node is a destination.
    """

    def __init__(
        self,
        env: simpy.Environment,
        node_id: int,
        processing_delay: float = 0,
        buffer_size: float = float("inf"),
    ):
        """Initialize a network node.

        Args:
            env: SimPy environment.
            node_id: Unique identifier for the node.
            processing_delay: Time to process a packet in seconds.
            buffer_size: Maximum buffer size in bytes (default: infinite).
        """
        self.env = env
        self.id = node_id
        self.processing_delay = processing_delay
        self.buffer_size = buffer_size
        self.links: Dict[int, Link] = {}
        self.routing_table: Dict[int, int] = {}
        self.packets_received = 0
        self.packets_dropped = 0
        self.packets_forwarded = 0
        self.is_destination = False

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

    def __repr__(self) -> str:
        """Return string representation of the node.

        Returns:
            String representation of the node.
        """
        return f"Node({self.id})"
