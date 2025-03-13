"""Packet class for network simulation.

This module defines the Packet class, which represents a network packet
traveling through the simulated network.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Packet:
    """Represents a network packet.

    Attributes:
        source: Source node ID.
        destination: Destination node ID.
        size: Size of packet in bytes.
        priority: Priority level (higher means more important).
        creation_time: Time when packet was created.
        id: Unique identifier for the packet.
        current_node: Current node where the packet is located.
        hops: List of nodes visited by the packet.
        arrival_time: Time when packet arrived at destination.
        dropped: Whether the packet was dropped.
        queuing_delays: List of queuing delays experienced by the packet.
        flow_id: Identifier for the flow (source-destination pair).
    """

    source: int
    destination: int
    size: int
    priority: int = 0
    creation_time: float = 0
    id: int = field(init=False)
    current_node: int = field(init=False)
    hops: List[Tuple[int, float]] = field(default_factory=list)
    arrival_time: Optional[float] = None
    dropped: bool = False
    queuing_delays: List[float] = field(default_factory=list)
    flow_id: str = field(init=False)

    _id_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize derived attributes after initialization."""
        type(self)._id_counter += 1
        self.id = type(self)._id_counter
        self.current_node = self.source
        self.flow_id = f"{self.source}-{self.destination}"

    def record_hop(self, node: int, time: float) -> None:
        """Record a hop in the packet's journey.

        Args:
            node: Node ID where the packet has arrived.
            time: Current simulation time.
        """
        self.hops.append((node, time))
        self.current_node = node

    def record_queuing_delay(self, delay: float) -> None:
        """Record queuing delay at a node.

        Args:
            delay: Queuing delay in seconds.
        """
        self.queuing_delays.append(delay)

    def get_total_delay(self) -> Optional[float]:
        """Calculate total delay if packet has arrived.

        Returns:
            Total delay in seconds or None if packet hasn't arrived.
        """
        if self.arrival_time is None:
            return None
        return self.arrival_time - self.creation_time

    def get_hop_count(self) -> int:
        """Get number of hops taken.

        Returns:
            Number of hops taken by the packet.
        """
        return len(self.hops)
