"""Link class for network simulation.

This module defines the Link class, which represents a network link
between two nodes in the simulated network.
"""

import simpy
from typing import Optional

from network_sim.core.packet import Packet
from network_sim.core.scheduling_algorithms import SchedulingAlgorithm

class Link:
    """Represents a network link between nodes.

    Attributes:
        env: SimPy environment.
        source: Source node ID.
        target: Target node ID.
        capacity: Link capacity in bits per second.
        propagation_delay: Propagation delay in seconds.
        buffer_size: Maximum buffer size in bytes.
        buffer_usage: Current buffer usage in bytes.
        busy: Whether the link is currently transmitting.
        packets_dropped: Number of packets dropped by this link.
        packets_sent: Number of packets sent through this link.
        bytes_sent: Number of bytes sent through this link.
        resource: SimPy resource for link access control.
    """

    def __init__(
        self,
        env: simpy.Environment,
        source: int,
        target: int,
        capacity: float,
        propagation_delay: float,
        buffer_size: float = float("inf"),
    ):
        """Initialize a network link.

        Args:
            env: SimPy environment.
            source: Source node ID.
            target: Target node ID.
            capacity: Link capacity in bits per second.
            propagation_delay: Propagation delay in seconds.
            buffer_size: Maximum buffer size in bytes (default: infinite).
        """
        self.env = env
        self.source = source
        self.target = target
        self.capacity = capacity
        self.propagation_delay = propagation_delay
        self.buffer_size = buffer_size
        self.buffer_usage = 0
        self.busy = False
        self.packets_dropped = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.resource = simpy.Resource(env, capacity=1)

    def can_queue_packet(self, packet: Packet) -> bool:
        """Check if there's enough buffer space for the packet.

        Args:
            packet: The packet to check.

        Returns:
            True if there's enough buffer space, False otherwise.
        """
        return self.buffer_usage + packet.size <= self.buffer_size

    def calculate_transmission_delay(self, packet_size: int) -> float:
        """Calculate transmission delay based on packet size and link capacity.

        Args:
            packet_size: Size of the packet in bytes.

        Returns:
            Transmission delay in seconds.
        """
        return (packet_size * 8) / self.capacity

    def get_total_delay(self, packet: Packet) -> float:
        """Calculate total delay for a packet (transmission + propagation).

        Args:
            packet: The packet to calculate delay for.

        Returns:
            Total delay in seconds.
        """
        transmission_delay = self.calculate_transmission_delay(packet.size)
        return transmission_delay + self.propagation_delay

    def add_packet_to_queue(
        self,
        packet: Packet,
        scheduler: SchedulingAlgorithm
    ) -> None:
        """Add a packet to the appropriate queue based on scheduler type.

        Args:
            packet: The packet to add.
            scheduler: Scheduler object.
        """
        scheduler.add_packet(packet)

    def get_next_packet(self, scheduler: SchedulingAlgorithm) -> Optional[Packet]:
        """Get the next packet to transmit based on scheduler type.

        Args:
            scheduler: Scheduler object.

        Returns:
            The next packet to transmit or None if no packets are available.
        """
        return scheduler.select_next_packet(self)

    def __repr__(self) -> str:
        """Return string representation of the link.

        Returns:
            String representation of the link.
        """
        return f"Link({self.source}->{self.target}, {self.capacity/1000000:.1f}Mbps, {self.propagation_delay*1000:.1f}ms)"
