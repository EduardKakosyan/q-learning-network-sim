"""Link class for network simulation.

This module defines the Link class, which represents a network link
between two nodes in the simulated network.
"""

import simpy

from network_sim.core.packet import Packet

class Link:
    """Represents a network link between nodes.

    Attributes:
        env: SimPy environment.
        source: Source node ID.
        destination: Destination node ID.
        capacity: Link capacity in bits per second.
        propagation_delay: Propagation delay in seconds.
        bytes_sent: Number of bytes sent through this link.
        resource: SimPy resource for link access control.
    """

    def __init__(
        self,
        env: simpy.Environment,
        source: int,
        destination: int,
        capacity: float,
        propagation_delay: float
    ):
        """Initialize a network link.

        Args:
            env: SimPy environment.
            source: Source node ID.
            destination: Destination node ID.
            capacity: Link capacity in bits per second.
            propagation_delay: Propagation delay in seconds.
        """
        self.env = env
        self.source = source
        self.destination = destination
        self.capacity = capacity
        self.propagation_delay = propagation_delay
        self.bytes_sent = 0
        self.resource = simpy.Resource(env, capacity=1)

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

    def __repr__(self) -> str:
        """Return string representation of the link.

        Returns:
            String representation of the link.
        """
        return f"Link({self.source}->{self.destination}, {self.capacity/1000000:.1f}Mbps, {self.propagation_delay*1000:.1f}ms)"
