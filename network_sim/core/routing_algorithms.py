"""Routing algorithms implementation for network simulation."""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np

from network_sim.core.packet import Packet
from network_sim.utils.rng import CustomRNG

if TYPE_CHECKING:
    from network_sim.core.node import Node
    from network_sim.core.simulator import NetworkSimulator


class Router(ABC):
    """Abstract base class for routing algorithms."""

    def __init__(self, node: Node) -> None:
        """Initialize the router.

        Args:
            node: The node associated with this router.
        """
        self.name = "Base Router"
        self.node = node
        self.extra_delay = 0.0

    @abstractmethod
    def route_packet(self, packet: Packet) -> tuple[int, float]:
        """Determine the next hop for a packet.

        Args:
            packet: The packet to be routed.

        Returns:
            Tuple containing:
                - The next hop node ID
                - The extra time spent performing routing tasks

        Raises:
            ValueError: If routing cannot be performed.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the router."""
        return self.name


class DijkstraRouter(Router):
    """Router using Dijkstra's algorithm for shortest path routing."""

    def __init__(self, node: Node) -> None:
        """Initialize the Dijkstra router.

        Args:
            node: The node associated with this router.
        """
        super().__init__(node)
        self.name = "Dijk"

    def route_packet(self, packet: Packet) -> tuple[int, float]:
        """Route the packet using Dijkstra's algorithm.

        Args:
            packet: The packet to be routed.

        Returns:
            Tuple containing:
                - The next hop node ID
                - The extra time spent performing routing tasks

        Raises:
            ValueError: If the buffer is empty or no route exists.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")
        next_hop = self.node.routing_table[packet.destination]
        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return next_hop, extra_delay


class OSPFRouter(Router):
    """OSPF-based routing algorithm using link-state information and Dijkstra's algorithm."""

    def __init__(self, node: Node, simulator: NetworkSimulator) -> None:
        """Initialize the OSPF router.

        Args:
            node: The node associated with this router.
            simulator: The network simulator instance.
        """
        super().__init__(node)
        self.name = "OSPF"
        self.simulator = simulator
        simulator.register_hook("sim_start", self.update_routing_table)

        def update() -> Generator[float, None, None]:
            """Periodically update the routing table based on buffer usage."""
            while True:
                timeout = max(self.node.buffer_usage(), 0.1)
                yield self.simulator.env.timeout(timeout)
                self.update_routing_table()

        self.simulator.env.process(update())

    def update_routing_table(self, sim_time: float | None = None) -> None:
        """Update the routing table using Dijkstra's algorithm.

        Args:
            sim_time: Current simulation time (optional).

        Raises:
            ValueError: If link costs are negative.
        """
        start_time = time.perf_counter()
        source = self.node.id
        queue = [(0, source, None)]
        distances: dict[int, tuple[float, int | None]] = {source: (0, None)}

        while queue:
            cost, current, first_hop = heapq.heappop(queue)
            current_node = self.simulator.nodes[current]

            for id, neighbour in current_node.neighbours.items():
                capacity = current_node.links[id].capacity
                queue_time = neighbour.buffer_used * 8 / capacity
                packet_size = 1500  # Standard packet size in bytes
                transmission_time = packet_size * 8 / capacity
                propagation_delay = current_node.links[id].propagation_delay
                link_cost = queue_time + transmission_time + propagation_delay

                if link_cost < 0:
                    raise ValueError("Link cost cannot be negative.")

                new_cost = cost + link_cost
                nh = id if current == source else first_hop

                if id not in distances or new_cost < distances[id][0]:
                    distances[id] = (new_cost, nh)
                    heapq.heappush(queue, (new_cost, id, nh))

        self.routing_table = distances
        self.extra_delay += time.perf_counter() - start_time

    def route_packet(self, packet: Packet) -> tuple[int, float]:
        """Select the next hop for the given packet.

        Args:
            packet: The packet to be routed.

        Returns:
            Tuple containing:
                - The next hop node ID
                - The extra time spent performing routing tasks

        Raises:
            ValueError: If the buffer is empty or no route exists.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")

        dest = packet.destination
        if dest not in self.routing_table:
            raise ValueError(f"No route found for destination {dest}.")

        next_hop = self.routing_table[dest][1]
        if next_hop is None:
            raise ValueError(
                "Packet is already at destination or no valid next hop found."
            )

        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return next_hop, extra_delay


class QRouter(Router):
    """Router using Q-Learning for adaptive routing."""

    def __init__(
        self,
        node: Node,
        simulator: NetworkSimulator,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        bins: int = 4,
        bin_base: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the Q-Learning router.

        Args:
            node: The node associated with this router.
            simulator: The network simulator instance.
            learning_rate: Learning rate for Q-Learning.
            discount_factor: Discount factor for future rewards.
            exploration_rate: Probability of exploring random actions.
            bins: Number of bins for discretizing buffer usage.
            bin_base: Base for logarithmic binning.
            seed: Seed for the random number generator.
        """
        super().__init__(node)
        self.name = "Q-Router"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.bins = [
            (float(x) - 1) / (bin_base - 1)
            for x in np.logspace(0, 1, bins + 1, base=bin_base)
        ][1:]
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.rng = CustomRNG(seed)
        self.packet_history: dict[int, tuple[tuple[int, ...], int, float]] = {}

        simulator.register_hook("packet_arrived", self.on_packet_arrived)
        simulator.register_hook("packet_dropped", self.on_packet_dropped)

    def _usage_to_bin(self, usage: float) -> int:
        """Map buffer usage to a discrete bin.

        Args:
            usage: Buffer usage as a float.

        Returns:
            The bin index.
        """
        for i, divider in enumerate(self.bins):
            if usage <= divider:
                return i
        return len(self.bins)

    def _get_state(self, destination: int) -> tuple[int, ...]:
        """Get the current state representation.

        Args:
            destination: The destination node ID.

        Returns:
            A tuple representing the state.
        """
        buffer_usages = [
            neighbour.buffer_usage() for _, neighbour in self.node.neighbours.items()
        ]
        bins = tuple(self._usage_to_bin(usage) for usage in buffer_usages)
        return (destination, *bins)

    def _get_actions(self) -> list[int]:
        """Get available actions in the current state.

        Returns:
            A list of available actions (next hop node IDs).
        """
        return list(self.node.links.keys())

    def route_packet(self, packet: Packet) -> tuple[int, float]:
        """Route the packet using Q-Learning.

        Args:
            packet: The packet to be routed.

        Returns:
            Tuple containing:
                - The next hop node ID
                - The extra time spent performing routing tasks

        Raises:
            ValueError: If the buffer is empty or no route exists.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")

        state = self._get_state(packet.destination)
        actions = self._get_actions()

        if not actions:
            raise ValueError("No available actions for routing.")

        if self.rng.random() < self.exploration_rate:
            next_hop = self.rng.choice(actions)
        else:
            q_values = [self.q_table[state][action] for action in actions]
            next_hop = actions[np.argmax(q_values)]

        self.packet_history[packet.id] = (state, next_hop, self.simulator.env.now)
        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return next_hop, extra_delay

    def on_packet_arrived(self, packet: Packet, node: Node, sim_time: float) -> None:
        """Handle packet arrival event.

        Args:
            packet: The arrived packet.
            node: The node where the packet arrived.
            sim_time: Current simulation time.
        """
        if packet.id in self.packet_history:
            state, action, start_time = self.packet_history[packet.id]
            time_diff = sim_time - start_time
            reward = self._calculate_hop_reward(packet, node, time_diff)
            self.update_q_table(state, action, reward)
            del self.packet_history[packet.id]

    def on_packet_dropped(
        self, packet: Packet, node: Node, reason: str, sim_time: float
    ) -> None:
        """Handle packet drop event.

        Args:
            packet: The dropped packet.
            node: The node where the packet was dropped.
            reason: Reason for packet drop.
            sim_time: Current simulation time.
        """
        if packet.id in self.packet_history:
            state, action, _ = self.packet_history[packet.id]
            reward = self._calculate_drop_reward(reason)
            self.update_q_table(state, action, reward)
            del self.packet_history[packet.id]

    def _calculate_hop_reward(
        self, packet: Packet, to_node: Node, time_diff: float
    ) -> float:
        """Calculate reward for successful packet hop.

        Args:
            packet: The packet that was routed.
            to_node: The destination node.
            time_diff: Time taken for the hop.

        Returns:
            The calculated reward value.
        """
        if packet.destination == to_node.id:
            return self._calculate_success_reward(packet)

        base_reward = -time_diff
        congestion_penalty = to_node.buffer_usage() * 10
        return base_reward - congestion_penalty

    def _calculate_success_reward(self, packet: Packet) -> float:
        """Calculate reward for successful packet delivery.

        Args:
            packet: The successfully delivered packet.

        Returns:
            The calculated reward value.
        """
        base_reward = 100.0
        time_penalty = (self.simulator.env.now - packet.creation_time) * 0.1
        return base_reward - time_penalty

    def _calculate_drop_reward(self, reason: str) -> float:
        """Calculate reward for packet drop.

        Args:
            reason: Reason for packet drop.

        Returns:
            The calculated reward value.
        """
        if reason == "buffer_full":
            return -50.0
        if reason == "ttl_expired":
            return -100.0
        return -25.0

    def update_q_table(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
    ) -> None:
        """Update Q-table with new experience.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
        """
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[state].values()) if self.q_table[state] else 0
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        self.q_table[state][action] = new_q


def router_factory(
    router_type: str,
    node: Node,
    simulator: NetworkSimulator | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> Router:
    """Create a router instance based on the specified type.

    Args:
        router_type: Type of router to create.
        node: The node associated with the router.
        simulator: The network simulator instance.
        seed: Random seed for stochastic routers.
        **kwargs: Additional arguments for router initialization.

    Returns:
        A new router instance.

    Raises:
        ValueError: If the router type is unknown.
    """
    if router_type == "dijkstra":
        return DijkstraRouter(node)
    if router_type == "ospf":
        if simulator is None:
            raise ValueError("OSPF router requires a simulator instance.")
        return OSPFRouter(node, simulator)
    if router_type == "q":
        if simulator is None:
            raise ValueError("Q-router requires a simulator instance.")
        return QRouter(node, simulator, seed=seed, **kwargs)

    raise ValueError(f"Unknown router type: {router_type}")
