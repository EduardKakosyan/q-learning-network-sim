from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple
from typing import TYPE_CHECKING
import numpy as np
import random

from network_sim.core.packet import Packet

if TYPE_CHECKING:
    from node import Node
    from simulator import NetworkSimulator


class Router(ABC):
    """Abstract base class for packet scheduling algorithms"""

    def __init__(self, node: "Node") -> None:
        """Initialize the scheduling algorithm

        Args:
            node: The node associated with this router.
        """
        self.name = "Base Router"
        self.node = node

    @abstractmethod
    def route_packet(self, packet: Packet) -> int:
        """Select the next packet to transmit from the queue

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID.
        """
        pass

    def __repr__(self) -> str:
        return self.name


class DijkstraRouter(Router):
    """Dijkstra-based routing algorithm"""

    def __init__(self, node: "Node") -> None:
        super().__init__(node)
        self.name = "Dijk"

    def route_packet(self, packet: Packet) -> int:
        """Select the next packet using Dijkstra's algorithm"""
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")
        next_hop = self.node.routing_table[packet.destination]
        return next_hop


class LeastCongestionFirstRouter(Router):
    """Least Congestion First (LCF) routing algorithm"""

    def __init__(self, node: "Node") -> None:
        super().__init__(node)
        self.name = "LCF"

    def route_packet(self, packet: Packet) -> int:
        """Select the next packet using LCF policy"""
        smallest_buffer_usage = min(
            [neighbour.buffer_usage() for _, neighbour in self.node.neighbours.items()]
        )

        min_ids = [
            id
            for id, neighbour in self.node.neighbours.items()
            if smallest_buffer_usage == neighbour.buffer_usage()
        ]
        if not min_ids:
            raise ValueError("No available neighbours to route the packet. Check network connectivity and routing table.")

        action = random.choice(min_ids)

        return action


class QRouter(Router):
    """Q-Learning based scheduling algorithm"""

    def __init__(
        self,
        node: "Node",
        simulator: "NetworkSimulator",
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        bins: int = 4,
        bin_base: int = 10,
    ) -> None:
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

        # State-action pair tracking for packets w timestamps
        # {packet_id: (state, action, timestamp)}
        self.packet_history = {}

        # Register hooks for packet events
        simulator.register_hook("packet_arrived", self.on_packet_arrived)
        simulator.register_hook("packet_dropped", self.on_packet_dropped)

    def _usage_to_bin(self, usage: float) -> int:
        for i, divider in enumerate(self.bins):
            if usage <= divider:
                return i
        return len(self.bins)

    def _get_state(self, packet: Packet) -> Tuple[int, int, Tuple[int, ...]]:
        """Get the current state representation"""

        buffer_usages = [
            neighbour.buffer_usage() for _, neighbour in self.node.neighbours.items()
        ]

        bins = tuple(self._usage_to_bin(usage) for usage in buffer_usages)

        state = (packet.source, packet.destination, bins)
        return state

    def _get_actions(self) -> List[int]:
        """Get available actions in the current state"""
        return list(self.node.links.keys())

    def route_packet(self, packet: Packet) -> int:
        """Select the next packet using Q-learning policy"""
        if not self.node.queue:
            raise ValueError("Cannot schedule a packet with an empty buffer.")

        state = self._get_state(packet)
        actions = self._get_actions()

        if not actions:
            raise ValueError("Router has no available links. Check the router's configuration and connectivity.")

        if random.random() < self.exploration_rate:
            # exploration
            action = random.choice(actions)
        else:
            # exploitation
            q_values = [self.q_table[state][a] for a in actions]
            if not q_values or all(q == 0 for q in q_values):
                action = random.choice(actions)
            else:
                max_q = max(q_values)
                max_actions = [a for q, a in zip(q_values, actions) if q == max_q]
                action = random.choice(max_actions)

        current_time = self.node.env.now
        self.packet_history[packet.id] = (state, action, current_time)

        return action

    def on_packet_hop(
        self, packet: Packet, from_node: "Node", to_node: "Node", time: float
    ) -> None:
        """Callback for packet_hop"""
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_hop_reward(packet, to_node, time - prev_time)

            self.update_q_table(prev_state, prev_action, reward)
            self.packet_history[packet.id] = (None, None, time)

    def on_packet_arrived(self, packet: Packet, node: "Node", time: float) -> None:
        """Callback for packet_arrived"""
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_success_reward(packet)

            self.update_q_table(prev_state, prev_action, reward, is_terminal=True)
            del self.packet_history[packet.id]

    def on_packet_dropped(
        self, packet: Packet, node: "Node", reason: str, time: float
    ) -> None:
        """Callback for packet_dropped"""
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_drop_reward(reason)

            self.update_q_table(prev_state, prev_action, reward, is_terminal=True)
            del self.packet_history[packet.id]

    def on_sim_end(self, metrics: dict) -> None:
        from pprint import pprint

        def convert_to_dict(d):
            if isinstance(d, defaultdict):
                d = dict({k: convert_to_dict(v) for k, v in d.items()})
            return d

        print(f"Node {self.node.id} q table:")
        pprint(convert_to_dict(self.q_table))

    def _calculate_hop_reward(
        self, packet: Packet, to_node: "Node", time_diff: float
    ) -> float:
        """Calculate reward for successful hop"""
        # small neg to encourage short paths
        reward = -1.0

        # pos if destination is next hop
        if packet.destination == to_node:
            reward += 5.0

        # neg for time to encourage faster routes
        reward -= time_diff

        # could add more complex rewards [congestion?]
        return reward

    def _calculate_success_reward(self, packet: Packet) -> float:
        """Calculate reward for successful packet delivery"""
        # big pos reward for arrival
        reward = 20.0

        # bonus for less hops
        hop_count = len(packet.hops)
        if hop_count > 0:
            efficiency_bonus = 10.0 / hop_count  # bigger bonus for less hops
            reward += efficiency_bonus

        # bonus for less delay
        total_delay = packet.get_total_delay()
        if total_delay:
            delay_bonus = 10.0 / (1.0 + total_delay)  # bigger bonus for less delay
            reward += delay_bonus

        return reward

    def _calculate_drop_reward(self, reason: str) -> float:
        """Calculate penalty for dropping a packet"""
        # big neg for dropping
        reward = -20.0

        # more penalty based on specific reasons
        if reason == "Buffer overflow":
            # neg for overwhelmed routes
            reward -= 5.0
        elif reason == "No route to destination":
            # big neg for sending to dead end
            reward -= 10.0

        return reward

    def update_q_table(
        self,
        state: Tuple[int, int, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Optional[Tuple[int, int, Tuple[int, ...]]] = None,
        is_terminal: bool = False,
    ) -> None:
        """Update Q-table using the Q-learning update rule

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state (None for terminal states)
            is_terminal: Whether this is a terminal state
        """
        if state is None or action is None:
            return

        if is_terminal or next_state is None:
            # for terminal states or unknown next states, use immediate reward
            max_next_q = 0
        else:
            # for normal transitions, include expected future rewards (approximating actions)
            next_actions = self._get_actions()
            if next_actions:
                next_q_values = [self.q_table[next_state][a] for a in next_actions]
                max_next_q = max(next_q_values) if next_q_values else 0
            else:
                max_next_q = 0

        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q


def router_factory(
    router_type: str,
    node: "Node",
    simulator: Optional["NetworkSimulator"] = None,
    **kwargs,
) -> Router:
    """
    Factory function to create the appropriate router

    Args:
        router_type: Type of the router ("RR", "QL", or other for FIFO)
        node: The node to which the router is attached.
        simulator: The network simulator instance.

    Returns:
        An instance of the selected scheduling algorithm
    """
    if router_type == "LCF":
        return LeastCongestionFirstRouter(node)
    elif router_type == "QL":
        if simulator is None:
            raise ValueError("Simulator is required for QRouter.")
        return QRouter(node, simulator, **kwargs)
    else:
        # Default to FIFO
        return DijkstraRouter(node)
