from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Tuple
from typing import TYPE_CHECKING
import numpy as np

from network_sim.core.packet import Packet
from network_sim.utils.rng import CustomRNG

if TYPE_CHECKING:
    from node import Node
    from simulator import NetworkSimulator


class Router(ABC):
    """Abstract base class for routing algorithms."""

    def __init__(self, node: "Node") -> None:
        """
        Initialize the router.

        Args:
            node: The node associated with this router.
        """
        self.name = "Base Router"
        self.node = node

    @abstractmethod
    def route_packet(self, packet: Packet) -> int:
        """
        Determine the next hop for a packet.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID.
        """
        pass

    def __repr__(self) -> str:
        return self.name


class DijkstraRouter(Router):
    """Router using Dijkstra's algorithm for shortest path routing."""

    def __init__(self, node: "Node") -> None:
        super().__init__(node)
        self.name = "Dijk"

    def route_packet(self, packet: Packet) -> int:
        """
        Route the packet using Dijkstra's algorithm.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")
        next_hop = self.node.routing_table[packet.destination]
        return next_hop


class LeastCongestionFirstRouter(Router):
    """Router using the Least Congestion First (LCF) routing algorithm."""

    def __init__(self, node: "Node", seed: int = 42) -> None:
        """
        Initialize the LCF router.

        Args:
            node: The node associated with this router.
            seed: Seed for the random number generator.
        """
        super().__init__(node)
        self.name = "LCF"
        self.rng = CustomRNG(seed)

    def route_packet(self, packet: Packet) -> int:
        """
        Route the packet using the LCF policy.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID.
        """
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

        action = self.rng.choice(min_ids)

        return action


class QRouter(Router):
    """Router using Q-Learning for adaptive routing."""

    def __init__(
        self,
        node: "Node",
        simulator: "NetworkSimulator",
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        bins: int = 4,
        bin_base: int = 10,
        seed: int = 42,
    ) -> None:
        """
        Initialize the Q-Learning router.

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

        # State-action pair tracking for packets with timestamps
        # {packet_id: (state, action, timestamp)}
        self.packet_history = {}

        # Register hooks for packet events
        simulator.register_hook("packet_arrived", self.on_packet_arrived)
        simulator.register_hook("packet_dropped", self.on_packet_dropped)

    def _usage_to_bin(self, usage: float) -> int:
        """
        Map buffer usage to a discrete bin.

        Args:
            usage: Buffer usage as a float.

        Returns:
            The bin index.
        """
        for i, divider in enumerate(self.bins):
            if usage <= divider:
                return i
        return len(self.bins)

    def _get_state(self, packet: Packet) -> Tuple[int, int, Tuple[int, ...]]:
        """
        Get the current state representation.

        Args:
            packet: The packet being routed.

        Returns:
            A tuple representing the state.
        """
        buffer_usages = [
            neighbour.buffer_usage() for _, neighbour in self.node.neighbours.items()
        ]

        bins = tuple(self._usage_to_bin(usage) for usage in buffer_usages)

        state = (packet.source, packet.destination, bins)
        return state

    def _get_actions(self) -> List[int]:
        """
        Get available actions in the current state.

        Returns:
            A list of available actions (next hop node IDs).
        """
        return list(self.node.links.keys())

    def route_packet(self, packet: Packet) -> int:
        """
        Route the packet using the Q-Learning policy.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule a packet with an empty buffer.")

        state = self._get_state(packet)
        actions = self._get_actions()

        if not actions:
            raise ValueError("Router has no available links. Check the router's configuration and connectivity.")

        if self.rng.random() < self.exploration_rate:
            # Exploration: choose a random action
            action = self.rng.choice(actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            q_values = [self.q_table[state][a] for a in actions]
            if not q_values or all(q == 0 for q in q_values):
                action = self.rng.choice(actions)
            else:
                max_q = max(q_values)
                max_actions = [a for q, a in zip(q_values, actions) if q == max_q]
                action = self.rng.choice(max_actions)

        current_time = self.node.env.now
        self.packet_history[packet.id] = (state, action, current_time)

        return action

    def on_packet_hop(
        self, packet: Packet, from_node: "Node", to_node: "Node", time: float
    ) -> None:
        """
        Callback for packet hop event.

        Args:
            packet: The packet being routed.
            from_node: The node from which the packet is hopping.
            to_node: The node to which the packet is hopping.
            time: The time of the hop event.
        """
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_hop_reward(packet, to_node, time - prev_time)

            self.update_q_table(prev_state, prev_action, reward)
            self.packet_history[packet.id] = (None, None, time)

    def on_packet_arrived(self, packet: Packet, node: "Node", time: float) -> None:
        """
        Callback for packet arrival event.

        Args:
            packet: The packet that arrived.
            node: The node where the packet arrived.
            time: The time of the arrival event.
        """
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_success_reward(packet)

            self.update_q_table(prev_state, prev_action, reward, is_terminal=True)
            del self.packet_history[packet.id]

    def on_packet_dropped(
        self, packet: Packet, node: "Node", reason: str, time: float
    ) -> None:
        """
        Callback for packet drop event.

        Args:
            packet: The packet that was dropped.
            node: The node where the packet was dropped.
            reason: The reason for dropping the packet.
            time: The time of the drop event.
        """
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_drop_reward(reason)

            self.update_q_table(prev_state, prev_action, reward, is_terminal=True)
            del self.packet_history[packet.id]

    def on_sim_end(self, metrics: dict) -> None:
        """
        Callback for simulation end event.

        Args:
            metrics: Simulation metrics collected during the run.
        """
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
        """
        Calculate reward for successful hop.

        Args:
            packet: The packet being routed.
            to_node: The node to which the packet is hopping.
            time_diff: Time difference since the last hop.

        Returns:
            The calculated reward.
        """
        # Small negative reward to encourage short paths
        reward = -1.0

        # Positive reward if destination is the next hop
        if packet.destination == to_node:
            reward += 5.0

        # Negative reward for time to encourage faster routes
        reward -= time_diff

        # Additional rewards could be added (e.g., based on congestion)
        return reward

    def _calculate_success_reward(self, packet: Packet) -> float:
        """
        Calculate reward for successful packet delivery.

        Args:
            packet: The packet that was successfully delivered.

        Returns:
            The calculated reward.
        """
        # Large positive reward for arrival
        reward = 20.0

        # Bonus for fewer hops
        hop_count = len(packet.hops)
        if hop_count > 0:
            efficiency_bonus = 10.0 / hop_count  # Larger bonus for fewer hops
            reward += efficiency_bonus

        # Bonus for less delay
        total_delay = packet.get_total_delay()
        if total_delay:
            delay_bonus = 10.0 / (1.0 + total_delay)  # Larger bonus for less delay
            reward += delay_bonus

        return reward

    def _calculate_drop_reward(self, reason: str) -> float:
        """
        Calculate penalty for dropping a packet.

        Args:
            reason: The reason for dropping the packet.

        Returns:
            The calculated penalty.
        """
        # Large negative reward for dropping
        reward = -20.0

        # Additional penalty based on specific reasons
        if reason == "Buffer overflow":
            # Negative reward for overwhelmed routes
            reward -= 5.0
        elif reason == "No route to destination":
            # Large negative reward for sending to a dead end
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
        """
        Update the Q-table using the Q-Learning update rule.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state (None for terminal states).
            is_terminal: Whether this is a terminal state.
        """
        if state is None or action is None:
            return

        if is_terminal or next_state is None:
            # For terminal states or unknown next states, use immediate reward
            max_next_q = 0
        else:
            # For normal transitions, include expected future rewards
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
    seed: int = 42,
    **kwargs,
) -> Router:
    """
    Factory function to create the appropriate router.

    Args:
        router_type: Type of the router ("LCF", "QL", or other for Dijkstra).
        node: The node to which the router is attached.
        simulator: The network simulator instance.
        seed: Seed for the random number generator.
        **kwargs: Additional arguments for specific router types.

    Returns:
        An instance of the selected routing algorithm.
    """
    if router_type == "LCF":
        return LeastCongestionFirstRouter(node, seed)
    elif router_type == "QL":
        if simulator is None:
            raise ValueError("Simulator is required for QRouter.")
        return QRouter(node, simulator, seed=seed, **kwargs)
    else:
        # Default to DijkstraRouter
        return DijkstraRouter(node)
