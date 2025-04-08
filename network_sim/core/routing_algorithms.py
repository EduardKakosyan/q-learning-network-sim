from abc import ABC, abstractmethod
from collections import defaultdict
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import heapq
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
        self.extra_delay = 0.0

    @abstractmethod
    def route_packet(self, packet: Packet) -> Tuple[int, float]:
        """
        Determine the next hop for a packet.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID and the extra time that was spent performing
            other tasks.
        """
        pass

    def __repr__(self) -> str:
        return self.name


class DijkstraRouter(Router):
    """Router using Dijkstra's algorithm for shortest path routing."""

    def __init__(self, node: "Node") -> None:
        super().__init__(node)
        self.name = "Dijk"

    def route_packet(self, packet: Packet) -> Tuple[int, float]:
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
        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return next_hop, extra_delay


class OSPFRouter(Router):
    """OSPF-based routing algorithm using link-state information and Dijkstra's algorithm."""

    def __init__(self, node: "Node", simulator: "NetworkSimulator") -> None:
        super().__init__(node)
        self.name = "OSPF"
        self.simulator = simulator
        simulator.register_hook("sim_start", self.update_routing_table)
        simulator.register_hook("sim_update", self.update_routing_table)

    def update_routing_table(self, sim_time: float = None) -> None:
        """
        Update the routing table using Dijkstra's algorithm based on the complete network topology.
            
        :param network_graph: A dictionary representing the network topology,
                            mapping node ids to their neighbours and associated link costs.
                            Example: {node_id: {neighbour_id: cost, ...}, ...}
        """
        start_time = time.perf_counter()
        source = self.node.id
        # Priority queue holds tuples of (accumulated_cost, current_node, first_hop)
        # For the source, first_hop is None.
        queue = [(0, source, None)]
        # distances store the best known cost and first hop to reach each node: {node: (cost, first_hop)}
        distances: Dict[int, Tuple[int, int]] = {source: (0, None)}

        while queue:
            cost, current, first_hop = heapq.heappop(queue)
            current_node = self.simulator.nodes[current]
            # Traverse all neighbours of the current node based on the network graph.
            for id, neighbour in current_node.neighbours.items():
                capacity = current_node.links[id].capacity
                queue_time = neighbour.buffer_used * 8 / capacity
                
                packet_size = 1500 # Assuming a standard packet size of 1500 bytes
                transmission_time = packet_size * 8 / capacity

                propagation_delay = current_node.links[id].propagation_delay
                
                link_cost = queue_time + transmission_time + propagation_delay
                
                # Compute the new accumulated cost to reach the neighbor
                new_cost = cost + link_cost
                # If we're at the source, the neighbour is the first hop. Otherwise, inherit the first hop.
                nh = id if current == source else first_hop
                if id not in distances or new_cost < distances[id][0]:
                    distances[id] = (new_cost, nh)
                    heapq.heappush(queue, (new_cost, id, nh))

        self.routing_table = distances
        
        self.extra_delay += time.perf_counter() - start_time

    def route_packet(self, packet: Packet) -> Tuple[int, float]:
        """
        Select the next hop for the given packet based on the current routing table.

        :param packet: The packet to be routed.
        :return: The next hop node id and the extra time incurred by the routing algorithm.
        :raises ValueError: If there are no packets in the buffer or if no route exists.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule packets with an empty buffer.")
        dest = packet.destination
        if dest not in self.routing_table:
            raise ValueError(f"No route found for destination {dest}.")
        next_hop = self.routing_table[dest][1]
        if next_hop is None:
            raise ValueError("Packet is already at the destination or no valid next hop found.")
        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return next_hop, extra_delay  # Return the extra delay incurred by the routing algorithm


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

    def _get_state(self, destination: int) -> Tuple[int, ...]:
        """
        Get the current state representation.

        Args:
            destination: The destination node ID.

        Returns:
            A tuple representing the state.
        """
        buffer_usages = [
            neighbour.buffer_usage() for _, neighbour in self.node.neighbours.items()
        ]

        bins = tuple(self._usage_to_bin(usage) for usage in buffer_usages)

        state = (destination, *bins)
        return state

    def _get_actions(self) -> List[int]:
        """
        Get available actions in the current state.

        Returns:
            A list of available actions (next hop node IDs).
        """
        return list(self.node.links.keys())

    def route_packet(self, packet: Packet) -> Tuple[int, float]:
        """
        Route the packet using the Q-Learning policy.

        Args:
            packet: The packet to be routed.

        Returns:
            The next hop node ID and the extra time that was spent performing
            other tasks.
        """
        if not self.node.queue:
            raise ValueError("Cannot schedule a packet with an empty buffer.")

        state = self._get_state(packet.destination)
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

        extra_delay, self.extra_delay = self.extra_delay, 0.0
        return action, extra_delay        

    def on_packet_arrived(self, packet: Packet, node: "Node", sim_time: float) -> None:
        """
        Callback for packet arrival event.

        Args:
            packet: The packet that arrived.
            node: The node where the packet arrived.
            sim_time: The time of the arrival event.
        """
        start_time = time.perf_counter()
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_success_reward(packet)

            self.update_q_table(prev_state, prev_action, reward)
            del self.packet_history[packet.id]
        self.extra_delay += time.perf_counter() - start_time

    def on_packet_dropped(
        self, packet: Packet, node: "Node", reason: str, sim_time: float
    ) -> None:
        """
        Callback for packet drop event.

        Args:
            packet: The packet that was dropped.
            node: The node where the packet was dropped.
            reason: The reason for dropping the packet.
            sim_time: The time of the drop event.
        """
        start_time = time.perf_counter()
        if packet.id in self.packet_history:
            prev_state, prev_action, prev_time = self.packet_history[packet.id]

            reward = self._calculate_drop_reward(reason)

            self.update_q_table(prev_state, prev_action, reward)
            del self.packet_history[packet.id]
        self.extra_delay += time.perf_counter() - start_time

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
        state: Tuple[int, ...],
        action: int,
        reward: float,
    ) -> None:
        """
        Update the Q-table using the Q-Learning update rule.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            is_terminal: Whether this is a terminal state.
        """
        if state is None or action is None:
            return

        max_next_q = 0.0

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
    if router_type == "OSPF":
        if simulator is None:
            raise ValueError("Simulator is required for OSPFRouter.")
        return OSPFRouter(node, simulator)
    elif router_type == "QL":
        if simulator is None:
            raise ValueError("Simulator is required for QRouter.")
        return QRouter(node, simulator, seed=seed, **kwargs)
    else:
        # Default to DijkstraRouter
        return DijkstraRouter(node)
