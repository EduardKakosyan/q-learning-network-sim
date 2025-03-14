from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, Tuple
from network_sim.core.packet import Packet
import random


class Router(ABC):
    """Abstract base class for packet scheduling algorithms"""

    def __init__(self):
        """
        Initialize the scheduling algorithm
        """
        self.name = "Base Router"

    @abstractmethod
    def route_packet(self, node, packet: Packet) -> int:
        """
        Select the next packet to transmit from the queue

        Args:
            queue: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        pass

    def __repr__(self) -> str:
        return self.name


class DijkstraRouter(Router):
    """First-In-First-Out (FIFO) scheduling algorithm"""

    def __init__(self):
        self.name = "Dijk"

    def route_packet(self, node, packet):
        """Select the next packet using FIFO policy"""
        if not node.queue:
            raise ValueError("Wtf")
        next_hop = node.routing_table[packet.destination]
        return next_hop


class LeastCongestionFirstRouter(Router):
    """Round Robin (RR) scheduling algorithm"""

    def __init__(self):
        self.name = "LCF"

    def route_packet(self, node, packet):
        smallest_buffer_usage = min(
            [neighbour.buffer_usage() for _, neighbour in node.neighbours.items()]
        )
        
        min_ids = [id for id, neighbour in node.neighbours.items() if smallest_buffer_usage == neighbour.buffer_usage()]    
        if not min_ids:
            raise ValueError("wtf")
        
        action = random.choice(min_ids)
        
        return action


class QRouter(Router):
    """Q-Learning based scheduling algorithm"""

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.name = "Q-Router"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _get_state(self, node, packet: Packet) -> Tuple[int, int, Any]:
        """Get the current state representation"""
        
        buffer_usages = [neighbour.buffer_usage() for _, neighbour in node.neighbours.items()]

        state = [packet.source, packet.destination] + buffer_usages
        return tuple(state)

    def _get_actions(self, node) -> List[int]:
        """Get available actions in the current state"""
        return [id for id, _ in node.links.items()]

    def route_packet(self, node, packet):
        """Select the next packet using Q-learning policy"""
        if not node.queue:
            return None

        state = self._get_state(node, packet)
        actions = self._get_actions(node)

        if not actions:
            return None

        if random.random() < self.exploration_rate:
            action = random.choice(actions)
        else:
            q_values = [self.q_table[state][a] for a in actions]
            max_q = max(q_values)
            max_actions = [a for q, a in zip(q_values, actions) if q == max_q]
            action = random.choice(max_actions)

        return action

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning update rule"""
        next_actions = self.get_actions()
        if next_actions:
            next_q_values = [self.q_table[next_state][a] for a in next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0

        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q


def router_factory(
    router_type: str,
    **kwargs
) -> Router:
    """
    Factory function to create the appropriate router

    Args:
        router_type: Type of the router ("RR", "QL", or other for FIFO)
        env: SimPy environment

    Returns:
        An instance of the selected scheduling algorithm
    """
    if router_type == "LCF":
        return LeastCongestionFirstRouter()
    elif router_type == "QL":
        return QRouter(kwargs)
    else:
        # Default to FIFO
        return DijkstraRouter()
