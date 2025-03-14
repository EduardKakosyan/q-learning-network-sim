from abc import ABC, abstractmethod
from collections import defaultdict, deque
from network_sim.core.packet import Packet
from typing import Deque, Optional
import random


class SchedulingAlgorithm(ABC):
    """Abstract base class for packet scheduling algorithms"""

    def __init__(self):
        """
        Initialize the scheduling algorithm
        """
        self.name = "Base Scheduler"

    @abstractmethod
    def select_next_packet(self, node) -> Optional[Packet]:
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


class FIFOScheduler(SchedulingAlgorithm):
    """First-In-First-Out (FIFO) scheduling algorithm"""

    def __init__(self):
        self.name = "FIFO"

    def select_next_packet(self, node):
        """Select the next packet using FIFO policy"""
        if not node.queue:
            return None
        return node.queue[0] # Return the first packet in the queue


class RoundRobinScheduler(SchedulingAlgorithm):
    """Round Robin (RR) scheduling algorithm"""

    def __init__(self):
        self.name = "Round Robin"
        self.flow_ids: Deque[str] = deque()

    def _next_flow_id(self) -> None:
        front = self.flow_ids.popleft()
        self.flow_ids.append(front)

    def select_next_packet(self, node):
        """Select the next packet using Round Robin policy"""
        if not node.queue:
            return None

        for _ in range(len(self.flow_ids)):
            current_flow_id = self.flow_ids[0]
            for packet in node.queue:
                if packet.flow_id == current_flow_id:
                    self._next_flow_id()
                    return packet
            self._next_flow_id()

        return node.queue[0] # Default


class QLearningScheduler(SchedulingAlgorithm):
    """Q-Learning based scheduling algorithm"""

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.name = "Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _get_state(self, node):
        """Get the current state representation"""
        queue_length = len(node.queue)
        buffer_usage = node.buffer_usage / node.buffer_size if node.buffer_size != float("inf") else 0

        queue_length_discrete = min(queue_length // 5, 5)
        buffer_usage_discrete = min(int(buffer_usage * 10), 9)

        return (queue_length_discrete, buffer_usage_discrete)

    def _get_actions(self, node):
        """Get available actions in the current state"""
        return list(range(len(node.queue))) if node.queue else []

    def select_next_packet(self, node):
        """Select the next packet using Q-learning policy"""
        if not node.queue:
            return None

        state = self._get_state(node)
        actions = self._get_actions(node)

        if not actions:
            return None

        if random.random() < self.exploration_rate:
            action = random.choice(actions)
        else:
            q_values = [self.q_table[state][a] for a in actions]
            max_q = max(q_values)
            max_indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_index = random.choice(max_indices)
            action = actions[action_index]

        return node.queue[action]

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


def scheduling_algorithm_factory(
    scheduler_type: str,
    **kwargs
) -> SchedulingAlgorithm:
    """
    Factory function to create the appropriate scheduler

    Args:
        scheduler_type: Type of the scheduler ("RR", "QL", or other for FIFO)
        env: SimPy environment

    Returns:
        An instance of the selected scheduling algorithm
    """
    if scheduler_type == "RR":
        return RoundRobinScheduler()
    elif scheduler_type == "QL":
        return QLearningScheduler()
    else:
        # Default to FIFO
        return FIFOScheduler()
