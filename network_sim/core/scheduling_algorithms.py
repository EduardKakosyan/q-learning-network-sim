from abc import ABC, abstractmethod
from collections import deque, defaultdict
from network_sim.core.packet import Packet
from typing import Deque, Dict, Optional
import random
import simpy


class SchedulingAlgorithm(ABC):
    """Abstract base class for packet scheduling algorithms"""

    def __init__(self, env: simpy.Environment):
        """
        Initialize the scheduling algorithm

        Args:
            env: SimPy environment
        """
        self.env = env
        self.name = "Base Scheduler"
        self.queue: Deque[Packet] = deque()

    @abstractmethod
    def select_next_packet(self, link) -> Optional[Packet]:
        """
        Select the next packet to transmit from the queue

        Args:
            link: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        pass

    def add_packet(self, packet: Packet):
        """
        Add a packet to the queue

        Args:
            packet: Packet to be added
        """
        self.queue.append(packet)

    def __repr__(self) -> str:
        return self.name


class FIFOScheduler(SchedulingAlgorithm):
    """First-In-First-Out (FIFO) scheduling algorithm"""

    def __init__(self, env):
        super().__init__(env)
        self.name = "FIFO"

    def select_next_packet(self, link):
        """Select the next packet using FIFO policy"""
        if not self.queue:
            return None
        packet = self.queue[0]  # Return the first packet in the queue
        if packet:
            self.queue.remove(packet)
        return packet


class RoundRobinScheduler(SchedulingAlgorithm):
    """Round Robin (RR) scheduling algorithm"""

    def __init__(self, env):
        super().__init__(env)
        self.name = "Round Robin"
        self.flow_queues: Dict[str, Deque[Packet]] = defaultdict(deque)
        self.current_queue_index = 0

    def add_packet(self, packet):
        """Add a packet to the appropriate flow queue"""
        flow_id = packet.flow_id
        self.flow_queues[flow_id].append(packet)

    def select_next_packet(self, link):
        """Select the next packet using Round Robin policy"""
        if not self.flow_queues:
            return None

        flow_ids = list(self.flow_queues.keys())
        for _ in range(len(flow_ids)):
            flow_id = flow_ids[self.current_queue_index]
            self.current_queue_index = (self.current_queue_index + 1) % len(flow_ids)
            if self.flow_queues[flow_id]:
                packet = self.flow_queues[flow_id][0]
                self.flow_queues[flow_id].remove(packet)
                return packet
        return None


class QLearningScheduler(SchedulingAlgorithm):
    """Q-Learning based scheduling algorithm"""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        super().__init__(env)
        self.name = "Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _get_state(self, link):
        """Get the current state representation"""
        queue_length = len(self.queue)
        buffer_usage = link.buffer_usage / link.buffer_size if link.buffer_size != float("inf") else 0

        queue_length_discrete = min(queue_length // 5, 5)
        buffer_usage_discrete = min(int(buffer_usage * 10), 9)

        return (queue_length_discrete, buffer_usage_discrete)

    def _get_actions(self):
        """Get available actions in the current state"""
        return list(range(len(self.queue))) if self.queue else []

    def select_next_packet(self, link):
        """Select the next packet using Q-learning policy"""
        if not self.queue:
            return None

        state = self._get_state(link)
        actions = self._get_actions()

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

        return self.queue[action]

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
    env: simpy.Environment
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
        return RoundRobinScheduler(env)
    elif scheduler_type == "QL":
        return QLearningScheduler(env)
    else:
        # Default to FIFO
        return FIFOScheduler(env)
