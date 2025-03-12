from abc import ABC, abstractmethod
from collections import deque, defaultdict
import random
import numpy as np


class SchedulingAlgorithm(ABC):
    """Abstract base class for packet scheduling algorithms"""

    def __init__(self, env):
        """
        Initialize the scheduling algorithm

        Args:
            env: SimPy environment
        """
        self.env = env
        self.name = "Base Scheduler"

    @abstractmethod
    def select_next_packet(self, queue, link):
        """
        Select the next packet to transmit from the queue

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        pass

    def __repr__(self):
        return self.name


class FIFOScheduler(SchedulingAlgorithm):
    """
    First-In-First-Out (FIFO) scheduling algorithm

    This is a placeholder implementation. The actual FIFO logic would be
    integrated into the network simulator's packet processing.
    """

    def __init__(self, env):
        super().__init__(env)
        self.name = "FIFO"

    def select_next_packet(self, queue, link):
        """
        Select the next packet using FIFO policy

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            The first packet in the queue or None if queue is empty
        """
        if not queue:
            return None
        return queue[0]  # Return the first packet in the queue


class RoundRobinScheduler(SchedulingAlgorithm):
    """
    Round Robin (RR) scheduling algorithm

    This is a placeholder implementation. The actual RR logic would need to be
    integrated into the network simulator to maintain separate queues.
    """

    def __init__(self, env):
        super().__init__(env)
        self.name = "Round Robin"
        self.queues = defaultdict(deque)  # Separate queues for different flows
        self.current_queue_index = 0

    def select_next_packet(self, queue, link):
        """
        Select the next packet using Round Robin policy

        Note: This is a simplified version. In a real implementation, we would
        need to maintain separate queues for different flows and cycle through them.

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        # This is a placeholder. In a real implementation, we would:
        # 1. Group packets by flow (source-destination pair)
        # 2. Maintain a pointer to the current flow
        # 3. Select one packet from the current flow
        # 4. Move the pointer to the next flow

        if not queue:
            return None
        return queue[0]  


class QLearningScheduler(SchedulingAlgorithm):
    """
    Q-Learning based scheduling algorithm

    This is a placeholder implementation. The actual Q-learning implementation
    would require state representation, action selection, and Q-table updates.
    """

    def __init__(
        self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1
    ):
        super().__init__(env)
        self.name = "Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_state(self, queue, link):
        """
        Get the current state representation

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            A tuple representing the state
        """
        # This is a placeholder. In a real implementation, we would:
        # 1. Extract relevant features from the queue and link
        # 2. Discretize continuous values if necessary
        # 3. Return a hashable state representation

        queue_length = len(queue)
        buffer_usage = (
            link.buffer_usage / link.buffer_size
            if link.buffer_size != float("inf")
            else 0
        )

        # Discretize state space
        queue_length_discrete = min(
            queue_length // 5, 5
        )  # 0, 1, 2, 3, 4, 5 (5+ packets)
        buffer_usage_discrete = min(
            int(buffer_usage * 10), 9
        )  # 0, 1, ..., 9 (0-100% in 10% increments)

        return (queue_length_discrete, buffer_usage_discrete)

    def get_actions(self, queue):
        """
        Get available actions in the current state

        Args:
            queue: Queue of packets waiting for transmission

        Returns:
            List of available actions (packet indices)
        """
        return list(range(len(queue))) if queue else []

    def select_next_packet(self, queue, link):
        """
        Select the next packet using Q-learning policy

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        if not queue:
            return None

        state = self.get_state(queue, link)
        actions = self.get_actions(queue)

        if not actions:
            return None

        # Exploration: random action
        if random.random() < self.exploration_rate:
            action = random.choice(actions)
        # Exploitation: best known action
        else:
            # Get Q-values for all actions in this state
            q_values = [self.q_table[state][a] for a in actions]

            # Find the action with the highest Q-value
            max_q = max(q_values)
            max_indices = [i for i, q in enumerate(q_values) if q == max_q]

            # If multiple actions have the same Q-value, choose randomly
            action_index = random.choice(max_indices)
            action = actions[action_index]

        # Return the selected packet
        return queue[action]

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning update rule

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get the best Q-value for the next state
        next_actions = self.get_actions(next_state)
        if next_actions:
            next_q_values = [self.q_table[next_state][a] for a in next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0

        # Q-learning update rule
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-table
        self.q_table[state][action] = new_q
