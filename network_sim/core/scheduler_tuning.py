"""Parameter tuning for scheduling algorithms.

This module provides functionality for automatically tuning the parameters
of scheduling algorithms, particularly the Q-Learning router.
"""

import random
from typing import Dict, List, Tuple
import simpy

from network_sim.core.routing_algorithms import QLearningRouter


class TunedQLearningRouter(QLearningRouter):
    """Q-Learning router with automatic parameter tuning.

    This router extends the base QLearningRouter with the ability to
    automatically tune its parameters based on network conditions.
    """

    def __init__(
        self,
        env: simpy.Environment,
        adaptation_interval: float = 1.0,
        population_size: int = 5,
        generations: int = 3,
    ):
        """Initialize the tuned Q-Learning router.

        Args:
            env: SimPy environment.
            adaptation_interval: Time interval for parameter adaptation in seconds.
            population_size: Number of parameter sets to evaluate in each generation.
            generations: Number of generations for parameter optimization.
        """
        # Initialize with default parameters
        super().__init__(env)
        self.name = "Tuned Q-Learning"

        # Tuning parameters
        self.adaptation_interval = adaptation_interval
        self.population_size = population_size
        self.generations = generations

        # Performance tracking
        self.performance_history: List[float] = []
        self.parameter_history: List[Dict[str, float]] = []
        self.current_parameters = {
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "exploration_rate": 0.1,
        }

        # Apply initial parameters
        self.learning_rate = self.current_parameters["learning_rate"]
        self.discount_factor = self.current_parameters["discount_factor"]
        self.exploration_rate = self.current_parameters["exploration_rate"]

        # Schedule parameter adaptation
        self.env.process(self._adapt_parameters())

    def _adapt_parameters(self):
        """Periodically adapt parameters based on performance."""
        while True:
            # Wait for the adaptation interval
            yield self.env.timeout(self.adaptation_interval)

            # Evaluate current performance
            current_performance = self._evaluate_performance()
            self.performance_history.append(current_performance)
            self.parameter_history.append(self.current_parameters.copy())

            # Generate and evaluate new parameter sets
            best_parameters, best_performance = self._optimize_parameters()

            # Update parameters if better performance is found
            if best_performance > current_performance:
                self.current_parameters = best_parameters
                self.learning_rate = best_parameters["learning_rate"]
                self.discount_factor = best_parameters["discount_factor"]
                self.exploration_rate = best_parameters["exploration_rate"]

    def _evaluate_performance(self) -> float:
        """Evaluate the current performance of the router.

        Returns:
            Performance score (higher is better).
        """
        # This is a simplified performance metric based on Q-table utilization
        # In a real implementation, you would use metrics from the network simulation
        if not self.q_table:
            return 0.0

        # Calculate average Q-value as a performance indicator
        total_q = 0.0
        count = 0
        for state in self.q_table:
            for action, q_value in self.q_table[state].items():
                total_q += q_value
                count += 1

        return total_q / max(1, count)

    def _optimize_parameters(self) -> Tuple[Dict[str, float], float]:
        """Optimize parameters using a genetic algorithm approach.

        Returns:
            Tuple of (best parameters, best performance).
        """
        # Initialize population with random parameter sets
        population = []
        for _ in range(self.population_size):
            params = {
                "learning_rate": random.uniform(0.01, 0.5),
                "discount_factor": random.uniform(0.5, 0.99),
                "exploration_rate": random.uniform(0.01, 0.3),
            }
            population.append(params)

        # Add current parameters to the population
        population.append(self.current_parameters.copy())

        best_params = self.current_parameters.copy()
        best_performance = self._evaluate_performance()

        # Evolve the population for several generations
        for _ in range(self.generations):
            # Evaluate each parameter set
            performances = []
            for params in population:
                # Temporarily apply parameters
                old_lr = self.learning_rate
                old_df = self.discount_factor
                old_er = self.exploration_rate

                self.learning_rate = params["learning_rate"]
                self.discount_factor = params["discount_factor"]
                self.exploration_rate = params["exploration_rate"]

                # Evaluate performance
                performance = self._evaluate_performance()
                performances.append(performance)

                # Restore original parameters
                self.learning_rate = old_lr
                self.discount_factor = old_df
                self.exploration_rate = old_er

                # Update best parameters if better performance is found
                if performance > best_performance:
                    best_performance = performance
                    best_params = params.copy()

            # Create next generation through selection, crossover, and mutation
            next_population = [best_params.copy()]  # Elitism

            # Select parents based on performance (tournament selection)
            while len(next_population) < self.population_size:
                # Tournament selection
                idx1, idx2 = random.sample(range(len(population)), 2)
                parent1 = (
                    population[idx1]
                    if performances[idx1] > performances[idx2]
                    else population[idx2]
                )

                idx1, idx2 = random.sample(range(len(population)), 2)
                parent2 = (
                    population[idx1]
                    if performances[idx1] > performances[idx2]
                    else population[idx2]
                )

                # Crossover
                child = {}
                for key in parent1:
                    if random.random() < 0.5:
                        child[key] = parent1[key]
                    else:
                        child[key] = parent2[key]

                # Mutation
                for key in child:
                    if random.random() < 0.2:  # 20% mutation rate
                        if key == "learning_rate":
                            child[key] = random.uniform(0.01, 0.5)
                        elif key == "discount_factor":
                            child[key] = random.uniform(0.5, 0.99)
                        elif key == "exploration_rate":
                            child[key] = random.uniform(0.01, 0.3)

                next_population.append(child)

            population = next_population

        return best_params, best_performance

    def select_next_packet(self, queue, link):
        """Select the next packet using the tuned Q-learning policy.

        This method overrides the base class method to use the tuned parameters.

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            Selected packet or None if no packet should be transmitted
        """
        # Use the parent class implementation with tuned parameters
        return super().select_next_packet(queue, link)


class RewardBasedQLearningRouter(QLearningRouter):
    """Q-Learning router with a sophisticated reward function.

    This router extends the base QLearningRouter with a more
    sophisticated reward function that considers multiple network metrics.
    """

    def __init__(
        self,
        env: simpy.Environment,
        learning_rate: float = 0.2,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.15,
    ):
        """Initialize the reward-based Q-Learning router.

        Args:
            env: SimPy environment.
            learning_rate: Learning rate for Q-learning.
            discount_factor: Discount factor for future rewards.
            exploration_rate: Probability of taking a random action.
        """
        super().__init__(env, learning_rate, discount_factor, exploration_rate)
        self.name = "Reward-Based Q-Learning"

        # Additional state tracking
        self.last_state = None
        self.last_action = None
        self.last_queue_length = 0
        self.last_buffer_usage = 0
        self.packet_delays = {}  # Track packet delays

    def get_state(self, queue, link):
        """Get a more detailed state representation.

        Args:
            queue: Queue of packets waiting for transmission
            link: Link object where the packet will be transmitted

        Returns:
            A tuple representing the state
        """
        queue_length = len(queue)
        buffer_usage = (
            link.buffer_usage / link.buffer_size
            if link.buffer_size != float("inf")
            else 0
        )

        # More fine-grained discretization
        queue_length_discrete = min(queue_length // 3, 9)  # 0-9 (0-27+ packets)
        buffer_usage_discrete = min(
            int(buffer_usage * 20), 19
        )  # 0-19 (0-100% in 5% increments)

        # Add link utilization as a state component
        link_utilization = link.bytes_sent * 8 / max(1, self.env.now * link.capacity)
        link_utilization_discrete = min(
            int(link_utilization * 10), 9
        )  # 0-9 (0-100% in 10% increments)

        # Store current state for reward calculation
        self.last_queue_length = queue_length
        self.last_buffer_usage = buffer_usage

        return (queue_length_discrete, buffer_usage_discrete, link_utilization_discrete)

    def select_next_packet(self, queue, link):
        """Select the next packet using the reward-based Q-learning policy.

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

        # Calculate packet priorities based on waiting time and size
        priorities = []
        for i, packet in enumerate(queue):
            # Prioritize packets that have been waiting longer
            waiting_time = self.env.now - packet.creation_time
            # Prioritize smaller packets (to reduce delay)
            size_factor = 1.0 / max(1, packet.size / 1000)
            # Combined priority
            priority = waiting_time * size_factor
            priorities.append((i, priority))

        # Sort actions by priority (descending)
        sorted_actions = [
            a for a, _ in sorted(priorities, key=lambda x: x[1], reverse=True)
        ]

        # Exploration: random action
        if random.random() < self.exploration_rate:
            # Biased exploration: favor high-priority packets
            if random.random() < 0.7:  # 70% chance to pick from top half
                top_half = max(1, len(sorted_actions) // 2)
                action = sorted_actions[random.randint(0, top_half - 1)]
            else:
                action = random.choice(actions)
        # Exploitation: best known action
        else:
            # Get Q-values for all actions in this state
            q_values = [self.q_table[state][a] for a in actions]

            # Find the action with the highest Q-value
            max_q = max(q_values)
            max_indices = [i for i, q in enumerate(q_values) if q == max_q]

            # If multiple actions have the same Q-value, choose the one with highest priority
            if len(max_indices) > 1:
                # Find the highest priority action among those with max Q-value
                best_priority = -1
                best_action = actions[max_indices[0]]

                for idx in max_indices:
                    action_idx = actions[idx]
                    priority = priorities[action_idx][1]
                    if priority > best_priority:
                        best_priority = priority
                        best_action = action_idx

                action = best_action
            else:
                action_index = max_indices[0]
                action = actions[action_index]

        # Store state and action for reward calculation
        self.last_state = state
        self.last_action = action

        # Track packet for delay calculation
        selected_packet = queue[action]
        self.packet_delays[selected_packet.id] = {
            "selection_time": self.env.now,
            "queue_length": len(queue),
            "buffer_usage": link.buffer_usage,
        }

        # Return the selected packet
        return selected_packet

    def update_reward(self, packet_id: int, transmission_time: float, success: bool):
        """Update Q-table with reward for a transmitted packet.

        Args:
            packet_id: ID of the packet that was transmitted
            transmission_time: Time taken to transmit the packet
            success: Whether the packet was successfully transmitted
        """
        if (
            packet_id not in self.packet_delays
            or self.last_state is None
            or self.last_action is None
        ):
            return

        packet_info = self.packet_delays.pop(packet_id)

        # Calculate waiting time (time from selection to transmission)
        waiting_time = transmission_time - packet_info["selection_time"]

        # Calculate queue length change
        queue_change = self.last_queue_length - packet_info["queue_length"]

        # Calculate buffer usage change
        buffer_change = self.last_buffer_usage - packet_info["buffer_usage"]

        # Calculate reward based on multiple factors
        if success:
            # Positive reward for successful transmission
            base_reward = 1.0

            # Bonus for reducing queue length
            queue_bonus = 0.2 * max(0, queue_change)

            # Bonus for reducing buffer usage
            buffer_bonus = 0.2 * max(0, buffer_change)

            # Penalty for long waiting time
            waiting_penalty = 0.5 * min(1.0, waiting_time)

            reward = base_reward + queue_bonus + buffer_bonus - waiting_penalty
        else:
            # Negative reward for failed transmission
            reward = -1.0

        # Update Q-table
        # For simplicity, we're using a dummy next state
        # In a real implementation, you would track the actual next state
        next_state = self.get_state([], None) if self.last_state else None

        if next_state:
            self.update_q_table(self.last_state, self.last_action, reward, next_state)

        # Reset state tracking
        self.last_state = None
        self.last_action = None
