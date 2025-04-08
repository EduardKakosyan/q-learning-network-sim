from itertools import product
from typing import Dict, List
from example import run_simulation, simulator_creator
from network_sim.utils.metrics import calculate_fairness_index

# Topology parameters
num_nodes: int = 8
excess_edges: int = 15
num_generators: int = 4

# Simulation parameters
router_time_scale: float = 1.0
duration: float = 10.0

# Define the grid of hyperparameters
param_grid: Dict[str, List[float]] = {
    "learning_rate": [0.01, 0.1, 0.2, 0.5],
    "discount_factor": [0.8, 0.9, 0.99],
    "exploration_rate": [0.01, 0.1, 0.2],
    "bins": [4, 8, 16],
    "bin_base": [10, 20, 30],
}


def grid_search_q_learning(
    param_grid: Dict[str, List[float]], router_type: str = "QL"
) -> Dict[str, float]:
    """Perform a grid search to find the best Q-learning parameters.

    Args:
        param_grid: Dictionary of hyperparameters to search over.
        router_type: Type of router to use in the simulation.

    Returns:
        The best set of Q-learning parameters found.
    """
    best_params: Dict[str, float] = None
    best_score: float = float("-inf")

    param_combos = list(product(*param_grid.values()))
    num_param_combos: int = len(param_combos)
    for i, params in enumerate(param_combos):
        ql_params: Dict[str, float] = dict(zip(param_grid.keys(), params))
        print(f"Testing Q-learning parameters: {ql_params}")
        print(f"Percent done: {i / num_param_combos * 100:.2f}%")

        simulator_func = simulator_creator(
            num_nodes,
            excess_edges,
            num_generators,
            router_time_scale,
            ql_params,
            show=False,
        )
        simulator = run_simulation(simulator_func, router_type, duration)

        performance: float = simulator.metrics["average_delay"]
        packet_loss: float = simulator.metrics["packet_loss_rate"]
        fairness: float = calculate_fairness_index(simulator)

        # Calculate a combined score
        score: float = -performance - packet_loss + fairness

        if score > best_score:
            best_score = score
            best_params = ql_params

    print(f"Best Q-learning parameters: {best_params} with score: {best_score}")
    return best_params


# Run the grid search
best_ql_params: Dict[str, float] = grid_search_q_learning(param_grid)
