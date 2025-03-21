from itertools import product

from example import run_simulation, simulator_creator
from network_sim.utils.metrics import calculate_fairness_index

# Topology parameters
num_nodes = 8
excess_edges = 18
num_generators = 5

# Simulation parameters
router_time_scale = 1.0
duration = 10.0

# Define the grid of hyperparameters
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2, 0.5],
    "discount_factor": [0.8, 0.9, 0.99],
    "exploration_rate": [0.01, 0.1, 0.2],
    "bins": [4, 8, 16],
    "bin_base": [10, 20, 30],
}

def grid_search_q_learning(param_grid, router_type="QL"):
    best_params = None
    best_score = float('-inf')

    param_combos = product(*param_grid.values())
    num_param_combos = len(param_combos)
    for i, params in enumerate(param_combos):
        ql_params = dict(zip(param_grid.keys(), params))
        print(f"Testing Q-learning parameters: {ql_params}")
        print(f"Percent done: {i / num_param_combos * 100}%")

        simulator_func = simulator_creator(num_nodes, excess_edges, num_generators, router_time_scale, ql_params)
        simulator = run_simulation(simulator_func, router_type, duration)

        performance = simulator.metrics["average_delay"]
        packet_loss = simulator.metrics["packet_loss_rate"]
        fairness = calculate_fairness_index(simulator)

        # Calculate a combined score
        score = -performance - packet_loss + fairness

        if score > best_score:
            best_score = score
            best_params = ql_params

    print(f"Best Q-learning parameters: {best_params} with score: {best_score}")
    return best_params

# Run the grid search
best_ql_params = grid_search_q_learning(param_grid)
