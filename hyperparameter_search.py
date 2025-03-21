#!/usr/bin/env python3
"""Hyper-parameter search for Q-learning router using Optuna."""

import optuna

from example import simulator_creator
from network_sim.utils.metrics import calculate_fairness_index

def objective(trial: optuna.Trial):
    # Topology parameters
    num_nodes = 8
    excess_edges = 18
    num_generators = 5

    # Simulation parameters
    router_time_scale = 1.0
    duration = 10.0

    # Q Learning parameters
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2, 0.5],
        "discount_factor": [0.8, 0.9, 0.99],
        "exploration_rate": [0.01, 0.1, 0.2],
        "bins": [4, 8, 16],
        "bin_base": [10, 20, 30],
    }
    def get(name: str):
        return trial.suggest_categorical(name, param_grid[name])
    ql_params = {
        "learning_rate": get("learning_rate"),
        "discount_factor": get("discount_factor"),
        "exploration_rate": get("exploration_rate"),
        "bins": get("bins"),
        "bin_base": get("bin_base"),
    }

    simulator_func = simulator_creator(num_nodes, excess_edges, num_generators, router_time_scale, ql_params)
    simulator = simulator_func("QL")

    simulator.run(duration, updates=True)

    delay = simulator.metrics["average_delay"]
    packet_loss = simulator.metrics["packet_loss_rate"]
    throughput = simulator.metrics["throughput"]
    fairness = calculate_fairness_index(simulator)

    # Objective is to minimize delay and packet loss, and maximize throughput and fairness
    return delay, packet_loss, throughput, fairness

def main():
    study = optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize"])
    study.optimize(objective, n_trials=100)

    best_trials = study.best_trials
    for trial in best_trials:
        print("Best parameters:", trial.params)
        print("Best values:", trial.values)

if __name__ == "__main__":
    main()
