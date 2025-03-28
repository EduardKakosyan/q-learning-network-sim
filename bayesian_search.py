#!/usr/bin/env python3
"""Hyper-parameter search for Q-learning router using Optuna."""

import random
import optuna
from typing import Dict, List
from example import simulator_creator

seed: int = random.randint(0, 2**32 - 1)


def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    """Objective function for the Optuna study.

    Args:
        trial: An Optuna trial object.

    Returns:
        A tuple containing average delay, packet loss rate, and throughput.
    """
    # Topology parameters
    num_nodes: int = 8
    excess_edges: int = 15
    num_generators: int = 4

    # Simulation parameters
    router_time_scale: float = 0.0
    duration: float = 10.0

    # Q Learning parameters
    param_grid: Dict[str, List[float]] = {
        "learning_rate": [0.01, 0.1, 0.2, 0.5],
        "discount_factor": [0.8, 0.9, 0.99],
        "exploration_rate": [0.01, 0.1, 0.2, 0.5],
        "bins": [4, 8, 16],
        "bin_base": [10, 20, 30],
    }

    def get(name: str) -> float:
        return trial.suggest_categorical(name, param_grid[name])

    ql_params: Dict[str, float] = {
        "learning_rate": get("learning_rate"),
        "discount_factor": get("discount_factor"),
        "exploration_rate": get("exploration_rate"),
        "bins": get("bins"),
        "bin_base": get("bin_base"),
    }

    simulator_func = simulator_creator(
        num_nodes,
        excess_edges,
        num_generators,
        router_time_scale,
        ql_params,
        seed=seed,
        show=False,
    )
    simulator = simulator_func("QL", 4)

    simulator.run(duration, updates=True)

    delay: float = simulator.metrics["average_delay"]
    packet_loss: float = simulator.metrics["packet_loss_rate"]
    throughput: float = simulator.metrics["throughput"]

    return delay, packet_loss, throughput


def main() -> None:
    """Create and optimize an Optuna study for Q-learning parameters."""
    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
    study.optimize(objective, n_trials=100)

    best_trials = study.best_trials
    for trial in best_trials:
        print("Best parameters:", trial.params)
        print("Best values:", trial.values)


if __name__ == "__main__":
    main()
