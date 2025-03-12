import os
import json
import argparse
from network_simulator import TrafficPattern
from example_simulation import (
    run_simulation,
    compare_traffic_patterns,
    compare_topologies,
)
from algorithm_comparison import compare_schedulers, run_fairness_test
from visualize_results import visualize_simulation_results


def save_metrics_to_json(metrics, filename):
    """
    Save metrics to a JSON file

    Args:
        metrics: Metrics dictionary
        filename: Output filename
    """
    # Convert non-serializable objects to strings
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == "link_utilization":
            # Convert tuple keys to strings
            serializable_metrics[key] = {
                f"{src}->{dst}": util for (src, dst), util in value.items()
            }
        else:
            serializable_metrics[key] = value

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to file
    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=4)


def run_basic_simulation():
    """Run a basic simulation with default parameters"""
    print("\n=== Running Basic Simulation ===")
    metrics = run_simulation(
        topology_type="dumbbell", traffic_pattern=TrafficPattern.CONSTANT, duration=30
    )

    # Save metrics
    save_metrics_to_json(metrics, "results/basic_simulation.json")

    return metrics


def run_traffic_pattern_comparison():
    """Run simulations with different traffic patterns"""
    print("\n=== Comparing Traffic Patterns ===")
    compare_traffic_patterns()


def run_topology_comparison():
    """Run simulations with different topologies"""
    print("\n=== Comparing Network Topologies ===")
    compare_topologies()


def run_scheduler_comparison():
    """Run simulations with different scheduling algorithms"""
    print("\n=== Comparing Scheduling Algorithms ===")
    results = compare_schedulers()

    # Save results
    save_metrics_to_json(results, "results/scheduler_comparison.json")

    return results


def run_fairness_comparison():
    """Run fairness comparison between FIFO and RR"""
    print("\n=== Running Fairness Comparison ===")
    results = run_fairness_test()

    # Save results
    save_metrics_to_json(results, "results/fairness_comparison.json")

    return results


def main():
    """Main function to run simulations"""
    parser = argparse.ArgumentParser(description="Network Simulation Environment")
    parser.add_argument("--all", action="store_true", help="Run all simulations")
    parser.add_argument("--basic", action="store_true", help="Run basic simulation")
    parser.add_argument(
        "--traffic", action="store_true", help="Compare traffic patterns"
    )
    parser.add_argument("--topology", action="store_true", help="Compare topologies")
    parser.add_argument("--scheduler", action="store_true", help="Compare schedulers")
    parser.add_argument(
        "--fairness", action="store_true", help="Run fairness comparison"
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument(
        "--fifo", action="store_true", help="Run simulation with FIFO scheduler"
    )
    parser.add_argument(
        "--rr", action="store_true", help="Run simulation with Round Robin scheduler"
    )

    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run selected simulations
    if args.all or args.basic:
        run_basic_simulation()

    if args.all or args.traffic:
        run_traffic_pattern_comparison()

    if args.all or args.topology:
        run_topology_comparison()

    if args.all or args.scheduler:
        run_scheduler_comparison()

    if args.all or args.fairness:
        run_fairness_comparison()

    if args.all or args.visualize:
        visualize_simulation_results()

    # Run specific scheduler simulations
    if args.fifo:
        from algorithm_comparison import run_simulation_with_scheduler

        print("\n=== Running FIFO Scheduler Simulation ===")
        metrics = run_simulation_with_scheduler("FIFO", TrafficPattern.MIXED, 30)
        save_metrics_to_json(metrics, "results/fifo_simulation.json")

    if args.rr:
        from algorithm_comparison import run_simulation_with_scheduler

        print("\n=== Running Round Robin Scheduler Simulation ===")
        metrics = run_simulation_with_scheduler("RR", TrafficPattern.MIXED, 30)
        save_metrics_to_json(metrics, "results/rr_simulation.json")

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
