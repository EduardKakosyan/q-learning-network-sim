import matplotlib.pyplot as plt
import numpy as np
import json
import os


def plot_metrics_comparison(metrics_data, title, filename):
    """
    Plot comparison of metrics across different scenarios

    Args:
        metrics_data: Dictionary with scenario names as keys and metrics as values
        title: Title for the plot
        filename: Filename to save the plot
    """
    scenarios = list(metrics_data.keys())

    # Extract metrics
    throughputs = [metrics_data[s]["throughput"] / 1000 for s in scenarios]  # KB/s
    delays = [metrics_data[s]["average_delay"] * 1000 for s in scenarios]  # ms
    loss_rates = [metrics_data[s]["packet_loss_rate"] * 100 for s in scenarios]  # %

    # Create figure
    plt.figure(figsize=(15, 5))

    # Throughput subplot
    plt.subplot(1, 3, 1)
    plt.bar(scenarios, throughputs, color="skyblue")
    plt.title("Throughput")
    plt.ylabel("Throughput (KB/s)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Delay subplot
    plt.subplot(1, 3, 2)
    plt.bar(scenarios, delays, color="lightgreen")
    plt.title("Average Delay")
    plt.ylabel("Delay (ms)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Packet loss subplot
    plt.subplot(1, 3, 3)
    plt.bar(scenarios, loss_rates, color="salmon")
    plt.title("Packet Loss Rate")
    plt.ylabel("Loss Rate (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    plt.savefig(filename)
    plt.close()


def plot_time_series(time_series_data, title, filename):
    """
    Plot time series data

    Args:
        time_series_data: Dictionary with scenario names as keys and time series data as values
        title: Title for the plot
        filename: Filename to save the plot
    """
    scenarios = list(time_series_data.keys())
    metrics = ["throughput", "delay", "packet_loss"]

    # Create figure
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)

        for scenario in scenarios:
            times = time_series_data[scenario]["times"]
            values = time_series_data[scenario][metric]
            plt.plot(times, values, label=scenario)

        plt.title(f"{metric.capitalize()} over Time")
        plt.xlabel("Time (s)")

        if metric == "throughput":
            plt.ylabel("Throughput (KB/s)")
        elif metric == "delay":
            plt.ylabel("Delay (ms)")
        else:
            plt.ylabel("Packet Loss Rate (%)")

        plt.grid(linestyle="--", alpha=0.7)
        plt.legend()

    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    plt.savefig(filename)
    plt.close()


def plot_network_utilization(utilization_data, title, filename):
    """
    Plot network link utilization

    Args:
        utilization_data: Dictionary with link names as keys and utilization values as values
        title: Title for the plot
        filename: Filename to save the plot
    """
    links = list(utilization_data.keys())
    utilizations = [
        utilization_data[link] * 100 for link in links
    ]  # Convert to percentage

    # Sort by utilization
    sorted_indices = np.argsort(utilizations)
    links = [links[i] for i in sorted_indices]
    utilizations = [utilizations[i] for i in sorted_indices]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Horizontal bar chart
    plt.barh(links, utilizations, color="lightblue")
    plt.title("Link Utilization")
    plt.xlabel("Utilization (%)")
    plt.ylabel("Link")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Add values to bars
    for i, v in enumerate(utilizations):
        plt.text(v + 1, i, f"{v:.1f}%", va="center")

    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    plt.savefig(filename)
    plt.close()


def generate_sample_data():
    """Generate sample data for visualization"""
    # Sample metrics data for different schedulers
    scheduler_metrics = {
        "FIFO": {
            "throughput": 500000,  # 500 KB/s
            "average_delay": 0.05,  # 50 ms
            "packet_loss_rate": 0.05,  # 5%
        },
        "Round Robin": {
            "throughput": 550000,  # 550 KB/s
            "average_delay": 0.04,  # 40 ms
            "packet_loss_rate": 0.04,  # 4%
        },
        "Q-Learning": {
            "throughput": 600000,  # 600 KB/s
            "average_delay": 0.03,  # 30 ms
            "packet_loss_rate": 0.03,  # 3%
        },
    }

    # Sample time series data
    time_series = {
        "FIFO": {
            "times": list(range(0, 100, 5)),
            "throughput": [500 + np.random.normal(0, 50) for _ in range(20)],
            "delay": [50 + np.random.normal(0, 10) for _ in range(20)],
            "packet_loss": [5 + np.random.normal(0, 1) for _ in range(20)],
        },
        "Round Robin": {
            "times": list(range(0, 100, 5)),
            "throughput": [550 + np.random.normal(0, 50) for _ in range(20)],
            "delay": [40 + np.random.normal(0, 10) for _ in range(20)],
            "packet_loss": [4 + np.random.normal(0, 1) for _ in range(20)],
        },
        "Q-Learning": {
            "times": list(range(0, 100, 5)),
            "throughput": [600 + np.random.normal(0, 50) for _ in range(20)],
            "delay": [30 + np.random.normal(0, 10) for _ in range(20)],
            "packet_loss": [3 + np.random.normal(0, 1) for _ in range(20)],
        },
    }

    # Sample link utilization data
    link_utilization = {
        "S1->R1": 0.45,
        "S2->R1": 0.55,
        "S3->R1": 0.65,
        "R1->R2": 0.95,  # Bottleneck link
        "R2->D1": 0.35,
        "R2->D2": 0.40,
        "R2->D3": 0.45,
    }

    return scheduler_metrics, time_series, link_utilization


def save_results_to_json(data, filename):
    """Save results to a JSON file"""
    # Convert tuples in keys to strings
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(k, tuple):
                new_key = f"{k[0]}->{k[1]}"
            else:
                new_key = k

            if isinstance(v, dict):
                new_data[new_key] = save_results_to_json(v, None)
            else:
                new_data[new_key] = v
        data = new_data

    if filename:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    return data


def load_results_from_json(filename):
    """Load results from a JSON file"""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def visualize_simulation_results(results_dir="results"):
    """
    Visualize simulation results from JSON files

    Args:
        results_dir: Directory containing result JSON files
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Generate sample data if no real data exists
    if not os.path.exists(f"{results_dir}/scheduler_metrics.json"):
        print("Generating sample data...")
        scheduler_metrics, time_series, link_utilization = generate_sample_data()

        # Save sample data
        save_results_to_json(scheduler_metrics, f"{results_dir}/scheduler_metrics.json")
        save_results_to_json(time_series, f"{results_dir}/time_series.json")
        save_results_to_json(link_utilization, f"{results_dir}/link_utilization.json")
    else:
        # Load data from JSON files
        print("Loading data from JSON files...")
        scheduler_metrics = load_results_from_json(
            f"{results_dir}/scheduler_metrics.json"
        )
        time_series = load_results_from_json(f"{results_dir}/time_series.json")
        link_utilization = load_results_from_json(
            f"{results_dir}/link_utilization.json"
        )

    # Create visualizations
    print("Creating visualizations...")

    # Scheduler comparison
    plot_metrics_comparison(
        scheduler_metrics,
        "Comparison of Scheduling Algorithms",
        f"{results_dir}/scheduler_comparison.png",
    )

    # Time series
    plot_time_series(
        time_series, "Performance Metrics Over Time", f"{results_dir}/time_series.png"
    )

    # Link utilization
    plot_network_utilization(
        link_utilization,
        "Network Link Utilization",
        f"{results_dir}/link_utilization.png",
    )

    print(f"Visualizations saved to {results_dir}/ directory")


if __name__ == "__main__":
    visualize_simulation_results()
