# Network Routing Optimization with Q-Learning

A comparative study of reinforcement learning-based routing (Q-routing) against traditional routing algorithms (Dijkstra and OSPF) in dynamic network environments. This project implements and evaluates these routing strategies across various network topologies and traffic patterns using a custom simulator.

## Authors

- Ethan Rozee
- Jack Whitmar
- Eduard Kakosyan

## Project Overview

### Problem Statement
Traditional routing methods (Dijkstra, OSPF) often struggle with dynamic adaptation to fluctuating traffic and complex network topologies. This project explores Q-routing, a reinforcement learning approach, as an alternative that enables adaptive decision-making through iterative learning without prior knowledge of network dynamics.

### Key Features
- Implementation of a comprehensive simulator using Simpy and Networkx:
  - Simpy is used to model the network as a continuous event-driven system
  - Networkx is used to model the network topology and the traffic patterns
  - The simulator is used to simulate the network traffic and the routing algorithms
- Implementation of three routing algorithms:
  - Q-routing (Reinforcement Learning based)
  - Dijkstra's Shortest Path
  - Open Shortest Path First (OSPF)
- Support for multiple network topologies:
  - Sparse mesh-like networks
  - Dense mesh-like networks
- Various traffic pattern simulations:
  - Smooth (steady) traffic
  - Periodic traffic
  - Burst traffic
- Comprehensive performance metrics:
  - Throughput
  - Delay
  - Packet loss rate
  - Link utilization

## Technical Architecture

### System Components
1. **Network Scheduler**
   - Generates network traffic with different patterns
   - Interacts with the RL agent for bandwidth distribution
   - Manages traffic pattern generation (smooth, periodic, burst)

2. **Access Point Pool**
   - Collection of simulated access points
   - Handles task execution and routing
   - Manages resource utilization

3. **Reinforcement Learning Agent**
   - Implements Q-learning algorithm
   - Monitors real-time metrics
   - Makes dynamic routing decisions

## Prerequisites

- Python 3.13.1+
- pyenv
- uv 

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/EduardKakosyan/q-learning-network-sim.git
cd q-learning-network-sim
```

2. Install Python 3.13.1 using pyenv:
```bash
pyenv install 3.13.1
pyenv virtualenv 3.13.1 q-learning-network-sim
pyenv local q-learning-network-sim
```

3. Install dependencies using uv:
```bash
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

4. Verify the installation (optional):
```bash
mypy .
ruff check .
pytest
```
## Usage
### Running Simulations

1. Run a simulation:
```bash
python demo.py
```

### Configuration Options

- `--topology`: Network topology type (sparse/dense)
- `--traffic`: Traffic pattern type (smooth/periodic/burst)
- `--nodes`: Number of nodes in the network
- `--duration`: Simulation duration in seconds