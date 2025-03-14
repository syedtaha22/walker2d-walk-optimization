"""
Keyframe Optimization and Execution for Walker2D Environment

This script performs keyframe-based optimization for controlling a Walker2D agent 
using a genetic algorithm. It evolves keyframe sequences to maximize an agent's 
performance in a reinforcement learning environment.

Overview:
    1. Load Keyframes: 
        - Reads initial keyframes from CSV files.
        - Keyframes define the agent's action sequences and duration.

    2. Initialize Components:
        - `WalkerAgent`: Controls the Walker2D environment and evaluates keyframes.
        - `GeneticAlgorithm`: Implements genetic operators (mutation, crossover, selection).
        - `KeyframeOptimizer`: Runs the evolutionary optimization process.

    3. Optimization:
        - Runs a genetic algorithm to evolve keyframe sequences.
        - Selects and refines the best keyframes based on agent performance.
        - Saves optimized keyframes to a CSV file.

    4. Execution of the Best Individual:
        - Loads the best evolved keyframe sequence.
        - Runs the agent in the Walker2D environment.
        - Loops through the keyframe sequence to control the agent.

Dependencies:
    - `WalkerAgent`: Handles agent interaction with the environment.
    - `GeneticAlgorithm`: Provides genetic operators for evolution.
    - `KeyframeOptimizer`: Manages the optimization loop.
    - `numpy`, `csv`: For data processing and file I/O.
    - `gymnasium`: For running the Walker2D simulation.
"""

import csv
import numpy as np

from WalkerAgent import WalkerAgent
from KeyframeOptimizer import KeyframeOptimizer
from GeneticAlgorithm import GeneticAlgorithm

# Configuration Constants
INITIAL_KEYFRAME_FILE = "keyframes/init.csv"
INTERMEDIATE_KEYFRAME_FILE = "keyframes/1500.csv" 
NUM_KEYFRAMES = 2
OPTIONAL_KEYFRAME_PERCENTAGE = 0.5
TARGET_FITNESS = 2500

BLX_ALPHA = 0.5
MUTATION_RATE = 0.1
MUTATION_MAGNITUDE = 0.2

CTRL_COST_WEIGHT = 1e-3
FORWARD_REWARD_WEIGHT = 1


def load_keyframes(filename: str, num_keyframes: int = 10) -> np.ndarray:
    """
    Loads keyframe data from a CSV file.

    Args:
        filename (str): Path to the CSV file containing keyframe data.
        num_keyframes (int): Number of keyframes to load from the file.

    Returns:
        np.ndarray: Array of keyframe data (shape: [num_keyframes, action_dim + 1]).
                    Each row represents an action with the last column as duration.
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        keyframes = [list(map(float, row)) for row in reader]
    return np.array(keyframes[:num_keyframes])


if __name__ == "__main__":
    # Load keyframes
    initial_keyframes = load_keyframes(INITIAL_KEYFRAME_FILE, NUM_KEYFRAMES)
    optional_keyframes = load_keyframes(INTERMEDIATE_KEYFRAME_FILE, NUM_KEYFRAMES)

    # Initialize components
    agent = WalkerAgent(ctrl_cost_weight=CTRL_COST_WEIGHT, forward_reward_weight=FORWARD_REWARD_WEIGHT)
    ga = GeneticAlgorithm(mutation_rate=MUTATION_RATE, mutation_magnitude=MUTATION_MAGNITUDE)
    optimizer = KeyframeOptimizer(
        initial_keyframes,
        agent,
        ga,
        optional_keyframes=optional_keyframes,
        optional_percentage=OPTIONAL_KEYFRAME_PERCENTAGE,
        target_fitness=TARGET_FITNESS
    )

    # Start optimization
    optimizer.optimize(blx_alpha=BLX_ALPHA)

    # Run the best individual in the environment
    observation, _ = agent.env.reset()
    keyframe_index = 0
    remaining_steps = int(optimizer.population[0][0, -1])
    episode_reward = 0

    print("Running best individual...")
    while True:
        # Retrieve action from the best individual’s keyframe sequence
        action = optimizer.population[0][keyframe_index, :-1]
        observation, reward, terminated, truncated, _ = agent.env.step(action)
        episode_reward += reward

        agent.env.render()

        remaining_steps -= 1
        if remaining_steps <= 0:
            keyframe_index = (keyframe_index + 1) % len(optimizer.population[0])
            remaining_steps = int(optimizer.population[0][keyframe_index, -1])

        if terminated or truncated:
            print(f"Episode ended. Total reward: {episode_reward}")

            # Reset for a new episode
            episode_reward = 0
            observation, _ = agent.env.reset()
            keyframe_index = 0
            remaining_steps = int(optimizer.population[0][0, -1])

    agent.env.close()
