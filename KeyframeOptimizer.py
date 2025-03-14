import numpy as np
import gymnasium as gym
import csv

from GeneticAlgorithm import GeneticAlgorithm
from WalkerAgent import WalkerAgent

class KeyframeOptimizer:
    """
    Manages the optimization loop for evolving keyframes using a genetic algorithm.

    This class runs an evolutionary process to optimize keyframe sequences for an agent 
    operating in a reinforcement learning environment. It maintains a population of 
    keyframe sequences and applies selection, crossover, and mutation to improve their 
    performance over generations.

    Attributes:
        initial_keyframes (np.ndarray): The initial set of keyframes to evolve.
        agent (object): The agent that evaluates keyframes in a given environment.
        ga (GeneticAlgorithm): The genetic algorithm used for evolution.
        optional_keyframes (np.ndarray, optional): An alternative set of keyframes 
            to include in the initial population.
        optional_percentage (float): The percentage of the population initialized 
            using optional_keyframes.
        target_fitness (float): The fitness threshold that stops the optimization loop.
        population (list of np.ndarray): The current population of keyframes.
        best_fitness (float): The highest recorded fitness score.
        generation (int): The current generation number.
        render_mode_on (bool): Whether rendering is enabled for visualization.
    """

    def __init__(self, initial_keyframes, agent, ga, optional_keyframes=None, optional_percentage=0.5,
                 target_fitness=1000):
        """
        Initializes the KeyframeOptimizer with an initial keyframe population.

        Args:
            initial_keyframes (np.ndarray): The base keyframe sequence to mutate.
            agent (WalkerAgent): The agent responsible for evaluating keyframes.
            ga (GeneticAlgorithm): The genetic algorithm used for optimization.
            optional_keyframes (np.ndarray, optional): An additional keyframe sequence 
                to introduce diversity in the initial population. Defaults to None.
            optional_percentage (float): The fraction of the population initialized 
                using optional_keyframes. Defaults to 0.5.
            target_fitness (float): The stopping criteria based on the best fitness score. Defaults to 1000.
        """
        self.initial_keyframes = initial_keyframes
        self.agent = agent
        self.ga = ga
        self.target_fitness = target_fitness

        if optional_keyframes is not None:
            self.population = [ga.mutate(initial_keyframes.copy()) 
                               for _ in range(int(ga.population_size * (1 - optional_percentage)))]
            self.population.extend([ga.mutate(optional_keyframes.copy()) 
                                    for _ in range(int(ga.population_size * optional_percentage))])
        else:
            self.population = [ga.mutate(initial_keyframes.copy()) for _ in range(ga.population_size)]
        
        self.best_fitness = -np.inf
        self.generation = 0
        self.render_mode_on = False
    
    def optimize(self, filename="data/best_individual.csv", blx_alpha=0.5):
        """
        Runs the genetic algorithm until a keyframe sequence reaches the target fitness.

        This function iterates through generations, evaluating fitness, selecting top 
        individuals, performing crossover and mutation, and logging the results. The 
        process stops when the best fitness exceeds the target_fitness.

        Args:
            filename (str): The file where the best individual keyframe sequence is saved. Defaults to "best_individual.csv".
            blx_alpha (float): The alpha value for BLX-alpha crossover. Defaults to 0.5.
        """
        while self.best_fitness < self.target_fitness:
            fitnesses = np.array([self.agent.evaluate(ind) for ind in self.population])
            
            # Track and store the best individual of the generation
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                best_individual = self.population[best_idx]
                print(f"Gen {self.generation}: {self.best_fitness}")
                self.append_to_csv(best_individual, filename)
            
            # Select top-performing individuals as parents
            top_parents = [self.population[i] for i in np.argsort(fitnesses)[-5:]]
            new_population = list(top_parents)
            
            # Generate new individuals through crossover and mutation
            while len(new_population) < self.ga.population_size:
                p1, p2 = self.ga.tournament_selection(self.population, fitnesses), \
                         self.ga.tournament_selection(self.population, fitnesses)
                child = (self.ga.keyframe_swap_crossover(p1, p2) if np.random.rand() < 0.5 
                         else self.ga.blxalpha_crossover(p1, p2, alpha=0.5))
                new_population.append(self.ga.mutate(child))
            
            self.population = new_population
            self.log_to_csv("data/fitness_log.csv", self.generation, self.best_fitness)
            self.generation += 1
        
        # Enable rendering for the best individual
        self.render_mode_on = True

        # Recreate the environment with rendering enabled
        self.agent.env.close()
        self.agent.env = gym.make("Walker2d-v5", render_mode="human")
    
    def append_to_csv(self, individual, filename):
        """
        Saves the best-performing keyframe sequence to a CSV file.

        Each row in the file represents an individual's flattened keyframe parameters.

        Args:
            individual (np.ndarray): The best keyframe sequence from the generation.
            filename (str): The CSV file where the keyframes are stored.
        """
        with open(filename, 'a') as f:
            f.write(','.join(map(str, individual.flatten().tolist())) + '\n')
    
    def log_to_csv(self, filename, generation, fitness):
        """
        Logs the best fitness score for each generation to a CSV file.

        Args:
            filename (str): The file to store generation fitness progress.
            generation (int): The generation number.
            fitness (float): The best fitness score of the generation.
        """
        with open(filename, 'a') as f:
            f.write(f"{generation},{fitness}\n")
