import gymnasium as gym

class WalkerAgent:
    """
    Handles interactions with the Walker2d environment, including evaluation of individuals.

    This class initializes a Gymnasium Walker2d environment with customizable reward and cost weights.
    It provides a method to evaluate an individual's performance by simulating multiple runs.

    Attributes:
        env (gym.Env): The Gymnasium environment for the Walker2d agent.
    """
    def __init__(self, env_name="Walker2d-v5", forward_reward_weight=1, ctrl_cost_weight=1e-3):
        """
        Initializes the WalkerAgent with a specified environment and reward parameters.

        Args:
            env_name (str): Name of the Gym environment to use. Default is "Walker2d-v5".
            forward_reward_weight (float): Weight for the forward velocity reward. Default is 1.
            ctrl_cost_weight (float): Weight for the control cost penalty. Default is 1e-3.
        """
        self.env = gym.make(env_name, forward_reward_weight=forward_reward_weight, ctrl_cost_weight=ctrl_cost_weight)
    
    def evaluate(self, individual, runs=3):
        """
        Evaluates an individual's performance over multiple runs and returns the average reward.

        The individual represents a set of keyframe-based actions, where each row contains an action vector 
        followed by a duration for which that action should be executed before moving to the next keyframe.

        The evaluation follows these steps:
        1. The environment is reset.
        2. The agent sequentially executes the actions defined by the individual.
        3. Each action is applied for a duration specified in the last column of the individual's row.
        4. If the episode terminates or is truncated, the run ends.
        5. The total reward is averaged over the specified number of runs.

        The reward function includes:
        - The total environment reward accumulated across runs.

        Args:
            individual (np.ndarray): A 2D array where each row represents a keyframe.
                                     The last column of each row specifies the duration for which the action is executed.
            runs (int): The number of evaluation runs to average over. Default is 3.

        Returns:
            float: The average modified reward over all runs.
        """
        total_reward = 0

        for _ in range(runs):
            observation, _ = self.env.reset()
            keyframe_index, remaining_steps = 0, int(individual[0, -1])

            while True:
                action = individual[keyframe_index, :-1]  # Extract action (excluding last column)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                
                # Decrement step counter for current keyframe
                remaining_steps -= 1
                if remaining_steps <= 0:
                    keyframe_index = (keyframe_index + 1) % len(individual)  # Move to the next keyframe
                    remaining_steps = int(individual[keyframe_index, -1])  # Reset step counter
                
                if terminated or truncated:
                    break
        
        return total_reward / runs