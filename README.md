# **Keyframe Optimization for Walker2D using Evolutionary Algorithms**  

## **Introduction**  
This project employs an evolutionary algorithm to optimize keyframe-based motion for a Walker2D agent. Unlike conventional reinforcement learning approaches, this method evolves sequences of keyframes—each representing a set of joint angles and durations—until an optimal walking cycle is found. The process begins with manually approximated keyframes derived from human observation and iteratively refines them through selection, crossover, and mutation.  

## **Evolutionary Algorithm Overview**  
Evolutionary algorithms are a class of optimization techniques inspired by the process of natural selection. Given an initial population of potential solutions, the algorithm iteratively applies selection, crossover, and mutation to improve solutions over successive generations.  

Let the population at generation \( t \) be denoted as:  
\[
P(t) = \{X_1^t, X_2^t, ..., X_N^t\}
\]
where each \( X_i^t \) represents an individual (keyframe sequence). The algorithm seeks to maximize a fitness function \( f(X) \), which evaluates how well a given sequence enables the Walker2D to move effectively.  

The evolutionary process consists of:  
1. **Selection**: Choosing parent individuals based on fitness.  
2. **Crossover**: Combining parent keyframes to generate offspring.  
3. **Mutation**: Introducing small variations to maintain genetic diversity.  

This cycle repeats until a stopping condition (e.g., target fitness or generation limit) is met.  

---

## **Selection: Binary Tournament Selection**  
The algorithm employs **binary tournament selection**, a stochastic method that favors higher-fitness individuals while preserving diversity.  

### **Mathematical Formulation**  
Two individuals \( X_a \) and \( X_b \) are randomly selected from the population. Their fitness values \( f(X_a) \) and \( f(X_b) \) are compared, and the individual with higher fitness is chosen:  

\[
X_{\text{selected}} =
\begin{cases} 
X_a, & \text{if } f(X_a) > f(X_b) \\
X_b, & \text{otherwise}
\end{cases}
\]

This selection process is repeated to generate the mating pool. The stochastic nature of tournament selection helps maintain a balance between exploration and exploitation.  

---

## **Crossover: Generating New Individuals**  
Crossover combines genetic information from two parents to produce offspring. Two different crossover techniques are used with equal probability:  

### **1. Keyframe Swap Crossover**  
A segment of keyframes from one parent is inserted into the other, preserving temporal structure while allowing genetic mixing.  

#### **Mathematical Formulation**  
Let \( X_A \) and \( X_B \) be two parent sequences, each consisting of \( k \) keyframes:  

\[
X_A = [K_1^A, K_2^A, ..., K_k^A], \quad X_B = [K_1^B, K_2^B, ..., K_k^B]
\]

Two crossover points \( p_1, p_2 \) are selected:  
\[
1 \leq p_1 < p_2 \leq k
\]

The offspring is formed as:  
\[
X_{\text{child}} = [K_1^A, ..., K_{p_1}^A, K_{p_1+1}^B, ..., K_{p_2}^B, K_{p_2+1}^A, ..., K_k^A]
\]

This allows sections of a well-performing parent to propagate while introducing variation.  

### **2. Blend Crossover (BLX-α)**  
The **BLX-α** method generates offspring by extrapolating between parental values. This is particularly useful for real-valued parameters, ensuring smooth transitions between keyframe actions.  

#### **Mathematical Formulation**  
For each action dimension \( j \) in a keyframe:  

\[
X_{\text{child},j} = (1 + \alpha)X_{A,j} - \alpha X_{B,j}
\]

where \( \alpha \) is a hyperparameter (typically \( \alpha = 0.3 \)) controlling the blending range. Duration values are inherited randomly from either parent.  

BLX-α encourages exploration by allowing offspring to take values outside the direct parental range while ensuring smooth transitions between actions.  

Each crossover method is applied with a **50% probability**, ensuring genetic diversity.  

---

## **Mutation: Introducing Variability**  
Mutation prevents premature convergence by applying random perturbations to offspring.  

### **Mathematical Formulation**  
Mutation is applied with probability \( P_m \). If mutation occurs:  
1. **Action values** receive Gaussian noise:  
   \[
   X_{\text{mut},j} = X_{\text{orig},j} + \mathcal{N}(0, \sigma^2)
   \]
   where \( \sigma \) (mutation magnitude) controls the deviation.  
   
2. **Duration values** receive uniform noise:  
   \[
   d_{\text{mut}} = d_{\text{orig}} + U(0, 10)
   \]

All values are clipped within valid ranges to ensure physically feasible motions.  

---

## **Key Parameters and Their Roles**  

| **Parameter** | **Description** |
|--------------|----------------|
| **Population Size** | Number of individuals per generation. Larger values improve diversity but increase computation. |
| **Number of Keyframes** | Each individual consists of 7 keyframes. |
| **Tournament Size** | Fixed at 2, ensuring binary competition. |
| **Crossover Probability** | 50% chance of using either Swap Crossover or BLX-α. |
| **Mutation Rate** | Probability of applying mutation to an individual. |
| **Mutation Magnitude** | Controls the variance of Gaussian noise in actions. |
| **Target Fitness** | The optimization process stops when this value is reached. |

---

## **How to Run the Project**  

1. Ensure the following dependencies are installed:  
   ```bash
   pip install numpy gymnasium
   ```

2. Prepare keyframe initialization files:  
   - `keyframes/init.csv` contains the manually approximated keyframes.  
   - `keyframes/intermediate.csv` stores additional keyframes used in early evolution stages.  

3. Run the optimization process:  
   ```bash
   python main.py
   ```
   - This executes the evolutionary optimization loop.  
   - The best keyframe sequence is saved automatically.  

4. Execute the best-performing keyframe sequence:  
   - The script will load the optimized keyframes and animate the Walker2D agent in its environment.  
   - The simulation runs until termination, displaying the agent's movements.  

5. Analyze results:  
   - `fitness_log.csv` stores fitness progression over generations.  
   - The `plot.ipynb` notebook visualizes the fitness curve.
---

## **Conclusion**  
This project demonstrates the efficacy of evolutionary algorithms for optimizing keyframe-based motion. Instead of relying on traditional reinforcement learning, the approach refines an initially approximated motion cycle through iterative evolution. The use of **binary tournament selection, swap crossover, BLX-α crossover, and Gaussian mutation** ensures a balance between exploration and exploitation.  

The resulting motion is a dynamically optimized walking sequence, improving upon initial human-observed approximations.