import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from collections import defaultdict
import pandas as pd
import random
import time
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_1_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introduction to Reinforcement Learning
print_step_header(1, "Introduction to Reinforcement Learning")

print("In this problem, we're exploring reinforcement learning (RL) in the context of a video game.")
print("Reinforcement learning is a type of machine learning where an agent learns to make decisions")
print("by taking actions in an environment to maximize cumulative reward.")
print()
print("Key components of reinforcement learning:")
print("1. Agent: The learner or decision-maker")
print("2. Environment: The external system the agent interacts with")
print("3. State: The current situation or configuration")
print("4. Action: The set of possible moves the agent can make")
print("5. Reward: Feedback from the environment (positive or negative)")
print("6. Policy: The agent's strategy to choose actions based on states")
print()

# Step 2: Create a simple game environment (Grid World)
print_step_header(2, "Creating a Grid World Environment")

class GridWorld:
    """A simple grid world environment for reinforcement learning."""
    
    def __init__(self, width=5, height=5, deterministic=True):
        # Environment parameters
        self.width = width
        self.height = height
        self.deterministic = deterministic  # If False, actions have randomness
        self.action_randomness = 0.2 if not deterministic else 0
        
        # Define states
        self.states = [(x, y) for x in range(width) for y in range(height)]
        
        # Define actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_names = ["Up", "Right", "Down", "Left"]
        
        # Define obstacles
        self.obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
        
        # Define terminal states with rewards
        self.goal_state = (width-1, height-1)
        self.goal_reward = 10
        self.hole_states = [(1, 3)]
        self.hole_reward = -10
        
        # Default step reward
        self.step_reward = -0.1
        
        # Current state
        self.current_state = (0, 0)
        
        # Transition counts for visualization
        self.state_visits = {state: 0 for state in self.states}
    
    def reset(self):
        """Reset the environment to the initial state."""
        self.current_state = (0, 0)
        self.state_visits = {state: 0 for state in self.states}
        return self.current_state
    
    def step(self, action):
        """
        Take an action in the environment and return the new state, reward, and done flag.
        If the environment is stochastic, the action may have randomness.
        """
        # Increase visit count for the current state
        self.state_visits[self.current_state] += 1
        
        # Check if current state is terminal
        if self.current_state == self.goal_state:
            return self.current_state, self.goal_reward, True
        if self.current_state in self.hole_states:
            return self.current_state, self.hole_reward, True
        
        # Process the action (possibly with randomness)
        if not self.deterministic and np.random.random() < self.action_randomness:
            # Choose a random action different from the intended one
            possible_actions = [a for a in self.actions if a != action]
            action = np.random.choice(possible_actions)
        
        # Calculate the next state based on the action
        x, y = self.current_state
        if action == 0:  # Up
            next_state = (max(x-1, 0), y)
        elif action == 1:  # Right
            next_state = (x, min(y+1, self.width-1))
        elif action == 2:  # Down
            next_state = (min(x+1, self.height-1), y)
        elif action == 3:  # Left
            next_state = (x, max(y-1, 0))
        
        # Check if the next state is an obstacle
        if next_state in self.obstacles:
            next_state = self.current_state  # Stay in the same place
        
        # Update the current state
        self.current_state = next_state
        
        # Determine reward and done flag
        reward = self.step_reward
        done = False
        
        if self.current_state == self.goal_state:
            reward = self.goal_reward
            done = True
        elif self.current_state in self.hole_states:
            reward = self.hole_reward
            done = True
        
        return self.current_state, reward, done
    
    def render(self, q_values=None, policy=None, title="Grid World"):
        """Render the grid world environment with optional Q-values or policy."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the grid
        for x in range(self.width + 1):
            ax.axhline(x, color='black', linestyle='-', alpha=0.3)
        for y in range(self.height + 1):
            ax.axvline(y, color='black', linestyle='-', alpha=0.3)
        
        # Fill cells with colors based on their type
        cmap = ListedColormap(['white', 'darkgrey', 'green', 'red'])
        grid_data = np.zeros((self.height, self.width))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid_data[obs[0], obs[1]] = 1
        
        # Mark goal
        grid_data[self.goal_state[0], self.goal_state[1]] = 2
        
        # Mark holes
        for hole in self.hole_states:
            grid_data[hole[0], hole[1]] = 3
        
        # Plot the grid
        ax.imshow(grid_data, cmap=cmap, origin='upper', extent=[0, self.width, self.height, 0])
        
        # Draw state visit counts or q-values
        for state in self.states:
            x, y = state
            if state not in self.obstacles:
                if q_values is not None:
                    # Draw Q-values for each action
                    q_vals = q_values.get(state, [0, 0, 0, 0])
                    for a, q_val in enumerate(q_vals):
                        dx, dy = 0, 0
                        if a == 0:  # Up
                            dx, dy = 0, -0.2
                            va, ha = 'bottom', 'center'
                        elif a == 1:  # Right
                            dx, dy = 0.2, 0
                            va, ha = 'center', 'left'
                        elif a == 2:  # Down
                            dx, dy = 0, 0.2
                            va, ha = 'top', 'center'
                        elif a == 3:  # Left
                            dx, dy = -0.2, 0
                            va, ha = 'center', 'right'
                        
                        ax.text(y + 0.5 + dy, x + 0.5 + dx, f"{q_val:.1f}", fontsize=9,
                                ha=ha, va=va, color='blue')
                
                if policy is not None:
                    # Draw the policy direction
                    action = policy.get(state, 0)
                    dx, dy = 0, 0
                    if action == 0:  # Up
                        dx, dy = 0, -0.35
                    elif action == 1:  # Right
                        dx, dy = 0.35, 0
                    elif action == 2:  # Down
                        dx, dy = 0, 0.35
                    elif action == 3:  # Left
                        dx, dy = -0.35, 0
                    
                    ax.arrow(y + 0.5, x + 0.5, dy, dx, head_width=0.15, head_length=0.15, fc='black', ec='black')
                
                if state == (0, 0):
                    ax.text(y + 0.5, x + 0.5, "Start", fontsize=12,
                            ha='center', va='center', color='purple')
        
        # Set the title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xticks(np.arange(0.5, self.width, 1))
        ax.set_yticks(np.arange(0.5, self.height, 1))
        ax.set_xticklabels(np.arange(self.width))
        ax.set_yticklabels(np.arange(self.height))
        ax.set_xlabel("Y-coordinate", fontsize=12)
        ax.set_ylabel("X-coordinate", fontsize=12)
        
        # Display a legend
        obstacle_patch = Rectangle((0, 0), 1, 1, facecolor='darkgrey', label='Obstacle')
        goal_patch = Rectangle((0, 0), 1, 1, facecolor='green', label='Goal')
        hole_patch = Rectangle((0, 0), 1, 1, facecolor='red', label='Hole')
        start_patch = Rectangle((0, 0), 1, 1, facecolor='white', label='Start')
        
        ax.legend(handles=[start_patch, obstacle_patch, goal_patch, hole_patch],
                  loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
        plt.tight_layout()
        return fig, ax

# Create and display the grid world
env = GridWorld(width=5, height=5, deterministic=True)
fig, ax = env.render(title="Grid World Environment")
# Save the figure
file_path = os.path.join(save_dir, "grid_world_environment.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Define the key components of the reinforcement learning problem
print_step_header(3, "Defining RL Components for the Game")

# Create a figure to show the RL components
plt.figure(figsize=(12, 8))

# Define components
components = {
    "Agent": "The learning entity (player) that makes decisions and takes actions",
    "Environment": "The game world (grid world) with rules, obstacles, and rewards",
    "State": "Current position in the grid (x, y coordinates)",
    "Actions": "Up, Right, Down, Left movements",
    "Rewards": "+10 for reaching goal, -10 for falling in hole, -0.1 for each step",
    "Policy": "The strategy that determines which action to take in each state"
}

# Draw the components and definitions
for i, (component, definition) in enumerate(components.items()):
    plt.text(0.1, 0.9 - i * 0.15, f"{component}:", fontsize=14, fontweight='bold')
    plt.text(0.3, 0.9 - i * 0.15, definition, fontsize=12)

plt.axis('off')
plt.title("Key Components of the Reinforcement Learning Problem", fontsize=16)
# Save the figure
file_path = os.path.join(save_dir, "rl_components.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: The Exploration-Exploitation Tradeoff
print_step_header(4, "Exploration-Exploitation Tradeoff")

print("The exploration-exploitation tradeoff is a fundamental challenge in reinforcement learning:")
print("- Exploration: Trying new actions to discover potentially better strategies")
print("- Exploitation: Using known good strategies to maximize reward")
print()
print("Balancing these aspects is crucial for effective learning:")
print("- Too much exploration: Wastes time on suboptimal strategies")
print("- Too much exploitation: May miss better strategies")
print()

# Create a figure to illustrate the exploration-exploitation tradeoff
plt.figure(figsize=(12, 8))

# Define epsilon values and their performance over time
episodes = np.arange(1, 1001)
high_epsilon = 0.9 * np.power(0.99, episodes)  # High exploration
balanced_epsilon = 0.9 * np.power(0.995, episodes)  # Balanced approach
low_epsilon = 0.1 * np.ones_like(episodes)  # High exploitation

# Performance curves (stylized)
np.random.seed(42)
perf_noise = np.random.normal(0, 5, len(episodes))
high_exploration_perf = 80 - 75 * np.exp(-episodes / 400) + perf_noise * 0.3
balanced_perf = 95 - 90 * np.exp(-episodes / 200) + perf_noise * 0.2
high_exploitation_perf = 60 - 20 * np.exp(-episodes / 50) + perf_noise * 0.1

# Plot the performance curves
plt.subplot(2, 1, 1)
plt.plot(episodes, high_epsilon, 'r-', label='High Exploration (ε starts high, decays slowly)')
plt.plot(episodes, balanced_epsilon, 'g-', label='Balanced Approach (ε decays moderately)')
plt.plot(episodes, low_epsilon, 'b-', label='High Exploitation (ε stays low)')
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Epsilon (ε) Value', fontsize=12)
plt.title('Epsilon Decay Strategies', fontsize=14)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(episodes, high_exploration_perf, 'r-', label='High Exploration Performance')
plt.plot(episodes, balanced_perf, 'g-', label='Balanced Approach Performance')
plt.plot(episodes, high_exploitation_perf, 'b-', label='High Exploitation Performance')
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('Performance vs. Episodes', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Exploration-Exploitation Tradeoff', fontsize=16, y=1.02)
# Save the figure
file_path = os.path.join(save_dir, "exploration_exploitation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Q-learning Implementation
print_step_header(5, "Q-learning Implementation")

def q_learning(env, num_episodes=500, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, decay_rate=0.99):
    """
    Implement Q-learning algorithm for the grid world environment.
    
    Args:
        env: The environment
        num_episodes: Number of episodes to run
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Exploration rate
        decay_rate: Rate at which epsilon decays
    
    Returns:
        Q-table, rewards per episode
    """
    # Initialize Q-table with zeros
    q_table = defaultdict(lambda: np.zeros(len(env.actions)))
    
    # Track rewards per episode for plotting
    rewards_per_episode = []
    
    # Track epsilon values
    epsilon_values = []
    
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Collect current epsilon
        epsilon_values.append(epsilon)
        
        while not done:
            # Choose action using epsilon-greedy policy
            if np.random.random() < epsilon:
                # Explore: choose a random action
                action = np.random.choice(env.actions)
            else:
                # Exploit: choose the best action
                action = np.argmax(q_table[state])
            
            # Take the action and observe the new state and reward
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update the Q-value
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            
            # Q-learning update rule
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state][action] = new_value
            
            # Move to the next state
            state = next_state
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * decay_rate)
        
        # Record the total reward for this episode
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {np.mean(rewards_per_episode[-100:]):.2f}")
    
    return q_table, rewards_per_episode, epsilon_values

# Run Q-learning for the deterministic environment
print("Running Q-learning for a deterministic environment...")
deterministic_env = GridWorld(deterministic=True)
q_table_det, rewards_det, epsilon_det = q_learning(deterministic_env, num_episodes=500, epsilon=0.9, decay_rate=0.99)

# Extract the policy from the Q-table
policy_det = {state: np.argmax(q_table_det[state]) for state in deterministic_env.states}

# Visualize the learned policy
fig, ax = deterministic_env.render(q_values=q_table_det, policy=policy_det, 
                               title="Q-learning Policy in Deterministic Environment")
# Save the figure
file_path = os.path.join(save_dir, "q_learning_deterministic.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Run Q-learning for the stochastic environment
print("\nRunning Q-learning for a stochastic environment...")
stochastic_env = GridWorld(deterministic=False)
q_table_stoch, rewards_stoch, epsilon_stoch = q_learning(stochastic_env, num_episodes=500, epsilon=0.9, decay_rate=0.99)

# Extract the policy from the Q-table
policy_stoch = {state: np.argmax(q_table_stoch[state]) for state in stochastic_env.states}

# Visualize the learned policy
fig, ax = stochastic_env.render(q_values=q_table_stoch, policy=policy_stoch, 
                           title="Q-learning Policy in Stochastic Environment")
# Save the figure
file_path = os.path.join(save_dir, "q_learning_stochastic.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards_det, label='Deterministic Environment')
plt.plot(rewards_stoch, label='Stochastic Environment')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Q-learning Performance: Deterministic vs. Stochastic Environment', fontsize=14)
plt.legend()
plt.grid(True)
# Save the figure
file_path = os.path.join(save_dir, "q_learning_rewards.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Q-learning vs. Policy Gradient Methods
print_step_header(6, "Q-learning vs. Policy Gradient Methods")

# Create a figure to compare Q-learning and Policy Gradient methods
plt.figure(figsize=(12, 10))

# Comparison table structure
comparison = [
    ["Aspect", "Q-learning", "Policy Gradient Methods"],
    ["Approach", "Value-based", "Policy-based"],
    ["What is learned", "Q-values for state-action pairs", "Policy function mapping states to actions"],
    ["Exploration", "Typically epsilon-greedy", "Stochastic policy"],
    ["Memory usage", "Stores Q-values for each state-action pair", "Stores policy parameters"],
    ["Handling continuous actions", "Difficult", "Natural"],
    ["Convergence", "Guaranteed in tabular case with sufficient exploration", "Local optimum"],
    ["Sample efficiency", "More efficient", "Less efficient, but improves with actor-critic"],
    ["Example algorithms", "Q-learning, DQN, Double DQN", "REINFORCE, A2C, PPO, TRPO"]
]

# Draw the comparison table
plt.axis('off')
table = plt.table(cellText=comparison, loc='center', cellLoc='center', colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_text_props(fontproperties=dict(weight='bold'))
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    elif key[1] == 0:  # First column
        cell.set_text_props(fontproperties=dict(weight='bold'))
        cell.set_facecolor('#D9E1F2')
    elif key[0] % 2 == 0:  # Even rows
        cell.set_facecolor('#E9EDF4')

plt.suptitle('Comparison of Q-learning and Policy Gradient Methods', fontsize=16, y=0.98)
# Save the figure
file_path = os.path.join(save_dir, "q_learning_vs_policy_gradient.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Deterministic vs. Stochastic Environments
print_step_header(7, "Deterministic vs. Stochastic Environments")

# Visual comparison of deterministic and stochastic environments
plt.figure(figsize=(14, 10))

# Deterministic environment
plt.subplot(2, 2, 1)
plt.imshow(np.zeros((5, 5)), cmap='binary', alpha=0.1)
plt.title('Deterministic Environment', fontsize=14)
plt.text(2, 2, "Actions always have\nthe same outcome", ha='center', fontsize=10)
plt.axis('off')

# Stochastic environment
plt.subplot(2, 2, 2)
plt.imshow(np.random.rand(5, 5), cmap='binary', alpha=0.3)
plt.title('Stochastic Environment', fontsize=14)
plt.text(2, 2, "Actions have random\noutcomes sometimes", ha='center', fontsize=10)
plt.axis('off')

# Learning curves
episodes = np.arange(1, 501)
plt.subplot(2, 1, 2)
plt.plot(episodes, np.convolve(rewards_det, np.ones(20)/20, mode='same'), 'b-', 
         label='Deterministic Environment (Smoothed)')
plt.plot(episodes, np.convolve(rewards_stoch, np.ones(20)/20, mode='same'), 'r-', 
         label='Stochastic Environment (Smoothed)')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Reward (Moving Average)', fontsize=12)
plt.title('Learning Performance Comparison', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle('Deterministic vs. Stochastic Environments', fontsize=16, y=1.02)
# Save the figure
file_path = os.path.join(save_dir, "deterministic_vs_stochastic.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Exploration and Convergence Analysis
print_step_header(8, "Exploration and Convergence Analysis")

# Plot epsilon decay and convergence
plt.figure(figsize=(12, 10))

# Epsilon decay
plt.subplot(3, 1, 1)
plt.plot(epsilon_det, 'b-')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Epsilon Value', fontsize=12)
plt.title('Epsilon Decay Over Time', fontsize=14)
plt.grid(True)

# Q-value convergence
# Track one specific state-action pair's Q-value
state_to_track = (0, 1)  # Coordinate (0,1)
action_to_track = 1  # Right

# Run another Q-learning trial and track Q-values for the specific state-action pair
env = GridWorld(deterministic=True)
q_values_history = []
q_table = defaultdict(lambda: np.zeros(len(env.actions)))
epsilon = 0.9
decay_rate = 0.99
learning_rate = 0.1
discount_factor = 0.99

for episode in range(500):
    state = env.reset()
    done = False
    
    while not done:
        if np.random.random() < epsilon:
            action = np.random.choice(env.actions)
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done = env.step(action)
        
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state][action] = new_value
        
        # Track Q-value for the specific state-action pair
        if state == state_to_track and action == action_to_track:
            q_values_history.append(new_value)
        
        state = next_state
    
    # Decay epsilon
    epsilon = max(0.01, epsilon * decay_rate)
    
    # If we haven't updated our tracked state-action pair this episode, record the last value
    if len(q_values_history) <= episode:
        if len(q_values_history) > 0:
            q_values_history.append(q_values_history[-1])
        else:
            q_values_history.append(0)

# Plot Q-value convergence
plt.subplot(3, 1, 2)
plt.plot(q_values_history, 'g-')
plt.xlabel('Episode', fontsize=12)
plt.ylabel(f'Q-value for State {state_to_track}, Action {env.action_names[action_to_track]}', fontsize=12)
plt.title('Q-value Convergence for a Specific State-Action Pair', fontsize=14)
plt.grid(True)

# Optimal policy visualization
plt.subplot(3, 1, 3)
plt.axis('off')
plt.text(0.5, 0.9, 'Characteristics of an Optimal Policy in Reinforcement Learning:', 
         ha='center', fontsize=12, fontweight='bold')

# List characteristics
characteristics = [
    "1. Maximizes expected cumulative reward",
    "2. Balances short-term and long-term rewards",
    "3. Adapts to environment dynamics",
    "4. In deterministic environments: Often deterministic actions",
    "5. In stochastic environments: May involve probabilistic decisions",
    "6. Convergence: Q-learning converges to the optimal policy with sufficient exploration"
]

for i, char in enumerate(characteristics):
    plt.text(0.1, 0.8 - i*0.1, char, fontsize=11)

plt.tight_layout()
plt.suptitle('Exploration and Convergence Analysis', fontsize=16, y=1.02)
# Save the figure
file_path = os.path.join(save_dir, "exploration_convergence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Summary and Key Points
print_step_header(9, "Summary and Key Points")

print("Summary of Key Points:")
print()
print("1. Reinforcement Learning Components in a Video Game:")
print("   - Agent: The player or AI controller")
print("   - Environment: The game world with its rules and mechanics")
print("   - State: Current game situation (position, health, inventory, etc.)")
print("   - Actions: Possible moves or commands (move, attack, use item, etc.)")
print("   - Rewards: Points, achievements, level completion, or penalties")
print("   - Policy: Strategy for choosing actions based on game state")
print()
print("2. Exploration-Exploitation Tradeoff:")
print("   - Exploration: Trying new strategies to potentially find better solutions")
print("   - Exploitation: Using known good strategies to maximize immediate reward")
print("   - Balance needed: Too much exploration wastes time, too much exploitation misses opportunities")
print("   - Implementation: Often through epsilon-greedy policy with decaying exploration")
print()
print("3. Q-learning vs. Policy Gradient Methods:")
print("   - Q-learning (Value-based): Learns state-action values, indirectly determines policy")
print("   - Policy Gradient (Policy-based): Directly learns policy function mapping states to actions")
print("   - Usage contexts: Q-learning for discrete action spaces, Policy Gradients for continuous")
print("   - Both have variants for handling complex state spaces (DQN, A2C, PPO, etc.)")
print()
print("4. Deterministic vs. Stochastic Environments:")
print("   - Deterministic: Same action in same state always produces same outcome")
print("   - Stochastic: Actions may have random outcomes")
print("   - Learning approach differences:")
print("     * Deterministic: Simpler learning, more predictable convergence")
print("     * Stochastic: Requires more exploration, robust policy, slower convergence")
print("     * Stochastic environments need algorithms that handle uncertainty well")
print()
print("Conclusion: Reinforcement learning provides powerful tools for agents to learn optimal behaviors")
print("in game environments by interacting and receiving feedback, with approaches that can be tailored")
print("to the specific characteristics of the game and the desired learning outcomes.") 