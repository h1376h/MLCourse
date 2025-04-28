import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Function to save figures
def save_figure(fig, filename):
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")

# Problem setup
print("\n===== Problem Definition =====")
print("A factory produces electronic components that can be in one of three states:")
print("- State 0: functioning perfectly")
print("- State 1: minor defect")
print("- State 2: major defect")
print("\nPossible actions:")
print("- Action a₀: ship the component")
print("- Action a₁: perform minor repairs")
print("- Action a₂: discard the component")
print("\nInitial probability distribution:")
print("P(θ = 0) = 0.7, P(θ = 1) = 0.2, and P(θ = 2) = 0.1")

# Define the loss function matrix
loss_matrix = np.array([
    [0, 10, 50],    # Perfect (state 0): Ship, Repair, Discard
    [30, 5, 40],    # Minor defect (state 1): Ship, Repair, Discard
    [100, 60, 20]   # Major defect (state 2): Ship, Repair, Discard
])

print("\nLoss function:")
actions = ["Ship (a₀)", "Repair (a₁)", "Discard (a₂)"]
states = ["Perfect (θ = 0)", "Minor defect (θ = 1)", "Major defect (θ = 2)"]
print("┌─────────────────┬───────────┬────────────┬─────────────┐")
print("│      Loss       │ Ship (a₀) │ Repair (a₁)│ Discard (a₂)│")
print("├─────────────────┼───────────┼────────────┼─────────────┤")
for i, state in enumerate(states):
    print(f"│ {state:<15} │ {loss_matrix[i,0]:<9} │ {loss_matrix[i,1]:<10} │ {loss_matrix[i,2]:<11} │")
print("└─────────────────┴───────────┴────────────┴─────────────┘")

# Task 1: Calculate expected loss for each action with initial probabilities
print("\n===== Task 1: Calculate expected loss for each action =====")
initial_probs = np.array([0.7, 0.2, 0.1])

expected_losses = np.zeros(3)
for action in range(3):
    expected_losses[action] = np.sum(initial_probs * loss_matrix[:, action])
    print(f"\nExpected loss for {actions[action]}:")
    for state in range(3):
        print(f"  P(θ={state}) × Loss(θ={state}, a_{action}) = {initial_probs[state]:.1f} × {loss_matrix[state, action]} = {initial_probs[state] * loss_matrix[state, action]:.1f}")
    print(f"  Total expected loss = {expected_losses[action]:.1f}")

# Task 2: Determine which action minimizes Bayes risk
print("\n===== Task 2: Determine optimal action =====")
optimal_action = np.argmin(expected_losses)
print(f"Expected losses: Ship = {expected_losses[0]:.1f}, Repair = {expected_losses[1]:.1f}, Discard = {expected_losses[2]:.1f}")
print(f"The action that minimizes the Bayes risk is: {actions[optimal_action]}")

# Visualize expected losses for initial probabilities
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(actions, expected_losses, color=['blue', 'green', 'red'])
bars[optimal_action].set_color('gold')
ax.set_ylabel('Expected Loss (Bayes Risk)')
ax.set_title('Expected Loss for Each Action with Initial Probabilities')
for i, v in enumerate(expected_losses):
    ax.text(i, v + 0.5, f"{v:.1f}", ha='center')
plt.tight_layout()
save_figure(fig, "initial_expected_losses.png")
plt.close(fig)

# Task 3: Change in decision with new probabilities
print("\n===== Task 3: Decision change with new probabilities =====")
new_probs = np.array([0.5, 0.3, 0.2])
print(f"New probability distribution: P(θ = 0) = {new_probs[0]}, P(θ = 1) = {new_probs[1]}, and P(θ = 2) = {new_probs[2]}")

new_expected_losses = np.zeros(3)
for action in range(3):
    new_expected_losses[action] = np.sum(new_probs * loss_matrix[:, action])
    print(f"\nExpected loss for {actions[action]} with new probabilities:")
    for state in range(3):
        print(f"  P(θ={state}) × Loss(θ={state}, a_{action}) = {new_probs[state]:.1f} × {loss_matrix[state, action]} = {new_probs[state] * loss_matrix[state, action]:.1f}")
    print(f"  Total expected loss = {new_expected_losses[action]:.1f}")

new_optimal_action = np.argmin(new_expected_losses)
print(f"\nNew expected losses: Ship = {new_expected_losses[0]:.1f}, Repair = {new_expected_losses[1]:.1f}, Discard = {new_expected_losses[2]:.1f}")
print(f"The new optimal action is: {actions[new_optimal_action]}")

# Visualize expected losses for new probabilities
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(actions, new_expected_losses, color=['blue', 'green', 'red'])
bars[new_optimal_action].set_color('gold')
ax.set_ylabel('Expected Loss (Bayes Risk)')
ax.set_title('Expected Loss for Each Action with New Probabilities')
for i, v in enumerate(new_expected_losses):
    ax.text(i, v + 0.5, f"{v:.1f}", ha='center')
plt.tight_layout()
save_figure(fig, "new_expected_losses.png")
plt.close(fig)

# Comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(actions))
width = 0.35
ax.bar(x - width/2, expected_losses, width, label='Initial Probabilities', color='skyblue')
ax.bar(x + width/2, new_expected_losses, width, label='New Probabilities', color='salmon')
ax.set_ylabel('Expected Loss (Bayes Risk)')
ax.set_title('Comparison of Expected Losses')
ax.set_xticks(x)
ax.set_xticklabels(actions)
ax.legend()
for i, v in enumerate(expected_losses):
    ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha='center')
for i, v in enumerate(new_expected_losses):
    ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha='center')
plt.tight_layout()
save_figure(fig, "comparison_expected_losses.png")
plt.close(fig)

# Task 4: Range of P(θ=0) for which shipping is optimal
print("\n===== Task 4: Find the range of P(θ=0) for which shipping is optimal =====")
print("Assuming P(θ = 1) = P(θ = 2) = (1-P(θ=0))/2")

p0_values = np.linspace(0, 1, 1000)
expected_loss_ship = np.zeros_like(p0_values)
expected_loss_repair = np.zeros_like(p0_values)
expected_loss_discard = np.zeros_like(p0_values)

for i, p0 in enumerate(p0_values):
    p1 = (1 - p0) / 2
    p2 = (1 - p0) / 2
    probs = np.array([p0, p1, p2])
    
    expected_loss_ship[i] = np.sum(probs * loss_matrix[:, 0])
    expected_loss_repair[i] = np.sum(probs * loss_matrix[:, 1])
    expected_loss_discard[i] = np.sum(probs * loss_matrix[:, 2])

# Find the range where shipping is optimal
ship_optimal = np.logical_and(
    expected_loss_ship < expected_loss_repair,
    expected_loss_ship < expected_loss_discard
)

ship_optimal_p0_min = p0_values[ship_optimal][0] if np.any(ship_optimal) else None
ship_optimal_p0_max = p0_values[ship_optimal][-1] if np.any(ship_optimal) else None

print(f"\nShipping is the optimal decision when P(θ=0) is in the range [{ship_optimal_p0_min:.4f}, {ship_optimal_p0_max:.4f}]")

# Calculate the exact thresholds algebraically
print("\nAlgebraic derivation of the thresholds:")
print("For shipping to be better than repair:")
print("  E[Loss(Ship)] < E[Loss(Repair)]")
print("  p0 × 0 + p1 × 30 + p2 × 100 < p0 × 10 + p1 × 5 + p2 × 60")
print("  p1 × 30 + p2 × 100 < p0 × 10 + p1 × 5 + p2 × 60")
print("  p1 × 30 - p1 × 5 + p2 × 100 - p2 × 60 < p0 × 10")
print("  p1 × 25 + p2 × 40 < p0 × 10")
print("Since p1 = p2 = (1-p0)/2:")
print("  (1-p0)/2 × 25 + (1-p0)/2 × 40 < p0 × 10")
print("  (1-p0) × (25 + 40)/2 < p0 × 10")
print("  (1-p0) × 32.5 < p0 × 10")
print("  32.5 - 32.5p0 < 10p0")
print("  32.5 < 10p0 + 32.5p0")
print("  32.5 < 42.5p0")
print(f"  p0 > {32.5/42.5:.4f}")

print("\nFor shipping to be better than discarding:")
print("  E[Loss(Ship)] < E[Loss(Discard)]")
print("  p0 × 0 + p1 × 30 + p2 × 100 < p0 × 50 + p1 × 40 + p2 × 20")
print("  p1 × 30 + p2 × 100 < p0 × 50 + p1 × 40 + p2 × 20")
print("  p1 × 30 - p1 × 40 + p2 × 100 - p2 × 20 < p0 × 50")
print("  p1 × (-10) + p2 × 80 < p0 × 50")
print("Since p1 = p2 = (1-p0)/2:")
print("  (1-p0)/2 × (-10) + (1-p0)/2 × 80 < p0 × 50")
print("  (1-p0) × (80 - 10)/2 < p0 × 50")
print("  (1-p0) × 35 < p0 × 50")
print("  35 - 35p0 < 50p0")
print("  35 < 50p0 + 35p0")
print("  35 < 85p0")
print(f"  p0 > {35/85:.4f}")

# Determine the final range
ship_repair_threshold = 32.5/42.5
ship_discard_threshold = 35/85
ship_optimal_p0_min_algebraic = max(ship_repair_threshold, ship_discard_threshold)

print(f"\nBased on algebraic calculations:")
print(f"  For shipping to be better than repair: p0 > {ship_repair_threshold:.4f}")
print(f"  For shipping to be better than discarding: p0 > {ship_discard_threshold:.4f}")
print(f"  Therefore, shipping is optimal when p0 > {ship_optimal_p0_min_algebraic:.4f}")

# Plot the expected losses as a function of P(θ=0)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p0_values, expected_loss_ship, 'b-', label='Ship (a₀)', linewidth=2)
ax.plot(p0_values, expected_loss_repair, 'g-', label='Repair (a₁)', linewidth=2)
ax.plot(p0_values, expected_loss_discard, 'r-', label='Discard (a₂)', linewidth=2)

# Add vertical lines for the thresholds
ax.axvline(x=ship_optimal_p0_min, color='purple', linestyle='--', 
           label=f'Shipping optimal threshold: p0 > {ship_optimal_p0_min:.4f}')

# Highlight the region where shipping is optimal
ax.axvspan(ship_optimal_p0_min, 1, alpha=0.2, color='blue')

# Mark the original probability scenario and the new scenario
ax.plot(0.7, np.sum(np.array([0.7, 0.2, 0.1]) * loss_matrix[:, 0]), 'bo', markersize=8, label='Initial scenario (p0=0.7)')
ax.plot(0.5, np.sum(np.array([0.5, 0.3, 0.2]) * loss_matrix[:, 1]), 'go', markersize=8, label='New scenario (p0=0.5)')

ax.set_xlim(0, 1)
ax.set_ylim(0, 70)
ax.set_xlabel('P(θ=0) - Probability of Perfect Component', fontsize=12)
ax.set_ylabel('Expected Loss (Bayes Risk)', fontsize=12)
ax.set_title('Expected Loss vs. Probability of Perfect Component', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
save_figure(fig, "expected_loss_vs_p0.png")
plt.close(fig)

# Create a decision region plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate optimal action for each p0 value
optimal_actions = np.zeros_like(p0_values, dtype=int)
for i, p0 in enumerate(p0_values):
    p1 = (1 - p0) / 2
    p2 = (1 - p0) / 2
    probs = np.array([p0, p1, p2])
    
    action_losses = np.array([
        np.sum(probs * loss_matrix[:, 0]),  # Ship
        np.sum(probs * loss_matrix[:, 1]),  # Repair
        np.sum(probs * loss_matrix[:, 2])   # Discard
    ])
    
    optimal_actions[i] = np.argmin(action_losses)

# Create a horizontal bar to show decision regions
y_base = np.zeros_like(p0_values)
y_height = np.ones_like(p0_values)

# Find the transition points
transitions = []
for i in range(1, len(optimal_actions)):
    if optimal_actions[i] != optimal_actions[i-1]:
        transitions.append((p0_values[i], optimal_actions[i-1], optimal_actions[i]))

# Plot regions
ax.fill_between(p0_values, y_base, y_height, 
                where=(optimal_actions == 0), 
                color='blue', alpha=0.5, label='Ship ($a_0$)')
ax.fill_between(p0_values, y_base, y_height, 
                where=(optimal_actions == 1), 
                color='green', alpha=0.5, label='Repair ($a_1$)')
ax.fill_between(p0_values, y_base, y_height, 
                where=(optimal_actions == 2), 
                color='red', alpha=0.5, label='Discard ($a_2$)')

# Add vertical lines for transitions
for p0, _, _ in transitions:
    ax.axvline(x=p0, color='black', linestyle='--')

# Add text for the transition values
for i, (p0, _, _) in enumerate(transitions):
    ax.text(p0, 1.05, f'{p0:.4f}', ha='center', rotation=90, fontsize=9)

# Mark the initial and new probability scenarios
ax.scatter([0.7], [0.5], s=100, color='blue', edgecolor='black', zorder=5, label='Initial: $p_0=0.7$')
ax.scatter([0.5], [0.5], s=100, color='green', edgecolor='black', zorder=5, label='New: $p_0=0.5$')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel('$P(\\theta=0)$ - Probability of Perfect Component', fontsize=12)
ax.set_title('Optimal Decision Regions', fontsize=14)

# Improve legend placement and formatting
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                  ncol=3, fontsize=10, frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_alpha(0.9)

plt.tight_layout()
save_figure(fig, "decision_regions.png")
plt.close(fig)

# Task 5: MAP estimation and its relationship to optimal decision under 0-1 loss
print("\n===== Task 5: MAP Estimation and Relation to 0-1 Loss =====")
print("Formulating MAP estimation approach with uniform prior over states")

# Calculate MAP estimate for initial probabilities
initial_map_state = np.argmax(initial_probs)
print(f"\nInitial probability distribution: P(θ = 0) = {initial_probs[0]}, P(θ = 1) = {initial_probs[1]}, P(θ = 2) = {initial_probs[2]}")
print(f"MAP estimate (state with highest probability): θ = {initial_map_state} ({states[initial_map_state].split(' ')[0]})")

# Calculate MAP estimate for new probabilities
new_map_state = np.argmax(new_probs)
print(f"\nNew probability distribution: P(θ = 0) = {new_probs[0]}, P(θ = 1) = {new_probs[1]}, P(θ = 2) = {new_probs[2]}")
print(f"MAP estimate (state with highest probability): θ = {new_map_state} ({states[new_map_state].split(' ')[0]})")

# Define 0-1 loss for component states
zero_one_loss = np.identity(3) 
zero_one_loss = 1 - zero_one_loss  # Convert to 0-1 loss (0 on diagonal, 1 elsewhere)

print("\n0-1 Loss matrix for state estimation:")
print("┌─────────────────┬──────────┬──────────┬──────────┐")
print("│    State Loss   │ Est θ=0  │ Est θ=1  │ Est θ=2  │")
print("├─────────────────┼──────────┼──────────┼──────────┤")
for i, state in enumerate(states):
    print(f"│ True {state:<11} │ {zero_one_loss[i,0]:<8} │ {zero_one_loss[i,1]:<8} │ {zero_one_loss[i,2]:<8} │")
print("└─────────────────┴──────────┴──────────┴──────────┘")

# Calculate expected 0-1 loss for each state estimate with initial probabilities
initial_expected_01_losses = np.zeros(3)
for state_est in range(3):
    initial_expected_01_losses[state_est] = np.sum(initial_probs * zero_one_loss[:, state_est])
    print(f"\nExpected 0-1 loss for estimating state {state_est} (initial probabilities):")
    for state_true in range(3):
        print(f"  P(θ={state_true}) × Loss(θ={state_true}, est={state_est}) = {initial_probs[state_true]:.2f} × {zero_one_loss[state_true, state_est]} = {initial_probs[state_true] * zero_one_loss[state_true, state_est]:.2f}")
    print(f"  Total expected 0-1 loss = {initial_expected_01_losses[state_est]:.2f}")

optimal_state_est_initial = np.argmin(initial_expected_01_losses)
print(f"\nState estimate that minimizes expected 0-1 loss (initial): θ = {optimal_state_est_initial}")
print(f"This matches the MAP estimate: {optimal_state_est_initial == initial_map_state}")

# Calculate expected 0-1 loss for each state estimate with new probabilities
new_expected_01_losses = np.zeros(3)
for state_est in range(3):
    new_expected_01_losses[state_est] = np.sum(new_probs * zero_one_loss[:, state_est])

optimal_state_est_new = np.argmin(new_expected_01_losses)
print(f"\nState estimate that minimizes expected 0-1 loss (new): θ = {optimal_state_est_new}")
print(f"This matches the MAP estimate: {optimal_state_est_new == new_map_state}")

# Visualize comparison of MAP estimation and optimal 0-1 loss decision
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot for initial probabilities
bar_width = 0.35
x = np.arange(3)
axs[0].bar(x, initial_probs, bar_width, label='Initial Probabilities', color='skyblue')
axs[0].bar(x + bar_width, initial_expected_01_losses, bar_width, label='Expected 0-1 Loss', color='salmon')
axs[0].set_xticks(x + bar_width/2)
axs[0].set_xticklabels([f'State {i}' for i in range(3)])
axs[0].set_ylabel('Probability / Expected Loss')
axs[0].set_title('MAP Estimation vs. 0-1 Loss: Initial Probabilities')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# Highlight the MAP estimate and optimal 0-1 loss decision
axs[0].plot(initial_map_state, initial_probs[initial_map_state], 'bo', markersize=10)
axs[0].plot(optimal_state_est_initial + bar_width, initial_expected_01_losses[optimal_state_est_initial], 'ro', markersize=10)

# Annotate
axs[0].annotate('MAP Estimate', xy=(initial_map_state, initial_probs[initial_map_state]), 
                xytext=(initial_map_state-0.5, initial_probs[initial_map_state]+0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
axs[0].annotate('Optimal 0-1', xy=(optimal_state_est_initial + bar_width, initial_expected_01_losses[optimal_state_est_initial]), 
                xytext=(optimal_state_est_initial + bar_width+0.3, initial_expected_01_losses[optimal_state_est_initial]+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Plot for new probabilities
axs[1].bar(x, new_probs, bar_width, label='New Probabilities', color='skyblue')
axs[1].bar(x + bar_width, new_expected_01_losses, bar_width, label='Expected 0-1 Loss', color='salmon')
axs[1].set_xticks(x + bar_width/2)
axs[1].set_xticklabels([f'State {i}' for i in range(3)])
axs[1].set_ylabel('Probability / Expected Loss')
axs[1].set_title('MAP Estimation vs. 0-1 Loss: New Probabilities')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

# Highlight the MAP estimate and optimal 0-1 loss decision
axs[1].plot(new_map_state, new_probs[new_map_state], 'bo', markersize=10)
axs[1].plot(optimal_state_est_new + bar_width, new_expected_01_losses[optimal_state_est_new], 'ro', markersize=10)

# Annotate
axs[1].annotate('MAP Estimate', xy=(new_map_state, new_probs[new_map_state]), 
                xytext=(new_map_state-0.5, new_probs[new_map_state]+0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
axs[1].annotate('Optimal 0-1', xy=(optimal_state_est_new + bar_width, new_expected_01_losses[optimal_state_est_new]), 
                xytext=(optimal_state_est_new + bar_width+0.3, new_expected_01_losses[optimal_state_est_new]+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
save_figure(fig, "map_vs_01_loss.png")
plt.close(fig)

# Create a visualization showing the relationship between MAP and optimal decisions
# Varying probability of perfect component (p0) and keeping p1=p2=(1-p0)/2
p0_range = np.linspace(0, 1, 1000)
map_estimates = np.zeros_like(p0_range, dtype=int)

for i, p0 in enumerate(p0_range):
    p1 = (1 - p0) / 2
    p2 = (1 - p0) / 2
    probs = np.array([p0, p1, p2])
    map_estimates[i] = np.argmax(probs)

# Create a more readable visualization for MAP vs optimal actions
fig, ax = plt.subplots(figsize=(10, 5))

# Create a cleaner, more organized layout
# First, add a title and axis labels
ax.set_title('MAP Estimates vs. Optimal Actions by Probability', fontsize=14, pad=20)
ax.set_xlabel('$P(\\theta=0)$ - Probability of Perfect Component', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_yticks([])

# Plot the MAP estimate regions with cleaner labels
for state in range(3):
    mask = (map_estimates == state)
    if np.any(mask):
        p0_start = p0_range[mask][0] if mask[0] else p0_range[np.where(~mask[:-1] & mask[1:])[0][0]+1]
        p0_end = p0_range[mask][-1] if mask[-1] else p0_range[np.where(mask[:-1] & ~mask[1:])[0][0]]
        ax.axvspan(p0_start, p0_end, alpha=0.3, color=['blue', 'green', 'red'][state],
                   label=f'MAP = State {state}')

# Add labels for MAP regions at the top of the plot
ax.text(0.17, 0.92, "MAP: State 1 or 2", fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.67, 0.92, "MAP: State 0", fontsize=10, ha='center', transform=ax.transAxes)

# Add vertical lines for the MAP transition points with clearer labels
map_transitions = []
for i in range(1, len(map_estimates)):
    if map_estimates[i] != map_estimates[i-1]:
        transition_p0 = p0_range[i]
        map_transitions.append(transition_p0)
        ax.axvline(x=transition_p0, color='black', linestyle='-.', linewidth=1.5, zorder=3)
        # Add the transition value label at the bottom
        ax.text(transition_p0, 0.05, f'MAP threshold:\n$p_0 = {transition_p0:.4f}$', 
                ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=3))

# Add optimal decision boundaries from previous task with distinct formatting
optimal_action_labels = ["Ship", "Repair", "Discard"]
for i, (p0, prev_action, next_action) in enumerate(transitions):
    ax.axvline(x=p0, color='purple', linestyle='--', linewidth=1.5, zorder=3)
    # Add transition label with a box for better readability
    ax.text(p0, 0.25, f'Action threshold:\n{optimal_action_labels[prev_action]} → {optimal_action_labels[next_action]}\n$p_0 = {p0:.4f}$', 
            ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=3))

# Add labels for optimal action regions at the center of the plot
for i, region in enumerate([(0, transitions[0][0]), 
                           (transitions[0][0], transitions[1][0]),
                           (transitions[1][0], 1)]):
    center = (region[0] + region[1]) / 2
    ax.text(center, 0.75, f"Optimal: {optimal_action_labels[i]}", 
            fontsize=10, ha='center', color=['red', 'green', 'blue'][i],
            bbox=dict(facecolor='white', alpha=0.7, pad=3))

# Mark the specific probability scenarios with clearer labeling
scatter_points = ax.scatter([0.7, 0.5], [0.55, 0.55], s=100, 
                           color=['blue', 'green'], edgecolor='black', zorder=5)
ax.text(0.7, 0.6, 'Initial: $p_0=0.7$', ha='center', fontsize=10,
       bbox=dict(facecolor='white', alpha=0.7, pad=2))
ax.text(0.5, 0.6, 'New: $p_0=0.5$', ha='center', fontsize=10,
       bbox=dict(facecolor='white', alpha=0.7, pad=2))

# Add a simple legend for the transition lines
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', linestyle='-.', lw=1.5),
                Line2D([0], [0], color='purple', linestyle='--', lw=1.5)]
ax.legend(custom_lines, ['MAP Threshold', 'Action Threshold'], 
         loc='lower right', frameon=True, fancybox=True)

plt.tight_layout()
save_figure(fig, "map_vs_optimal_actions.png")
plt.close(fig)

print("\n===== Summary =====")
print("1. Initial probabilities [0.7, 0.2, 0.1]:")
print(f"   Expected losses: Ship = {expected_losses[0]:.1f}, Repair = {expected_losses[1]:.1f}, Discard = {expected_losses[2]:.1f}")
print(f"   Optimal action: {actions[optimal_action]}")
print(f"   MAP estimate: State {initial_map_state} ({states[initial_map_state].split(' ')[0]})")
print("2. New probabilities [0.5, 0.3, 0.2]:")
print(f"   Expected losses: Ship = {new_expected_losses[0]:.1f}, Repair = {new_expected_losses[1]:.1f}, Discard = {new_expected_losses[2]:.1f}")
print(f"   Optimal action: {actions[new_optimal_action]}")
print(f"   MAP estimate: State {new_map_state} ({states[new_map_state].split(' ')[0]})")
print(f"3. Shipping is optimal when P(θ=0) > {ship_optimal_p0_min_algebraic:.4f}")
print(f"   (when P(θ=1) = P(θ=2) = (1-P(θ=0))/2)")
print("4. MAP estimate matches the state that minimizes expected 0-1 loss")
print(f"   MAP transitions occur at P(θ=0) = {map_transitions[0]:.4f}") 