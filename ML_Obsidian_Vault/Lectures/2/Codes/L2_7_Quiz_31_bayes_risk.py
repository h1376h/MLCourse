import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified location"""
    fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight", dpi=300)
    plt.close(fig)

def print_step_header(step_number, step_title):
    """Print a formatted header for each main step"""
    print(f"\n\n{'='*80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'='*80}\n")

def print_substep(substep_title):
    """Print a formatted header for each substep"""
    print(f"\n{'-'*50}")
    print(f"{substep_title}")
    print(f"{'-'*50}\n")

# Define the problem parameters
states = ["Benign (C₁)", "Early-stage (C₂)", "Advanced (C₃)"]
actions = ["No treatment (a₁)", "Mild treatment (a₂)", "Aggressive treatment (a₃)"]

# Define the loss matrix: [action][state]
loss_matrix = np.array([
    [0, 50, 100],    # a₁: No treatment
    [10, 5, 40],     # a₂: Mild treatment
    [20, 15, 10]     # a₃: Aggressive treatment
])

# Initial posterior probabilities: P(Cj|x)
initial_posteriors = np.array([0.7, 0.2, 0.1])  # [P(C₁|x), P(C₂|x), P(C₃|x)]

# Updated posterior probabilities for Task 3
updated_posteriors = np.array([0.5, 0.3, 0.2])  # [P(C₁|x), P(C₂|x), P(C₃|x)]

def calculate_expected_loss(loss_matrix, posteriors):
    """Calculate expected loss for each action"""
    return np.dot(loss_matrix, posteriors)

def visualize_loss_matrix(loss_matrix, states, actions, filename="loss_matrix.png"):
    """Create a heatmap visualization of the loss matrix"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(loss_matrix, cmap="YlOrRd")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Loss Value", rotation=-90, va="bottom")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(states)))
    ax.set_yticks(np.arange(len(actions)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(actions)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(actions)):
        for j in range(len(states)):
            text = ax.text(j, i, loss_matrix[i, j],
                          ha="center", va="center", color="black" if loss_matrix[i, j] < 50 else "white")
    
    ax.set_title("Loss Matrix for Oncology Treatment Decisions")
    fig.tight_layout()
    save_figure(fig, filename)
    return fig

def visualize_expected_loss(expected_losses, actions, title, filename="expected_loss.png"):
    """Create a bar chart of expected losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(actions, expected_losses, color='skyblue')
    
    # Add value labels on top of each bar
    for bar, value in zip(bars, expected_losses):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 1, 
                f'{value:.2f}', 
                ha='center', va='bottom')
    
    # Highlight the minimum expected loss
    min_idx = np.argmin(expected_losses)
    bars[min_idx].set_color('green')
    
    ax.set_ylabel('Expected Loss (Bayes Risk)')
    ax.set_title(title)
    ax.set_ylim(0, max(expected_losses) * 1.2)  # Add some space for labels
    
    # Add text note about the optimal decision
    ax.text(0.5, 0.95, f"Optimal decision: {actions[min_idx]}", 
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7))
    
    fig.tight_layout()
    save_figure(fig, filename)
    return fig

def visualize_decision_boundaries(loss_matrix, states, actions, filename="decision_boundaries.png"):
    """Visualize decision boundaries as function of P(C₁|x)"""
    # Create a range of P(C₁|x) values
    p_c1 = np.linspace(0, 1, 1000)
    
    # Calculate P(C₂|x) and P(C₃|x) assuming they are equal: (1-P(C₁|x))/2
    p_c2 = (1 - p_c1) / 2
    p_c3 = (1 - p_c1) / 2
    
    # Calculate expected loss for each action across all P(C₁|x) values
    expected_loss_a1 = loss_matrix[0, 0] * p_c1 + loss_matrix[0, 1] * p_c2 + loss_matrix[0, 2] * p_c3
    expected_loss_a2 = loss_matrix[1, 0] * p_c1 + loss_matrix[1, 1] * p_c2 + loss_matrix[1, 2] * p_c3
    expected_loss_a3 = loss_matrix[2, 0] * p_c1 + loss_matrix[2, 1] * p_c2 + loss_matrix[2, 2] * p_c3
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(p_c1, expected_loss_a1, label=actions[0], linewidth=2)
    ax.plot(p_c1, expected_loss_a2, label=actions[1], linewidth=2)
    ax.plot(p_c1, expected_loss_a3, label=actions[2], linewidth=2)
    
    # Find the optimal action for each P(C₁|x)
    optimal_action = np.argmin([expected_loss_a1, expected_loss_a2, expected_loss_a3], axis=0)
    
    # Find the decision boundaries (where the expected loss curves intersect)
    decision_boundaries = []
    for i in range(1, len(p_c1)):
        if optimal_action[i] != optimal_action[i-1]:
            # Linear interpolation to find more precise boundary
            p_boundary = (p_c1[i-1] + p_c1[i]) / 2
            decision_boundaries.append(p_boundary)
    
    # Shade regions with different optimal actions
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for i in range(len(decision_boundaries) + 1):
        start = 0 if i == 0 else decision_boundaries[i-1]
        end = 1 if i == len(decision_boundaries) else decision_boundaries[i]
        
        idx_range = np.where((p_c1 >= start) & (p_c1 <= end))[0]
        if len(idx_range) > 0:
            action_idx = optimal_action[idx_range[0]]
            ax.fill_between(p_c1, 0, np.maximum.reduce([expected_loss_a1, expected_loss_a2, expected_loss_a3]), 
                           where=(p_c1 >= start) & (p_c1 <= end),
                           color=colors[action_idx], alpha=0.3)
            # Add text label for region
            mid_point = (start + end) / 2
            y_pos = np.min([expected_loss_a1[np.abs(p_c1 - mid_point).argmin()], 
                          expected_loss_a2[np.abs(p_c1 - mid_point).argmin()],
                          expected_loss_a3[np.abs(p_c1 - mid_point).argmin()]]) / 2
            ax.text(mid_point, y_pos, f"Optimal: {actions[action_idx]}", 
                   ha='center', va='center', fontweight='bold')
    
    # Mark the decision boundaries with vertical lines
    for boundary in decision_boundaries:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.7)
        # Add text for the boundary value
        ax.text(boundary, ax.get_ylim()[1]*0.95, f"{boundary:.3f}", ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
    
    # Formatting
    ax.set_xlabel("P(C₁|x) - Probability of Benign")
    ax.set_ylabel("Expected Loss (Bayes Risk)")
    ax.set_title("Expected Loss vs. P(C₁|x) with Decision Boundaries")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # For Task 1: Mark the initial posterior P(C₁|x) = 0.7
    ax.axvline(x=0.7, color='purple', linestyle='-', alpha=0.7)
    ax.text(0.7, ax.get_ylim()[1]*0.9, "Initial\nP(C₁|x)=0.7", ha='center', va='top',
           bbox=dict(boxstyle="round,pad=0.3", fc="lavender", ec="purple", alpha=0.7))
    
    # For Task 3: Mark the updated posterior P(C₁|x) = 0.5
    ax.axvline(x=0.5, color='darkblue', linestyle='-', alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1]*0.85, "Updated\nP(C₁|x)=0.5", ha='center', va='top',
           bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="darkblue", alpha=0.7))
    
    fig.tight_layout()
    save_figure(fig, filename)
    return fig, decision_boundaries

def compare_zero_one_loss():
    """Compare standard loss matrix with zero-one loss matrix"""
    # Define zero-one loss matrix
    zero_one_loss_matrix = np.array([
        [0, 1, 1],  # a₁ correct if C₁, wrong otherwise
        [1, 0, 1],  # a₂ correct if C₂, wrong otherwise
        [1, 1, 0]   # a₃ correct if C₃, wrong otherwise
    ])
    
    # Calculate expected loss for each action using initial posteriors
    expected_losses_01 = calculate_expected_loss(zero_one_loss_matrix, initial_posteriors)
    
    # Find MAP estimate
    map_estimate = np.argmax(initial_posteriors)
    
    # Create a figure showing the comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original loss matrix visualization
    im1 = ax1.imshow(loss_matrix, cmap="YlOrRd")
    ax1.set_xticks(np.arange(len(states)))
    ax1.set_yticks(np.arange(len(actions)))
    ax1.set_xticklabels(states)
    ax1.set_yticklabels(actions)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(actions)):
        for j in range(len(states)):
            text = ax1.text(j, i, loss_matrix[i, j],
                          ha="center", va="center", color="black" if loss_matrix[i, j] < 50 else "white")
    ax1.set_title("Original Loss Matrix")
    
    # Zero-one loss matrix visualization
    im2 = ax2.imshow(zero_one_loss_matrix, cmap="Blues")
    ax2.set_xticks(np.arange(len(states)))
    ax2.set_yticks(np.arange(len(actions)))
    ax2.set_xticklabels(states)
    ax2.set_yticklabels(actions)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(actions)):
        for j in range(len(states)):
            text = ax2.text(j, i, zero_one_loss_matrix[i, j],
                          ha="center", va="center", color="white" if zero_one_loss_matrix[i, j] == 1 else "black")
    ax2.set_title("Zero-One Loss Matrix")
    
    fig.tight_layout()
    save_figure(fig, "loss_matrix_comparison.png")
    
    # Bar chart to compare expected losses
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(actions))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, calculate_expected_loss(loss_matrix, initial_posteriors), width, label='Original Loss')
    rects2 = ax.bar(x + width/2, expected_losses_01, width, label='Zero-One Loss')
    
    ax.set_ylabel('Expected Loss')
    ax.set_title('Comparison of Expected Losses: Original vs Zero-One Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend()
    
    # Highlight the minimum expected loss for each loss function
    min_idx1 = np.argmin(calculate_expected_loss(loss_matrix, initial_posteriors))
    min_idx2 = np.argmin(expected_losses_01)
    rects1[min_idx1].set_color('darkgreen')
    rects2[min_idx2].set_color('darkblue')
    
    # Highlight the MAP decision
    rects2[map_estimate].set_edgecolor('red')
    rects2[map_estimate].set_linewidth(2)
    
    # Add annotation about MAP
    annotation_text = f"MAP estimate = {states[map_estimate]} (max posterior = {initial_posteriors[map_estimate]:.1f})"
    ax.text(0.5, 0.95, annotation_text, 
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.7))
    
    # Show the numerical values on top of each bar
    for rect in rects1:
        ax.annotate(f'{rect.get_height():.2f}',
                   xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    for rect in rects2:
        ax.annotate(f'{rect.get_height():.2f}',
                   xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    fig.tight_layout()
    save_figure(fig, "expected_loss_comparison.png")
    
    return map_estimate, min_idx2, expected_losses_01

def run_analysis():
    """Run the entire analysis for Question 31"""
    print_step_header(0, "Problem Setup")
    print("Loss Matrix:")
    print(loss_matrix)
    print("\nInitial posterior probabilities:")
    for i, state in enumerate(states):
        print(f"{state}: {initial_posteriors[i]}")
    
    # Visualize the loss matrix
    visualize_loss_matrix(loss_matrix, states, actions)
    
    # Task 1: Calculate expected loss for each action
    print_step_header(1, "Calculate Expected Loss (Bayes Risk) for Each Treatment Decision")
    expected_losses = calculate_expected_loss(loss_matrix, initial_posteriors)
    
    print("Expected Loss (Bayes Risk) for each action:")
    for i, action in enumerate(actions):
        print(f"{action}: {expected_losses[i]:.2f}")
    
    # Visualize expected losses
    visualize_expected_loss(expected_losses, actions, 
                           "Expected Loss for Each Treatment Decision\nWith Initial Posteriors",
                           "expected_loss_initial.png")
    
    # Task 2: Determine which treatment minimizes Bayes risk
    print_step_header(2, "Determine Treatment that Minimizes Bayes Risk")
    min_risk_idx = np.argmin(expected_losses)
    print(f"The treatment that minimizes Bayes risk is: {actions[min_risk_idx]}")
    print(f"With an expected loss of: {expected_losses[min_risk_idx]:.2f}")
    
    # Task 3: Recalculate with updated posterior probabilities
    print_step_header(3, "Recalculate with Updated Posterior Probabilities")
    print("\nUpdated posterior probabilities:")
    for i, state in enumerate(states):
        print(f"{state}: {updated_posteriors[i]}")
    
    updated_expected_losses = calculate_expected_loss(loss_matrix, updated_posteriors)
    
    print("\nUpdated Expected Loss (Bayes Risk) for each action:")
    for i, action in enumerate(actions):
        print(f"{action}: {updated_expected_losses[i]:.2f}")
    
    # Visualize updated expected losses
    visualize_expected_loss(updated_expected_losses, actions, 
                           "Expected Loss for Each Treatment Decision\nWith Updated Posteriors",
                           "expected_loss_updated.png")
    
    updated_min_risk_idx = np.argmin(updated_expected_losses)
    print(f"\nThe treatment that now minimizes Bayes risk is: {actions[updated_min_risk_idx]}")
    print(f"With an expected loss of: {updated_expected_losses[updated_min_risk_idx]:.2f}")
    
    # Task 4: Find range of P(C₁|x) values that make "no treatment" optimal
    print_step_header(4, "Find Range of P(C₁|x) Values for Optimal 'No Treatment'")
    
    fig, decision_boundaries = visualize_decision_boundaries(loss_matrix, states, actions)
    
    a1_optimal_range = []
    for i in range(len(decision_boundaries) + 1):
        start = 0 if i == 0 else decision_boundaries[i-1]
        end = 1 if i == len(decision_boundaries) else decision_boundaries[i]
        
        # Test a point in this range to see which action is optimal
        test_point = (start + end) / 2
        test_posteriors = np.array([test_point, (1-test_point)/2, (1-test_point)/2])
        test_expected_losses = calculate_expected_loss(loss_matrix, test_posteriors)
        optimal_action_idx = np.argmin(test_expected_losses)
        
        if optimal_action_idx == 0:  # If a₁ (No treatment) is optimal
            a1_optimal_range.append((start, end))
    
    if a1_optimal_range:
        print(f"'No treatment' is optimal when P(C₁|x) is in the range:")
        for start, end in a1_optimal_range:
            print(f"{start:.4f} to {end:.4f}")
    else:
        print("'No treatment' is not optimal for any value of P(C₁|x).")
    
    # Task 5: Use zero-one loss function and compare with MAP
    print_step_header(5, "Compare with Zero-One Loss and MAP Estimation")
    
    map_estimate, min_idx_01, expected_losses_01 = compare_zero_one_loss()
    
    print(f"MAP estimate (class with highest posterior): {states[map_estimate]}")
    print(f"Decision under zero-one loss: {actions[min_idx_01]}")
    print(f"Original decision under asymmetric loss: {actions[min_risk_idx]}")
    
    if min_idx_01 == map_estimate:
        print("\nThe decision under zero-one loss matches the MAP estimate, as expected.")
    else:
        print("\nUnexpected result: The decision under zero-one loss doesn't match the MAP estimate.")
    
    # Task 6: Explain how loss values serve as a "prior"
    print_step_header(6, "Loss Values as Prior in Decision-Making")
    
    # This is mostly an explanatory task, but we can illustrate with a visualization
    print("This is primarily an explanatory task. The explanation will discuss how the loss values")
    print("influence decision boundaries similar to how priors influence posterior probabilities.")
    print("The visualization of decision boundaries already illustrates this relationship.")

if __name__ == "__main__":
    run_analysis() 