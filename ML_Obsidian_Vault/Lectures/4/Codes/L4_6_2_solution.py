import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Define the score functions for each class
scores = {
    'A': 2.1,
    'B': 1.7,
    'C': -0.5,
    'D': 0.8
}

print("Step 1: Score functions for each class")
for class_name, score in scores.items():
    print(f"f_{class_name}(x) = {score}")

# Find the class with the highest score
predicted_class = max(scores, key=scores.get)
print(f"\nThe predicted class is: {predicted_class} (Score: {scores[predicted_class]})")

# Step 2: Convert scores to probabilities using the sigmoid function
def sigmoid(z):
    """Compute sigmoid function σ(z) = 1/(1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

print("\nStep 2: Convert scores to probabilities using the sigmoid function")
probabilities = {class_name: sigmoid(score) for class_name, score in scores.items()}

print("\nProbabilities for each binary classifier:")
for class_name, prob in probabilities.items():
    print(f"P(y = {class_name} | x) = σ(f_{class_name}(x)) = σ({scores[class_name]:.2f}) = {prob:.6f}")

# Verify that the class with highest probability is the predicted class
max_prob_class = max(probabilities, key=probabilities.get)
print(f"\nClass with highest probability: {max_prob_class} (Probability: {probabilities[max_prob_class]:.6f})")
print(f"This matches our predicted class from Step 1: {max_prob_class == predicted_class}")

# Visualization 1: Bar plot comparing the scores
plt.figure(figsize=(10, 6))
classes = list(scores.keys())
score_values = list(scores.values())

# Create bars with different colors
bars = plt.bar(classes, score_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Adding value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    if height < 0:
        va = 'top'
        y_pos = height - 0.1
    else:
        va = 'bottom'
        y_pos = height + 0.1
    plt.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{height:.1f}', ha='center', va=va)

# Adding a horizontal line at y=0
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Highlighting the highest score
max_score_idx = score_values.index(max(score_values))
bars[max_score_idx].set_color('darkblue')
bars[max_score_idx].set_edgecolor('black')
bars[max_score_idx].set_linewidth(1.5)

plt.xlabel('Class')
plt.ylabel('Score $f_k(x)$')
plt.title('One-vs-All: Score Functions for Each Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(min(score_values) - 0.5, max(score_values) + 0.5)

plt.savefig(os.path.join(save_dir, 'ova_scores.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Bar plot comparing the probabilities
plt.figure(figsize=(10, 6))
prob_values = list(probabilities.values())

# Create bars with different colors
bars = plt.bar(classes, prob_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Adding value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{height:.4f}', ha='center', va='bottom')

# Highlighting the highest probability
max_prob_idx = prob_values.index(max(prob_values))
bars[max_prob_idx].set_color('darkblue')
bars[max_prob_idx].set_edgecolor('black')
bars[max_prob_idx].set_linewidth(1.5)

plt.xlabel('Class')
plt.ylabel('Probability $P(y = k | x)$')
plt.title('One-vs-All: Converted Probabilities for Each Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1.1)

plt.savefig(os.path.join(save_dir, 'ova_probabilities.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Combined visualization with scores and probabilities
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Scores
bars1 = ax1.bar(classes, score_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
for bar in bars1:
    height = bar.get_height()
    if height < 0:
        va = 'top'
        y_pos = height - 0.1
    else:
        va = 'bottom'
        y_pos = height + 0.1
    ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{height:.1f}', ha='center', va=va)

ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.set_ylabel('Score $f_k(x)$')
ax1.set_title('One-vs-All: Score Functions for Each Class')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim(min(score_values) - 0.5, max(score_values) + 0.5)

# Highlight highest score
bars1[max_score_idx].set_color('darkblue')
bars1[max_score_idx].set_edgecolor('black')
bars1[max_score_idx].set_linewidth(1.5)

# Plot 2: Probabilities
bars2 = ax2.bar(classes, prob_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{height:.4f}', ha='center', va='bottom')

ax2.set_xlabel('Class')
ax2.set_ylabel('Probability $P(y = k | x)$')
ax2.set_title('One-vs-All: Converted Probabilities for Each Class')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim(0, 1.1)

# Highlight highest probability
bars2[max_prob_idx].set_color('darkblue')
bars2[max_prob_idx].set_edgecolor('black')
bars2[max_prob_idx].set_linewidth(1.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ova_combined.png'), dpi=300, bbox_inches='tight')

# Visualization 4: OVA binary classification visualization
plt.figure(figsize=(12, 10))

# Create 2x2 grid for the 4 classes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Create a simple visualization of the binary classification decision boundary
x = np.linspace(-5, 5, 1000)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (class_name, score) in enumerate(scores.items()):
    ax = axes[i]
    
    # Plot the sigmoid function
    sigmoid_curve = sigmoid(x)
    ax.plot(x, sigmoid_curve, 'k-', lw=2)
    
    # Mark the score and its probability
    ax.plot([score], [sigmoid(score)], 'o', ms=10, color=colors[i])
    
    # Add a vertical line at the score
    ax.axvline(x=score, color=colors[i], linestyle='--', alpha=0.7, 
               label=f'$f_{{{class_name}}}(x) = {score:.1f}$')
    
    # Add a horizontal line from (score, probability) to y-axis
    ax.axhline(y=sigmoid(score), color=colors[i], linestyle='--', alpha=0.7,
               label=f'$P(y={class_name}|x) = {sigmoid(score):.4f}$')
    
    # Set title and labels
    ax.set_title(f'Class {class_name} vs Rest')
    ax.set_xlabel('Score value')
    ax.set_ylabel('Probability')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Set decision threshold at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Threshold = 0.5')
    
    # Mark the decision region
    if sigmoid(score) > 0.5:
        ax.text(3, 0.25, f'Predict NOT Class {class_name}', ha='center', fontsize=10)
        ax.text(3, 0.75, f'Predict Class {class_name}', ha='center', fontsize=12, weight='bold')
    else:
        ax.text(3, 0.25, f'Predict NOT Class {class_name}', ha='center', fontsize=12, weight='bold')
        ax.text(3, 0.75, f'Predict Class {class_name}', ha='center', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ova_binary_classifiers.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Conflict scenarios in OVA - Print instead of visualize
print("\nStep 5: Conflict Scenarios in One-vs-All Classification")

# Create a hypothetical conflict scenario
conflict_scores = {
    'Case 1': {'A': 1.1, 'B': 1.0, 'C': 0.9, 'D': 0.8},
    'Case 2': {'A': 0.1, 'B': 0.4, 'C': -0.2, 'D': -0.5},
    'Case 3': {'A': 2.5, 'B': -1.0, 'C': -1.5, 'D': -2.0}
}

# Convert to probabilities
conflict_probs = {
    case: {cls: sigmoid(score) for cls, score in case_scores.items()}
    for case, case_scores in conflict_scores.items()
}

# Print column headers for LaTeX table
print("\nConflict Scenarios Table (LaTeX format for Obsidian):")
print("| Scenario | Score A | Score B | Score C | Score D | Predicted Class | Max Probability | Num Classes with P > 0.5 |")
print("| :------ | :-----: | :-----: | :-----: | :-----: | :------------: | :------------: | :---------------------: |")

# Print each row
for case in conflict_scores:
    row = [case]
    for cls in 'ABCD':
        row.append(f"{conflict_scores[case][cls]:.2f}")
    
    max_cls = max(conflict_scores[case], key=conflict_scores[case].get)
    row.append(max_cls)
    
    all_probs = [conflict_probs[case][cls] for cls in 'ABCD']
    max_prob = max(all_probs)
    row.append(f"{max_prob:.4f}")
    
    # Count classes with probability > 0.5
    count_above = sum(prob > 0.5 for prob in all_probs)
    row.append(f"{count_above}")
    
    # Format for LaTeX table
    print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} |")

print(f"\nAll visualizations have been saved to: {save_dir}")

# Create a scenario that demonstrates when OVA might fail
print("\nStep 3: Demonstrating when OVA might fail to provide a clear decision")
ambiguous_scores = {'A': 1.1, 'B': 1.2, 'C': 1.0, 'D': 1.3}
print("Consider a scenario with these scores:")
for class_name, score in ambiguous_scores.items():
    print(f"f_{class_name}(x) = {score}")

ambiguous_probs = {class_name: sigmoid(score) for class_name, score in ambiguous_scores.items()}
print("\nProbabilities for each binary classifier:")
for class_name, prob in ambiguous_probs.items():
    print(f"P(y = {class_name} | x) = {prob:.6f}")

# Count classes above threshold
above_threshold = sum(1 for p in ambiguous_probs.values() if p > 0.5)
print(f"\nNumber of classes with probability > 0.5: {above_threshold}")

if above_threshold > 1:
    print("This is an ambiguous situation where multiple binary classifiers predict their class.")

# Step 4: Approaches to resolve ambiguities
print("\nStep 4: Approaches to resolve ambiguities in OVA predictions")
print("1. Select the class with the highest probability (winner-takes-all)")
print("2. Use a calibration method to ensure better probability estimates")
print("3. Implement One-vs-One (OVO) approach instead of OVA")
print("4. Use multi-class classification directly with methods like multinomial logistic regression")
print("5. Implement a hierarchical classification approach") 