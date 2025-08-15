import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=== Netflix Decision Tree Overfitting Analysis ===\n")

# Given data
tree_depths = np.array([1, 2, 3, 4, 5, 6])
training_accuracy = np.array([0.65, 0.78, 0.89, 0.95, 0.98, 0.99])
user_satisfaction = np.array([0.68, 0.75, 0.82, 0.79, 0.76, 0.74])
user_complaints = np.array([15, 12, 8, 11, 18, 25])

# Create DataFrame for easier analysis
data = pd.DataFrame({
    'Tree_Depth': tree_depths,
    'Training_Accuracy': training_accuracy,
    'User_Satisfaction': user_satisfaction,
    'User_Complaints': user_complaints
})

print("Given Data:")
print(data.to_string(index=False))
print()

# Step 1: Identify overfitting point
print("=== Step 1: Identifying Overfitting Point ===")
print("Overfitting occurs when training accuracy increases but validation performance decreases.")
print("We need to find where user satisfaction starts declining while training accuracy continues improving.")

# Calculate the point where user satisfaction starts decreasing
satisfaction_peaks_at = np.argmax(user_satisfaction) + 1  # +1 because array is 0-indexed
print(f"User satisfaction peaks at depth {satisfaction_peaks_at} with value {user_satisfaction[satisfaction_peaks_at-1]:.2f}")

# Find where overfitting begins
overfitting_depth = None
for i in range(1, len(tree_depths)):
    if user_satisfaction[i] < user_satisfaction[i-1] and training_accuracy[i] > training_accuracy[i-1]:
        overfitting_depth = tree_depths[i]
        break

print(f"Overfitting begins at depth {overfitting_depth}")
print("Justification: At depth 4, training accuracy increases from 0.89 to 0.95,")
print("but user satisfaction decreases from 0.82 to 0.79.")
print()

# Step 2: Find optimal tree depth
print("=== Step 2: Finding Optimal Tree Depth ===")
optimal_depth = tree_depths[satisfaction_peaks_at - 1]
optimal_satisfaction = user_satisfaction[satisfaction_peaks_at - 1]
optimal_complaints = user_complaints[satisfaction_peaks_at - 1]

print(f"Optimal tree depth: {optimal_depth}")
print(f"Reason: This depth achieves the highest user satisfaction ({optimal_satisfaction:.2f})")
print(f"while maintaining reasonable training accuracy ({training_accuracy[satisfaction_peaks_at-1]:.2f})")
print(f"and keeping complaints low ({optimal_complaints}%)")
print()

# Step 3: Bias-Variance Tradeoff Analysis
print("=== Step 3: Bias-Variance Tradeoff Analysis ===")
print("Bias-Variance Tradeoff Demonstration:")
print("- Low depth (1-2): High bias, low variance")
print("  * Underfitting: Model is too simple, misses patterns")
print("  * Low training accuracy but reasonable generalization")
print()
print("- Medium depth (3): Balanced bias and variance")
print("  * Optimal complexity: Captures true patterns without noise")
print("  * Best validation performance (user satisfaction)")
print()
print("- High depth (4-6): Low bias, high variance")
print("  * Overfitting: Model memorizes training data")
print("  * High training accuracy but poor generalization")
print("  * Increasing user complaints due to poor recommendations")
print()

# Step 4: Create training vs validation accuracy plot
print("=== Step 4: Creating Training vs Validation Accuracy Plot ===")

plt.figure(figsize=(12, 8))

# Plot training accuracy
plt.subplot(2, 2, 1)
plt.plot(tree_depths, training_accuracy, 'b-o', linewidth=2, markersize=8, label=r'Training Accuracy')
plt.plot(tree_depths, user_satisfaction, 'r-s', linewidth=2, markersize=8, label=r'User Satisfaction (Validation)')
plt.axvline(x=optimal_depth, color='g', linestyle='--', alpha=0.7, label=rf'Optimal Depth ({optimal_depth})')
plt.axvline(x=overfitting_depth, color='orange', linestyle='--', alpha=0.7, label=rf'Overfitting Begins ({overfitting_depth})')

plt.xlabel(r'Tree Depth')
plt.ylabel(r'Accuracy/Score')
plt.title(r'Training vs Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 1.0)

# Add annotations with LaTeX
for i, (depth, train_acc, val_acc) in enumerate(zip(tree_depths, training_accuracy, user_satisfaction)):
    plt.annotate(rf'${train_acc:.2f}$', (depth, train_acc), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)
    plt.annotate(rf'${val_acc:.2f}$', (depth, val_acc), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=9)

# Step 5: User complaints analysis
plt.subplot(2, 2, 2)
plt.plot(tree_depths, user_complaints, 'm-^', linewidth=2, markersize=8, label=r'User Complaints (\%)')
plt.axvline(x=optimal_depth, color='g', linestyle='--', alpha=0.7, label=rf'Optimal Depth ({optimal_depth})')
plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label=r'10\% Complaint Threshold')

plt.xlabel(r'Tree Depth')
plt.ylabel(r'User Complaints (\%)')
plt.title(r'User Complaints vs Tree Depth')
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations with LaTeX
for i, (depth, complaints) in enumerate(zip(tree_depths, user_complaints)):
    plt.annotate(rf'${complaints}\%$', (depth, complaints), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# Step 6: Gap analysis (overfitting visualization)
plt.subplot(2, 2, 3)
gap = training_accuracy - user_satisfaction
plt.plot(tree_depths, gap, 'purple', linewidth=2, markersize=8, marker='o', label=r'Accuracy Gap')
plt.axvline(x=optimal_depth, color='g', linestyle='--', alpha=0.7, label=rf'Optimal Depth ({optimal_depth})')
plt.axvline(x=overfitting_depth, color='orange', linestyle='--', alpha=0.7, label=rf'Overfitting Begins ({overfitting_depth})')

plt.xlabel(r'Tree Depth')
plt.ylabel(r'Training - Validation Gap')
plt.title(r'Overfitting Gap Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations with LaTeX
for i, (depth, gap_val) in enumerate(zip(tree_depths, gap)):
    plt.annotate(rf'${gap_val:.2f}$', (depth, gap_val), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# Step 7: Cost analysis
plt.subplot(2, 2, 4)
# Calculate costs for each depth
total_users = 100_000_000
cost_per_complaint = 2
costs = (user_complaints / 100) * total_users * cost_per_complaint

plt.plot(tree_depths, costs, 'c-D', linewidth=2, markersize=8, label=r'Total Cost (USD)')
plt.axvline(x=optimal_depth, color='g', linestyle='--', alpha=0.7, label=rf'Optimal Depth ({optimal_depth})')
plt.axvline(x=overfitting_depth, color='orange', linestyle='--', alpha=0.7, label=rf'Overfitting Begins ({overfitting_depth})')

plt.xlabel(r'Tree Depth')
plt.ylabel(r'Total Cost (USD)')
plt.title(r'Customer Service Cost Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Add cost annotations with LaTeX
for i, (depth, cost) in enumerate(zip(tree_depths, costs)):
    plt.annotate(rf'\${cost/1e6:.1f}M', (depth, cost), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_decision_tree_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 5: Maximum acceptable tree depth for <10% complaints
print("=== Step 5: Maximum Acceptable Tree Depth ===")
max_acceptable_depth = None
for i, complaints in enumerate(user_complaints):
    if complaints <= 10:
        max_acceptable_depth = tree_depths[i]
    else:
        break

if max_acceptable_depth:
    print(f"Maximum acceptable tree depth: {max_acceptable_depth}")
    print(f"At this depth: {user_complaints[tree_depths == max_acceptable_depth][0]}% complaints")
else:
    print("No depth achieves <10% complaints")
print()

# Step 6: Cost calculation for overfitting at depth 6
print("=== Step 6: Cost of Overfitting at Depth 6 ===")
depth_6_complaints = user_complaints[-1]  # 25%
optimal_complaints_at_depth_3 = user_complaints[2]  # 8%

excess_complaints = depth_6_complaints - optimal_complaints_at_depth_3
excess_complaints_absolute = (excess_complaints / 100) * total_users
total_cost = excess_complaints_absolute * cost_per_complaint

print(f"Excess complaints at depth 6: {excess_complaints}%")
print(f"Excess complaints in absolute numbers: {excess_complaints_absolute:,}")
print(f"Total cost of overfitting: ${total_cost:,}")
print(f"Cost in millions: ${total_cost/1e6:.1f}M")
print()

# Additional analysis: Create detailed comparison table
print("=== Detailed Analysis Summary ===")
analysis_df = pd.DataFrame({
    'Tree_Depth': tree_depths,
    'Training_Accuracy': training_accuracy,
    'User_Satisfaction': user_satisfaction,
    'User_Complaints_%': user_complaints,
    'Accuracy_Gap': gap,
    'Total_Cost_$': costs,
    'Status': ['Underfitting' if i < 2 else 'Optimal' if i == 2 else 'Overfitting' for i in range(len(tree_depths))]
})

print(analysis_df.to_string(index=False))
print()

# Create separate focused visualizations

# 1. Main Performance Analysis Plot
plt.figure(figsize=(10, 8))
plt.plot(tree_depths, training_accuracy, 'b-o', linewidth=3, markersize=10, label=r'Training Accuracy')
plt.plot(tree_depths, user_satisfaction, 'r-s', linewidth=3, markersize=10, label=r'User Satisfaction (Validation)')
plt.fill_between(tree_depths, training_accuracy, user_satisfaction, alpha=0.2, color='purple', label=r'Overfitting Gap')
plt.axvline(x=optimal_depth, color='g', linewidth=3, linestyle='--', label=rf'Optimal: Depth {optimal_depth}')
plt.axvline(x=overfitting_depth, color='orange', linewidth=2, linestyle='--', label=rf'Overfitting: Depth {overfitting_depth}')

plt.xlabel(r'Tree Depth', fontsize=14)
plt.ylabel(r'Accuracy/Score', fontsize=14)
plt.title(r'Netflix Decision Tree Performance Analysis', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 1.0)

# Add annotations with LaTeX
for i, (depth, train_acc, val_acc) in enumerate(zip(tree_depths, training_accuracy, user_satisfaction)):
    plt.annotate(rf'${train_acc:.2f}$', (depth, train_acc), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)
    plt.annotate(rf'${val_acc:.2f}$', (depth, val_acc), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_performance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. User Complaints Analysis
plt.figure(figsize=(10, 8))
bars = plt.bar(tree_depths, user_complaints, color=['green' if x <= 10 else 'orange' if x <= 15 else 'red' for x in user_complaints])
plt.axhline(y=10, color='red', linestyle='--', linewidth=2, label=r'10\% Threshold')
plt.xlabel(r'Tree Depth', fontsize=14)
plt.ylabel(r'User Complaints (\%)', fontsize=14)
plt.title(r'User Complaints by Tree Depth', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, complaint in zip(bars, user_complaints):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             rf'${complaint}\%$', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_complaints_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Cost Analysis
plt.figure(figsize=(10, 8))
cost_bars = plt.bar(tree_depths, costs/1e6, color=['green' if x == optimal_depth else 'orange' if x <= overfitting_depth else 'red' for x in tree_depths])
plt.xlabel(r'Tree Depth', fontsize=14)
plt.ylabel(r'Total Cost (Millions \$)', fontsize=14)
plt.title(r'Customer Service Cost by Tree Depth', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, cost in zip(cost_bars, costs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             rf'\${cost/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_cost_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Overfitting Gap Analysis
plt.figure(figsize=(10, 8))
plt.plot(tree_depths, gap, 'purple', linewidth=3, marker='o', markersize=10)
plt.fill_between(tree_depths, gap, alpha=0.3, color='purple')
plt.axvline(x=optimal_depth, color='g', linewidth=2, linestyle='--', label=rf'Optimal: Depth {optimal_depth}')
plt.xlabel(r'Tree Depth', fontsize=14)
plt.ylabel(r'Training - Validation Gap', fontsize=14)
plt.title(r'Overfitting Gap Analysis', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add annotations
for i, (depth, gap_val) in enumerate(zip(tree_depths, gap)):
    plt.annotate(rf'${gap_val:.2f}$', (depth, gap_val), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_gap_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Optimal vs Worst Performance Comparison
plt.figure(figsize=(10, 8))
metrics = [r'Training\\Accuracy', r'User\\Satisfaction', r'Complaints\\(\%)', r'Cost\\(\$M)']
optimal_values = [
    training_accuracy[tree_depths == optimal_depth][0],
    user_satisfaction[tree_depths == optimal_depth][0],
    user_complaints[tree_depths == optimal_depth][0],
    costs[tree_depths == optimal_depth][0]/1e6
]
worst_values = [
    training_accuracy[-1],
    user_satisfaction[-1],
    user_complaints[-1],
    costs[-1]/1e6
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, optimal_values, width, label=rf'Optimal (Depth {optimal_depth})', color='green', alpha=0.7)
plt.bar(x + width/2, worst_values, width, label=r'Worst (Depth 6)', color='red', alpha=0.7)

plt.xlabel(r'Metrics', fontsize=14)
plt.ylabel(r'Values', fontsize=14)
plt.title(r'Optimal vs Worst Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(x, metrics)
plt.legend(fontsize=12)

# Add value labels
for i, (opt, worst) in enumerate(zip(optimal_values, worst_values)):
    plt.text(i - width/2, opt + 0.01, rf'${opt:.2f}$', ha='center', va='bottom', fontweight='bold')
    plt.text(i + width/2, worst + 0.01, rf'${worst:.2f}$', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'netflix_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All plots saved to: {save_dir}")
print("\n=== Analysis Complete ===")
print("The comprehensive analysis shows:")
print(f"1. Overfitting begins at depth {overfitting_depth}")
print(f"2. Optimal depth is {optimal_depth}")
print(f"3. Maximum acceptable depth for <10% complaints: {max_acceptable_depth}")
print(f"4. Cost of overfitting at depth 6: ${total_cost/1e6:.1f}M")
print("5. Clear bias-variance tradeoff demonstrated")
print("6. Visual evidence of the overfitting problem")
