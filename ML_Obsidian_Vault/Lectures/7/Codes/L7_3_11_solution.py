import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid encoding issues
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 11: FRAUD DETECTION USING RANDOM FOREST")
print("=" * 80)

# Given data: 7 trees, 4 transactions
# Each row represents a transaction, each column represents a tree
fraud_probabilities = np.array([
    [0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3],  # Transaction A
    [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7],  # Transaction B
    [0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4],  # Transaction C
    [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]   # Transaction D
])

transaction_names = ['A', 'B', 'C', 'D']
n_trees = 7
n_transactions = 4

print(f"Random Forest with {n_trees} trees analyzing {n_transactions} transactions")
print(f"Fraud threshold: 0.5")
print()

# Display the data in a clear format
print("FRAUD PROBABILITIES FROM EACH TREE:")
print("-" * 50)
print("Transaction | Tree 1 | Tree 2 | Tree 3 | Tree 4 | Tree 5 | Tree 6 | Tree 7")
print("-" * 80)
for i, name in enumerate(transaction_names):
    probs = fraud_probabilities[i]
    print(f"     {name}     | {probs[0]:6.1f} | {probs[1]:6.1f} | {probs[2]:6.1f} | {probs[3]:6.1f} | {probs[4]:6.1f} | {probs[5]:6.1f} | {probs[6]:6.1f}")
print()

# Task 1: Calculate ensemble fraud probability for each transaction
print("TASK 1: ENSEMBLE FRAUD PROBABILITIES")
print("=" * 50)

ensemble_probs = np.mean(fraud_probabilities, axis=1)
print("Ensemble probability = Average of all tree predictions")
print()

for i, name in enumerate(transaction_names):
    probs = fraud_probabilities[i]
    mean_prob = ensemble_probs[i]
    print(f"Transaction {name}:")
    print(f"  Individual probabilities: {probs}")
    print(f"  Sum: {np.sum(probs):.1f}")
    print(f"  Count: {len(probs)}")
    print(f"  Ensemble probability: {np.sum(probs)}/{len(probs)} = {mean_prob:.3f}")
    print()

# Task 2: Determine which transactions are flagged as suspicious (threshold 0.5)
print("TASK 2: FRAUD DETECTION WITH THRESHOLD 0.5")
print("=" * 50)

fraud_threshold = 0.5
flagged_transactions = ensemble_probs >= fraud_threshold

print(f"Fraud threshold: {fraud_threshold}")
print("Decision rule: If ensemble probability ≥ threshold, flag as suspicious")
print()

for i, name in enumerate(transaction_names):
    prob = ensemble_probs[i]
    is_flagged = flagged_transactions[i]
    status = "SUSPICIOUS" if is_flagged else "CLEAN"
    print(f"Transaction {name}: {prob:.3f} {'≥' if is_flagged else '<'} {fraud_threshold} → {status}")

print(f"\nSummary: {np.sum(flagged_transactions)} out of {n_transactions} transactions flagged as suspicious")

# Task 3: Find transaction with highest disagreement (variance)
print("\nTASK 3: HIGHEST DISAGREEMENT AMONG TREES")
print("=" * 50)

variances = np.var(fraud_probabilities, axis=1)
print("Variance measures disagreement among trees (higher = more disagreement)")
print()

for i, name in enumerate(transaction_names):
    probs = fraud_probabilities[i]
    variance = variances[i]
    print(f"Transaction {name}:")
    print(f"  Probabilities: {probs}")
    print(f"  Variance: {variance:.4f}")
    print()

# Find transaction with highest variance
max_var_idx = np.argmax(variances)
max_var_transaction = transaction_names[max_var_idx]
print(f"Transaction {max_var_transaction} shows the highest disagreement (variance: {variances[max_var_idx]:.4f})")

# Task 4: Prioritize 2 transactions for investigation
print("\nTASK 4: PRIORITIZE 2 TRANSACTIONS FOR INVESTIGATION")
print("=" * 50)

# Create a scoring system: high probability + high uncertainty = high priority
# Priority score = ensemble_probability + normalized_variance
normalized_variances = variances / np.max(variances)  # Normalize to [0,1]
priority_scores = ensemble_probs + normalized_variances

print("Priority scoring system:")
print("  Priority Score = Ensemble Probability + Normalized Variance")
print("  Higher score = Higher priority for investigation")
print()

for i, name in enumerate(transaction_names):
    prob = ensemble_probs[i]
    var = variances[i]
    norm_var = normalized_variances[i]
    score = priority_scores[i]
    print(f"Transaction {name}:")
    print(f"  Ensemble probability: {prob:.3f}")
    print(f"  Variance: {var:.4f}")
    print(f"  Normalized variance: {norm_var:.3f}")
    print(f"  Priority score: {prob:.3f} + {norm_var:.3f} = {score:.3f}")
    print()

# Sort by priority score
priority_order = np.argsort(priority_scores)[::-1]
print("Ranking by priority score (highest to lowest):")
for i, idx in enumerate(priority_order):
    name = transaction_names[idx]
    score = priority_scores[idx]
    print(f"  {i+1}. Transaction {name}: {score:.3f}")

print(f"\nTop 2 priorities for investigation:")
print(f"  1. Transaction {transaction_names[priority_order[0]]} (score: {priority_scores[priority_order[0]]:.3f})")
print(f"  2. Transaction {transaction_names[priority_order[1]]} (score: {priority_scores[priority_order[1]]:.3f})")

# Task 5: Calculate standard deviation and rank by uncertainty
print("\nTASK 5: STANDARD DEVIATION AND UNCERTAINTY RANKING")
print("=" * 50)

std_deviations = np.std(fraud_probabilities, axis=1)
print("Standard deviation measures uncertainty in predictions")
print()

for i, name in enumerate(transaction_names):
    probs = fraud_probabilities[i]
    std_dev = std_deviations[i]
    print(f"Transaction {name}:")
    print(f"  Probabilities: {probs}")
    print(f"  Standard deviation: {std_dev:.4f}")
    print()

# Rank by standard deviation (uncertainty)
uncertainty_order = np.argsort(std_deviations)[::-1]
print("Ranking by uncertainty (highest to lowest):")
for i, idx in enumerate(uncertainty_order):
    name = transaction_names[idx]
    std_dev = std_deviations[idx]
    print(f"  {i+1}. Transaction {name}: {std_dev:.4f}")

print(f"\nTransaction with highest uncertainty: {transaction_names[uncertainty_order[0]]} (std = {std_deviations[uncertainty_order[0]]:.4f})")
print(f"Transaction with lowest uncertainty: {transaction_names[uncertainty_order[-1]]} (std = {std_deviations[uncertainty_order[-1]]:.4f})")

# VISUALIZATIONS

# 1. Bar plot of ensemble probabilities
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
bars = plt.bar(transaction_names, ensemble_probs, color=['red' if x >= fraud_threshold else 'green' for x in ensemble_probs])
plt.axhline(y=fraud_threshold, color='black', linestyle='--', label=f'Threshold ({fraud_threshold})')
plt.xlabel('Transaction')
plt.ylabel('Ensemble Fraud Probability')
plt.title('Ensemble Fraud Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, prob in zip(bars, ensemble_probs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{prob:.3f}', ha='center', va='bottom')

# 2. Box plot showing distribution of probabilities
plt.subplot(2, 2, 2)
box_data = [fraud_probabilities[i] for i in range(n_transactions)]
bp = plt.boxplot(box_data, tick_labels=transaction_names, patch_artist=True)
colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Transaction')
plt.ylabel('Fraud Probability')
plt.title('Distribution of Tree Predictions')
plt.grid(True, alpha=0.3)

# 3. Heatmap of individual tree predictions
plt.subplot(2, 2, 3)
sns.heatmap(fraud_probabilities, annot=True, fmt='.1f', cmap='RdYlBu_r',
            xticklabels=[f'Tree {i+1}' for i in range(n_trees)],
            yticklabels=[f'Transaction {name}' for name in transaction_names],
            cbar_kws={'label': 'Fraud Probability'})
plt.title('Individual Tree Predictions')
plt.xlabel('Tree')
plt.ylabel('Transaction')

# 4. Priority scores and uncertainty comparison
plt.subplot(2, 2, 4)
x = np.arange(n_transactions)
width = 0.35

plt.bar(x - width/2, ensemble_probs, width, label='Ensemble Probability', alpha=0.7)
plt.bar(x + width/2, normalized_variances, width, label='Normalized Variance', alpha=0.7)
plt.xlabel('Transaction')
plt.ylabel('Score')
plt.title('Priority Score Components')
plt.xticks(x, transaction_names)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fraud_detection_overview.png'), dpi=300, bbox_inches='tight')

# Detailed analysis plots

# 1. Individual transaction analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Detailed Analysis of Each Transaction', fontsize=16)

for i, name in enumerate(transaction_names):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    probs = fraud_probabilities[i]
    mean_prob = ensemble_probs[i]
    std_prob = std_deviations[i]
    
    # Bar plot of individual tree predictions
    bars = ax.bar(range(1, n_trees + 1), probs, color='skyblue', alpha=0.7)
    ax.axhline(y=mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.3f}')
    ax.axhline(y=fraud_threshold, color='black', linestyle=':', alpha=0.7, label=f'Threshold: {fraud_threshold}')
    
    # Color bars based on threshold
    for j, (bar, prob) in enumerate(zip(bars, probs)):
        if prob >= fraud_threshold:
            bar.set_color('lightcoral')
        else:
            bar.set_color('lightgreen')
    
    ax.set_xlabel('Tree Number')
    ax.set_ylabel('Fraud Probability')
    ax.set_title(f'Transaction {name} - std = {std_prob:.3f}')
    ax.set_xticks(range(1, n_trees + 1))
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for j, prob in enumerate(probs):
        ax.text(j + 1, prob + 0.02, f'{prob:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'individual_transaction_analysis.png'), dpi=300, bbox_inches='tight')

# 2. Summary statistics comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Statistical Comparison of Transactions', fontsize=16)

# Ensemble probabilities
axes[0, 0].bar(transaction_names, ensemble_probs, color=['red' if x >= fraud_threshold else 'green' for x in ensemble_probs])
axes[0, 0].axhline(y=fraud_threshold, color='black', linestyle='--', label=f'Threshold ({fraud_threshold})')
axes[0, 0].set_title('Ensemble Fraud Probabilities')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Variances
axes[0, 1].bar(transaction_names, variances, color='orange', alpha=0.7)
axes[0, 1].set_title('Variance (Disagreement Among Trees)')
axes[0, 0].set_ylabel('Variance')
axes[0, 1].grid(True, alpha=0.3)

# Standard deviations
axes[1, 0].bar(transaction_names, std_deviations, color='purple', alpha=0.7)
axes[1, 0].set_title('Standard Deviation (Uncertainty)')
axes[1, 0].set_ylabel('Standard Deviation')
axes[1, 0].grid(True, alpha=0.3)

# Priority scores
axes[1, 1].bar(transaction_names, priority_scores, color='brown', alpha=0.7)
axes[1, 1].set_title('Priority Scores for Investigation')
axes[1, 1].set_ylabel('Priority Score')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')

# 3. Decision boundary visualization
plt.figure(figsize=(10, 8))
plt.scatter(ensemble_probs, std_deviations, s=200, c=['red' if x >= fraud_threshold else 'green' for x in ensemble_probs], alpha=0.7)

# Add transaction labels
for i, name in enumerate(transaction_names):
    plt.annotate(f'Transaction {name}', (ensemble_probs[i], std_deviations[i]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.axvline(x=fraud_threshold, color='black', linestyle='--', alpha=0.7, label=f'Fraud Threshold ({fraud_threshold})')
plt.xlabel('Ensemble Fraud Probability')
plt.ylabel('Standard Deviation (Uncertainty)')
plt.title('Fraud Detection Decision Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Add quadrants
plt.axhline(y=np.mean(std_deviations), color='gray', linestyle=':', alpha=0.5)
plt.text(0.25, np.mean(std_deviations) + 0.01, 'High Uncertainty', ha='center', va='bottom', 
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7))
plt.text(0.75, np.mean(std_deviations) + 0.01, 'High Uncertainty', ha='center', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7))

plt.savefig(os.path.join(save_dir, 'decision_space_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"1. Ensemble Probabilities: {dict(zip(transaction_names, ensemble_probs))}")
print(f"2. Flagged as Suspicious: {[name for name, flagged in zip(transaction_names, flagged_transactions) if flagged]}")
print(f"3. Highest Disagreement: Transaction {max_var_transaction} (variance: {variances[max_var_idx]:.4f})")
print(f"4. Top 2 Investigation Priorities: {transaction_names[priority_order[0]]}, {transaction_names[priority_order[1]]}")
print(f"5. Highest Uncertainty: Transaction {transaction_names[uncertainty_order[0]]} (σ = {std_deviations[uncertainty_order[0]]:.4f})")

print(f"\nAll plots saved to: {save_dir}")
print("=" * 80)
