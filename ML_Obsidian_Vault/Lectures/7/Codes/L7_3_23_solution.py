import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 23: Fraud Detection using Random Forest Analysis")
print("=" * 60)

# Given data: fraud probabilities from 7 trees for 4 transactions
transactions = {
    'A': [0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3],
    'B': [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7],
    'C': [0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4],
    'D': [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]
}

print("\nGiven fraud probabilities from 7 trees:")
for tx, probs in transactions.items():
    print(f"Transaction {tx}: {probs}")

# Convert to numpy arrays for easier computation
tx_arrays = {tx: np.array(probs) for tx, probs in transactions.items()}

print("\n" + "="*60)
print("STEP 1: Calculate Ensemble Fraud Probability for Each Transaction")
print("="*60)

# Calculate ensemble probabilities (mean of all trees)
ensemble_probs = {}
for tx, probs in tx_arrays.items():
    ensemble_probs[tx] = np.mean(probs)
    print(f"Transaction {tx}:")
    print(f"  Individual probabilities: {probs}")
    print(f"  Ensemble probability = mean({probs}) = {ensemble_probs[tx]:.3f}")
    print()

print("Summary of ensemble probabilities:")
for tx, prob in ensemble_probs.items():
    print(f"  Transaction {tx}: {prob:.3f}")

print("\n" + "="*60)
print("STEP 2: Fraud Detection with Threshold 0.5")
print("="*60)

threshold = 0.5
flagged_transactions = []
non_flagged_transactions = []

for tx, prob in ensemble_probs.items():
    if prob >= threshold:
        flagged_transactions.append(tx)
        print(f"Transaction {tx}: {prob:.3f} >= {threshold} → FLAGGED as suspicious")
    else:
        non_flagged_transactions.append(tx)
        print(f"Transaction {tx}: {prob:.3f} < {threshold} → NOT flagged")

print(f"\nSummary:")
print(f"  Flagged transactions: {flagged_transactions}")
print(f"  Non-flagged transactions: {non_flagged_transactions}")

print("\n" + "="*60)
print("STEP 3: Calculate Variance (Disagreement Among Trees)")
print("="*60)

variances = {}
for tx, probs in tx_arrays.items():
    variance = np.var(probs, ddof=1)  # Sample variance
    variances[tx] = variance
    print(f"Transaction {tx}:")
    print(f"  Probabilities: {probs}")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Variance = {variance:.4f}")
    print(f"  Standard Deviation = {np.std(probs, ddof=1):.4f}")
    print()

# Find transaction with highest variance
highest_variance_tx = max(variances.keys(), key=lambda x: variances[x])
print(f"Transaction with highest disagreement (variance): {highest_variance_tx}")
print(f"Variance: {variances[highest_variance_tx]:.4f}")

print("\n" + "="*60)
print("STEP 4: Transaction Prioritization for Investigation")
print("="*60)

# Create a scoring system for prioritization
# Higher ensemble probability and higher variance should get higher priority
priorities = {}
for tx in transactions.keys():
    # Priority score = ensemble_prob * 0.7 + variance * 10 (scaled)
    # Higher ensemble probability = higher fraud risk
    # Higher variance = more uncertainty = need for investigation
    priority_score = ensemble_probs[tx] * 0.7 + variances[tx] * 10
    priorities[tx] = priority_score
    print(f"Transaction {tx}:")
    print(f"  Ensemble probability: {ensemble_probs[tx]:.3f}")
    print(f"  Variance: {variances[tx]:.4f}")
    print(f"  Priority score: {priority_score:.4f}")
    print()

# Sort by priority score
sorted_priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
print("Prioritization for investigation (top 2):")
for i, (tx, score) in enumerate(sorted_priorities[:2]):
    print(f"  {i+1}. Transaction {tx} (Priority score: {score:.4f})")

print("\n" + "="*60)
print("STEP 5: Calculate 95% Confidence Intervals using t-distribution")
print("="*60)

confidence_intervals = {}
alpha = 0.05  # 95% confidence level
df = len(tx_arrays['A']) - 1  # degrees of freedom = n-1 = 6

t_critical = stats.t.ppf(1 - alpha/2, df)
print(f"t-critical value for 95% CI with {df} degrees of freedom: {t_critical:.4f}")

for tx, probs in tx_arrays.items():
    n = len(probs)
    mean_prob = np.mean(probs)
    std_error = np.std(probs, ddof=1) / np.sqrt(n)
    margin_of_error = t_critical * std_error
    
    ci_lower = mean_prob - margin_of_error
    ci_upper = mean_prob + margin_of_error
    confidence_intervals[tx] = (ci_lower, ci_upper)
    
    print(f"Transaction {tx}:")
    print(f"  Sample mean: {mean_prob:.4f}")
    print(f"  Standard error: {std_error:.4f}")
    print(f"  Margin of error: {margin_of_error:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

print("\n" + "="*60)
print("VISUALIZATIONS")
print("="*60)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Fraud Detection Analysis: Random Forest with 7 Trees', fontsize=16, fontweight='bold')

# 1. Individual tree probabilities
ax1 = axes[0, 0]
x_pos = np.arange(len(transactions))
width = 0.1

for i in range(len(transactions['A'])):  # For each tree
    tree_probs = [transactions[tx][i] for tx in transactions.keys()]
    ax1.bar(x_pos + i*width, tree_probs, width, label=f'Tree {i+1}', alpha=0.7)

ax1.set_xlabel('Transaction')
ax1.set_ylabel('Fraud Probability')
ax1.set_title('Individual Tree Predictions')
ax1.set_xticks(x_pos + width * 3)
ax1.set_xticklabels(list(transactions.keys()))
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Ensemble probabilities comparison
ax2 = axes[0, 1]
bars = ax2.bar(ensemble_probs.keys(), ensemble_probs.values(), 
                color=['red' if p >= threshold else 'green' for p in ensemble_probs.values()],
                alpha=0.7)
ax2.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
ax2.set_xlabel('Transaction')
ax2.set_ylabel('Ensemble Fraud Probability')
ax2.set_title('Ensemble Fraud Probabilities')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, prob in zip(bars, ensemble_probs.values()):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{prob:.3f}', ha='center', va='bottom')

# 3. Variance analysis
ax3 = axes[0, 2]
bars = ax3.bar(variances.keys(), variances.values(), 
                color=['orange' if tx == highest_variance_tx else 'blue' for tx in variances.keys()],
                alpha=0.7)
ax3.set_xlabel('Transaction')
ax3.set_ylabel('Variance')
ax3.set_title('Variance Among Trees (Disagreement)')
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for bar, var in zip(bars, variances.values()):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{var:.4f}', ha='center', va='bottom')

# 4. Priority scores
ax4 = axes[1, 0]
bars = ax4.bar(priorities.keys(), priorities.values(), 
                color=['red', 'orange', 'yellow', 'green'][:len(priorities)],
                alpha=0.7)
ax4.set_xlabel('Transaction')
ax4.set_ylabel('Priority Score')
ax4.set_title('Investigation Priority Scores')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, priorities.values()):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# 5. Confidence intervals
ax5 = axes[1, 1]
x_pos = np.arange(len(confidence_intervals))
ci_lower = [ci[0] for ci in confidence_intervals.values()]
ci_upper = [ci[1] for ci in confidence_intervals.values()]
means = [ensemble_probs[tx] for tx in confidence_intervals.keys()]

ax5.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                 np.array(ci_upper) - np.array(means)],
             fmt='o', capsize=5, capthick=2, markersize=8)
ax5.set_xlabel('Transaction')
ax5.set_ylabel('Fraud Probability')
ax5.set_title('95% Confidence Intervals')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(list(confidence_intervals.keys()))
ax5.grid(True, alpha=0.3)

# 6. Heatmap of all probabilities
ax6 = axes[1, 2]
prob_matrix = np.array([transactions[tx] for tx in transactions.keys()])
im = ax6.imshow(prob_matrix, cmap='RdYlBu_r', aspect='auto')
ax6.set_xlabel('Tree Number')
ax6.set_ylabel('Transaction')
ax6.set_title('Fraud Probability Heatmap')
ax6.set_xticks(range(len(transactions['A'])))
ax6.set_xticklabels([f'T{i+1}' for i in range(len(transactions['A']))])
ax6.set_yticks(range(len(transactions)))
ax6.set_yticklabels(list(transactions.keys()))

# Add colorbar
cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('Fraud Probability')

# Add text annotations to heatmap
for i in range(len(transactions)):
    for j in range(len(transactions['A'])):
        text = ax6.text(j, i, f'{prob_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fraud_detection_analysis.png'), dpi=300, bbox_inches='tight')

# Create additional detailed plots
plt.figure(figsize=(15, 10))

# Box plot showing distribution of probabilities for each transaction
plt.subplot(2, 2, 1)
prob_data = [transactions[tx] for tx in transactions.keys()]
bp = plt.boxplot(prob_data, labels=list(transactions.keys()), patch_artist=True)
colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Transaction')
plt.ylabel('Fraud Probability')
plt.title('Distribution of Tree Predictions')
plt.grid(True, alpha=0.3)

# Scatter plot of mean vs variance
plt.subplot(2, 2, 2)
for tx in transactions.keys():
    plt.scatter(ensemble_probs[tx], variances[tx], s=100, label=f'Transaction {tx}')
    plt.annotate(tx, (ensemble_probs[tx], variances[tx]), xytext=(5, 5), 
                 textcoords='offset points', fontsize=12, fontweight='bold')

plt.xlabel('Ensemble Probability')
plt.ylabel('Variance')
plt.title('Risk vs Uncertainty Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Bar plot comparing all metrics
plt.subplot(2, 2, 3)
x = np.arange(len(transactions))
width = 0.25

plt.bar(x - width, [ensemble_probs[tx] for tx in transactions.keys()], 
        width, label='Ensemble Probability', alpha=0.7)
plt.bar(x, [variances[tx] for tx in transactions.keys()], 
        width, label='Variance', alpha=0.7)
plt.bar(x + width, [priorities[tx] for tx in transactions.keys()], 
        width, label='Priority Score', alpha=0.7)

plt.xlabel('Transaction')
plt.ylabel('Value')
plt.title('Comparison of All Metrics')
plt.xticks(x, list(transactions.keys()))
plt.legend()
plt.grid(True, alpha=0.3)

# Confidence intervals with individual points
plt.subplot(2, 2, 4)
for i, tx in enumerate(transactions.keys()):
    # Plot individual tree predictions
    plt.scatter([i]*len(transactions[tx]), transactions[tx], 
                alpha=0.6, s=50, label=f'Tree predictions' if i == 0 else "")
    
    # Plot ensemble mean
    plt.scatter(i, ensemble_probs[tx], color='red', s=200, marker='*', 
                label=f'Ensemble mean' if i == 0 else "")
    
    # Plot confidence interval
    ci_lower, ci_upper = confidence_intervals[tx]
    plt.vlines(i, ci_lower, ci_upper, color='red', linewidth=3, alpha=0.7)

plt.xlabel('Transaction')
plt.ylabel('Fraud Probability')
plt.title('Individual Predictions with Confidence Intervals')
plt.xticks(range(len(transactions)), list(transactions.keys()))
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fraud_detection_detailed_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Print final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("1. Ensemble Fraud Probabilities:")
for tx, prob in ensemble_probs.items():
    status = "FLAGGED" if prob >= threshold else "NOT FLAGGED"
    print(f"   Transaction {tx}: {prob:.3f} → {status}")

print(f"\n2. Transactions flagged as suspicious (threshold {threshold}): {flagged_transactions}")

print(f"\n3. Transaction with highest disagreement: {highest_variance_tx} (variance: {variances[highest_variance_tx]:.4f})")

print("\n4. Investigation priority (top 2):")
for i, (tx, score) in enumerate(sorted_priorities[:2]):
    print(f"   {i+1}. Transaction {tx} (Priority: {score:.4f})")

print("\n5. 95% Confidence Intervals:")
for tx, (ci_lower, ci_upper) in confidence_intervals.items():
    print(f"   Transaction {tx}: [{ci_lower:.4f}, {ci_upper:.4f}]")
