import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 21: RANDOM FOREST RAINFALL PREDICTION")
print("=" * 80)

# Given data: Daily rainfall predictions from 10 trees
daily_predictions = {
    'Day 1': [0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2],
    'Day 2': [0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7],
    'Day 3': [0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4]
}

print("\nGIVEN DATA:")
print("-" * 40)
for day, predictions in daily_predictions.items():
    print(f"{day}: {predictions}")

# Convert to numpy arrays for easier computation
daily_arrays = {day: np.array(preds) for day, preds in daily_predictions.items()}

print("\n" + "=" * 80)
print("MATHEMATICAL CONCEPTS AND FORMULAS")
print("=" * 80)

print("\nKey Formulas Used:")
print("1. Ensemble Mean: μ = (1/n) × Σx_i")
print("2. Sample Variance: σ² = Σ(x_i - μ)² / (n-1)")
print("3. Sample Standard Deviation: σ = √σ²")
print("4. Coefficient of Variation: CV = σ/μ")
print("5. Entropy: H = -Σ p_i × log₂(p_i)")

print("\nWhy n-1 for sample variance?")
print("- We use n-1 (Bessel's correction) because we're estimating the population")
print("- variance from a sample, and this gives an unbiased estimator")
print("- If we used n, we'd systematically underestimate the true variance")

print("\n" + "=" * 80)
print("STEP-BY-STEP SOLUTION")
print("=" * 80)

# Task 1: Calculate ensemble rainfall probability and uncertainty for each day
print("\nTASK 1: ENSEMBLE RAINFALL PROBABILITY AND UNCERTAINTY")
print("-" * 60)

ensemble_results = {}

for day, predictions in daily_arrays.items():
    print(f"\n{day}:")
    print(f"  Individual predictions: {predictions}")
    
    # Calculate ensemble mean (ensemble rainfall probability)
    ensemble_mean = np.mean(predictions)
    print(f"  Ensemble mean = Σ(predictions) / n = {np.sum(predictions)} / {len(predictions)} = {ensemble_mean:.3f}")
    
    # Detailed step-by-step standard deviation calculation
    print(f"  Standard deviation calculation:")
    print(f"    Step 1: Calculate mean = {ensemble_mean:.3f}")
    
    # Calculate deviations from mean
    deviations = predictions - ensemble_mean
    print(f"    Step 2: Calculate deviations from mean:")
    for i, (pred, dev) in enumerate(zip(predictions, deviations)):
        print(f"      Tree {i+1}: {pred:.1f} - {ensemble_mean:.3f} = {dev:.3f}")
    
    # Calculate squared deviations
    squared_deviations = deviations ** 2
    print(f"    Step 3: Square the deviations:")
    for i, (dev, sq_dev) in enumerate(zip(deviations, squared_deviations)):
        print(f"      Tree {i+1}: ({dev:.3f})² = {sq_dev:.4f}")
    
    # Sum of squared deviations
    sum_squared_deviations = np.sum(squared_deviations)
    print(f"    Step 4: Sum of squared deviations = {sum_squared_deviations:.4f}")
    
    # Calculate variance (using n-1 for sample variance)
    n = len(predictions)
    ensemble_var = sum_squared_deviations / (n - 1)
    print(f"    Step 5: Variance = {sum_squared_deviations:.4f} / ({n} - 1) = {ensemble_var:.4f}")
    
    # Calculate standard deviation
    ensemble_std = np.sqrt(ensemble_var)
    print(f"    Step 6: Standard deviation = √{ensemble_var:.4f} = {ensemble_std:.3f}")
    
    # Calculate coefficient of variation (relative uncertainty)
    cv = ensemble_std / ensemble_mean if ensemble_mean != 0 else float('inf')
    print(f"    Step 7: Coefficient of variation = {ensemble_std:.3f} / {ensemble_mean:.3f} = {cv:.3f}")
    
    # Store results
    ensemble_results[day] = {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'variance': ensemble_var,
        'cv': cv,
        'predictions': predictions
    }

# Task 2: Rain warning for probabilities > 0.5
print("\n" + "-" * 60)
print("TASK 2: RAIN WARNING FOR PROBABILITIES > 0.5")
print("-" * 60)

print("\nRain Warning Analysis:")
print("Threshold: P(rain) > 0.5")
print("Decision rule: Issue warning if ensemble probability > 0.5")

warnings = {}
for day, results in ensemble_results.items():
    mean_prob = results['mean']
    warning_needed = mean_prob > 0.5
    warnings[day] = warning_needed
    
    print(f"\n{day}:")
    print(f"  Ensemble probability: {mean_prob:.3f}")
    print(f"  Warning threshold: 0.500")
    print(f"  Decision rule: Issue warning if P(rain) > 0.5")
    print(f"  Comparison: {mean_prob:.3f} vs 0.500")
    print(f"  Decision: {'WARNING' if warning_needed else 'NO WARNING'}")
    print(f"  Reasoning: {mean_prob:.3f} {'>' if warning_needed else '≤'} 0.500")
    print(f"  Explanation: {'High probability of rain detected' if warning_needed else 'Rain probability below warning threshold'}")

# Task 3: Most reliable prediction (lowest variance)
print("\n" + "-" * 60)
print("TASK 3: MOST RELIABLE PREDICTION (LOWEST VARIANCE)")
print("-" * 60)

print("\nReliability Analysis (Lower variance = Higher reliability):")
print("-" * 50)
print("Reliability measures how consistent the predictions are across trees.")
print("Lower variance indicates more agreement between trees, making predictions more reliable.")

# Sort days by variance (ascending - most reliable first)
sorted_reliability = sorted(ensemble_results.items(), key=lambda x: x[1]['variance'])

for i, (day, results) in enumerate(sorted_reliability):
    rank = i + 1
    print(f"\n{rank}. {day}:")
    print(f"   Variance: {results['variance']:.4f}")
    print(f"   Standard deviation: {results['std']:.4f}")
    print(f"   Coefficient of variation: {results['cv']:.3f}")
    print(f"   Reliability ranking: {'Most' if rank == 1 else 'Middle' if rank == 2 else 'Least'} reliable")
    
    # Add interpretation
    if rank == 1:
        print(f"   Interpretation: Most reliable - trees show highest agreement")
    elif rank == 2:
        print(f"   Interpretation: Middle reliability - moderate tree agreement")
    else:
        print(f"   Interpretation: Least reliable - trees show lowest agreement")

most_reliable_day = sorted_reliability[0][0]
print(f"\nCONCLUSION: {most_reliable_day} has the most reliable prediction (lowest variance)")

# Task 4: Most confident prediction
print("\n" + "-" * 60)
print("TASK 4: MOST CONFIDENT PREDICTION")
print("-" * 60)

print("\nConfidence Analysis:")
print("Confidence is inversely related to uncertainty (standard deviation)")
print("-" * 50)
print("Confidence measures how certain we are about the ensemble prediction.")
print("Lower standard deviation indicates less spread in predictions, giving higher confidence.")

# Sort by standard deviation (ascending - most confident first)
sorted_confidence = sorted(ensemble_results.items(), key=lambda x: x[1]['std'])

for i, (day, results) in enumerate(sorted_confidence):
    rank = i + 1
    print(f"\n{rank}. {day}:")
    print(f"   Standard deviation: {results['std']:.4f}")
    print(f"   Variance: {results['variance']:.4f}")
    print(f"   Coefficient of variation: {results['cv']:.3f}")
    print(f"   Confidence ranking: {'Most' if rank == 1 else 'Middle' if rank == 2 else 'Least'} confident")
    
    # Add interpretation
    if rank == 1:
        print(f"   Interpretation: Most confident - lowest prediction uncertainty")
    elif rank == 2:
        print(f"   Interpretation: Middle confidence - moderate prediction uncertainty")
    else:
        print(f"   Interpretation: Least confident - highest prediction uncertainty")

most_confident_day = sorted_confidence[0][0]
print(f"\nCONCLUSION: {most_confident_day} has the most confident prediction (lowest uncertainty)")

# Task 5: Calculate entropy of predictions
print("\n" + "-" * 60)
print("TASK 5: ENTROPY OF PREDICTIONS")
print("-" * 60)

print("\nEntropy Calculation:")
print("Formula: H = -Σ p_i log₂(p_i)")
print("Higher entropy = Higher uncertainty in predictions")
print("-" * 50)

def calculate_entropy(predictions):
    """Calculate entropy of predictions using histogram approach"""
    # Create histogram bins (0-0.1, 0.1-0.2, etc.)
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    hist, _ = np.histogram(predictions, bins=bins)
    
    # Calculate probabilities for each bin
    probs = hist / len(predictions)
    
    # Calculate entropy (avoid log(0))
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy, probs, bins

entropy_results = {}
for day, results in ensemble_results.items():
    predictions = results['predictions']
    entropy, probs, bins = calculate_entropy(predictions)
    entropy_results[day] = entropy
    
    print(f"\n{day}:")
    print(f"  Predictions: {predictions}")
    
    # Show histogram bins and probabilities
    print(f"  Histogram bins: {bins}")
    print(f"  Bin probabilities: {probs}")
    
    # Calculate entropy step by step with detailed explanation
    print(f"  Entropy calculation:")
    print(f"    Formula: H = -Σ p_i × log₂(p_i)")
    print(f"    Where p_i are the probabilities of each bin")
    print(f"    Note: 0 × log₂(0) is undefined, so we ignore bins with 0 probability")
    
    entropy_terms = []
    for i, p in enumerate(probs):
        if p > 0:
            # Calculate log₂(p) step by step
            log2_p = np.log2(p)
            print(f"    Bin {i}: p = {p:.3f}")
            print(f"      Step 1: log₂({p:.3f}) = {log2_p:.4f}")
            
            # Calculate the entropy term
            term = -p * log2_p
            entropy_terms.append(term)
            print(f"      Step 2: -{p:.3f} × {log2_p:.4f} = {term:.4f}")
        else:
            print(f"    Bin {i}: p = {p:.3f} → 0 × log₂(0) = 0 (undefined, ignored)")
    
    total_entropy = sum(entropy_terms)
    print(f"    Step 3: Sum all terms = {sum(entropy_terms):.4f}")
    print(f"    Final entropy: H = {total_entropy:.4f} bits")

# Sort by entropy (ascending - lowest uncertainty first)
sorted_entropy = sorted(entropy_results.items(), key=lambda x: x[1])
print(f"\nEntropy Ranking (Lower entropy = Lower uncertainty):")
for i, (day, entropy) in enumerate(sorted_entropy):
    rank = i + 1
    print(f"{rank}. {day}: H = {entropy:.4f}")

# Create comprehensive visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# 1. Individual predictions comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Rainfall Predictions Analysis', fontsize=16, fontweight='bold')

# Plot 1: Individual predictions for each day
ax1 = axes[0, 0]
for i, (day, predictions) in enumerate(daily_arrays.items()):
    ax1.plot(range(1, len(predictions) + 1), predictions, 'o-', 
             label=day, linewidth=2, markersize=8)
    ax1.axhline(y=np.mean(predictions), linestyle='--', alpha=0.7, 
                label=f'{day} Mean')

ax1.set_xlabel('Tree Number')
ax1.set_ylabel('Rainfall Probability')
ax1.set_title('Individual Tree Predictions by Day')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: Ensemble statistics comparison
ax2 = axes[0, 1]
days = list(ensemble_results.keys())
means = [results['mean'] for results in ensemble_results.values()]
stds = [results['std'] for results in ensemble_results.values()]

x_pos = np.arange(len(days))
bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
               color=['skyblue', 'lightcoral', 'lightgreen'])
ax2.set_xlabel('Day')
ax2.set_ylabel('Rainfall Probability')
ax2.set_title('Ensemble Predictions with Uncertainty')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(days)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax2.text(bar.get_x() + bar.get_width()/2, mean + std + 0.02, 
             f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Variance comparison
ax3 = axes[1, 0]
variances = [results['variance'] for results in ensemble_results.values()]
colors = ['red' if day == most_reliable_day else 'blue' for day in days]

bars = ax3.bar(days, variances, color=colors, alpha=0.7)
ax3.set_xlabel('Day')
ax3.set_ylabel('Variance')
ax3.set_title('Prediction Variance (Lower = More Reliable)')
ax3.grid(True, alpha=0.3, axis='y')

# Highlight most reliable
ax3.text(days.index(most_reliable_day), variances[days.index(most_reliable_day)] + 0.001, 
         'MOST RELIABLE', ha='center', va='bottom', fontweight='bold', color='red')

# Plot 4: Entropy comparison
ax4 = axes[1, 1]
entropies = [entropy_results[day] for day in days]
colors = ['green' if day == sorted_entropy[0][0] else 'blue' for day in days]

bars = ax4.bar(days, entropies, color=colors, alpha=0.7)
ax4.set_xlabel('Day')
ax4.set_ylabel('Entropy (bits)')
ax4.set_title('Prediction Entropy (Lower = Less Uncertainty)')
ax4.grid(True, alpha=0.3, axis='y')

# Highlight lowest entropy
ax4.text(days.index(sorted_entropy[0][0]), entropies[days.index(sorted_entropy[0][0])] + 0.01, 
         'LOWEST UNCERTAINTY', ha='center', va='bottom', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'rainfall_predictions_analysis.png'), dpi=300, bbox_inches='tight')

# 2. Detailed uncertainty analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Detailed Uncertainty Analysis', fontsize=16, fontweight='bold')

# Coefficient of variation
ax1 = axes[0]
cvs = [results['cv'] for results in ensemble_results.values()]
bars = ax1.bar(days, cvs, color=['orange', 'purple', 'brown'], alpha=0.7)
ax1.set_xlabel('Day')
ax1.set_ylabel('Coefficient of Variation')
ax1.set_title('Relative Uncertainty (CV = $\\sigma/\\mu$)')
ax1.grid(True, alpha=0.3, axis='y')

# Add CV values on bars
for i, (bar, cv) in enumerate(zip(bars, cvs)):
    ax1.text(bar.get_x() + bar.get_width()/2, cv + 0.01, 
             f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')

# Standard deviation comparison
ax2 = axes[1]
bars = ax2.bar(days, stds, color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.7)
ax2.set_xlabel('Day')
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Absolute Uncertainty')
ax2.grid(True, alpha=0.3, axis='y')

# Add std values on bars
for i, (bar, std) in enumerate(zip(bars, stds)):
    ax2.text(bar.get_x() + bar.get_width()/2, std + 0.001, 
             f'{std:.3f}', ha='center', va='bottom', fontweight='bold')

# Entropy comparison
ax3 = axes[2]
bars = ax3.bar(days, entropies, color=['gold', 'lightgray', 'orange'], alpha=0.7)
ax3.set_xlabel('Day')
ax3.set_ylabel('Entropy (bits)')
ax3.set_title('Information Uncertainty')
ax3.grid(True, alpha=0.3, axis='y')

# Add entropy values on bars
for i, (bar, entropy) in enumerate(zip(bars, entropies)):
    ax3.text(bar.get_x() + bar.get_width()/2, entropy + 0.01, 
             f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')

# 3. Decision boundary visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create decision regions
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Decision function: distance from warning threshold
Z = np.abs(X - 0.5)

# Plot decision regions
contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
ax.contour(X, Y, Z, levels=[0.5], colors='red', linewidths=2, linestyles='--')

# Plot ensemble predictions
for i, (day, results) in enumerate(ensemble_results.items()):
    mean_prob = results['mean']
    std_prob = results['std']
    
    # Color based on warning decision
    color = 'red' if mean_prob > 0.5 else 'green'
    marker = '^' if mean_prob > 0.5 else 'o'
    
    ax.errorbar(mean_prob, std_prob, xerr=0, yerr=0, 
                marker=marker, markersize=12, color=color, 
                capsize=5, capthick=2, linewidth=2,
                label=f'{day}: {mean_prob:.3f}±{std_prob:.3f}')
    
    # Add day label
    ax.annotate(day, (mean_prob, std_prob), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Add warning threshold line
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Warning Threshold (0.5)')

ax.set_xlabel('Ensemble Probability')
ax.set_ylabel('Standard Deviation (Uncertainty)')
ax.set_title('Rain Warning Decision Space')
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Distance from Decision Boundary')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_space.png'), dpi=300, bbox_inches='tight')

# 4. Summary statistics table
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_data = []
headers = ['Day', 'Mean', 'Std Dev', 'Variance', 'CV', 'Entropy', 'Warning', 'Reliability Rank', 'Confidence Rank']

for day in days:
    results = ensemble_results[day]
    warning_status = "YES" if warnings[day] else "NO"
    reliability_rank = sorted_reliability.index((day, results)) + 1
    confidence_rank = sorted_confidence.index((day, results)) + 1
    
    row = [
        day,
        f"{results['mean']:.3f}",
        f"{results['std']:.3f}",
        f"{results['variance']:.4f}",
        f"{results['cv']:.3f}",
        f"{entropy_results[day]:.3f}",
        warning_status,
        f"{reliability_rank}",
        f"{confidence_rank}"
    ]
    table_data.append(row)

# Create table
table = ax.table(cellText=table_data, colLabels=headers, 
                cellLoc='center', loc='center',
                colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12, 0.12])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the table
for i, row in enumerate(table_data):
    for j in range(len(headers)):
        cell = table[(i+1, j)]  # +1 because row 0 is headers
        # Color code based on values
        if j == 1:  # Mean column
            if float(row[j]) > 0.5:
                cell.set_facecolor('lightcoral')  # Warning
            else:
                cell.set_facecolor('lightgreen')  # No warning
        elif j == 3:  # Variance column
            if i == 0:  # Most reliable
                cell.set_facecolor('lightgreen')
            elif i == 2:  # Least reliable
                cell.set_facecolor('lightcoral')
            else:
                cell.set_facecolor('lightyellow')
        elif j == 5:  # Entropy column
            if i == 0:  # Lowest uncertainty
                cell.set_facecolor('lightgreen')
            elif i == 2:  # Highest uncertainty
                cell.set_facecolor('lightcoral')
            else:
                cell.set_facecolor('lightyellow')

ax.set_title('Summary Statistics and Rankings', fontsize=16, fontweight='bold', pad=20)
plt.savefig(os.path.join(save_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n1. ENSEMBLE PROBABILITIES:")
for day, results in ensemble_results.items():
    print(f"   {day}: {results['mean']:.3f} ± {results['std']:.3f}")

print(f"\n2. RAIN WARNINGS (P > 0.5):")
for day, warning in warnings.items():
    status = "WARNING" if warning else "NO WARNING"
    print(f"   {day}: {status}")

print(f"\n3. RELIABILITY RANKING (by variance):")
for i, (day, results) in enumerate(sorted_reliability):
    print(f"   {i+1}. {day}: variance = {results['variance']:.4f}")

print(f"\n4. CONFIDENCE RANKING (by std dev):")
for i, (day, results) in enumerate(sorted_confidence):
    print(f"   {i+1}. {day}: std = {results['std']:.4f}")

print(f"\n5. UNCERTAINTY RANKING (by entropy):")
for i, (day, entropy) in enumerate(sorted_entropy):
    print(f"   {i+1}. {day}: H = {entropy:.3f} bits")

print(f"\nKEY INSIGHTS:")
print(f"   • Most reliable prediction: {most_reliable_day}")
print(f"   • Most confident prediction: {most_confident_day}")
print(f"   • Lowest uncertainty: {sorted_entropy[0][0]}")
print(f"   • Days requiring warnings: {[day for day, warn in warnings.items() if warn]}")

# plt.show()  # Commented out to only save plots without displaying
