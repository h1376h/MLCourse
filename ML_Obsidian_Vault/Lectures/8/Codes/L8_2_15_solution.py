import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 15: IRRELEVANT FEATURES IMPACT")
print("=" * 80)

# Given parameters
total_samples = 1000
total_features = 100
relevant_features = 20
irrelevant_features = total_features - relevant_features
noise_per_irrelevant_feature = 0.01  # 1%

print(f"Dataset Parameters:")
print(f"  Total samples: {total_samples}")
print(f"  Total features: {total_features}")
print(f"  Relevant features: {relevant_features}")
print(f"  Irrelevant features: {irrelevant_features}")
print(f"  Noise per irrelevant feature: {noise_per_irrelevant_feature:.1%}")

print("\n" + "=" * 80)
print("TASK 1: PERCENTAGE OF IRRELEVANT FEATURES")
print("=" * 80)

# Task 1: Percentage of irrelevant features
irrelevant_percentage = (irrelevant_features / total_features) * 100
relevant_percentage = (relevant_features / total_features) * 100

print(f"Calculation:")
print(f"  Irrelevant features = {irrelevant_features}")
print(f"  Total features = {total_features}")
print(f"  Irrelevant percentage = (irrelevant_features / total_features) $\\times$ 100")
print(f"  Irrelevant percentage = ({irrelevant_features} / {total_features}) $\\times$ 100")
print(f"  Irrelevant percentage = {irrelevant_percentage:.1f}%")

print(f"\nAnswer: {irrelevant_percentage:.1f}% of features are irrelevant")

# Create visualization for Task 1
plt.figure(figsize=(12, 8))

# Pie chart
plt.subplot(2, 2, 1)
labels = ['Relevant Features', 'Irrelevant Features']
sizes = [relevant_features, irrelevant_features]
colors = ['lightgreen', 'lightcoral']
explode = (0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, shadow=True)
plt.title('Feature Distribution')

# Bar chart
plt.subplot(2, 2, 2)
categories = ['Relevant', 'Irrelevant']
counts = [relevant_features, irrelevant_features]
bars = plt.bar(categories, counts, color=['lightgreen', 'lightcoral'], alpha=0.7)
plt.title('Feature Counts')
plt.ylabel('Number of Features')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

# Percentage comparison
plt.subplot(2, 2, 3)
percentages = [relevant_percentage, irrelevant_percentage]
bars = plt.bar(categories, percentages, color=['lightgreen', 'lightcoral'], alpha=0.7)
plt.title('Feature Percentages')
plt.ylabel('Percentage (%)')

# Add percentage labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# Text summary
plt.subplot(2, 2, 4)
plt.axis('off')
summary_text = f"""Feature Analysis Summary:

Total Features: {total_features}
Relevant Features: {relevant_features} ({relevant_percentage:.1f}%)
Irrelevant Features: {irrelevant_features} ({irrelevant_percentage:.1f}%)

Key Insight:
{irrelevant_percentage:.1f}% of features add noise
to the dataset and may harm model performance."""
plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='center', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(save_dir, 'feature_distribution_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "=" * 80)
print("TASK 2: TOTAL NOISE LEVEL")
print("=" * 80)

# Task 2: Total noise level
total_noise = irrelevant_features * noise_per_irrelevant_feature
total_noise_percentage = total_noise * 100

print(f"Calculation:")
print(f"  Each irrelevant feature adds {noise_per_irrelevant_feature:.1%} noise")
print(f"  Number of irrelevant features = {irrelevant_features}")
print(f"  Total noise = irrelevant_features $\\times$ noise_per_feature")
print(f"  Total noise = {irrelevant_features} $\\times$ {noise_per_irrelevant_feature:.1%}")
print(f"  Total noise = {total_noise:.3f} = {total_noise_percentage:.1f}%")

print(f"\nAnswer: Total noise level is {total_noise_percentage:.1f}%")

# Create visualization for Task 2
plt.figure(figsize=(15, 6))

# Noise accumulation visualization
plt.subplot(1, 3, 1)
feature_indices = np.arange(1, total_features + 1)
noise_levels = np.zeros(total_features)

# Set noise for irrelevant features (assuming first 80 are irrelevant for visualization)
for i in range(irrelevant_features):
    noise_levels[i] = noise_per_irrelevant_feature

plt.bar(feature_indices, noise_levels, color='lightcoral', alpha=0.7, 
        label=f'Noise per feature: {noise_per_irrelevant_feature:.1%}')
plt.axhline(y=total_noise, color='red', linestyle='--', linewidth=2, 
            label=f'Total noise: {total_noise:.3f}')
plt.xlabel('Feature Index')
plt.ylabel('Noise Level')
plt.title('Noise Accumulation Across Features')
plt.legend()
plt.grid(True, alpha=0.3)

# Cumulative noise
plt.subplot(1, 3, 2)
cumulative_noise = np.cumsum(noise_levels)
plt.plot(feature_indices, cumulative_noise, 'b-', linewidth=2, marker='o', markersize=4)
plt.axhline(y=total_noise, color='red', linestyle='--', linewidth=2, 
            label=f'Total noise: {total_noise:.3f}')
plt.xlabel('Feature Index')
plt.ylabel('Cumulative Noise')
plt.title('Cumulative Noise Accumulation')
plt.legend()
plt.grid(True, alpha=0.3)

# Noise vs Signal comparison
plt.subplot(1, 3, 3)
signal = relevant_features  # Assuming each relevant feature contributes 1 unit of signal
noise_units = total_noise * 100  # Convert to same scale as signal

categories = ['Signal (Relevant)', 'Noise (Irrelevant)']
values = [signal, noise_units]
colors = ['lightgreen', 'lightcoral']

bars = plt.bar(categories, values, color=colors, alpha=0.7)
plt.title('Signal vs Noise Comparison')
plt.ylabel('Magnitude')

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(save_dir, 'noise_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "=" * 80)
print("TASK 3: SIGNAL-TO-NOISE RATIO")
print("=" * 80)

# Task 3: Signal-to-noise ratio
signal_all_features = relevant_features  # Signal from relevant features
noise_all_features = total_noise  # Noise from irrelevant features
snr_all_features = signal_all_features / noise_all_features if noise_all_features > 0 else float('inf')

signal_relevant_only = relevant_features  # Signal from relevant features only
noise_relevant_only = 0  # No noise when using only relevant features
snr_relevant_only = float('inf') if noise_relevant_only == 0 else signal_relevant_only / noise_relevant_only

print(f"Signal-to-Noise Ratio Analysis:")
print(f"\nWith All Features:")
print(f"  Signal = {signal_all_features} (from relevant features)")
print(f"  Noise = {noise_all_features:.3f} (from irrelevant features)")
print(f"  SNR = Signal / Noise = {signal_all_features} / {noise_all_features:.3f}")
print(f"  SNR = {snr_all_features:.2f}")

print(f"\nWith Relevant Features Only:")
print(f"  Signal = {signal_relevant_only} (from relevant features)")
print(f"  Noise = {noise_relevant_only} (no irrelevant features)")
print(f"  SNR = Signal / Noise = {signal_relevant_only} / {noise_relevant_only}")
print(f"  SNR = Inf (infinite, no noise)")

print(f"\nImprovement Factor:")
if noise_all_features > 0:
    improvement = snr_relevant_only / snr_all_features if snr_all_features != 0 else float('inf')
    print(f"  Improvement = SNR_relevant_only / SNR_all_features")
    print(f"  Improvement = Inf / {snr_all_features:.2f} = Inf")
    print(f"  Using only relevant features eliminates all noise!")

# Create visualization for Task 3
plt.figure(figsize=(15, 6))

# SNR comparison
plt.subplot(1, 3, 1)
scenarios = ['All Features', 'Relevant Only']
snr_values = [snr_all_features, 100]  # Cap at 100 for visualization
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(scenarios, snr_values, color=colors, alpha=0.7)
plt.title('Signal-to-Noise Ratio Comparison')
plt.ylabel('SNR (Signal/Noise)')

# Add value labels
for bar, value, scenario in zip(bars, snr_values, scenarios):
    height = bar.get_height()
    if scenario == 'Relevant Only':
        label = 'Inf'
    else:
        label = f'{value:.2f}'
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             label, ha='center', va='bottom', fontweight='bold')

# Signal vs Noise visualization
plt.subplot(1, 3, 2)
scenarios = ['All Features', 'Relevant Only']
signal_values = [signal_all_features, signal_relevant_only]
noise_values = [noise_all_features, noise_relevant_only]

x = np.arange(len(scenarios))
width = 0.35

bars1 = plt.bar(x - width/2, signal_values, width, label='Signal', color='lightgreen', alpha=0.7)
bars2 = plt.bar(x + width/2, noise_values, width, label='Noise', color='lightcoral', alpha=0.7)

plt.xlabel('Scenario')
plt.ylabel('Magnitude')
plt.title('Signal vs Noise by Scenario')
plt.xticks(x, scenarios)
plt.legend()

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# Feature efficiency
plt.subplot(1, 3, 3)
efficiency_all = (signal_all_features / total_features) * 100
efficiency_relevant = (signal_relevant_only / relevant_features) * 100

efficiency_scenarios = ['All Features', 'Relevant Only']
efficiency_values = [efficiency_all, efficiency_relevant]
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(efficiency_scenarios, efficiency_values, color=colors, alpha=0.7)
plt.title('Feature Efficiency')
plt.ylabel('Efficiency (%)')

# Add value labels
for bar, value in zip(bars, efficiency_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(save_dir, 'signal_noise_ratio_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "=" * 80)
print("TASK 4: BINOMIAL DISTRIBUTION CALCULATIONS")
print("=" * 80)

# Task 4: Binomial distribution calculations
n_features_selected = 20
p_relevant = relevant_features / total_features  # Probability of selecting a relevant feature
k_exact = 15  # Exactly 15 relevant features

print(f"Binomial Distribution Parameters:")
print(f"  n = {n_features_selected} (features selected)")
print(f"  p = {p_relevant:.3f} (probability of relevant feature)")
print(f"  k = {k_exact} (exactly relevant features)")

print(f"\nCalculations:")

# Probability of exactly 15 relevant features
prob_exact_15 = stats.binom.pmf(k_exact, n_features_selected, p_relevant)
print(f"\n1. Probability of exactly {k_exact} relevant features:")
print(f"   P(X = {k_exact}) = C({n_features_selected}, {k_exact}) $\\times$ p^{k_exact} $\\times$ (1-p)^{n_features_selected - k_exact}")
print(f"   P(X = {k_exact}) = C({n_features_selected}, {k_exact}) $\\times$ {p_relevant:.3f}^{k_exact} $\\times$ {1-p_relevant:.3f}^{n_features_selected - k_exact}")

# Calculate combination manually
from math import comb
combination = comb(n_features_selected, k_exact)
print(f"   C({n_features_selected}, {k_exact}) = {combination}")

# Calculate probability manually
prob_manual = combination * (p_relevant ** k_exact) * ((1 - p_relevant) ** (n_features_selected - k_exact))
print(f"   P(X = {k_exact}) = {combination} $\\times$ {p_relevant ** k_exact:.6f} $\\times$ {((1 - p_relevant) ** (n_features_selected - k_exact)):.6f}")
print(f"   P(X = {k_exact}) = {prob_manual:.6f}")

print(f"\n   Using scipy.stats.binom.pmf: {prob_exact_15:.6f}")
print(f"   Answer: P(X = 15) = {prob_exact_15:.6f}")

# Probability of at least 15 relevant features
prob_at_least_15 = 1 - stats.binom.cdf(k_exact - 1, n_features_selected, p_relevant)
print(f"\n2. Probability of at least {k_exact} relevant features:")
print(f"   P(X $\\geq$ {k_exact}) = 1 - P(X < {k_exact})")
print(f"   P(X $\\geq$ {k_exact}) = 1 - P(X $\\leq$ {k_exact - 1})")

# Calculate cumulative probability manually
cumulative_prob = 0
for i in range(k_exact):
    cumulative_prob += stats.binom.pmf(i, n_features_selected, p_relevant)

prob_at_least_manual = 1 - cumulative_prob
print(f"   P(X $\\leq$ {k_exact - 1}) = $\\Sigma$ P(X = i) for i = 0 to {k_exact - 1}")
print(f"   P(X $\\leq$ {k_exact - 1}) = {cumulative_prob:.6f}")
print(f"   P(X $\\geq$ {k_exact}) = 1 - {cumulative_prob:.6f} = {prob_at_least_manual:.6f}")

print(f"\n   Using scipy.stats.binom.sf: {prob_at_least_15:.6f}")
print(f"   Answer: P(X $\\geq$ 15) = {prob_at_least_15:.6f}")

# Create separate visualizations for Task 4

# 1. Binomial PMF
plt.figure(figsize=(10, 6))
k_values = np.arange(0, n_features_selected + 1)
pmf_values = stats.binom.pmf(k_values, n_features_selected, p_relevant)

plt.bar(k_values, pmf_values, color='skyblue', alpha=0.7, edgecolor='navy')
plt.axvline(x=k_exact, color='red', linestyle='--', linewidth=2, 
            label=f'P(X = {k_exact}) = {prob_exact_15:.6f}')
plt.xlabel('Number of Relevant Features ($k$)')
plt.ylabel('Probability $P(X = k)$')
plt.title(f'Binomial PMF: n={n_features_selected}, p={p_relevant:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
plt.savefig(os.path.join(save_dir, 'binomial_pmf.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Highlight exact probability
plt.figure(figsize=(8, 6))
highlight_k = [k_exact]
highlight_prob = [prob_exact_15]

plt.bar(highlight_k, highlight_prob, color='red', alpha=0.8, edgecolor='darkred')
plt.xlabel('Number of Relevant Features ($k$)')
plt.ylabel('Probability $P(X = k)$')
plt.title(f'Probability of Exactly {k_exact} Relevant Features')
plt.grid(True, alpha=0.3)
try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
plt.savefig(os.path.join(save_dir, 'binomial_exact_probability.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Cumulative distribution function
plt.figure(figsize=(10, 6))
cdf_values = stats.binom.cdf(k_values, n_features_selected, p_relevant)

plt.plot(k_values, cdf_values, 'b-', linewidth=2, marker='o', markersize=4)
plt.axvline(x=k_exact - 1, color='orange', linestyle='--', linewidth=2, 
            label=f'P(X $\\leq$ {k_exact - 1}) = {cumulative_prob:.6f}')
plt.axvline(x=k_exact, color='red', linestyle='--', linewidth=2, 
            label=f'P(X $\\geq$ {k_exact}) = {prob_at_least_15:.6f}')
plt.xlabel('Number of Relevant Features ($k$)')
plt.ylabel('Cumulative Probability $P(X \\leq k)$')
plt.title(f'Binomial CDF: n={n_features_selected}, p={p_relevant:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
plt.savefig(os.path.join(save_dir, 'binomial_cdf.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Survival function (1 - CDF)
plt.figure(figsize=(10, 6))
sf_values = 1 - cdf_values

plt.plot(k_values, sf_values, 'g-', linewidth=2, marker='s', markersize=4)
plt.axvline(x=k_exact, color='red', linestyle='--', linewidth=2, 
            label=f'P(X $\\geq$ {k_exact}) = {prob_at_least_15:.6f}')
plt.xlabel('Number of Relevant Features ($k$)')
plt.ylabel('Survival Probability $P(X \\geq k)$')
plt.title(f'Binomial Survival Function: n={n_features_selected}, p={p_relevant:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
plt.savefig(os.path.join(save_dir, 'binomial_survival_function.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Probability comparison
plt.figure(figsize=(10, 6))
comparison_labels = ['P(X = 15)', 'P(X $\\geq$ 15)']
comparison_values = [prob_exact_15, prob_at_least_15]
colors = ['red', 'green']

# Handle case where probabilities are extremely small (effectively zero)
if max(comparison_values) < 1e-10:
    # Create a visualization showing that probabilities are essentially zero
    plt.figure(figsize=(10, 6))
    
    # Create bars with a small height for visualization
    display_values = [1e-6, 1e-6]  # Small values for display purposes
    bars = plt.bar(comparison_labels, display_values, color=colors, alpha=0.7)
    
    plt.title('Probability Comparison (Extremely Small Values)', fontsize=14, fontweight='bold')
    plt.xlabel('Probability Type', fontsize=12)
    plt.ylabel('Probability Value (Log Scale)', fontsize=12)
    
    # Set y-axis to log scale to better show small values
    plt.yscale('log')
    plt.ylim(1e-7, 1e-5)
    
    # Add value labels showing actual values
    for i, (bar, value) in enumerate(zip(bars, comparison_values)):
        plt.text(bar.get_x() + bar.get_width()/2., display_values[i] * 2,
                 f'Actual: {value:.2e}\\n≈ 0.000000', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add explanatory text
    plt.text(0.5, 0.8, 'Note: Both probabilities are effectively zero\\n(less than 1 in a million)', 
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
             
else:
    bars = plt.bar(comparison_labels, comparison_values, color=colors, alpha=0.7)
    plt.title('Probability Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Probability Type', fontsize=12)
    plt.ylabel('Probability Value', fontsize=12)

    # Add value labels
    for bar, value in zip(bars, comparison_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(comparison_values) * 0.05,
                 f'{value:.6f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Set y-axis limits to make values visible
    plt.ylim(0, max(comparison_values) * 1.2)

# Add grid for better readability
plt.grid(True, alpha=0.3)

plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
plt.savefig(os.path.join(save_dir, 'binomial_probability_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Expected value and variance
plt.figure(figsize=(10, 8))
expected_value = n_features_selected * p_relevant
variance = n_features_selected * p_relevant * (1 - p_relevant)
std_dev = np.sqrt(variance)

stats_text = f"""Binomial Distribution Statistics:

$n = {n_features_selected}$ features
$p = {p_relevant:.3f}$ (probability of relevant)

Expected Value $E[X] = n \\times p$
$E[X] = {n_features_selected} \\times {p_relevant:.3f} = {expected_value:.1f}$

Variance $\\mathrm{{Var}}[X] = n \\times p \\times (1-p)$
$\\mathrm{{Var}}[X] = {n_features_selected} \\times {p_relevant:.3f} \\times {1-p_relevant:.3f} = {variance:.3f}$

Standard Deviation $\\sigma = \\sqrt{{\\mathrm{{Var}}[X]}}$
$\\sigma = \\sqrt{{{variance:.3f}}} = {std_dev:.3f}$

For $k = {k_exact}$:
$P(X = {k_exact}) = {prob_exact_15:.6f}$
$P(X \\geq {k_exact}) = {prob_at_least_15:.6f}$"""

plt.axis('off')
plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='center', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

try:
    plt.tight_layout()
except:
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
plt.savefig(os.path.join(save_dir, 'binomial_statistics.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create summary visualization
plt.figure(figsize=(18, 12))

# Summary of all results
summary_data = {
    'Task 1: Irrelevant Features (%)': [irrelevant_percentage],
    'Task 2: Total Noise (%)': [total_noise_percentage],
    'Task 3: SNR (All Features)': [snr_all_features],
    'Task 4: P(X = 15)': [prob_exact_15 * 1000000],  # Scale for visibility
    'Task 4: P(X $\\geq$ 15)': [prob_at_least_15 * 1000000]  # Scale for visibility
}

tasks = list(summary_data.keys())
values = [summary_data[task][0] for task in tasks]
colors = ['lightcoral', 'lightcoral', 'lightcoral', 'lightgreen', 'lightgreen']

bars = plt.bar(tasks, values, color=colors, alpha=0.7)
plt.title('Summary of All Task Results', fontsize=16, fontweight='bold')
plt.ylabel('Value')
plt.xticks(rotation=45, ha='right')

# Add value labels
for bar, value, task in zip(bars, values, tasks):
    height = bar.get_height()
    if 'P(X' in task:
        # For probability tasks, show original probability
        if 'P(X = 15)' in task:
            original_value = prob_exact_15
        else:
            original_value = prob_at_least_15
        label = f'{original_value:.6f}'
    else:
        label = f'{value:.1f}'
    
    plt.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
             label, ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.85, wspace=0.3, hspace=0.3)
plt.savefig(os.path.join(save_dir, 'summary_all_tasks.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "=" * 80)
print("SUMMARY OF ALL RESULTS")
print("=" * 80)
print(f"Task 1: {irrelevant_percentage:.1f}% of features are irrelevant")
print(f"Task 2: Total noise level is {total_noise_percentage:.1f}%")
print(f"Task 3: SNR with all features = {snr_all_features:.2f}, with relevant only = Inf")
print(f"Task 4: P(X = 15) = {prob_exact_15:.6f}, P(X $\\geq$ 15) = {prob_at_least_15:.6f}")

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
