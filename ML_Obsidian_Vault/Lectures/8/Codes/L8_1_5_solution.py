import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 5: IRRELEVANT FEATURES IMPACT ANALYSIS")
print("=" * 80)

# Given parameters
total_samples = 1000
total_features = 100
relevant_features = 20
irrelevant_features = total_features - relevant_features
noise_per_feature = 0.01  # 1% per irrelevant feature

print(f"Dataset Parameters:")
print(f"  Total samples: {total_samples}")
print(f"  Total features: {total_features}")
print(f"  Relevant features: {relevant_features}")
print(f"  Irrelevant features: {irrelevant_features}")
print(f"  Noise per irrelevant feature: {noise_per_feature:.1%}")

print("\n" + "=" * 80)
print("STEP-BY-STEP SOLUTION")
print("=" * 80)

# Task 1: Percentage of irrelevant features
print("\n1. PERCENTAGE OF IRRELEVANT FEATURES")
print("-" * 40)
print("Step-by-step calculation:")
print("Let:")
print("  • n_total = total number of features")
print("  • n_relevant = number of relevant features")
print("  • n_irrelevant = number of irrelevant features")
print()
print("We know that:")
print("  n_irrelevant = n_total - n_relevant")
print(f"  n_irrelevant = {total_features} - {relevant_features} = {irrelevant_features}")
print()
print("The percentage of irrelevant features is:")
print("  Percentage = (n_irrelevant / n_total) × 100%")
print(f"  Percentage = ({irrelevant_features} / {total_features}) × 100%")

irrelevant_percentage = (irrelevant_features / total_features) * 100
print(f"  Percentage = {irrelevant_features/total_features:.2f} × 100% = {irrelevant_percentage:.1f}%")
print()
print(f"Answer: {irrelevant_percentage:.1f}% of features are irrelevant")

# Task 2: Total noise level
print("\n2. TOTAL NOISE LEVEL")
print("-" * 40)
print("Step-by-step calculation:")
print("Let:")
print("  • η = noise contribution per irrelevant feature")
print("  • n_irrelevant = number of irrelevant features")
print("  • N_total = total noise level")
print()
print("Given:")
print(f"  η = {noise_per_feature:.1%} per feature")
print(f"  n_irrelevant = {irrelevant_features} features")
print()
print("Assuming noise accumulates linearly:")
print("  N_total = n_irrelevant × η")
print(f"  N_total = {irrelevant_features} × {noise_per_feature:.1%}")

total_noise = irrelevant_features * noise_per_feature
print(f"  N_total = {irrelevant_features} × {noise_per_feature:.3f} = {total_noise:.3f}")
print(f"  N_total = {total_noise:.1%}")
print()
print(f"Answer: Total noise level is {total_noise:.1%}")

# Task 3: Model performance and training time impact
print("\n3. IMPACT ON MODEL PERFORMANCE AND TRAINING TIME")
print("-" * 40)
print("Model Performance Impact:")
print(f"  • {irrelevant_percentage:.1f}% of features contribute only noise")
print(f"  • Total noise level: {total_noise:.1%}")
print(f"  • Signal-to-noise ratio decreases significantly")
print(f"  • Model may overfit to noise in irrelevant features")
print(f"  • Generalization performance likely to suffer")

print("\nTraining Time Impact:")
print(f"  • Computational complexity: O(n²) to O(n³) where n = {total_features}")
print(f"  • Memory usage increases with feature count")
print(f"  • Convergence may take longer due to noise")
print(f"  • Risk of local minima increases")

# Task 4: Signal-to-noise ratio comparison
print("\n4. SIGNAL-TO-NOISE RATIO COMPARISON")
print("-" * 40)
print("Step-by-step calculation:")
print("Let:")
print("  • S = signal strength (relevant features)")
print("  • N = noise level (irrelevant features)")
print("  • SNR = Signal-to-Noise Ratio")
print()
print("Case 1: Using all features")
print("-------")
print("  S = n_relevant")
print("  N = n_irrelevant")
print(f"  S = {relevant_features}")
print(f"  N = {irrelevant_features}")
print()
print("  SNR = S / N")
print(f"  SNR = {relevant_features} / {irrelevant_features}")

snr_all_simple = relevant_features / irrelevant_features
print(f"  SNR = {snr_all_simple:.3f}")
print()
print("Case 2: Using only relevant features")
print("-------")
print("  S = n_relevant")
print("  N = 0 (no irrelevant features)")
print(f"  S = {relevant_features}")
print("  N = 0")
print()
print("  SNR = S / N = S / 0")
print(f"  SNR = {relevant_features} / 0 = ∞ (infinite)")
print()
print("Summary:")
print(f"  • SNR with all features: {snr_all_simple:.3f}")
print("  • SNR with relevant only: ∞")

# Task 5: Probability of random selection
print("\n5. PROBABILITY OF SELECTING ONLY RELEVANT FEATURES BY RANDOM CHANCE")
print("-" * 40)
print("Step-by-step calculation:")
print("This is a hypergeometric distribution problem.")
print()
print("Let:")
print("  • N = total number of features")
print("  • K = number of relevant features")
print("  • n = number of features we select")
print("  • k = number of relevant features we want to select")
print()
print("Given:")
print(f"  • N = {total_features} (total features)")
print(f"  • K = {relevant_features} (relevant features)")
print(f"  • n = {relevant_features} (features we select)")
print(f"  • k = {relevant_features} (relevant features we want)")
print()
print("The hypergeometric probability formula is:")
print("  P(X = k) = [C(K,k) × C(N-K,n-k)] / C(N,n)")
print()
print("Where C(a,b) is the binomial coefficient 'a choose b'")
print()
print("Substituting our values:")
print(f"  P(X = {relevant_features}) = [C({relevant_features},{relevant_features}) × C({total_features}-{relevant_features},{relevant_features}-{relevant_features})] / C({total_features},{relevant_features})")
print(f"  P(X = {relevant_features}) = [C({relevant_features},{relevant_features}) × C({irrelevant_features},0)] / C({total_features},{relevant_features})")
print()
print("Calculating each binomial coefficient:")
print(f"  • C({relevant_features},{relevant_features}) = 1 (choosing all from all)")
print(f"  • C({irrelevant_features},0) = 1 (choosing none from irrelevant)")

# Using hypergeometric distribution
def hypergeometric_prob(k, N, K, n):
    """Calculate hypergeometric probability P(X = k)"""
    return (comb(K, k) * comb(N-K, n-k)) / comb(N, n)

prob_exact = hypergeometric_prob(relevant_features, total_features, relevant_features, relevant_features)
total_combinations = comb(total_features, relevant_features)
print(f"  • C({total_features},{relevant_features}) = {total_combinations:.0f}")
print()
print("Therefore:")
print(f"  P(X = {relevant_features}) = (1 × 1) / {total_combinations:.0f}")
print(f"  P(X = {relevant_features}) = 1 / {total_combinations:.0f}")
print(f"  P(X = {relevant_features}) = {prob_exact:.2e}")
print()
print(f"Answer: The probability is {prob_exact:.2e} (extremely small!)")

# Task 6: SNR calculation with specific values
print("\n6. SNR CALCULATION WITH SPECIFIC VALUES")
print("-" * 40)
print("Step-by-step calculation:")
print("Now we consider specific variance values for signal and noise.")
print()
print("Let:")
print("  • σ²_s = variance of signal features")
print("  • σ²_n = variance of noise features")
print("  • S_total = total signal strength")
print("  • N_total = total noise strength")
print("  • SNR = Signal-to-Noise Ratio")
print()

sigma_s_squared = 4  # Signal variance
sigma_n_squared = 1  # Noise variance

print("Given:")
print(f"  • σ²_s = {sigma_s_squared}")
print(f"  • σ²_n = {sigma_n_squared}")
print(f"  • n_relevant = {relevant_features}")
print(f"  • n_irrelevant = {irrelevant_features}")
print()
print("Case 1: Using all features")
print("-------")
print("Total signal strength:")
print("  S_total = n_relevant × σ²_s")
print(f"  S_total = {relevant_features} × {sigma_s_squared} = {relevant_features * sigma_s_squared}")
print()
print("Total noise strength:")
print("  N_total = n_irrelevant × σ²_n")
print(f"  N_total = {irrelevant_features} × {sigma_n_squared} = {irrelevant_features * sigma_n_squared}")
print()
print("SNR calculation:")

# With all features
signal_all = relevant_features * sigma_s_squared
noise_all = irrelevant_features * sigma_n_squared
snr_all = signal_all / noise_all

print("  SNR = S_total / N_total")
print(f"  SNR = {signal_all} / {noise_all} = {snr_all:.3f}")
print()
print("Case 2: Using only relevant features")
print("-------")
print("Total signal strength:")
print("  S_total = n_relevant × σ²_s")
print(f"  S_total = {relevant_features} × {sigma_s_squared} = {relevant_features * sigma_s_squared}")
print()
print("Total noise strength:")
print("  N_total = 0 × σ²_n = 0 (no irrelevant features)")
print()
print("SNR calculation:")

# With only relevant features
signal_relevant = relevant_features * sigma_s_squared
noise_relevant = 0  # No irrelevant features
snr_relevant = float('inf') if noise_relevant == 0 else signal_relevant / noise_relevant

print("  SNR = S_total / N_total")
print(f"  SNR = {signal_relevant} / 0 = ∞ (infinite)")
print()
print("SNR Improvement Analysis:")
print("-------")
print("Improvement factor = SNR_relevant / SNR_all")
print(f"Improvement factor = ∞ / {snr_all:.3f} = ∞")
print()
print("This means:")
print("  • The SNR improvement is infinite")
print("  • Signal quality becomes perfect when noise is eliminated")
print("  • Any finite SNR becomes infinite when denominator → 0")
print()
print("Summary:")
print(f"  • SNR with all features: {snr_all:.3f}")
print("  • SNR with relevant only: ∞")
print("  • Improvement: Infinite")

print("\n" + "=" * 80)
print("VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Feature Distribution
plt.figure(figsize=(15, 10))

# Subplot 1: Pie chart of feature distribution
plt.subplot(2, 3, 1)
labels = ['Relevant Features', 'Irrelevant Features']
sizes = [relevant_features, irrelevant_features]
colors = ['#2E8B57', '#DC143C']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Feature Distribution', fontsize=14, fontweight='bold')

# Subplot 2: Bar chart of feature counts
plt.subplot(2, 3, 2)
categories = ['Relevant', 'Irrelevant']
counts = [relevant_features, irrelevant_features]
bars = plt.bar(categories, counts, color=['#2E8B57', '#DC143C'], alpha=0.7)
plt.title('Feature Counts', fontsize=14, fontweight='bold')
plt.ylabel('Number of Features')
plt.ylim(0, total_features + 5)

# Add value labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', va='bottom', fontweight='bold')

# Subplot 3: Noise accumulation
plt.subplot(2, 3, 3)
x_noise = np.arange(1, irrelevant_features + 1)
cumulative_noise = x_noise * noise_per_feature

plt.plot(x_noise, cumulative_noise, 'r-', linewidth=2, marker='o')
plt.fill_between(x_noise, cumulative_noise, alpha=0.3, color='red')
plt.title('Cumulative Noise Accumulation', fontsize=14, fontweight='bold')
plt.xlabel('Number of Irrelevant Features')
plt.ylabel('Total Noise Level')
plt.grid(True, alpha=0.3)

# Subplot 4: SNR comparison
plt.subplot(2, 3, 4)
snr_categories = ['All Features', 'Relevant Only']
snr_values = [snr_all, snr_relevant if snr_relevant != float('inf') else 100]

# Handle infinite SNR for plotting
snr_plot_values = [snr_all, 100]  # Cap at 100 for visualization
bars = plt.bar(snr_categories, snr_plot_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
plt.title('Signal-to-Noise Ratio Comparison', fontsize=14, fontweight='bold')
plt.ylabel('SNR Value')
plt.ylim(0, 110)

# Add value labels on bars
for bar, value in zip(bars, snr_values):
    if value == float('inf'):
        label = '∞'
    else:
        label = f'{value:.2f}'
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             label, ha='center', va='bottom', fontweight='bold')

# Subplot 5: Feature selection probability
plt.subplot(2, 3, 5)
# Calculate probabilities for different numbers of relevant features selected
k_values = np.arange(0, relevant_features + 1)
probabilities = [hypergeometric_prob(k, total_features, relevant_features, relevant_features) 
                for k in k_values]

plt.bar(k_values, probabilities, color='#9B59B6', alpha=0.7)
plt.title('Probability of Selecting k Relevant Features', fontsize=14, fontweight='bold')
plt.xlabel('Number of Relevant Features Selected')
plt.ylabel('Probability')
plt.xticks(k_values[::2])  # Show every other tick to avoid crowding

# Highlight the case where all relevant features are selected
max_prob_idx = np.argmax(probabilities)
plt.bar(k_values[max_prob_idx], probabilities[max_prob_idx], color='#E74C3C', alpha=0.8)

# Subplot 6: Model complexity vs performance trade-off
plt.subplot(2, 3, 6)
feature_counts = np.arange(20, 101, 5)
noise_levels = (feature_counts - relevant_features) * noise_per_feature
signal_strength = relevant_features * sigma_s_squared
snr_values_plot = signal_strength / (noise_levels + 1e-10)  # Avoid division by zero

plt.plot(feature_counts, snr_values_plot, 'b-', linewidth=2, marker='o')
plt.axvline(x=relevant_features, color='r', linestyle='--', alpha=0.7, 
            label=f'Optimal: {relevant_features} features')
plt.title('SNR vs Number of Features', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('Signal-to-Noise Ratio')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_analysis_overview.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Detailed SNR Analysis
plt.figure(figsize=(15, 10))

# Subplot 1: Signal and noise components
plt.subplot(2, 2, 1)
components = ['Signal\n(Relevant Features)', 'Noise\n(Irrelevant Features)']
values = [signal_all, noise_all]
colors = ['#2E8B57', '#DC143C']

bars = plt.bar(components, values, color=colors, alpha=0.7)
plt.title('Signal vs Noise Components', fontsize=14, fontweight='bold')
plt.ylabel('Magnitude')
plt.ylim(0, max(values) * 1.1)

# Add value labels
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(value), ha='center', va='bottom', fontweight='bold')

# Subplot 2: SNR improvement visualization
plt.subplot(2, 2, 2)
improvement_categories = ['Current SNR\n(All Features)', 'Improved SNR\n(Relevant Only)']
snr_current = snr_all
snr_improved = 100  # Cap for visualization

bars = plt.bar(improvement_categories, [snr_current, snr_improved], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
plt.title('SNR Improvement', fontsize=14, fontweight='bold')
plt.ylabel('SNR Value')
plt.ylim(0, 110)

# Add value labels
plt.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 2,
         f'{snr_current:.2f}', ha='center', va='bottom', fontweight='bold')
plt.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 2,
         '∞', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Feature selection probability distribution
plt.subplot(2, 2, 3)
k_values_extended = np.arange(0, relevant_features + 1)
probabilities_extended = [hypergeometric_prob(k, total_features, relevant_features, relevant_features) 
                         for k in k_values_extended]

plt.bar(k_values_extended, probabilities_extended, color='#9B59B6', alpha=0.7)
plt.title('Feature Selection Probability Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Number of Relevant Features Selected')
plt.ylabel('Probability')
plt.xticks(k_values_extended)

# Highlight the optimal case
plt.bar(k_values_extended[-1], probabilities_extended[-1], color='#E74C3C', alpha=0.8)
plt.text(k_values_extended[-1], probabilities_extended[-1] + 1e-6,
         f'P(X={relevant_features})\n={probabilities_extended[-1]:.2e}', 
         ha='center', va='bottom', fontsize=10)

# Subplot 4: Cumulative noise impact
plt.subplot(2, 2, 4)
x_range = np.arange(0, irrelevant_features + 1)
noise_impact = x_range * noise_per_feature
signal_quality = signal_all / (noise_impact + 1e-10)

plt.plot(x_range, signal_quality, 'g-', linewidth=2, marker='o', label='Signal Quality')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Optimal: No irrelevant features')
plt.title('Signal Quality vs Irrelevant Features', fontsize=14, fontweight='bold')
plt.xlabel('Number of Irrelevant Features')
plt.ylabel('Signal Quality (SNR)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_snr_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Training Impact Analysis
plt.figure(figsize=(15, 8))

# Subplot 1: Computational complexity
plt.subplot(1, 3, 1)
feature_range = np.arange(20, 101, 5)
complexity_quadratic = feature_range ** 2
complexity_cubic = feature_range ** 3

plt.plot(feature_range, complexity_quadratic, 'b-', linewidth=2, marker='o', label='O(n²)')
plt.plot(feature_range, complexity_cubic, 'r-', linewidth=2, marker='s', label='O(n³)')
plt.axvline(x=relevant_features, color='g', linestyle='--', alpha=0.7, 
            label=f'Optimal: {relevant_features} features')
plt.title('Computational Complexity', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('Complexity (relative)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Memory usage
plt.subplot(1, 3, 2)
memory_usage = feature_range * total_samples * 8 / (1024 * 1024)  # MB assuming float64

plt.plot(feature_range, memory_usage, 'purple', linewidth=2, marker='o')
plt.axvline(x=relevant_features, color='g', linestyle='--', alpha=0.7, 
            label=f'Optimal: {relevant_features} features')
plt.title('Memory Usage', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('Memory (MB)')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Convergence time estimation
plt.subplot(1, 3, 3)
# Estimate convergence time based on feature count and noise
base_time = 100  # Base convergence time
noise_factor = 1 + (feature_range - relevant_features) * noise_per_feature
convergence_time = base_time * noise_factor

plt.plot(feature_range, convergence_time, 'orange', linewidth=2, marker='o')
plt.axvline(x=relevant_features, color='g', linestyle='--', alpha=0.7, 
            label=f'Optimal: {relevant_features} features')
plt.title('Estimated Convergence Time', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('Convergence Time (relative)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_impact_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"1. Irrelevant features: {irrelevant_percentage:.1f}%")
print(f"2. Total noise level: {total_noise:.1%}")
print(f"3. SNR with all features: {snr_all:.2f}")
print(f"4. SNR with relevant features only: ∞")
print(f"5. Probability of random selection: {prob_exact:.2e}")
print(f"6. SNR improvement: Infinite (∞)")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("• Feature selection is crucial: 80% of features are irrelevant")
print("• Noise accumulates linearly with irrelevant features")
print("• SNR improves dramatically with proper feature selection")
print("• Random feature selection is extremely unlikely to succeed")
print("• Computational and memory costs increase significantly")
print("• Model performance suffers due to noise and overfitting")

# Don't show plots, just save them
print(f"\nAll visualizations have been saved to: {save_dir}")
print("The plots demonstrate the dramatic impact of irrelevant features on model performance.")
