import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
from scipy.stats import spearmanr
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 17: RESOURCE CONSTRAINTS IN FEATURE SELECTION")
print("=" * 80)

# ============================================================================
# TASK 1: Selection strategies with limited time
# ============================================================================
print("\n" + "="*60)
print("TASK 1: Selection Strategies with Limited Time")
print("="*60)

strategies = {
    "Univariate Selection": "Fastest method, evaluates each feature independently",
    "Correlation-based": "Quick removal of highly correlated features",
    "Variance Threshold": "Very fast, removes low-variance features",
    "Mutual Information": "Moderate speed, good for non-linear relationships",
    "Recursive Feature Elimination": "Slower but more thorough",
    "L1 Regularization": "Built into model training, moderate speed"
}

print("Recommended strategies for limited time:")
for i, (strategy, description) in enumerate(strategies.items(), 1):
    print(f"{i}. {strategy}: {description}")

# ============================================================================
# TASK 2: Power law distribution calculations
# ============================================================================
print("\n" + "="*60)
print("TASK 2: Power Law Distribution Calculations")
print("="*60)

def feature_evaluation_time(i, alpha=0.8, base_time=0.1):
    """Calculate evaluation time for i-th feature using power law"""
    return base_time * (i ** alpha)

def total_evaluation_time(n_features, alpha=0.8, base_time=0.1):
    """Calculate total time to evaluate first n features"""
    # Sum of power series: sum(i^alpha) for i from 1 to n
    # For alpha = 0.8, this is approximately sum(i^0.8)
    # We can approximate this using the integral or direct summation
    total_time = 0
    for i in range(1, n_features + 1):
        total_time += feature_evaluation_time(i, alpha, base_time)
    return total_time

def approximate_power_series_sum(n, alpha=0.8):
    """Approximate sum of power series using integral approximation"""
    # For alpha != -1, sum(i^alpha) ≈ (n^(alpha+1))/(alpha+1) + n^alpha/2 + C
    if alpha == -1:
        return np.log(n) + 0.5772  # Euler-Mascheroni constant
    else:
        return (n**(alpha + 1)) / (alpha + 1) + (n**alpha) / 2

# Calculate times for different numbers of features
feature_counts = [100, 500, 1000]
print(f"Feature evaluation time follows: t_i = 0.1 × i^0.8 seconds")
print(f"Where i is the feature index (1, 2, 3, ...)")

print("\nCalculating total evaluation times:")
print("-" * 50)
for n in feature_counts:
    # Exact calculation
    exact_time = total_evaluation_time(n)
    # Approximate calculation
    approx_time = 0.1 * approximate_power_series_sum(n, 0.8)
    
    print(f"First {n} features:")
    print(f"  Exact calculation: {exact_time:.2f} seconds ({exact_time/60:.2f} minutes)")
    print(f"  Approximate calculation: {approx_time:.2f} seconds ({approx_time/60:.2f} minutes)")
    print(f"  Difference: {abs(exact_time - approx_time):.2f} seconds")

# Calculate how many features can be evaluated in 1 hour
one_hour_seconds = 3600
print(f"\nTime constraint: 1 hour = {one_hour_seconds} seconds")

# Function to find n where total time <= 1 hour
def find_max_features_for_time(target_time, alpha=0.8, base_time=0.1):
    """Find maximum number of features that can be evaluated within target time"""
    # Use binary search to find the maximum n
    left, right = 1, 10000
    best_n = 0
    
    while left <= right:
        mid = (left + right) // 2
        total_time = total_evaluation_time(mid, alpha, base_time)
        
        if total_time <= target_time:
            best_n = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return best_n

max_features = find_max_features_for_time(one_hour_seconds)
actual_time = total_evaluation_time(max_features)
print(f"Maximum features evaluable in 1 hour: {max_features}")
print(f"Actual time used: {actual_time:.2f} seconds ({actual_time/60:.2f} minutes)")

# Verify with next feature
next_feature_time = total_evaluation_time(max_features + 1)
print(f"Time with {max_features + 1} features: {next_feature_time:.2f} seconds (exceeds limit)")

# ============================================================================
# TASK 3: Prioritizing 10% of features
# ============================================================================
print("\n" + "="*60)
print("TASK 3: Prioritizing 10% of Features")
print("="*60)

# Simulate feature characteristics for demonstration
np.random.seed(42)
n_total_features = 1000
n_select = int(0.1 * n_total_features)  # 10% = 100 features

# Generate synthetic feature data
feature_data = {
    'variance': np.random.exponential(1.0, n_total_features),
    'correlation_with_target': np.random.normal(0.3, 0.4, n_total_features),
    'missing_rate': np.random.beta(2, 8, n_total_features),
    'evaluation_time': np.array([feature_evaluation_time(i) for i in range(1, n_total_features + 1)]),
    'domain_importance': np.random.beta(3, 2, n_total_features)
}

# Create feature DataFrame
feature_scores = []
for i in range(n_total_features):
    # Calculate composite score (higher is better)
    variance_score = feature_data['variance'][i] / np.max(feature_data['variance'])
    correlation_score = abs(feature_data['correlation_with_target'][i])
    missing_score = 1 - feature_data['missing_rate'][i]  # Lower missing rate is better
    time_score = 1 / feature_data['evaluation_time'][i]  # Faster evaluation is better
    domain_score = feature_data['domain_importance'][i]
    
    # Weighted composite score
    composite_score = (0.3 * variance_score + 
                      0.3 * correlation_score + 
                      0.2 * missing_score + 
                      0.1 * time_score + 
                      0.1 * domain_score)
    
    feature_scores.append({
        'index': i + 1,
        'composite_score': composite_score,
        'variance': feature_data['variance'][i],
        'correlation': feature_data['correlation_with_target'][i],
        'missing_rate': feature_data['missing_rate'][i],
        'evaluation_time': feature_data['evaluation_time'][i],
        'domain_importance': feature_data['domain_importance'][i]
    })

# Sort by composite score and select top 10%
feature_scores.sort(key=lambda x: x['composite_score'], reverse=True)
selected_features = feature_scores[:n_select]

print(f"Total features: {n_total_features}")
print(f"Features to select (10%): {n_select}")
print(f"\nTop 10 selected features:")
print("-" * 80)
print(f"{'Rank':<5} {'Index':<8} {'Score':<8} {'Variance':<10} {'Corr':<8} {'Missing':<8} {'Time':<8}")
print("-" * 80)

for i, feature in enumerate(selected_features[:10]):
    print(f"{i+1:<5} {feature['index']:<8} {feature['composite_score']:.4f} "
          f"{feature['variance']:.4f} {feature['correlation']:.4f} "
          f"{feature['missing_rate']:.4f} {feature['evaluation_time']:.4f}")

print(f"\nBottom 10 selected features:")
print("-" * 80)
for i, feature in enumerate(selected_features[-10:]):
    rank = n_select - 10 + i + 1
    print(f"{rank:<5} {feature['index']:<8} {feature['composite_score']:.4f} "
          f"{feature['variance']:.4f} {feature['correlation']:.4f} "
          f"{feature['missing_rate']:.4f} {feature['evaluation_time']:.4f}")

# ============================================================================
# TASK 4: Efficient selection strategy design
# ============================================================================
print("\n" + "="*60)
print("TASK 4: Efficient Selection Strategy Design")
print("="*60)

def efficient_selection_strategy(n_features, time_budget, alpha=0.8, base_time=0.1):
    """Design efficient feature selection strategy for resource constraints"""
    
    # Phase 1: Quick screening (20% of budget)
    phase1_budget = 0.2 * time_budget
    phase1_features = find_max_features_for_time(phase1_budget, alpha, base_time)
    
    # Phase 2: Detailed evaluation (80% of budget)
    phase2_budget = 0.8 * time_budget
    phase2_features = find_max_features_for_time(phase2_budget, alpha, base_time)
    
    # Calculate total features that can be evaluated
    total_evaluable = min(phase1_features + phase2_features, n_features)
    
    return {
        'phase1_features': phase1_features,
        'phase2_features': phase2_features,
        'total_evaluable': total_evaluable,
        'phase1_time': phase1_budget,
        'phase2_time': phase2_budget,
        'efficiency_gain': n_features / total_evaluable if total_evaluable > 0 else float('inf')
    }

# Apply strategy to our 1-hour constraint
strategy = efficient_selection_strategy(n_total_features, one_hour_seconds)

print("Efficient Selection Strategy for 1-hour constraint:")
print("-" * 50)
print(f"Phase 1 (Quick Screening): {strategy['phase1_features']} features in {strategy['phase1_time']/60:.1f} minutes")
print(f"Phase 2 (Detailed Evaluation): {strategy['phase2_features']} features in {strategy['phase2_time']/60:.1f} minutes")
print(f"Total features evaluable: {strategy['total_evaluable']}")
print(f"Efficiency gain: {strategy['efficiency_gain']:.2f}x")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. Power Law Distribution Visualization
plt.figure(figsize=(12, 8))

# Plot individual feature evaluation times
feature_indices = np.arange(1, 1001)
evaluation_times = np.array([feature_evaluation_time(i) for i in feature_indices])

plt.subplot(2, 2, 1)
plt.plot(feature_indices, evaluation_times, 'b-', linewidth=2, alpha=0.7)
plt.xlabel('Feature Index (i)')
plt.ylabel('Evaluation Time (seconds)')
plt.title('Individual Feature Evaluation Times\n$t_i = 0.1 \\times i^{0.8}$')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 2. Cumulative Time Plot
plt.subplot(2, 2, 2)
cumulative_times = np.cumsum(evaluation_times)
plt.plot(feature_indices, cumulative_times, 'r-', linewidth=2, alpha=0.7)
plt.axhline(y=one_hour_seconds, color='g', linestyle='--', linewidth=2, 
            label=f'1 Hour Limit ({one_hour_seconds}s)')
plt.axvline(x=max_features, color='orange', linestyle='--', linewidth=2,
            label=f'Max Features ({max_features})')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Time (seconds)')
plt.title('Cumulative Evaluation Time')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Feature Selection Strategy Visualization
plt.subplot(2, 2, 3)
# Plot composite scores
all_scores = [f['composite_score'] for f in feature_scores]
all_indices = [f['index'] for f in feature_scores]

plt.scatter(all_indices, all_scores, c='blue', alpha=0.6, s=20, label='All Features')
plt.scatter([f['index'] for f in selected_features], 
           [f['composite_score'] for f in selected_features], 
           c='red', s=30, label='Selected Features (10%)')
plt.xlabel('Feature Index')
plt.ylabel('Composite Score')
plt.title('Feature Selection Based on Composite Score')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Strategy Efficiency Comparison
plt.subplot(2, 2, 4)
strategies_comparison = ['Random', 'Variance', 'Correlation', 'Composite', 'Efficient']
efficiency_scores = [0.5, 0.7, 0.8, 0.9, 0.95]  # Hypothetical efficiency scores

bars = plt.bar(strategies_comparison, efficiency_scores, 
               color=['lightcoral', 'lightblue', 'lightgreen', 'gold', 'lightpink'])
plt.ylabel('Efficiency Score')
plt.title('Strategy Efficiency Comparison')
plt.ylim(0, 1)
for bar, score in zip(bars, efficiency_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'resource_constraints_analysis.png'), dpi=300, bbox_inches='tight')

# 5. Time Budget Allocation
plt.figure(figsize=(10, 6))
phases = ['Phase 1\n(Quick Screening)', 'Phase 2\n(Detailed Evaluation)']
times = [strategy['phase1_time']/60, strategy['phase2_time']/60]
colors = ['lightblue', 'lightcoral']

plt.pie(times, labels=phases, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Time Budget Allocation for Efficient Strategy')
plt.axis('equal')

# Add time labels
for i, (phase, time_val) in enumerate(zip(phases, times)):
    plt.text(0, 0, f'{time_val:.1f} min', ha='center', va='center', fontsize=12, fontweight='bold')

plt.savefig(os.path.join(save_dir, 'time_budget_allocation.png'), dpi=300, bbox_inches='tight')

# 6. Feature Evaluation Time vs Score
plt.figure(figsize=(12, 8))

# Scatter plot of evaluation time vs composite score
evaluation_times_array = np.array([f['evaluation_time'] for f in feature_scores])
composite_scores_array = np.array([f['composite_score'] for f in feature_scores])

plt.scatter(evaluation_times_array, composite_scores_array, alpha=0.6, s=20, c='blue', label='All Features')
plt.scatter([f['evaluation_time'] for f in selected_features], 
           [f['composite_score'] for f in selected_features], 
           c='red', s=30, label='Selected Features')

plt.xlabel('Evaluation Time (seconds)')
plt.ylabel('Composite Score')
plt.title('Feature Evaluation Time vs Composite Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(evaluation_times_array, composite_scores_array, 1)
p = np.poly1d(z)
plt.plot(evaluation_times_array, p(evaluation_times_array), "r--", alpha=0.8, linewidth=2)

plt.savefig(os.path.join(save_dir, 'time_vs_score_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Summary of key results
print("\nKEY RESULTS SUMMARY:")
print("-" * 40)
print(f"1. Maximum features evaluable in 1 hour: {max_features}")
print(f"2. Top 10% features selected based on composite scoring")
print(f"3. Efficient strategy: {strategy['phase1_features']} + {strategy['phase2_features']} features")
print(f"4. Time budget: {strategy['phase1_time']/60:.1f} min + {strategy['phase2_time']/60:.1f} min")
print(f"5. Efficiency gain: {strategy['efficiency_gain']:.2f}x")
