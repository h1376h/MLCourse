import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 1: FEATURE SELECTION FUNDAMENTALS")
print("=" * 80)

# ============================================================================
# PART 1: Three Main Benefits of Feature Selection
# ============================================================================
print("\n" + "="*60)
print("PART 1: THREE MAIN BENEFITS OF FEATURE SELECTION")
print("="*60)

benefits = [
    "Improved Model Performance",
    "Reduced Overfitting", 
    "Enhanced Interpretability"
]

print("The three main benefits of feature selection are:")
for i, benefit in enumerate(benefits, 1):
    print(f"{i}. {benefit}")

# Parameters for visualization
max_features = 100
optimal_features = 25
feature_range = np.linspace(0, max_features, 100)

# Create visualization of benefits
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Three Main Benefits of Feature Selection', fontsize=16, fontweight='bold')

# Benefit 1: Improved Model Performance
# Model performance peaks at optimal feature count, then declines due to curse of dimensionality
y_performance = 0.8 + 0.15 * np.exp(-feature_range/30) + 0.05 * np.random.normal(0, 0.01, len(feature_range))
optimal_idx = np.argmax(y_performance)
optimal_point = feature_range[optimal_idx]
max_performance = y_performance[optimal_idx]

axes[0].plot(feature_range, y_performance, 'b-', linewidth=2, label='Performance Curve')
axes[0].scatter([optimal_point], [max_performance], color='red', s=150, zorder=5, label=f'Optimal: {optimal_point:.0f} features')
axes[0].set_xlabel('Number of Features')
axes[0].set_ylabel('Model Performance (Accuracy)')
axes[0].set_title('1. Improved Model Performance')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].text(optimal_point + 15, max_performance - 0.02, f'Peak at\n{optimal_point:.0f} features', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

# Benefit 2: Reduced Overfitting
# Training performance stays high, test performance degrades with too many features
y_train = 0.95 - 0.001 * feature_range + 0.0001 * feature_range**2 + 0.02 * np.random.normal(0, 0.01, len(feature_range))
y_test = 0.85 - 0.002 * feature_range + 0.0002 * feature_range**2 + 0.01 * np.random.normal(0, 0.01, len(feature_range))

# Find point where gap becomes significant
gap = y_train - y_test
max_gap_idx = np.argmax(gap)
max_gap_features = feature_range[max_gap_idx]
max_gap_value = gap[max_gap_idx]

axes[1].plot(feature_range, y_train, 'g-', linewidth=2, label='Training Performance')
axes[1].plot(feature_range, y_test, 'r-', linewidth=2, label='Test Performance')
axes[1].fill_between(feature_range, y_train, y_test, alpha=0.3, color='orange', label='Overfitting Gap')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Model Performance')
axes[1].set_title('2. Reduced Overfitting')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].text(max_gap_features - 15, 0.75, f'Max Gap:\n{max_gap_value:.2f}', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.7))

# Benefit 3: Enhanced Interpretability
# Interpretability decreases as model complexity increases
y_interpretability = 1.0 - 0.008 * feature_range + 0.00005 * feature_range**2

# Find threshold where interpretability becomes very low
interpret_threshold = 0.3
low_interpret_idx = np.where(y_interpretability < interpret_threshold)[0]
if len(low_interpret_idx) > 0:
    threshold_features = feature_range[low_interpret_idx[0]]
else:
    threshold_features = max_features

axes[2].plot(feature_range, y_interpretability, 'purple', linewidth=2, label='Interpretability')
axes[2].axhline(y=interpret_threshold, color='red', linestyle='--', alpha=0.7, label=f'Low Interpretability Threshold')
axes[2].set_xlabel('Number of Features')
axes[2].set_ylabel('Model Interpretability')
axes[2].set_title('3. Enhanced Interpretability')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].text(threshold_features + 10, 0.15, f'Low interpretability\nafter {threshold_features:.0f} features', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_benefits.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 2: How Feature Selection Improves Model Interpretability
# ============================================================================
print("\n" + "="*60)
print("PART 2: FEATURE SELECTION AND MODEL INTERPRETABILITY")
print("="*60)

print("Feature selection improves model interpretability by:")
print("1. Reducing complexity - fewer features mean simpler decision boundaries")
print("2. Highlighting important variables - focusing on most relevant features")
print("3. Eliminating noise - removing irrelevant features that confuse interpretation")

# Create synthetic dataset to demonstrate feature importance
np.random.seed(42)
n_samples = 1000
n_total_features = 20
n_informative = 5
n_redundant = 10

X, y = make_classification(n_samples=n_samples, n_features=n_total_features, n_informative=n_informative, 
                          n_redundant=n_redundant, n_clusters_per_class=1, random_state=42)

# Split data
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Feature selection using Random Forest importance
n_estimators = 100
rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = rf.feature_importances_
feature_names = [f'Feature_{i+1}' for i in range(n_total_features)]

# Create feature importance plot
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Before Selection')
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], rotation=45)

# Calculate selection threshold based on top features
selection_percentile = 75
threshold = np.percentile(feature_importance, selection_percentile)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Selection Threshold ({threshold:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_before.png'), dpi=300, bbox_inches='tight')

# Select top features
top_features = feature_importance > threshold
selected_features = np.where(top_features)[0]
n_selected = len(selected_features)
selection_ratio = n_selected / n_total_features

print(f"\nSelected {n_selected} out of {n_total_features} features ({selection_ratio:.1%}) based on importance threshold {threshold:.3f}")
print(f"Selected features: {[feature_names[i] for i in selected_features]}")
print(f"This means {(1-selection_ratio):.1%} of features contribute little to predictive power")

# ============================================================================
# PART 3: Importance for Real-time Applications
# ============================================================================
print("\n" + "="*60)
print("PART 3: FEATURE SELECTION FOR REAL-TIME APPLICATIONS")
print("="*60)

print("Feature selection is crucial for real-time applications because:")
print("1. Faster inference - fewer features mean quicker predictions")
print("2. Lower computational cost - reduced memory and processing requirements")
print("3. Better scalability - models can handle higher throughput")

# Demonstrate computational cost reduction with parameterized model
min_features = 10
max_features_demo = 200
step_size = max_features_demo // 8
feature_counts = list(range(min_features, max_features_demo + 1, step_size))
inference_times = []
memory_usage = []

# Model parameters
base_time = 0.001  # 1ms base time
quadratic_factor = 0.0001
bytes_per_feature = 8

for n_features in feature_counts:
    # Simulate inference time (linear + quadratic component)
    inference_time = base_time * n_features + quadratic_factor * n_features**2
    inference_times.append(inference_time)
    
    # Simulate memory usage (linear relationship)
    memory_per_sample = n_features * bytes_per_feature
    memory_usage.append(memory_per_sample)

print(f"\nComputational analysis for {len(feature_counts)} feature count scenarios:")
print(f"Feature range: {min(feature_counts)} to {max(feature_counts)} features")
print(f"Inference time model: {base_time}*n + {quadratic_factor}*n^2")
print(f"Memory model: {bytes_per_feature}*n bytes per sample")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Real-time Application Benefits of Feature Selection', fontsize=16, fontweight='bold')

# Inference time
ax1.plot(feature_counts, inference_times, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Inference Time (seconds)')
ax1.set_title('Inference Time vs Feature Count')
ax1.grid(True, alpha=0.3)
# Find the feature count with highest inference time
max_time_idx = np.argmax(inference_times)
max_time_features = feature_counts[max_time_idx]
max_time_value = inference_times[max_time_idx]

ax1.text(max_time_features * 0.7, max_time_value * 0.8, 
         f'Max: {max_time_value:.3f}s\nat {max_time_features} features', 
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

# Memory usage
ax2.plot(feature_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Memory per Sample (bytes)')
ax2.set_title('Memory Usage vs Feature Count')
ax2.grid(True, alpha=0.3)
# Find the feature count with highest memory usage
max_memory_idx = np.argmax(memory_usage)
max_memory_features = feature_counts[max_memory_idx]
max_memory_value = memory_usage[max_memory_idx]

ax2.text(max_memory_features * 0.7, max_memory_value * 0.8, 
         f'Max: {max_memory_value} bytes\nat {max_memory_features} features', 
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'realtime_benefits.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: Training Time Estimation
# ============================================================================
print("\n" + "="*60)
print("PART 4: TRAINING TIME ESTIMATION")
print("="*60)

# Parameters from the problem
original_features = 100
original_time = 5  # minutes
new_features = 25

# Calculate scaling assuming linear relationship
time_ratio = new_features / original_features
estimated_time = original_time * time_ratio
time_reduction = original_time - estimated_time
time_improvement_pct = (time_reduction / original_time) * 100

print(f"Given parameters:")
print(f"  - Original training time: {original_time} minutes with {original_features} features")
print(f"  - Target feature count: {new_features} features")
print(f"\nLinear scaling calculation:")
print(f"  - Time ratio: {new_features}/{original_features} = {time_ratio:.3f}")
print(f"  - Estimated training time: {original_time} × {time_ratio:.3f} = {estimated_time:.2f} minutes")
print(f"  - Time reduction: {original_time} - {estimated_time:.2f} = {time_reduction:.2f} minutes")
print(f"  - Improvement: {time_improvement_pct:.1f}%")

# Create visualization with broader range
plt.figure(figsize=(10, 6))
min_viz_features = 10
max_viz_features = 150
viz_step = 15
feature_counts_viz = list(range(min_viz_features, max_viz_features + 1, viz_step))
training_times = [original_time * (f/original_features) for f in feature_counts_viz]

plt.plot(feature_counts_viz, training_times, 'bo-', linewidth=2, markersize=6, label='Linear Scaling Model')
plt.scatter([original_features], [original_time], color='red', s=200, zorder=5, label=f'Given: {original_features} features, {original_time} min')
plt.scatter([new_features], [estimated_time], color='green', s=200, zorder=5, label=f'Estimated: {new_features} features, {estimated_time:.2f} min')

plt.xlabel('Number of Features')
plt.ylabel('Training Time (minutes)')
plt.title('Training Time vs Feature Count (Linear Scaling)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(max_viz_features * 0.8, max(training_times) * 0.8, 
         f'Linear relationship: $Time \\propto Features$\\\\{time_improvement_pct:.1f}\\% improvement', 
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_time_estimation.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: Memory Reduction Calculation
# ============================================================================
print("\n" + "="*60)
print("PART 5: MEMORY REDUCTION CALCULATION")
print("="*60)

# Parameters from the problem
original_features_mem = 1000
new_features_mem = 100
bytes_per_feature_mem = 8

# Calculate memory usage
original_memory = original_features_mem * bytes_per_feature_mem
new_memory = new_features_mem * bytes_per_feature_mem
memory_reduction = original_memory - new_memory
reduction_percentage = (memory_reduction / original_memory) * 100
feature_reduction_ratio = new_features_mem / original_features_mem

print(f"Given parameters:")
print(f"  - Original features: {original_features_mem}")
print(f"  - New features: {new_features_mem}")
print(f"  - Bytes per feature: {bytes_per_feature_mem}")
print(f"\nMemory calculations:")
print(f"  - Original memory per sample: {original_features_mem} × {bytes_per_feature_mem} = {original_memory:,} bytes")
print(f"  - New memory per sample: {new_features_mem} × {bytes_per_feature_mem} = {new_memory:,} bytes")
print(f"  - Memory reduction: {original_memory:,} - {new_memory:,} = {memory_reduction:,} bytes")
print(f"  - Reduction percentage: ({memory_reduction:,}/{original_memory:,}) × 100 = {reduction_percentage:.1f}%")
print(f"  - Feature reduction ratio: {new_features_mem}/{original_features_mem} = {feature_reduction_ratio:.2f}")

# Create visualization with extended range
plt.figure(figsize=(10, 6))
min_mem_features = 50
max_mem_features = 1200
mem_step = 150
feature_counts_mem = list(range(min_mem_features, max_mem_features + 1, mem_step))
memory_usage_viz = [f * bytes_per_feature_mem for f in feature_counts_mem]

plt.plot(feature_counts_mem, memory_usage_viz, 'mo-', linewidth=2, markersize=6, label='Linear Memory Model')
plt.scatter([original_features_mem], [original_memory], color='red', s=200, zorder=5, 
           label=f'Original: {original_features_mem} features, {original_memory:,} bytes')
plt.scatter([new_features_mem], [new_memory], color='green', s=200, zorder=5, 
           label=f'Reduced: {new_features_mem} features, {new_memory:,} bytes')

plt.xlabel('Number of Features')
plt.ylabel('Memory per Sample (bytes)')
plt.title('Memory Usage vs Feature Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(max_mem_features * 0.7, max(memory_usage_viz) * 0.8, 
         f'$Memory = Features \\times {bytes_per_feature_mem}$ bytes\\\\{reduction_percentage:.1f}\\% reduction', 
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'memory_reduction.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: Neural Network Training Time Calculation
# ============================================================================
print("\n" + "="*60)
print("PART 6: NEURAL NETWORK TRAINING TIME CALCULATION")
print("="*60)

# Parameters from the problem
n_samples_nn = 1000
original_features_nn = 50
original_time_nn = 2  # hours
new_features_nn = 10

# Complexity: O(n² × d) for neural networks
# Calculate complexity ratios
original_complexity = n_samples_nn**2 * original_features_nn
new_complexity = n_samples_nn**2 * new_features_nn
complexity_ratio = new_complexity / original_complexity
feature_ratio = new_features_nn / original_features_nn

# Estimate new training time
estimated_time_nn = original_time_nn * complexity_ratio
time_improvement_nn = original_time_nn - estimated_time_nn
improvement_percentage_nn = (time_improvement_nn / original_time_nn) * 100

print(f"Given parameters:")
print(f"  - Number of samples (n): {n_samples_nn:,}")
print(f"  - Original features (d): {original_features_nn}")
print(f"  - Original training time: {original_time_nn} hours")
print(f"  - New features: {new_features_nn}")
print(f"\nComplexity analysis (O(n² × d)):")
print(f"  - Original complexity: O({n_samples_nn:,}² × {original_features_nn}) = O({original_complexity:,})")
print(f"  - New complexity: O({n_samples_nn:,}² × {new_features_nn}) = O({new_complexity:,})")
print(f"  - Complexity ratio: {new_complexity:,} / {original_complexity:,} = {complexity_ratio:.3f}")
print(f"  - Feature ratio: {new_features_nn}/{original_features_nn} = {feature_ratio:.2f}")
print(f"\nTraining time estimation:")
print(f"  - Estimated new time: {original_time_nn} × {complexity_ratio:.3f} = {estimated_time_nn:.3f} hours")
print(f"  - Time improvement: {original_time_nn} - {estimated_time_nn:.3f} = {time_improvement_nn:.3f} hours")
print(f"  - Improvement percentage: ({time_improvement_nn:.3f}/{original_time_nn}) × 100 = {improvement_percentage_nn:.1f}%")

# Create visualization with broader feature range
plt.figure(figsize=(12, 8))
min_nn_features = 5
max_nn_features = 60
nn_step = 5
feature_counts_nn = list(range(min_nn_features, max_nn_features + 1, nn_step))
complexities = [n_samples_nn**2 * f for f in feature_counts_nn]
training_times_nn = [original_time_nn * (c / (n_samples_nn**2 * original_features_nn)) for c in complexities]

plt.plot(feature_counts_nn, training_times_nn, 'co-', linewidth=2, markersize=6, label='Neural Network Training Time')
plt.scatter([original_features_nn], [original_time_nn], color='red', s=200, zorder=5, 
           label=f'Given: {original_features_nn} features, {original_time_nn} hours')
plt.scatter([new_features_nn], [estimated_time_nn], color='green', s=200, zorder=5, 
           label=f'Estimated: {new_features_nn} features, {estimated_time_nn:.3f} hours')

plt.xlabel('Number of Features')
plt.ylabel('Training Time (hours)')
plt.title(f'Neural Network Training Time vs Feature Count\nComplexity: $O(n^2 \\times d)$ with $n={n_samples_nn:,}$ samples')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(max_nn_features * 0.7, max(training_times_nn) * 0.8, 
         f'$Time \\propto Features$\\\\Complexity: $O(n^2 \\times d)$\\\\{improvement_percentage_nn:.1f}\\% improvement', 
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'neural_network_training_time.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 7: Feature Selection Strategy for Weather Prediction
# ============================================================================
print("\n" + "="*60)
print("PART 7: FEATURE SELECTION STRATEGY FOR WEATHER PREDICTION")
print("="*60)

print("Designing a feature selection strategy for smartphone weather prediction:")
print("\nConstraints:")
print("- Limited battery life")
print("- Must run on smartphone")
print("- Real-time predictions needed")

# Create synthetic weather dataset with realistic parameters
np.random.seed(42)
n_samples_weather = 1000
temp_mean, temp_std = 20, 10
humidity_min, humidity_max = 30, 90
pressure_mean, pressure_std = 1013, 20
wind_scale = 5
solar_mean, solar_std = 500, 200

weather_features = {
    'temperature': np.random.normal(temp_mean, temp_std, n_samples_weather),
    'humidity': np.random.uniform(humidity_min, humidity_max, n_samples_weather),
    'pressure': np.random.normal(pressure_mean, pressure_std, n_samples_weather),
    'wind_speed': np.random.exponential(wind_scale, n_samples_weather),
    'wind_direction': np.random.uniform(0, 360, n_samples_weather),
    'cloud_cover': np.random.uniform(0, 100, n_samples_weather),
    'precipitation': np.random.exponential(2, n_samples_weather),
    'solar_radiation': np.random.normal(solar_mean, solar_std, n_samples_weather),
    'time_of_day': np.random.uniform(0, 24, n_samples_weather),
    'day_of_year': np.random.uniform(1, 365, n_samples_weather)
}

print(f"\nWeather dataset created with {n_samples_weather} samples and {len(weather_features)} features:")
for feature, values in weather_features.items():
    print(f"  - {feature}: mean={np.mean(values):.1f}, std={np.std(values):.1f}")

# Create target variable (rain probability) based on domain knowledge
X_weather = pd.DataFrame(weather_features)
# Rain occurs when: high humidity AND low pressure OR significant precipitation
high_humidity_threshold = 70
low_pressure_threshold = 1000
precipitation_threshold = 1

y_weather = ((X_weather['humidity'] > high_humidity_threshold) & 
             (X_weather['pressure'] < low_pressure_threshold)) | \
            (X_weather['precipitation'] > precipitation_threshold)
y_weather = y_weather.astype(int)

rain_probability = np.mean(y_weather)
print(f"\nTarget variable (rain prediction):")
print(f"  - Rain threshold conditions: humidity > {high_humidity_threshold}% AND pressure < {low_pressure_threshold} hPa")
print(f"  - OR precipitation > {precipitation_threshold} mm")
print(f"  - Overall rain probability in dataset: {rain_probability:.1%}")

# Split data
X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(
    X_weather, y_weather, test_size=0.3, random_state=42
)

# Feature selection methods
print("\nFeature Selection Methods:")

# 1. Correlation-based selection
correlations = X_train_weather.corrwith(y_train_weather).abs()
correlation_threshold = 0.1
correlation_selected = correlations[correlations > correlation_threshold].index.tolist()
corr_selected_values = correlations[correlations > correlation_threshold].sort_values(ascending=False)

print(f"1. Correlation-based selection (threshold > {correlation_threshold}):")
print(f"   Selected features: {correlation_selected}")
print(f"   Number of features: {len(correlation_selected)} out of {len(X_train_weather.columns)}")
print(f"   Correlation values: {dict(corr_selected_values)}")
print(f"   Selection ratio: {len(correlation_selected)/len(X_train_weather.columns):.1%}")

# 2. Statistical test selection
k_best = 5
selector = SelectKBest(score_func=f_classif, k=k_best)
X_selected = selector.fit_transform(X_train_weather, y_train_weather)
statistical_selected = X_train_weather.columns[selector.get_support()].tolist()
feature_scores = dict(zip(X_train_weather.columns[selector.get_support()], selector.scores_[selector.get_support()]))

print(f"\n2. Statistical test selection (top {k_best} features):")
print(f"   Selected features: {statistical_selected}")
print(f"   Number of features: {len(statistical_selected)} out of {len(X_train_weather.columns)}")
print(f"   F-scores: {feature_scores}")
print(f"   Selection ratio: {len(statistical_selected)/len(X_train_weather.columns):.1%}")

# 3. Recursive feature elimination
n_rfe_features = 5
estimator_rfe = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator_rfe, n_features_to_select=n_rfe_features)
X_rfe = rfe.fit_transform(X_train_weather, y_train_weather)
rfe_selected = X_train_weather.columns[rfe.support_].tolist()
rfe_rankings = dict(zip(X_train_weather.columns, rfe.ranking_))

print(f"\n3. Recursive feature elimination (top {n_rfe_features} features):")
print(f"   Selected features: {rfe_selected}")
print(f"   Number of features: {len(rfe_selected)} out of {len(X_train_weather.columns)}")
print(f"   Feature rankings: {rfe_rankings}")
print(f"   Selection ratio: {len(rfe_selected)/len(X_train_weather.columns):.1%}")

# Compare methods
methods = ['Correlation', 'Statistical', 'RFE']
feature_counts = [len(correlation_selected), len(statistical_selected), len(rfe_selected)]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, feature_counts, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Feature Selection Method')
plt.ylabel('Number of Selected Features')
plt.title('Feature Selection Methods Comparison')
plt.ylim(0, max(feature_counts) + 1)

# Add value labels on bars with method details
method_details = [
    f'{len(correlation_selected)}\n(fastest)', 
    f'{len(statistical_selected)}\n(validated)', 
    f'{len(rfe_selected)}\n(model-aware)'
]
for bar, detail in zip(bars, method_details):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             detail, ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_methods_comparison.png'), dpi=300, bbox_inches='tight')

# Final strategy recommendation
# Calculate overlap between methods
overlap_corr_stat = set(correlation_selected) & set(statistical_selected)
overlap_corr_rfe = set(correlation_selected) & set(rfe_selected)
overlap_stat_rfe = set(statistical_selected) & set(rfe_selected)
overlap_all = set(correlation_selected) & set(statistical_selected) & set(rfe_selected)

print(f"\nFeature Selection Method Analysis:")
print(f"  - Correlation & Statistical overlap: {list(overlap_corr_stat)}")
print(f"  - Correlation & RFE overlap: {list(overlap_corr_rfe)}")
print(f"  - Statistical & RFE overlap: {list(overlap_stat_rfe)}")
print(f"  - All methods agree on: {list(overlap_all)}")

print(f"\nRECOMMENDED FEATURE SELECTION STRATEGY:")
print(f"1. Use correlation-based selection first (fastest: {len(correlation_selected)} features)")
print(f"2. Apply statistical test selection for validation ({len(statistical_selected)} features)")
print(f"3. Consider RFE for model-aware selection ({len(rfe_selected)} features)")
print(f"4. Target {min(len(correlation_selected), len(statistical_selected), len(rfe_selected))}-{max(len(correlation_selected), len(statistical_selected), len(rfe_selected))} features for smartphone constraints")
print(f"5. Prioritize features agreed upon by multiple methods: {list(overlap_all) if overlap_all else 'None in this case'}")
print(f"6. Implement feature importance monitoring for adaptive selection")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL CALCULATIONS AND RESULTS")
print("="*80)

summary_data = {
    'Question': [
        f'Training time ({original_features}→{new_features} features)',
        f'Memory reduction ({original_features_mem}→{new_features_mem} features)', 
        f'Neural network training ({original_features_nn}→{new_features_nn} features)',
        'Feature selection methods compared'
    ],
    'Result': [
        f'{estimated_time:.2f} min ({time_improvement_pct:.1f}% improvement)',
        f'{reduction_percentage:.1f}% reduction ({memory_reduction:,} bytes saved)',
        f'{improvement_percentage_nn:.1f}% improvement ({time_improvement_nn:.2f} hours saved)',
        f'{len(methods)} methods: {len(correlation_selected)}, {len(statistical_selected)}, {len(rfe_selected)} features'
    ],
    'Key Insight': [
        'Linear scaling: Time ∝ Features',
        'Linear scaling: Memory ∝ Features',
        'Quadratic scaling: O(n²×d)',
        'Correlation fastest, RFE most sophisticated'
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\nAll visualizations saved to: {save_dir}")
print(f"Generated {len([f for f in os.listdir(save_dir) if f.endswith('.png')])} visualizations")
print("=" * 80)
