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

# Create visualization of benefits
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Three Main Benefits of Feature Selection', fontsize=16, fontweight='bold')

# Benefit 1: Improved Model Performance
x = np.linspace(0, 100, 100)
y_performance = 0.8 + 0.15 * np.exp(-x/30) + 0.05 * np.random.normal(0, 0.01, 100)
axes[0].plot(x, y_performance, 'b-', linewidth=2)
axes[0].scatter([20, 50, 80], [0.92, 0.85, 0.82], color='red', s=100, zorder=5)
axes[0].set_xlabel('Number of Features')
axes[0].set_ylabel('Model Performance (Accuracy)')
axes[0].set_title('Improved Model Performance')
axes[0].grid(True, alpha=0.3)
axes[0].text(60, 0.88, 'Optimal\nFeature Count', ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

# Benefit 2: Reduced Overfitting
x = np.linspace(0, 100, 100)
y_train = 0.95 - 0.001 * x + 0.0001 * x**2 + 0.02 * np.random.normal(0, 1, 100)
y_test = 0.85 - 0.002 * x + 0.0002 * x**2 + 0.01 * np.random.normal(0, 1, 100)
axes[1].plot(x, y_train, 'g-', linewidth=2, label='Training Performance')
axes[1].plot(x, y_test, 'r-', linewidth=2, label='Test Performance')
axes[1].fill_between(x, y_train, y_test, alpha=0.3, color='orange')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Model Performance')
axes[1].set_title('Reduced Overfitting')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].text(70, 0.75, 'Overfitting\nGap', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.7))

# Benefit 3: Enhanced Interpretability
x = np.linspace(0, 100, 100)
y_interpretability = 1.0 - 0.008 * x + 0.00005 * x**2
axes[2].plot(x, y_interpretability, 'purple', linewidth=2)
axes[2].set_xlabel('Number of Features')
axes[2].set_ylabel('Model Interpretability')
axes[2].set_title('Enhanced Interpretability')
axes[2].grid(True, alpha=0.3)
axes[2].text(80, 0.3, 'Complex\nModels\nHard to\nInterpret', ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

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

# Create synthetic dataset to demonstrate
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                          n_redundant=10, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature selection using Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = rf.feature_importances_
feature_names = [f'Feature_{i+1}' for i in range(20)]

# Create feature importance plot
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Before Selection')
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], rotation=45)

# Add threshold line for selection
threshold = np.percentile(feature_importance, 75)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Selection Threshold ({threshold:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_before.png'), dpi=300, bbox_inches='tight')

# Select top features
top_features = feature_importance > threshold
selected_features = np.where(top_features)[0]
print(f"\nSelected {len(selected_features)} out of 20 features based on importance threshold {threshold:.3f}")
print(f"Selected features: {[feature_names[i] for i in selected_features]}")

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

# Demonstrate computational cost reduction
feature_counts = [10, 25, 50, 100, 200]
inference_times = []
memory_usage = []

for n_features in feature_counts:
    # Simulate inference time (proportional to number of features)
    base_time = 0.001  # 1ms base time
    inference_time = base_time * n_features + 0.0001 * n_features**2
    inference_times.append(inference_time)
    
    # Simulate memory usage (8 bytes per feature per sample)
    memory_per_sample = n_features * 8
    memory_usage.append(memory_per_sample)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Real-time Application Benefits of Feature Selection', fontsize=16, fontweight='bold')

# Inference time
ax1.plot(feature_counts, inference_times, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Inference Time (seconds)')
ax1.set_title('Inference Time vs Feature Count')
ax1.grid(True, alpha=0.3)
ax1.text(150, 0.08, 'Faster\nInference\nwith Fewer\nFeatures', ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

# Memory usage
ax2.plot(feature_counts, memory_usage, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Memory per Sample (bytes)')
ax2.set_title('Memory Usage vs Feature Count')
ax2.grid(True, alpha=0.3)
ax2.text(150, 1200, 'Lower\nMemory\nUsage\nwith Fewer\nFeatures', ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'realtime_benefits.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: Training Time Estimation
# ============================================================================
print("\n" + "="*60)
print("PART 4: TRAINING TIME ESTIMATION")
print("="*60)

# Given: 5 minutes with 100 features
original_features = 100
original_time = 5  # minutes
new_features = 25

# Assuming linear scaling
time_ratio = new_features / original_features
estimated_time = original_time * time_ratio

print(f"Original training time: {original_time} minutes with {original_features} features")
print(f"New feature count: {new_features} features")
print(f"Time ratio: {new_features}/{original_features} = {time_ratio}")
print(f"Estimated training time: {original_time} × {time_ratio} = {estimated_time} minutes")

# Create visualization
plt.figure(figsize=(10, 6))
feature_counts = [25, 50, 75, 100, 125, 150]
training_times = [original_time * (f/original_features) for f in feature_counts]

plt.plot(feature_counts, training_times, 'bo-', linewidth=2, markersize=8)
plt.scatter([100], [5], color='red', s=200, zorder=5, label='Given Point')
plt.scatter([25], [estimated_time], color='green', s=200, zorder=5, label='Estimated Point')

plt.xlabel('Number of Features')
plt.ylabel('Training Time (minutes)')
plt.title('Training Time vs Feature Count (Linear Scaling)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(120, 4, f'Linear relationship:\nTime \\propto Features', ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_time_estimation.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: Memory Reduction Calculation
# ============================================================================
print("\n" + "="*60)
print("PART 5: MEMORY REDUCTION CALCULATION")
print("="*60)

# Given parameters
original_features = 1000
new_features = 100
bytes_per_feature = 8

# Calculate memory usage
original_memory = original_features * bytes_per_feature
new_memory = new_features * bytes_per_feature
memory_reduction = original_memory - new_memory
reduction_percentage = (memory_reduction / original_memory) * 100

print(f"Original features: {original_features}")
print(f"New features: {new_features}")
print(f"Bytes per feature: {bytes_per_feature}")
print(f"Original memory per sample: {original_features} × {bytes_per_feature} = {original_memory} bytes")
print(f"New memory per sample: {new_features} × {bytes_per_feature} = {new_memory} bytes")
print(f"Memory reduction: {original_memory} - {new_memory} = {memory_reduction} bytes")
print(f"Reduction percentage: ({memory_reduction}/{original_memory}) × 100 = {reduction_percentage:.1f}%")

# Create visualization
plt.figure(figsize=(10, 6))
feature_counts = [100, 250, 500, 750, 1000]
memory_usage = [f * bytes_per_feature for f in feature_counts]

plt.plot(feature_counts, memory_usage, 'mo-', linewidth=2, markersize=8)
plt.scatter([1000], [8000], color='red', s=200, zorder=5, label='Original (1000 features)')
plt.scatter([100], [800], color='green', s=200, zorder=5, label='Reduced (100 features)')

plt.xlabel('Number of Features')
plt.ylabel('Memory per Sample (bytes)')
plt.title('Memory Usage vs Feature Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(750, 6000, f'Memory = Features \\times {bytes_per_feature} bytes', ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'memory_reduction.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: Neural Network Training Time Calculation
# ============================================================================
print("\n" + "="*60)
print("PART 6: NEURAL NETWORK TRAINING TIME CALCULATION")
print("="*60)

# Given parameters
n_samples = 1000
original_features = 50
original_time = 2  # hours
new_features = 10

# Complexity: O(n² × d)
# Calculate time ratios
original_complexity = n_samples**2 * original_features
new_complexity = n_samples**2 * new_features
complexity_ratio = new_complexity / original_complexity

# Estimate new training time
estimated_time = original_time * complexity_ratio
time_improvement = original_time - estimated_time
improvement_percentage = (time_improvement / original_time) * 100

print(f"Given parameters:")
print(f"  - Number of samples (n): {n_samples}")
print(f"  - Original features (d): {original_features}")
print(f"  - Original training time: {original_time} hours")
print(f"  - New features: {new_features}")
print(f"\nComplexity analysis:")
print(f"  - Original complexity: O(n² × d) = O({n_samples}² × {original_features}) = O({original_complexity:,})")
print(f"  - New complexity: O(n² × d) = O({n_samples}² × {new_features}) = O({new_complexity:,})")
print(f"  - Complexity ratio: {new_complexity:,} / {original_complexity:,} = {complexity_ratio:.3f}")
print(f"\nTraining time estimation:")
print(f"  - Estimated new time: {original_time} × {complexity_ratio:.3f} = {estimated_time:.3f} hours")
print(f"  - Time improvement: {original_time} - {estimated_time:.3f} = {time_improvement:.3f} hours")
print(f"  - Improvement percentage: ({time_improvement:.3f}/{original_time}) × 100 = {improvement_percentage:.1f}%")

# Create visualization
plt.figure(figsize=(12, 8))
feature_counts = [5, 10, 20, 30, 40, 50]
complexities = [n_samples**2 * f for f in feature_counts]
training_times = [2 * (c / (n_samples**2 * 50)) for c in complexities]

plt.plot(feature_counts, training_times, 'co-', linewidth=2, markersize=8)
plt.scatter([50], [2], color='red', s=200, zorder=5, label='Given Point (50 features, 2 hours)')
plt.scatter([10], [estimated_time], color='green', s=200, zorder=5, label=f'Estimated Point (10 features, {estimated_time:.3f} hours)')

plt.xlabel('Number of Features')
plt.ylabel('Training Time (hours)')
plt.title('Neural Network Training Time vs Feature Count\nComplexity: O(n² × d)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(35, 1.5, f'Time \\propto Features\n(quadratic relationship)', ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

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

# Create synthetic weather dataset
np.random.seed(42)
n_samples = 1000
weather_features = {
    'temperature': np.random.normal(20, 10, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'pressure': np.random.normal(1013, 20, n_samples),
    'wind_speed': np.random.exponential(5, n_samples),
    'wind_direction': np.random.uniform(0, 360, n_samples),
    'cloud_cover': np.random.uniform(0, 100, n_samples),
    'precipitation': np.random.exponential(2, n_samples),
    'solar_radiation': np.random.normal(500, 200, n_samples),
    'time_of_day': np.random.uniform(0, 24, n_samples),
    'day_of_year': np.random.uniform(1, 365, n_samples)
}

# Create target variable (rain probability)
X_weather = pd.DataFrame(weather_features)
y_weather = (X_weather['humidity'] > 70) & (X_weather['pressure'] < 1000) | (X_weather['precipitation'] > 1)
y_weather = y_weather.astype(int)

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

print(f"1. Correlation-based selection (threshold > {correlation_threshold}):")
print(f"   Selected features: {correlation_selected}")
print(f"   Number of features: {len(correlation_selected)}")

# 2. Statistical test selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_train_weather, y_train_weather)
statistical_selected = X_train_weather.columns[selector.get_support()].tolist()

print(f"\n2. Statistical test selection (top 5 features):")
print(f"   Selected features: {statistical_selected}")
print(f"   Number of features: {len(statistical_selected)}")

# 3. Recursive feature elimination
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train_weather, y_train_weather)
rfe_selected = X_train_weather.columns[rfe.support_].tolist()

print(f"\n3. Recursive feature elimination (top 5 features):")
print(f"   Selected features: {rfe_selected}")
print(f"   Number of features: {len(rfe_selected)}")

# Compare methods
methods = ['Correlation', 'Statistical', 'RFE']
feature_counts = [len(correlation_selected), len(statistical_selected), len(rfe_selected)]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, feature_counts, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Feature Selection Method')
plt.ylabel('Number of Selected Features')
plt.title('Feature Selection Methods Comparison')
plt.ylim(0, max(feature_counts) + 1)

# Add value labels on bars
for bar, count in zip(bars, feature_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(count), ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_methods_comparison.png'), dpi=300, bbox_inches='tight')

# Final strategy recommendation
print(f"\nRECOMMENDED FEATURE SELECTION STRATEGY:")
print(f"1. Use correlation-based selection first (fastest, lowest computational cost)")
print(f"2. Apply statistical test selection for validation")
print(f"3. Consider domain knowledge (e.g., temperature, humidity, pressure are most important)")
print(f"4. Target 5-7 features maximum for smartphone constraints")
print(f"5. Implement feature importance monitoring for adaptive selection")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL CALCULATIONS AND RESULTS")
print("="*80)

summary_data = {
    'Question': [
        'Training time with 25 features',
        'Memory reduction (1000→100 features)',
        'Neural network training time (50→10 features)',
        'Feature selection methods compared'
    ],
    'Result': [
        f'{estimated_time:.2f} minutes',
        f'{reduction_percentage:.1f}% reduction',
        f'{improvement_percentage:.1f}% improvement',
        f'{len(methods)} methods analyzed'
    ],
    'Key Insight': [
        'Linear scaling assumption',
        'Memory ∝ Features',
        'Complexity O(n²×d)',
        'Correlation method fastest'
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\nAll visualizations saved to: {save_dir}")
print("=" * 80)
