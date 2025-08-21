import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os
import time
from scipy.stats import pearsonr
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("Question 16: Feature Selection Timing in Machine Learning Pipeline")
print("=" * 70)

# Task 3: Pipeline timing calculations
print("\nTask 3: Pipeline Timing Analysis")
print("-" * 40)

# Given parameters
preprocessing_time_per_feature = 2  # minutes
selection_time_per_feature = 1      # minutes
total_features = 100
selected_features = 20
preprocessed_features = 50

# Strategy 1: Preprocess all 100 features then select 20
strategy1_time = (total_features * preprocessing_time_per_feature + 
                  total_features * selection_time_per_feature)
print(f"Strategy 1: Preprocess all {total_features} features then select {selected_features}")
print(f"  Time = {total_features} × {preprocessing_time_per_feature} + {total_features} × {selection_time_per_feature}")
print(f"  Time = {total_features * preprocessing_time_per_feature} + {total_features * selection_time_per_feature}")
print(f"  Total Time = {strategy1_time} minutes")

# Strategy 2: Select 20 features then preprocess them
strategy2_time = (total_features * selection_time_per_feature + 
                  selected_features * preprocessing_time_per_feature)
print(f"\nStrategy 2: Select {selected_features} features then preprocess them")
print(f"  Time = {total_features} × {selection_time_per_feature} + {selected_features} × {preprocessing_time_per_feature}")
print(f"  Time = {total_features * selection_time_per_feature} + {selected_features * preprocessing_time_per_feature}")
print(f"  Total Time = {strategy2_time} minutes")

# Strategy 3: Preprocess 50 features then select 20
strategy3_time = (preprocessed_features * preprocessing_time_per_feature + 
                  preprocessed_features * selection_time_per_feature)
print(f"\nStrategy 3: Preprocess {preprocessed_features} features then select {selected_features}")
print(f"  Time = {preprocessed_features} × {preprocessing_time_per_feature} + {preprocessed_features} × {selection_time_per_feature}")
print(f"  Time = {preprocessed_features * preprocessing_time_per_feature} + {preprocessed_features * selection_time_per_feature}")
print(f"  Total Time = {strategy3_time} minutes")

# Find fastest strategy
strategies = [
    ("Strategy 1", strategy1_time),
    ("Strategy 2", strategy2_time),
    ("Strategy 3", strategy3_time)
]

fastest_strategy = min(strategies, key=lambda x: x[1])
slowest_strategy = max(strategies, key=lambda x: x[1])

print(f"\nResults:")
print(f"  Fastest: {fastest_strategy[0]} ({fastest_strategy[1]} minutes)")
print(f"  Slowest: {slowest_strategy[0]} ({slowest_strategy[1]} minutes)")

# Calculate time savings
time_savings = {}
for name, time_val in strategies:
    if name != fastest_strategy[0]:
        savings = time_val - fastest_strategy[1]
        time_savings[name] = savings
        print(f"  {name} is {savings} minutes slower than {fastest_strategy[0]}")

# Visualization 1: Pipeline Timing Comparison
plt.figure(figsize=(12, 8))

# Create bar chart
strategy_names = [s[0] for s in strategies]
strategy_times = [s[1] for s in strategies]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = plt.bar(strategy_names, strategy_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, time_val in zip(bars, strategy_times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{time_val} min', ha='center', va='bottom', fontweight='bold')

# Highlight fastest strategy
fastest_idx = strategy_names.index(fastest_strategy[0])
bars[fastest_idx].set_edgecolor('green')
bars[fastest_idx].set_linewidth(3)

plt.xlabel('Pipeline Strategy', fontsize=14)
plt.ylabel('Total Time (minutes)', fontsize=14)
plt.title('Feature Selection Pipeline Timing Comparison', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(strategy_times) * 1.1)

# Add annotation for fastest strategy
plt.annotate(f'Fastest Strategy\n{fastest_strategy[0]}\n{fastest_strategy[1]} minutes',
             xy=(fastest_idx, fastest_strategy[1]), xycoords='data',
             xytext=(fastest_idx + 0.5, fastest_strategy[1] + 50), textcoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='green', lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", lw=2),
             fontsize=12, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pipeline_timing_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Detailed Pipeline Breakdown
plt.figure(figsize=(15, 10))

# Create subplots for each strategy
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Detailed Pipeline Breakdown by Strategy', fontsize=16, fontweight='bold')

# Strategy 1 breakdown
preprocess_time1 = total_features * preprocessing_time_per_feature
select_time1 = total_features * selection_time_per_feature

axes[0].pie([preprocess_time1, select_time1], 
            labels=[f'Preprocessing\n{preprocess_time1} min', f'Selection\n{select_time1} min'],
            colors=['#FF9999', '#66B2FF'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Strategy 1:\nPreprocess All → Select', fontweight='bold')

# Strategy 2 breakdown
select_time2 = total_features * selection_time_per_feature
preprocess_time2 = selected_features * preprocessing_time_per_feature

axes[1].pie([select_time2, preprocess_time2], 
            labels=[f'Selection\n{select_time2} min', f'Preprocessing\n{preprocess_time2} min'],
            colors=['#66B2FF', '#FF9999'], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Strategy 2:\nSelect → Preprocess Selected', fontweight='bold')

# Strategy 3 breakdown
preprocess_time3 = preprocessed_features * preprocessing_time_per_feature
select_time3 = preprocessed_features * selection_time_per_feature

axes[2].pie([preprocess_time3, select_time3], 
            labels=[f'Preprocessing\n{preprocess_time3} min', f'Selection\n{select_time3} min'],
            colors=['#FF9999', '#66B2FF'], autopct='%1.1f%%', startangle=90)
axes[2].set_title('Strategy 3:\nPreprocess 50 → Select', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pipeline_breakdown.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Feature Selection Pipeline Flow
plt.figure(figsize=(16, 10))

# Create a flowchart-style visualization
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Define positions for different pipeline stages
stages = {
    'raw_data': (1, 8),
    'preprocessing': (4, 8),
    'feature_selection': (7, 8),
    'model_training': (10, 8),
    'evaluation': (13, 8)
}

# Draw arrows and boxes for different strategies
strategies_flow = {
    'Strategy 1': {
        'path': [(1, 8), (4, 8), (7, 8), (10, 8), (13, 8)],
        'color': '#FF6B6B',
        'label': 'Preprocess All → Select'
    },
    'Strategy 2': {
        'path': [(1, 6), (7, 6), (4, 6), (10, 6), (13, 6)],
        'color': '#4ECDC4',
        'label': 'Select → Preprocess Selected'
    },
    'Strategy 3': {
        'path': [(1, 4), (4, 4), (7, 4), (10, 4), (13, 4)],
        'color': '#45B7D1',
        'label': 'Preprocess 50 → Select'
    }
}

# Draw stage boxes
for stage_name, (x, y) in stages.items():
    box = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor='lightgray', 
                         edgecolor='black', 
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, stage_name.replace('_', '\n'), ha='center', va='center', 
            fontweight='bold', fontsize=10)

# Draw strategy paths
for strategy_name, strategy_info in strategies_flow.items():
    path = strategy_info['path']
    color = strategy_info['color']
    label = strategy_info['label']
    
    # Draw arrows
    for i in range(len(path) - 1):
        start_x, start_y = path[i]
        end_x, end_y = path[i + 1]
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=3))
        
        # Add arrow label for first arrow
        if i == 0:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                   fontsize=9, color=color, fontweight='bold')

# Add time annotations
time_annotations = [
    (4, 7, f'Preprocessing:\n{preprocessing_time_per_feature} min/feature'),
    (7, 7, f'Selection:\n{selection_time_per_feature} min/feature'),
    (10, 7, 'Model Training'),
    (13, 7, 'Evaluation')
]

for x, y, text in time_annotations:
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

ax.set_xlim(0, 14)
ax.set_ylim(2, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Feature Selection Pipeline Strategies Flow', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pipeline_flow.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Time Complexity Analysis
plt.figure(figsize=(12, 8))

# Create a range of feature counts to analyze
feature_counts = np.arange(10, 201, 10)

# Calculate times for different strategies
strategy1_times = feature_counts * preprocessing_time_per_feature + feature_counts * selection_time_per_feature
strategy2_times = feature_counts * selection_time_per_feature + 20 * preprocessing_time_per_feature  # Assuming 20 selected features
strategy3_times = np.full_like(feature_counts, 50 * preprocessing_time_per_feature + 50 * selection_time_per_feature)  # Constant time for Strategy 3

plt.plot(feature_counts, strategy1_times, 'o-', color='#FF6B6B', linewidth=3, 
         markersize=6, label='Strategy 1: Preprocess All → Select')
plt.plot(feature_counts, strategy2_times, 's-', color='#4ECDC4', linewidth=3, 
         markersize=6, label='Strategy 2: Select → Preprocess Selected')
plt.plot(feature_counts, strategy3_times, '^-', color='#45B7D1', linewidth=3, 
         markersize=6, label='Strategy 3: Preprocess 50 → Select')

# Highlight the crossover point
crossover_idx = np.where(strategy1_times > strategy2_times)[0]
if len(crossover_idx) > 0:
    crossover_features = feature_counts[crossover_idx[0]]
    crossover_time = strategy2_times[crossover_idx[0]]
    plt.axvline(x=crossover_features, color='red', linestyle='--', alpha=0.7, 
                label=f'Crossover at {crossover_features} features')
    plt.axhline(y=crossover_time, color='red', linestyle='--', alpha=0.7)

plt.xlabel('Number of Features', fontsize=14)
plt.ylabel('Total Pipeline Time (minutes)', fontsize=14)
plt.title('Pipeline Time vs Number of Features', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(10, 200)

# Add annotation for crossover point
if len(crossover_idx) > 0:
    plt.annotate(f'Crossover Point:\n{crossover_features} features\n{crossover_time:.0f} minutes',
                 xy=(crossover_features, crossover_time), xycoords='data',
                 xytext=(crossover_features + 20, crossover_time + 50), textcoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", lw=2),
                 fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'time_complexity_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Online Learning Feature Selection
plt.figure(figsize=(14, 10))

# Simulate online learning scenario
np.random.seed(42)
n_samples = 1000
n_features = 50
n_relevant_features = 15

# Generate synthetic data
X = np.random.randn(n_samples, n_features)
# Make some features relevant
relevant_features = np.random.choice(n_features, n_relevant_features, replace=False)
for i, feat_idx in enumerate(relevant_features):
    X[:, feat_idx] = X[:, feat_idx] * (1 + 0.5 * i) + np.random.randn(n_samples) * 0.1

# Simulate online learning with feature selection
batch_size = 100
feature_importance_history = []
selected_features_history = []

for batch_start in range(0, n_samples, batch_size):
    batch_end = min(batch_start + batch_size, n_samples)
    X_batch = X[batch_start:batch_end]
    
    # Calculate feature importance for this batch (simplified)
    feature_importance = np.abs(np.mean(X_batch, axis=0))
    feature_importance_history.append(feature_importance)
    
    # Select top features for this batch
    top_features = np.argsort(feature_importance)[-n_relevant_features:]
    selected_features_history.append(top_features)

# Convert to arrays
feature_importance_history = np.array(feature_importance_history)
selected_features_history = np.array(selected_features_history)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Online Learning Feature Selection Analysis', fontsize=16, fontweight='bold')

# Plot 1: Feature importance evolution
im1 = axes[0, 0].imshow(feature_importance_history.T, aspect='auto', cmap='viridis')
axes[0, 0].set_xlabel('Batch Number', fontsize=12)
axes[0, 0].set_ylabel('Feature Index', fontsize=12)
axes[0, 0].set_title('Feature Importance Evolution Over Time', fontweight='bold')
plt.colorbar(im1, ax=axes[0, 0], label='Importance Score')

# Plot 2: Selected features over time
selected_features_matrix = np.zeros((n_features, len(selected_features_history)))
for i, selected in enumerate(selected_features_history):
    selected_features_matrix[selected, i] = 1

im2 = axes[0, 1].imshow(selected_features_matrix, aspect='auto', cmap='Reds')
axes[0, 1].set_xlabel('Batch Number', fontsize=12)
axes[0, 1].set_ylabel('Feature Index', fontsize=12)
axes[0, 1].set_title('Feature Selection Over Time', fontweight='bold')
axes[0, 1].set_yticks(range(0, n_features, 5))

# Plot 3: Stability of feature selection
feature_selection_frequency = np.sum(selected_features_matrix, axis=1)
axes[1, 0].bar(range(n_features), feature_selection_frequency, 
                color=['red' if i in relevant_features else 'blue' for i in range(n_features)],
                alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Feature Index', fontsize=12)
axes[1, 0].set_ylabel('Number of Times Selected', fontsize=12)
axes[1, 0].set_title('Feature Selection Frequency', fontweight='bold')
axes[1, 0].axhline(y=np.mean(feature_selection_frequency), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(feature_selection_frequency):.1f}')
axes[1, 0].legend()

# Plot 4: Convergence analysis
cumulative_importance = np.cumsum(feature_importance_history, axis=0)
axes[1, 1].plot(range(len(cumulative_importance)), 
                np.std(cumulative_importance, axis=1), 'o-', color='green', linewidth=2)
axes[1, 1].set_xlabel('Batch Number', fontsize=12)
axes[1, 1].set_ylabel('Std Dev of Feature Importance', fontsize=12)
axes[1, 1].set_title('Convergence of Feature Selection', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'online_learning_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary of results
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(f"1. Fastest Strategy: {fastest_strategy[0]} ({fastest_strategy[1]} minutes)")
print(f"2. Time Savings:")
for strategy, savings in time_savings.items():
    print(f"   - {strategy}: {savings} minutes slower")
print(f"3. Key Insight: {fastest_strategy[0]} is most efficient because it minimizes")
print(f"   the total number of features processed through the entire pipeline.")
print(f"4. Online Learning: Feature selection can be performed incrementally,")
print(f"   adapting to new data while maintaining stability over time.")
