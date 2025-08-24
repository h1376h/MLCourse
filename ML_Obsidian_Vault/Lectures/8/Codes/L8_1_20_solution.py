import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 20: Feature Selection Decision Analysis")
print("=" * 60)

# 1. Generate synthetic dataset with known feature importance
print("\n1. Generating synthetic dataset...")
np.random.seed(42)
X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=15,  # Only 15 features are actually informative
    n_redundant=10,    # 10 features are redundant (correlated)
    n_repeated=5,      # 5 features are repeated
    random_state=42
)

# Create feature names
feature_names = [f'Feature_{i+1}' for i in range(30)]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_names)}")
print(f"Samples: {len(df)}")
print(f"Target distribution: {np.bincount(y)}")

# 2. Feature Selection Analysis
print("\n2. Performing feature selection analysis...")

# 2.1 Correlation-based selection
print("\n2.1 Correlation-based selection...")
correlation_matrix = df.drop('target', axis=1).corr()
correlation_with_target = df.drop('target', axis=1).corrwith(df['target']).abs()

# Find highly correlated features (correlation > 0.8)
high_corr_features = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_features.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

print(f"Number of highly correlated feature pairs (|corr| > 0.8): {len(high_corr_features)}")

# Select top features by correlation with target
correlation_threshold = 0.1
correlation_selected = correlation_with_target[correlation_with_target > correlation_threshold].index.tolist()
print(f"Features selected by correlation (threshold > {correlation_threshold}): {len(correlation_selected)}")

# 2.2 Mutual Information
print("\n2.2 Mutual Information analysis...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

mi_threshold = np.percentile(mi_scores, 70)  # Top 30% features
mi_selected = mi_df[mi_df['MI_Score'] > mi_threshold]['Feature'].tolist()
print(f"Features selected by mutual information (top 30%): {len(mi_selected)}")

# 2.3 Recursive Feature Elimination
print("\n2.3 Recursive Feature Elimination...")
estimator = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator, n_features_to_select=15, step=1)
rfe.fit(X, y)

rfe_selected = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
print(f"Features selected by RFE: {len(rfe_selected)}")

# 3. Performance Comparison
print("\n3. Comparing model performance...")

def evaluate_features(feature_subset, X, y, method_name):
    if len(feature_subset) == 0:
        return 0.0
    
    # Get feature indices
    feature_indices = [feature_names.index(f) for f in feature_subset]
    X_subset = X[:, feature_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Evaluate with cross-validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    return scores.mean()

# Evaluate different feature selection methods
correlation_performance = evaluate_features(correlation_selected, X, y, "Correlation")
mi_performance = evaluate_features(mi_selected, X, y, "Mutual Information")
rfe_performance = evaluate_features(rfe_selected, X, y, "RFE")
all_features_performance = evaluate_features(feature_names, X, y, "All Features")

print(f"Performance with correlation selection: {correlation_performance:.4f}")
print(f"Performance with mutual information: {mi_performance:.4f}")
print(f"Performance with RFE: {rfe_performance:.4f}")
print(f"Performance with all features: {all_features_performance:.4f}")

# 4. Time Allocation Analysis
print("\n4. Time allocation analysis...")

# Given time constraints
total_project_time = 40  # hours
correlation_time = 1     # hour
mi_time = 3             # hours
rfe_time = 8            # hours
target_accuracy = 0.80  # 80%

# Calculate time for different approaches
time_allocation = {
    'Correlation': correlation_time,
    'Mutual Information': mi_time,
    'RFE': rfe_time
}

# Estimate model training and evaluation time
model_training_time = 20  # hours
feature_selection_time = min(correlation_time, mi_time, rfe_time)  # Choose fastest method

total_feature_selection_time = feature_selection_time
total_model_time = model_training_time

percentage_feature_selection = (total_feature_selection_time / total_project_time) * 100
percentage_model = (total_model_time / total_project_time) * 100

print(f"Time allocation:")
print(f"  Feature selection: {total_feature_selection_time} hours ({percentage_feature_selection:.1f}%)")
print(f"  Model training/evaluation: {total_model_time} hours ({percentage_model:.1f}%)")
print(f"  Remaining time: {total_project_time - total_feature_selection_time - total_model_time} hours")

# 5. Visualizations
print("\n5. Generating visualizations...")

# 5.1 Correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.2 Correlation with target
plt.figure(figsize=(12, 8))
correlation_with_target_sorted = correlation_with_target.sort_values(ascending=False)
plt.bar(range(len(correlation_with_target_sorted)), correlation_with_target_sorted.values)
plt.axhline(y=correlation_threshold, color='red', linestyle='--', label=f'Threshold ({correlation_threshold})')
plt.xlabel('Features')
plt.ylabel('Absolute Correlation with Target')
plt.title('Feature Correlation with Target Variable')
plt.xticks(range(len(correlation_with_target_sorted)), correlation_with_target_sorted.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_with_target.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.3 Mutual Information scores
plt.figure(figsize=(12, 8))
mi_df_sorted = mi_df.sort_values('MI_Score', ascending=False)
plt.bar(range(len(mi_df_sorted)), mi_df_sorted['MI_Score'].values)
plt.axhline(y=mi_threshold, color='red', linestyle='--', label=f'Threshold ({mi_threshold:.4f})')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.title('Feature Mutual Information Scores')
plt.xticks(range(len(mi_df_sorted)), mi_df_sorted['Feature'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mutual_information_scores.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.4 Performance comparison
plt.figure(figsize=(10, 8))
methods = ['Correlation', 'Mutual Info', 'RFE', 'All Features']
performances = [correlation_performance, mi_performance, rfe_performance, all_features_performance]
colors = ['skyblue', 'lightgreen', 'orange', 'red']

bars = plt.bar(methods, performances, color=colors, alpha=0.8)
plt.axhline(y=target_accuracy, color='red', linestyle='--', linewidth=2, label=f'Target Accuracy ({target_accuracy})')
plt.ylabel('Accuracy Score')
plt.title('Model Performance with Different Feature Selection Methods')
plt.ylim(0, 1)

# Add value labels on bars
for bar, perf in zip(bars, performances):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{perf:.3f}', ha='center', va='bottom')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.5 Time allocation pie chart
plt.figure(figsize=(10, 8))
labels = ['Feature Selection', 'Model Training/Evaluation', 'Remaining Time']
sizes = [total_feature_selection_time, total_model_time, total_project_time - total_feature_selection_time - total_model_time]
colors = ['lightcoral', 'lightblue', 'lightgreen']
explode = (0.1, 0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Project Time Allocation')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'time_allocation.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.6 Feature selection overlap
plt.figure(figsize=(12, 8))
from matplotlib_venn import venn3

# Convert to sets for Venn diagram
correlation_set = set(correlation_selected)
mi_set = set(mi_selected)
rfe_set = set(rfe_selected)

# Create Venn diagram
venn3([correlation_set, mi_set, rfe_set], 
      ('Correlation', 'Mutual Info', 'RFE'))
plt.title('Feature Selection Method Overlap')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_overlap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Summary and Recommendations
print("\n6. Summary and Recommendations:")
print("=" * 60)

print(f"\nFeature Selection Results:")
print(f"  Correlation method: {len(correlation_selected)} features, Accuracy: {correlation_performance:.4f}")
print(f"  Mutual Information: {len(mi_selected)} features, Accuracy: {mi_performance:.4f}")
print(f"  RFE method: {len(rfe_selected)} features, Accuracy: {rfe_performance:.4f}")
print(f"  All features: {len(feature_names)} features, Accuracy: {all_features_performance:.4f}")

print(f"\nTime Analysis:")
print(f"  Total project time: {total_project_time} hours")
print(f"  Recommended feature selection time: {total_feature_selection_time} hours ({percentage_feature_selection:.1f}%)")
print(f"  Model development time: {total_model_time} hours ({percentage_model:.1f}%)")

print(f"\nRecommendations:")
if correlation_performance >= target_accuracy:
    print(f"  ✓ Use correlation-based selection (fastest, meets accuracy target)")
elif mi_performance >= target_accuracy:
    print(f"  ✓ Use mutual information selection (good balance of speed and performance)")
elif rfe_performance >= target_accuracy:
    print(f"  ✓ Use RFE if time allows (best performance but slowest)")
else:
    print(f"  ⚠ Consider using all features or ensemble methods")

print(f"\nPlots saved to: {save_dir}")
print("Analysis complete!")
