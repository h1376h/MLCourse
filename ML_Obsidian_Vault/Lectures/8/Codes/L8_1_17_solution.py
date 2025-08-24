import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_17")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 17: Data Type-Specific Feature Selection Strategies")
print("=" * 60)

# Task 6: TF-IDF Calculation
print("\nTask 6: TF-IDF Calculation")
print("-" * 30)

N = 1000  # Total documents
DF = 100  # Document frequency (word appears in 100 documents)
TF = 5    # Term frequency in one document

# Calculate TF-IDF score
tf_idf_score = TF * np.log(N / DF)
print(f"Given parameters:")
print(f"  N (total documents) = {N}")
print(f"  DF (document frequency) = {DF}")
print(f"  TF (term frequency) = {TF}")
print(f"\nTF-IDF calculation:")
print(f"  TF-IDF = TF × log(N/DF)")
print(f"  TF-IDF = {TF} × log({N}/{DF})")
print(f"  TF-IDF = {TF} × log({N/DF:.2f})")
print(f"  TF-IDF = {TF} × {np.log(N/DF):.4f}")
print(f"  TF-IDF = {tf_idf_score:.4f}")

threshold = 2.0
print(f"\nThreshold = {threshold}")
if tf_idf_score >= threshold:
    print(f"✓ Word would be SELECTED as a feature (TF-IDF = {tf_idf_score:.4f} ≥ {threshold})")
else:
    print(f"✗ Word would NOT be selected as a feature (TF-IDF = {tf_idf_score:.4f} < {threshold})")

# Create comprehensive visualizations for different data types
print("\nCreating visualizations for different data types...")

# 1. Text Data Feature Selection
print("\n1. Text Data Feature Selection")

# Sample text data
documents = [
    "machine learning algorithms are powerful tools",
    "deep learning neural networks excel at pattern recognition",
    "natural language processing handles text data",
    "computer vision processes image data effectively",
    "reinforcement learning optimizes decision making",
    "supervised learning uses labeled training data",
    "unsupervised learning finds hidden patterns",
    "feature selection improves model performance",
    "dimensionality reduction reduces computational cost",
    "ensemble methods combine multiple models"
]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Calculate average TF-IDF scores for each feature
avg_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

# Select top features
top_k = 10
top_indices = np.argsort(avg_tfidf_scores)[-top_k:]
top_features = [feature_names[i] for i in top_indices]
top_scores = [avg_tfidf_scores[i] for i in top_indices]

# Plot text feature selection
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
bars = plt.barh(range(len(top_features)), top_scores, color='skyblue', edgecolor='navy')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Average TF-IDF Score')
plt.title('Text Data: Top Features by TF-IDF Score')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, top_scores)):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', ha='left', va='center', fontsize=9)

# 2. Numerical Data Feature Selection
print("\n2. Numerical Data Feature Selection")

# Generate sample numerical data
np.random.seed(42)
n_samples, n_features = 200, 15
X_numerical = np.random.randn(n_samples, n_features)
y_numerical = (X_numerical[:, 0] + 2*X_numerical[:, 1] + 0.5*X_numerical[:, 2] + 
               0.1*np.random.randn(n_samples) > 0).astype(int)

# Feature selection methods
# ANOVA F-test
f_scores, f_pvalues = f_classif(X_numerical, y_numerical)
# Mutual Information
mi_scores = mutual_info_classif(X_numerical, y_numerical, random_state=42)

# Create feature importance comparison
feature_names_num = [f'Feature_{i+1}' for i in range(n_features)]
feature_importance_df = pd.DataFrame({
    'Feature': feature_names_num,
    'F-Score': f_scores,
    'P-Value': f_pvalues,
    'Mutual_Info': mi_scores
})

# Select top features by F-score
top_f_features = feature_importance_df.nlargest(8, 'F-Score')

plt.subplot(2, 2, 2)
bars = plt.barh(range(len(top_f_features)), top_f_features['F-Score'], 
                color='lightgreen', edgecolor='darkgreen')
plt.yticks(range(len(top_f_features)), top_f_features['Feature'])
plt.xlabel('F-Score')
plt.title('Numerical Data: Top Features by F-Score')
plt.grid(True, alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top_f_features['F-Score'])):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{score:.1f}', ha='left', va='center', fontsize=9)

# 3. Image Data Feature Selection
print("\n3. Image Data Feature Selection")

# Simulate image features (pixel values, texture features, etc.)
n_image_features = 20
image_features = np.random.randn(100, n_image_features)
image_labels = np.random.randint(0, 2, 100)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)
pca_features = pca.fit_transform(image_features)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.subplot(2, 2, 3)
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 
         'bo-', linewidth=2, markersize=6)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Image Data: PCA Dimensionality Reduction')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
plt.legend()

# 4. Time Series Feature Selection
print("\n4. Time Series Feature Selection")

# Generate time series data
time_points = np.linspace(0, 10, 100)
np.random.seed(42)

# Create multiple time series with different patterns
ts1 = np.sin(2 * np.pi * time_points) + 0.1 * np.random.randn(100)  # Seasonal
ts2 = time_points + 0.2 * np.random.randn(100)  # Trend
ts3 = np.exp(-time_points/3) + 0.1 * np.random.randn(100)  # Decay
ts4 = 0.5 * np.random.randn(100)  # Noise

# Combine into feature matrix
X_timeseries = np.column_stack([ts1, ts2, ts3, ts4])
y_timeseries = (ts1 + ts2 > 1.5).astype(int)

# Calculate correlation with target
correlations = [np.corrcoef(X_timeseries[:, i], y_timeseries)[0, 1] for i in range(4)]
feature_names_ts = ['Seasonal', 'Trend', 'Decay', 'Noise']

plt.subplot(2, 2, 4)
bars = plt.barh(range(len(feature_names_ts)), np.abs(correlations), 
                color='lightcoral', edgecolor='darkred')
plt.yticks(range(len(feature_names_ts)), feature_names_ts)
plt.xlabel('|Correlation with Target|')
plt.title('Time Series: Feature-Target Correlation')
plt.grid(True, alpha=0.3)

# Add value labels
for i, (bar, corr) in enumerate(zip(bars, correlations)):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{corr:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_by_data_type.png'), dpi=300, bbox_inches='tight')

# 5. Mixed Data Types Feature Selection
print("\n5. Mixed Data Types Feature Selection")

# Create mixed data scenario
n_samples_mixed = 150
numerical_features = np.random.randn(n_samples_mixed, 5)
categorical_features = np.random.randint(0, 3, (n_samples_mixed, 3))
text_features = np.random.randint(0, 10, (n_samples_mixed, 4))  # Simplified text representation

# Combine features
X_mixed = np.column_stack([numerical_features, categorical_features, text_features])
y_mixed = (numerical_features[:, 0] + categorical_features[:, 0] > 1).astype(int)

# Apply different selection methods
# For numerical features
numerical_indices = list(range(5))
f_scores_mixed, _ = f_classif(X_mixed[:, numerical_indices], y_mixed)

# For categorical features
categorical_indices = list(range(5, 8))
chi2_scores = []
for i in categorical_indices:
    contingency_table = pd.crosstab(X_mixed[:, i], y_mixed)
    chi2, p_val = stats.chi2_contingency(contingency_table)[:2]
    chi2_scores.append(chi2)

# For text-like features
text_indices = list(range(8, 12))
mi_scores_mixed = mutual_info_classif(X_mixed[:, text_indices], y_mixed, random_state=42)

# Create comprehensive feature importance plot
plt.figure(figsize=(14, 10))

# Feature importance by type
feature_types = ['Numerical'] * 5 + ['Categorical'] * 3 + ['Text-like'] * 4
feature_names_mixed = ([f'Num_{i+1}' for i in range(5)] + 
                       [f'Cat_{i+1}' for i in range(3)] + 
                       [f'Text_{i+1}' for i in range(4)])

# Combine all scores
all_scores = list(f_scores_mixed) + chi2_scores + list(mi_scores_mixed)

# Create DataFrame for easier plotting
mixed_df = pd.DataFrame({
    'Feature': feature_names_mixed,
    'Type': feature_types,
    'Score': all_scores
})

# Plot by feature type
colors = {'Numerical': 'skyblue', 'Categorical': 'lightgreen', 'Text-like': 'lightcoral'}
plt.subplot(2, 1, 1)

for feature_type in ['Numerical', 'Categorical', 'Text-like']:
    type_data = mixed_df[mixed_df['Type'] == feature_type]
    plt.bar(range(len(type_data)), type_data['Score'], 
            label=feature_type, color=colors[feature_type], alpha=0.7)

plt.xlabel('Feature Index')
plt.ylabel('Feature Importance Score')
plt.title('Mixed Data Types: Feature Importance by Type')
plt.legend()
plt.grid(True, alpha=0.3)

# Top features overall
plt.subplot(2, 1, 2)
top_features_mixed = mixed_df.nlargest(8, 'Score')
bars = plt.barh(range(len(top_features_mixed)), top_features_mixed['Score'], 
                color=[colors[ft] for ft in top_features_mixed['Type']], alpha=0.7)

plt.yticks(range(len(top_features_mixed)), top_features_mixed['Feature'])
plt.xlabel('Feature Importance Score')
plt.title('Mixed Data Types: Top 8 Features Overall')
plt.grid(True, alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top_features_mixed['Score'])):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mixed_data_feature_selection.png'), dpi=300, bbox_inches='tight')

# 6. Feature Selection Strategy Comparison
print("\n6. Feature Selection Strategy Comparison")

# Compare different selection methods
methods = ['Correlation', 'F-Test', 'Mutual Info', 'Random Forest', 'PCA']
performance_metrics = [0.75, 0.82, 0.85, 0.88, 0.80]  # Example accuracy scores
time_complexity = [1, 2, 3, 4, 2]  # Relative time complexity
interpretability = [0.9, 0.8, 0.7, 0.6, 0.4]  # Interpretability scores

# Create comparison plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Performance comparison
bars1 = ax1.bar(methods, performance_metrics, color='lightblue', edgecolor='navy')
ax1.set_ylabel('Accuracy Score')
ax1.set_title('Performance Comparison')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
for bar, score in zip(bars1, performance_metrics):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.2f}', ha='center', va='bottom')

# Time complexity comparison
bars2 = ax2.bar(methods, time_complexity, color='lightgreen', edgecolor='darkgreen')
ax2.set_ylabel('Relative Time Complexity')
ax2.set_title('Computational Cost')
ax2.grid(True, alpha=0.3)
for bar, time in zip(bars2, time_complexity):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{time}', ha='center', va='bottom')

# Interpretability comparison
bars3 = ax3.bar(methods, interpretability, color='lightcoral', edgecolor='darkred')
ax3.set_ylabel('Interpretability Score')
ax3.set_title('Model Interpretability')
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)
for bar, score in zip(bars3, interpretability):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_methods_comparison.png'), dpi=300, bbox_inches='tight')

# 7. Detailed TF-IDF Analysis
print("\n7. Detailed TF-IDF Analysis")

# Create a more detailed TF-IDF example
sample_docs = [
    "machine learning algorithms",
    "deep learning neural networks",
    "natural language processing",
    "computer vision image recognition",
    "reinforcement learning optimization"
]

# Calculate TF-IDF for each document
tfidf_detailed = TfidfVectorizer(max_features=15, stop_words='english')
tfidf_matrix_detailed = tfidf_detailed.fit_transform(sample_docs)
feature_names_detailed = tfidf_detailed.get_feature_names_out()

# Create heatmap
plt.figure(figsize=(12, 8))
tfidf_array = tfidf_matrix_detailed.toarray()
sns.heatmap(tfidf_array, 
            xticklabels=feature_names_detailed,
            yticklabels=[f'Doc {i+1}' for i in range(len(sample_docs))],
            annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('TF-IDF Matrix Heatmap')
plt.xlabel('Features (Words)')
plt.ylabel('Documents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tfidf_heatmap.png'), dpi=300, bbox_inches='tight')

# Feature importance by average TF-IDF
avg_tfidf_detailed = np.mean(tfidf_array, axis=0)
top_indices_detailed = np.argsort(avg_tfidf_detailed)[-10:]
top_features_detailed = [feature_names_detailed[i] for i in top_indices_detailed]
top_scores_detailed = [avg_tfidf_detailed[i] for i in top_indices_detailed]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_features_detailed)), top_scores_detailed, 
                color='gold', edgecolor='orange')
plt.yticks(range(len(top_features_detailed)), top_features_detailed)
plt.xlabel('Average TF-IDF Score')
plt.title('Top 10 Features by Average TF-IDF Score')
plt.grid(True, alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top_scores_detailed)):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tfidf_feature_importance.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary of results
print("\n" + "="*60)
print("SUMMARY OF FEATURE SELECTION STRATEGIES BY DATA TYPE")
print("="*60)

print("\n1. TEXT DATA:")
print(f"   - Top feature: '{top_features[-1]}' (TF-IDF: {top_scores[-1]:.3f})")
print(f"   - Uses TF-IDF scoring with threshold consideration")
print(f"   - Example calculation: TF-IDF = {tf_idf_score:.4f} for given parameters")

print("\n2. NUMERICAL DATA:")
print(f"   - Top feature: '{top_f_features.iloc[-1]['Feature']}' (F-Score: {top_f_features.iloc[-1]['F-Score']:.1f})")
print(f"   - Uses statistical tests (F-test, mutual information)")
print(f"   - Considers correlation with target variable")

print("\n3. IMAGE DATA:")
print(f"   - PCA reduces {n_image_features} features to 10 components")
print(f"   - First 5 components explain {np.sum(explained_variance_ratio[:5]):.1%} of variance")
print(f"   - Focuses on dimensionality reduction and feature extraction")

print("\n4. TIME SERIES DATA:")
print(f"   - Top feature: '{feature_names_ts[np.argmax(np.abs(correlations))]}' (|Corr|: {np.max(np.abs(correlations)):.3f})")
print(f"   - Considers temporal patterns and correlations")
print(f"   - Uses lag features and statistical measures")

print("\n5. MIXED DATA TYPES:")
print(f"   - Combines multiple selection methods")
print(f"   - Top overall feature: '{top_features_mixed.iloc[-1]['Feature']}' (Score: {top_features_mixed.iloc[-1]['Score']:.3f})")
print(f"   - Adapts selection strategy to each data type")

print("\n6. METHOD COMPARISON:")
print(f"   - Best performance: Random Forest (0.88)")
print(f"   - Fastest: Correlation-based (1x)")
print(f"   - Most interpretable: Correlation-based (0.9)")

print(f"\nAll detailed visualizations and analysis saved to: {save_dir}")
