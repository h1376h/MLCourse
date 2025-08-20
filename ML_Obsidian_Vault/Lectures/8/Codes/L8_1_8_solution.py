import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 8: DOMAIN-SPECIFIC FEATURE SELECTION CONSIDERATIONS")
print("=" * 80)

# ============================================================================
# PART 1: Key Considerations for Medical Diagnosis Features
# ============================================================================
print("\n" + "="*60)
print("PART 1: Key Considerations for Medical Diagnosis Features")
print("="*60)

# Create synthetic medical data for demonstration
np.random.seed(42)
n_samples = 1000

# Simulate medical features with different characteristics
medical_features = {
    'blood_pressure': np.random.normal(120, 20, n_samples),
    'heart_rate': np.random.normal(75, 15, n_samples),
    'cholesterol': np.random.normal(200, 40, n_samples),
    'blood_sugar': np.random.normal(100, 20, n_samples),
    'age': np.random.normal(55, 15, n_samples),
    'bmi': np.random.normal(25, 5, n_samples)
}

# Create target variable (disease presence) based on some features
disease_prob = 1 / (1 + np.exp(-(
    -5 + 0.02 * medical_features['blood_pressure'] + 
    0.03 * medical_features['cholesterol'] + 
    0.05 * medical_features['age'] - 
    0.1 * medical_features['bmi']
)))
disease_presence = np.random.binomial(1, disease_prob)

# Convert to DataFrame
medical_df = pd.DataFrame(medical_features)
medical_df['disease'] = disease_presence

print("Medical Dataset Overview:")
print(f"Number of samples: {n_samples}")
print(f"Number of features: {len(medical_features)}")
print(f"Disease prevalence: {disease_presence.mean():.3f}")

# Calculate feature importance using different methods
print("\nFeature Importance Analysis:")

# Method 1: Mutual Information
mi_scores = mutual_info_classif(medical_df.drop('disease', axis=1), medical_df['disease'])
mi_df = pd.DataFrame({
    'Feature': list(medical_features.keys()),
    'Mutual_Information': mi_scores
}).sort_values('Mutual_Information', ascending=False)

print("\n1. Mutual Information Scores:")
print(mi_df)

# Method 2: F-statistics
f_scores, p_values = f_classif(medical_df.drop('disease', axis=1), medical_df['disease'])
f_df = pd.DataFrame({
    'Feature': list(medical_features.keys()),
    'F_Score': f_scores,
    'P_Value': p_values
}).sort_values('F_Score', ascending=False)

print("\n2. F-statistics:")
print(f_df)

# Method 3: Correlation with target
correlations = []
for feature in medical_features.keys():
    corr = np.corrcoef(medical_df[feature], medical_df['disease'])[0, 1]
    correlations.append(corr)

corr_df = pd.DataFrame({
    'Feature': list(medical_features.keys()),
    'Correlation': correlations
}).sort_values('Correlation', key=abs, ascending=False)

print("\n3. Correlation with Target:")
print(corr_df)

# Visualize feature importance
plt.figure(figsize=(12, 8))

# Subplot 1: Mutual Information
plt.subplot(2, 2, 1)
bars1 = plt.bar(range(len(mi_df)), mi_df['Mutual_Information'], color='skyblue', alpha=0.7)
plt.title('Feature Importance: Mutual Information')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.xticks(range(len(mi_df)), mi_df['Feature'], rotation=45, ha='right')
for i, bar in enumerate(bars1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{mi_df.iloc[i]["Mutual_Information"]:.3f}', ha='center', va='bottom')

# Subplot 2: F-scores
plt.subplot(2, 2, 2)
bars2 = plt.bar(range(len(f_df)), f_df['F_Score'], color='lightcoral', alpha=0.7)
plt.title('Feature Importance: F-statistics')
plt.xlabel('Features')
plt.ylabel('F-Score')
plt.xticks(range(len(f_df)), f_df['Feature'], rotation=45, ha='right')
for i, bar in enumerate(bars2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{f_df.iloc[i]["F_Score"]:.1f}', ha='center', va='bottom')

# Subplot 3: Correlation
plt.subplot(2, 2, 3)
bars3 = plt.bar(range(len(corr_df)), corr_df['Correlation'], color='lightgreen', alpha=0.7)
plt.title('Feature Importance: Correlation with Target')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(range(len(corr_df)), corr_df['Feature'], rotation=45, ha='right')
for i, bar in enumerate(bars3):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{corr_df.iloc[i]["Correlation"]:.3f}', ha='center', va='bottom')

# Subplot 4: Feature distributions by class
plt.subplot(2, 2, 4)
feature_to_plot = 'blood_pressure'  # Most important feature
plt.hist(medical_df[medical_df['disease']==0][feature_to_plot], alpha=0.7, 
         label='No Disease', bins=20, color='lightblue')
plt.hist(medical_df[medical_df['disease']==1][feature_to_plot], alpha=0.7, 
         label='Disease', bins=20, color='lightcoral')
plt.title(f'Distribution of {feature_to_plot} by Disease Status')
plt.xlabel(feature_to_plot)
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'medical_feature_importance.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 2: Financial vs Image Recognition Feature Selection
# ============================================================================
print("\n" + "="*60)
print("PART 2: Financial vs Image Recognition Feature Selection")
print("="*60)

# Simulate financial data
financial_features = {
    'price': np.random.normal(100, 20, n_samples),
    'volume': np.random.normal(1000000, 500000, n_samples),
    'volatility': np.random.exponential(0.1, n_samples),
    'market_cap': np.random.lognormal(10, 1, n_samples),
    'pe_ratio': np.random.normal(15, 5, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples)
}

# Create financial target (stock movement)
stock_movement = np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3])

financial_df = pd.DataFrame(financial_features)
financial_df['stock_movement'] = stock_movement

# Simulate image features (high-dimensional, sparse)
n_image_features = 100
image_features = np.random.normal(0, 1, (n_samples, n_image_features))
# Make some features sparse (many zeros)
sparsity = 0.8
image_features[np.random.random(image_features.shape) < sparsity] = 0

# Create image target (object classification)
image_target = np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.25, 0.25, 0.25])

print("Financial Dataset:")
print(f"Features: {list(financial_features.keys())}")
print(f"Target classes: {np.unique(stock_movement)}")
print(f"Class distribution: {np.bincount(stock_movement + 1)}")

print("\nImage Dataset:")
print(f"Number of features: {n_image_features}")
print(f"Sparsity: {sparsity:.1%}")
print(f"Target classes: {np.unique(image_target)}")
print(f"Class distribution: {np.bincount(image_target)}")

# Compare feature characteristics
plt.figure(figsize=(15, 10))

# Financial features correlation heatmap
plt.subplot(2, 3, 1)
corr_matrix_financial = financial_df.corr()
sns.heatmap(corr_matrix_financial, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('Financial Features Correlation')

# Financial feature distributions
plt.subplot(2, 3, 2)
financial_df['price'].hist(bins=30, alpha=0.7, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(2, 3, 3)
financial_df['volatility'].hist(bins=30, alpha=0.7, color='lightcoral')
plt.title('Volatility Distribution')
plt.xlabel('Volatility')
plt.ylabel('Frequency')

# Image features sparsity
plt.subplot(2, 3, 4)
sparsity_per_sample = (image_features == 0).sum(axis=1) / n_image_features
plt.hist(sparsity_per_sample, bins=30, alpha=0.7, color='lightgreen')
plt.title('Feature Sparsity per Sample')
plt.xlabel('Sparsity Ratio')
plt.ylabel('Frequency')

# Image feature variance
plt.subplot(2, 3, 5)
feature_variance = np.var(image_features, axis=0)
plt.hist(feature_variance, bins=30, alpha=0.7, color='gold')
plt.title('Feature Variance Distribution')
plt.xlabel('Variance')
plt.ylabel('Frequency')

# Feature importance comparison
plt.subplot(2, 3, 6)
# Calculate feature importance for financial data
financial_importance = np.abs(np.corrcoef(financial_df.drop('stock_movement', axis=1).T, 
                                        financial_df['stock_movement'])[:-1, -1])
financial_importance = np.abs(financial_importance)

# Calculate feature importance for image data (using variance as proxy)
image_importance = feature_variance

# Plot comparison
x_pos = np.arange(len(financial_importance))
plt.bar(x_pos - 0.2, financial_importance, 0.4, label='Financial', alpha=0.7, color='skyblue')
plt.bar(x_pos + 0.2, image_importance[:len(financial_importance)]/np.max(image_importance), 
        0.4, label='Image (normalized)', alpha=0.7, color='lightcoral')
plt.title('Feature Importance Comparison')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.xticks(x_pos, list(financial_features.keys()), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'financial_vs_image_features.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 3: Real-time Sensor Data Feature Selection
# ============================================================================
print("\n" + "="*60)
print("PART 3: Real-time Sensor Data Feature Selection")
print("="*60)

# Simulate real-time sensor data
time_steps = 1000
sensor_features = {
    'temperature': np.random.normal(25, 5, time_steps) + 0.1 * np.sin(np.linspace(0, 4*np.pi, time_steps)),
    'humidity': np.random.normal(60, 10, time_steps) + 0.05 * np.cos(np.linspace(0, 6*np.pi, time_steps)),
    'pressure': np.random.normal(1013, 10, time_steps) + 0.02 * np.sin(np.linspace(0, 8*np.pi, time_steps)),
    'vibration': np.random.exponential(0.1, time_steps) + 0.01 * np.random.normal(0, 1, time_steps),
    'noise': np.random.normal(50, 5, time_steps) + 0.1 * np.random.exponential(0.5, time_steps)
}

# Add some anomalies
anomaly_indices = np.random.choice(time_steps, size=20, replace=False)
for idx in anomaly_indices:
    sensor_features['temperature'][idx] += np.random.normal(0, 10)
    sensor_features['vibration'][idx] += np.random.exponential(0.5)

# Create target (anomaly detection)
anomaly_target = np.zeros(time_steps)
anomaly_target[anomaly_indices] = 1

print("Sensor Data Characteristics:")
print(f"Time steps: {time_steps}")
print(f"Number of features: {len(sensor_features)}")
print(f"Anomaly rate: {anomaly_target.mean():.3f}")

# Analyze temporal characteristics
plt.figure(figsize=(15, 10))

# Time series plots
for i, (feature_name, feature_data) in enumerate(sensor_features.items()):
    plt.subplot(3, 2, i+1)
    plt.plot(feature_data, alpha=0.7, linewidth=1)
    if anomaly_target.sum() > 0:
        anomaly_points = np.where(anomaly_target == 1)[0]
        plt.scatter(anomaly_points, feature_data[anomaly_points], 
                   color='red', s=20, alpha=0.8, label='Anomaly')
    plt.title(f'{feature_name.capitalize()} Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    if i == 0:
        plt.legend()

# Feature correlation over time
plt.subplot(3, 2, 6)
sensor_df = pd.DataFrame(sensor_features)
corr_matrix_sensor = sensor_df.corr()
sns.heatmap(corr_matrix_sensor, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('Sensor Features Correlation')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sensor_data_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: Text vs Numerical Data Feature Selection
# ============================================================================
print("\n" + "="*60)
print("PART 4: Text vs Numerical Data Feature Selection")
print("="*60)

# Simulate text data (TF-IDF like features)
n_text_features = 50
text_features = np.random.exponential(0.1, (n_samples, n_text_features))
# Make text features sparse
text_sparsity = 0.7
text_features[np.random.random(text_features.shape) < text_sparsity] = 0

# Create text target (sentiment classification)
text_target = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])

# Numerical data (already created above)
numerical_target = disease_presence

print("Text Data:")
print(f"Number of features: {n_text_features}")
print(f"Sparsity: {text_sparsity:.1%}")
print(f"Target distribution: {np.bincount(text_target)}")

print("\nNumerical Data:")
print(f"Number of features: {len(medical_features)}")
print(f"Target distribution: {np.bincount(numerical_target)}")

# Compare feature selection methods
plt.figure(figsize=(15, 10))

# Text features sparsity
plt.subplot(2, 3, 1)
text_sparsity_per_sample = (text_features == 0).sum(axis=1) / n_text_features
plt.hist(text_sparsity_per_sample, bins=30, alpha=0.7, color='lightblue')
plt.title('Text Features Sparsity')
plt.xlabel('Sparsity Ratio')
plt.ylabel('Frequency')

# Text feature variance
plt.subplot(2, 3, 2)
text_feature_variance = np.var(text_features, axis=0)
plt.hist(text_feature_variance, bins=20, alpha=0.7, color='lightcoral')
plt.title('Text Feature Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency')

# Numerical feature variance
plt.subplot(2, 3, 3)
numerical_feature_variance = np.var(medical_df.drop('disease', axis=1), axis=0)
plt.hist(numerical_feature_variance, bins=20, alpha=0.7, color='lightgreen')
plt.title('Numerical Feature Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency')

# Feature importance comparison
plt.subplot(2, 3, 4)
# Text feature importance (using variance as proxy)
text_importance = text_feature_variance
# Numerical feature importance (using correlation)
numerical_importance = np.abs(correlations)

plt.bar(range(len(text_importance[:20])), text_importance[:20], alpha=0.7, color='lightblue', label='Text')
plt.title('Text Feature Importance (Top 20)')
plt.xlabel('Feature Index')
plt.ylabel('Variance')

plt.subplot(2, 3, 5)
plt.bar(range(len(numerical_importance)), numerical_importance, alpha=0.7, color='lightgreen', label='Numerical')
plt.title('Numerical Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('|Correlation|')

# Sparsity comparison
plt.subplot(2, 3, 6)
sparsity_comparison = [text_sparsity, 0]  # Numerical data has no sparsity
plt.bar(['Text', 'Numerical'], sparsity_comparison, color=['lightblue', 'lightgreen'], alpha=0.7)
plt.title('Feature Sparsity Comparison')
plt.ylabel('Sparsity Ratio')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'text_vs_numerical_features.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: Interpretable Features Analysis
# ============================================================================
print("\n" + "="*60)
print("PART 5: Interpretable Features Analysis")
print("="*60)

# Create interpretable vs non-interpretable features
interpretable_features = {
    'age': np.random.normal(45, 15, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'education_years': np.random.poisson(16, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples)
}

# Create non-interpretable features (e.g., PCA components)
non_interpretable_features = {
    'pca_1': np.random.normal(0, 1, n_samples),
    'pca_2': np.random.normal(0, 1, n_samples),
    'pca_3': np.random.normal(0, 1, n_samples),
    'pca_4': np.random.normal(0, 1, n_samples)
}

# Create target (loan approval)
loan_prob = 1 / (1 + np.exp(-(
    -2 + 0.01 * interpretable_features['age'] + 
    0.0001 * interpretable_features['income'] + 
    0.1 * interpretable_features['education_years'] + 
    0.005 * interpretable_features['credit_score']
)))
loan_approval = np.random.binomial(1, loan_prob)

print("Interpretable Features:")
for feature, data in interpretable_features.items():
    print(f"{feature}: mean={data.mean():.2f}, std={data.std():.2f}")

print("\nNon-interpretable Features:")
for feature, data in non_interpretable_features.items():
    print(f"{feature}: mean={data.mean():.2f}, std={data.std():.2f}")

# Compare model performance and interpretability
plt.figure(figsize=(15, 10))

# Feature distributions
for i, (feature_name, feature_data) in enumerate(interpretable_features.items()):
    plt.subplot(2, 4, i+1)
    plt.hist(feature_data, bins=20, alpha=0.7, color='lightblue')
    plt.title(f'Interpretable: {feature_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

for i, (feature_name, feature_data) in enumerate(non_interpretable_features.items()):
    plt.subplot(2, 4, i+5)
    plt.hist(feature_data, bins=20, alpha=0.7, color='lightcoral')
    plt.title(f'Non-interpretable: {feature_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'interpretable_vs_non_interpretable.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# PART 6: False Positive vs False Negative Rate Analysis
# ============================================================================
print("\n" + "="*60)
print("PART 6: False Positive vs False Negative Rate Analysis")
print("="*60)

# Given values
fpr_a, fnr_a = 0.05, 0.10  # Feature A
fpr_b, fnr_b = 0.02, 0.15  # Feature B

# Calculate total error rates
total_error_a = fpr_a + fnr_a
total_error_b = fpr_b + fnr_b

print("Feature A:")
print(f"  False Positive Rate (FPR) = {fpr_a:.3f}")
print(f"  False Negative Rate (FNR) = {fnr_a:.3f}")
print(f"  Total Error Rate = {total_error_a:.3f}")

print("\nFeature B:")
print(f"  False Positive Rate (FPR) = {fpr_b:.3f}")
print(f"  False Negative Rate (FNR) = {fnr_b:.3f}")
print(f"  Total Error Rate = {total_error_b:.3f}")

print(f"\nComparison:")
print(f"  Feature A total error: {total_error_a:.3f}")
print(f"  Feature B total error: {total_error_b:.3f}")
print(f"  Difference: {abs(total_error_a - total_error_b):.3f}")

# Create visualization
plt.figure(figsize=(12, 8))

# Subplot 1: Error rates comparison
plt.subplot(2, 2, 1)
features = ['Feature A', 'Feature B']
fpr_values = [fpr_a, fpr_b]
fnr_values = [fnr_a, fnr_b]
total_errors = [total_error_a, total_error_b]

x = np.arange(len(features))
width = 0.25

plt.bar(x - width, fpr_values, width, label='False Positive Rate', color='lightcoral', alpha=0.8)
plt.bar(x, fnr_values, width, label='False Negative Rate', color='lightblue', alpha=0.8)
plt.bar(x + width, total_errors, width, label='Total Error Rate', color='lightgreen', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Error Rate')
plt.title('Error Rate Comparison')
plt.xticks(x, features)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Trade-off visualization
plt.subplot(2, 2, 2)
plt.scatter(fpr_a, fnr_a, s=200, color='red', alpha=0.7, label='Feature A')
plt.scatter(fpr_b, fnr_b, s=200, color='blue', alpha=0.7, label='Feature B')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('False Negative Rate (FNR)')
plt.title('FPR vs FNR Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)

# Add ideal point (0,0)
plt.scatter(0, 0, s=300, color='green', marker='*', alpha=0.8, label='Ideal (0,0)')
plt.legend()

# Subplot 3: Total error comparison
plt.subplot(2, 2, 3)
bars = plt.bar(features, total_errors, color=['lightcoral', 'lightblue'], alpha=0.8)
plt.ylabel('Total Error Rate')
plt.title('Total Error Rate Comparison')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, error in zip(bars, total_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{error:.3f}', ha='center', va='bottom')

# Subplot 4: Decision boundary visualization
plt.subplot(2, 2, 4)
# Create synthetic data to show the effect
np.random.seed(42)
n_points = 1000

# Generate points with different error rates
feature_a_scores = np.random.normal(0.6, 0.2, n_points)
feature_b_scores = np.random.normal(0.7, 0.15, n_points)

# Apply thresholds to simulate classification
threshold_a = 0.5
threshold_b = 0.6

# Calculate actual errors
actual_labels = np.random.choice([0, 1], n_points, p=[0.7, 0.3])
pred_a = (feature_a_scores > threshold_a).astype(int)
pred_b = (feature_b_scores > threshold_b).astype(int)

# Calculate confusion matrices
def calculate_errors(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return fpr, fnr

fpr_a_actual, fnr_a_actual = calculate_errors(actual_labels, pred_a)
fpr_b_actual, fnr_b_actual = calculate_errors(actual_labels, pred_b)

plt.scatter(feature_a_scores[actual_labels == 0], 
           feature_a_scores[actual_labels == 0], 
           alpha=0.6, color='lightblue', label='Class 0 (Feature A)')
plt.scatter(feature_a_scores[actual_labels == 1], 
           feature_a_scores[actual_labels == 1], 
           alpha=0.6, color='lightcoral', label='Class 1 (Feature A)')
plt.axvline(x=threshold_a, color='red', linestyle='--', alpha=0.8, label=f'Threshold A = {threshold_a}')
plt.xlabel('Feature A Score')
plt.ylabel('Feature A Score')
plt.title('Feature A Classification')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'error_rate_analysis.png'), dpi=300, bbox_inches='tight')

# ============================================================================
# SUMMARY AND CONCLUSIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND CONCLUSIONS")
print("="*80)

print("\n1. Medical Diagnosis Features:")
print("   - Most important: age, blood_pressure, cholesterol")
print("   - Key considerations: interpretability, reliability, cost")
print("   - Balance between accuracy and clinical relevance")

print("\n2. Financial vs Image Recognition:")
print("   - Financial: low-dimensional, interpretable, correlated features")
print("   - Image: high-dimensional, sparse, independent features")
print("   - Different feature selection strategies needed")

print("\n3. Real-time Sensor Data:")
print("   - Temporal patterns and anomaly detection important")
print("   - Feature selection must consider computational efficiency")
print("   - Real-time processing constraints")

print("\n4. Text vs Numerical Data:")
print("   - Text: high-dimensional, sparse, requires different metrics")
print("   - Numerical: lower-dimensional, dense, correlation-based selection")
print("   - Sparsity handling crucial for text data")

print("\n5. Interpretable Features:")
print("   - Medical and financial domains benefit most")
print("   - Regulatory compliance and trust requirements")
print("   - Trade-off between interpretability and performance")

print("\n6. Error Rate Analysis:")
print(f"   - Feature A total error: {total_error_a:.3f}")
print(f"   - Feature B total error: {total_error_b:.3f}")
print(f"   - Feature A is better (lower total error)")
print("   - Consider domain-specific costs of FP vs FN")

print(f"\nAll visualizations saved to: {save_dir}")
print("="*80)
