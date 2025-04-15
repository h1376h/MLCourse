import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

print("\n=== COVARIANCE, CORRELATION & INDEPENDENCE IN ML: STEP-BY-STEP EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "L2_1_ML")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: Feature Relationships in Housing Price Prediction
print("Example 1: Feature Relationships in Housing Price Prediction")
print("==========================================================\n")

# Create synthetic housing data
n_samples = 150
sq_footage = np.random.normal(2000, 500, n_samples)  # Square footage (mean 2000, std 500)
num_bedrooms = np.random.normal(3, 1, n_samples)     # Number of bedrooms (mean 3, std 1)
neighborhood_score = np.random.normal(7, 2, n_samples)  # Neighborhood score (1-10)

# Add correlation between features
# Larger houses tend to have more bedrooms
num_bedrooms = num_bedrooms + 0.002 * (sq_footage - 2000)
# Round bedrooms to make it more realistic
num_bedrooms = np.round(np.maximum(1, num_bedrooms))

# House price is influenced by all features with some noise
price = 100000 + 100 * sq_footage + 25000 * num_bedrooms + 15000 * neighborhood_score
# Add some random noise
price = price + np.random.normal(0, 50000, n_samples)

# Create DataFrame
housing_data = pd.DataFrame({
    'sq_footage': sq_footage,
    'num_bedrooms': num_bedrooms,
    'neighborhood_score': neighborhood_score,
    'price': price
})

print("Sample Housing Data:")
print(housing_data.head())
print("\nSummary Statistics:")
print(housing_data.describe().round(2))
print("\n")

# Step 1: Calculate Covariance and Correlation Matrix
print("Step 1: Calculating Covariance and Correlation Matrix")
print("---------------------------------------------------\n")

covariance_matrix = housing_data.cov()
correlation_matrix = housing_data.corr()

print("Covariance Matrix:")
print(covariance_matrix.round(2))
print("\nCorrelation Matrix:")
print(correlation_matrix.round(4))
print("\n")

# Explanation of covariance interpretation
print("Interpreting Covariance Values:")
print("- Positive covariance: Variables tend to increase together")
print("- Negative covariance: When one increases, the other tends to decrease")
print("- Zero covariance: No linear relationship between variables")
print("- Scale-dependent: The units matter (larger values for larger scales)")
print("\nInterpreting Correlation Values:")
print("- Always between -1 and 1")
print("- 1: Perfect positive correlation")
print("- -1: Perfect negative correlation")
print("- 0: No linear correlation")
print("- Scale-independent: Values standardized")
print("\n")

# Step 2: Visualize Relationships between Variables
print("Step 2: Visualizing Feature Relationships")
print("--------------------------------------\n")

# Plot correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('Correlation Matrix of Housing Features')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'housing_correlation_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create scatter plots with regression lines for key relationships
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Square Footage vs. Price
sns.regplot(x='sq_footage', y='price', data=housing_data, line_kws={"color":"red"}, ax=axes[0])
axes[0].set_title(f'Square Footage vs. Price\nCorrelation: {correlation_matrix.loc["sq_footage", "price"]:.3f}')

# Plot 2: Number of Bedrooms vs. Price
sns.regplot(x='num_bedrooms', y='price', data=housing_data, line_kws={"color":"red"}, ax=axes[1])
axes[1].set_title(f'Bedrooms vs. Price\nCorrelation: {correlation_matrix.loc["num_bedrooms", "price"]:.3f}')

# Plot 3: Neighborhood Score vs. Price
sns.regplot(x='neighborhood_score', y='price', data=housing_data, line_kws={"color":"red"}, ax=axes[2])
axes[2].set_title(f'Neighborhood Score vs. Price\nCorrelation: {correlation_matrix.loc["neighborhood_score", "price"]:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'housing_feature_relationships.png'), dpi=100, bbox_inches='tight')
plt.close()

# Step 3: Detecting Multicollinearity
print("Step 3: Detecting Multicollinearity")
print("--------------------------------\n")

# Create a new feature that is a combination of others (to demonstrate multicollinearity)
housing_data['weighted_size'] = housing_data['sq_footage'] * 0.8 + housing_data['num_bedrooms'] * 200

# New correlation matrix with the collinear feature
new_corr_matrix = housing_data.corr()
print("Correlation Matrix with Potentially Collinear Feature:")
print(new_corr_matrix.round(4))

# Calculate VIF (Variance Inflation Factor) for each feature
# VIF = 1 / (1 - R²) where R² is from regression of feature against all other features
features = ['sq_footage', 'num_bedrooms', 'neighborhood_score', 'weighted_size']
vif_data = pd.DataFrame()
vif_data['Feature'] = features
vif_values = []

for feature in features:
    y = housing_data[feature]
    X = housing_data[features].drop(feature, axis=1)
    r2 = r2_score(y, LinearRegression().fit(X, y).predict(X))
    vif = 1 / (1 - r2) if r2 < 1 else float('inf')
    vif_values.append(vif)

vif_data['VIF'] = vif_values
print("\nVariance Inflation Factors (VIF):")
print("- VIF = 1: No correlation with other features")
print("- VIF < 5: Moderate correlation, usually acceptable")
print("- VIF >= 5: High correlation, potential multicollinearity issue")
print("- VIF >= 10: Severe multicollinearity")
print(vif_data)
print("\n")

# Visualize multicollinearity
plt.figure(figsize=(10, 8))
sns.heatmap(new_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('Correlation Matrix with Multicollinear Feature')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'housing_multicollinearity.png'), dpi=100, bbox_inches='tight')
plt.close()

# Bar chart for VIF values
plt.figure(figsize=(10, 6))
plt.bar(vif_data['Feature'], vif_data['VIF'], color='skyblue')
plt.axhline(y=5, color='orange', linestyle='--', label='Moderate Multicollinearity Threshold')
plt.axhline(y=10, color='red', linestyle='--', label='Severe Multicollinearity Threshold')
plt.xlabel('Feature')
plt.ylabel('VIF')
plt.title('Variance Inflation Factors (VIF)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'housing_vif.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Dimensionality Reduction with PCA
print("Example 2: Dimensionality Reduction with PCA")
print("=========================================\n")

# Use a real dataset: breast cancer dataset
print("Using the Breast Cancer Dataset for PCA Analysis")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print("\n")

# Step 1: Compute and visualize the correlation matrix
print("Step 1: Compute Feature Correlations")
print("----------------------------------\n")

# Create a DataFrame with feature names
cancer_df = pd.DataFrame(X, columns=feature_names)
cancer_corr = cancer_df.corr()

print("Feature Correlation Matrix Sample (first 5x5):")
print(cancer_corr.iloc[:5, :5].round(3))
print("\n")

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cancer_corr, cmap='coolwarm', center=0, xticklabels=feature_names, yticklabels=feature_names)
plt.title('Breast Cancer Dataset Feature Correlation Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'cancer_correlation_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Step 2: Standardize features and apply PCA
print("Step 2: Apply PCA for Dimensionality Reduction")
print("-------------------------------------------\n")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by principal component:")
for i, var in enumerate(explained_variance[:5]):
    print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
print("... (showing first 5 components)")

# Find number of components needed for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumber of components needed for 95% variance: {n_components_95}")
print("\n")

# Visualize explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance threshold')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'Components needed: {n_components_95}')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'pca_explained_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Step 3: Visualize data in reduced dimensions
print("Step 3: Visualize Data in the Reduced Feature Space")
print("------------------------------------------------\n")

# Create a plot of the first two principal components
plt.figure(figsize=(10, 8))
colors = ['red', 'blue']
targets = ['Malignant', 'Benign']

for color, target, target_value in zip(colors, targets, [0, 1]):
    plt.scatter(X_pca[y == target_value, 0], X_pca[y == target_value, 1], 
                color=color, alpha=0.7, label=target)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'cancer_pca_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Step 4: Analyze component loadings
print("Step 4: Analyze Component Loadings")
print("--------------------------------\n")

# Get the component loadings (feature weights for each principal component)
component_loadings = pca.components_

# Create a DataFrame of loadings for the first two components
loadings_df = pd.DataFrame({
    'Feature': feature_names,
    'PC1': component_loadings[0],
    'PC2': component_loadings[1]
})

# Sort by absolute loading for PC1
loadings_df = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index)

print("Top 5 Features Contributing to First Principal Component:")
print(loadings_df[['Feature', 'PC1']].head())
print("\nTop 5 Features Contributing to Second Principal Component:")
print(loadings_df.sort_values(by='PC2', key=abs, ascending=False)[['Feature', 'PC2']].head())
print("\n")

# Visualize component loadings
plt.figure(figsize=(12, 8))
plt.scatter(component_loadings[0], component_loadings[1], alpha=0.8)
plt.grid(True, alpha=0.3)

# Add feature names as annotations
for i, feature_name in enumerate(feature_names):
    plt.annotate(feature_name, (component_loadings[0, i], component_loadings[1, i]))

# Add coordinate lines
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.xlabel('PC1 Loadings')
plt.ylabel('PC2 Loadings')
plt.title('PCA Component Loadings')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'pca_component_loadings.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Feature Importance vs. Correlation in ML
print("Example 3: Feature Importance vs. Correlation in ML")
print("================================================\n")

# Create a synthetic dataset with correlated and uncorrelated features
n_samples = 500
n_features = 10

# Generate base features with varying correlations to target
X_base = np.random.randn(n_samples, n_features)
y = 3*X_base[:, 0] + 2*X_base[:, 1] - 1.5*X_base[:, 2] + 0.5*X_base[:, 3] + np.random.randn(n_samples) * 0.5

# Add some correlated features
X_correlated = np.zeros((n_samples, n_features * 2))
X_correlated[:, :n_features] = X_base

# Create correlated features that have indirect relationships with target
X_correlated[:, n_features] = X_base[:, 0] * 0.9 + np.random.randn(n_samples) * 0.2  # Highly correlated with X0
X_correlated[:, n_features+1] = X_base[:, 1] * 0.8 + np.random.randn(n_samples) * 0.3  # Highly correlated with X1
X_correlated[:, n_features+2] = X_base[:, 3] * 0.7 + np.random.randn(n_samples) * 0.4  # Correlated with X3

# The rest are random with low correlation to both target and other features
X_correlated[:, n_features+3:] = np.random.randn(n_samples, n_features-3)

# Create feature names
feature_names = [f'X{i}' for i in range(X_correlated.shape[1])]

# Step 1: Calculate correlation with target
print("Step 1: Calculate Correlation with Target")
print("--------------------------------------\n")

X_df = pd.DataFrame(X_correlated, columns=feature_names)
X_df['Target'] = y

# Calculate correlation with target
target_correlation = X_df.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
print("Correlation with target (absolute values):")
print(target_correlation)
print("\n")

# Step 2: Calculate feature importance with Random Forest
print("Step 2: Calculate Feature Importance with Random Forest")
print("-------------------------------------------------\n")

X_train, X_test, y_train, y_test = train_test_split(X_correlated, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train > np.median(y_train))  # Convert to binary classification for simplicity

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Random Forest feature importances:")
print(feature_importances)
print("\n")

# Compute F-statistic and p-values for regression
f_statistics, p_values = f_regression(X_correlated, y)
f_stat_series = pd.Series(f_statistics, index=feature_names).sort_values(ascending=False)
print("F-statistics from linear regression:")
print(f_stat_series.head())
print("\n")

# Step 3: Compare correlation vs. importance
print("Step 3: Compare Correlation vs. Feature Importance")
print("-----------------------------------------------\n")

# Combine the metrics into a single DataFrame
comparison_df = pd.DataFrame({
    'Correlation': target_correlation,
    'RF_Importance': pd.Series(rf.feature_importances_, index=feature_names),
    'F_Statistic': f_statistics
}).sort_values(by='RF_Importance', ascending=False)

print("Comparison of different feature ranking methods:")
print(comparison_df.head(10))
print("\n")

# Visualize the comparison
plt.figure(figsize=(12, 6))
top_features = comparison_df.index[:10]
x = np.arange(len(top_features))
width = 0.3

# Normalize each metric to [0,1] for easier comparison
correlation_norm = comparison_df['Correlation'].loc[top_features] / comparison_df['Correlation'].max()
importance_norm = comparison_df['RF_Importance'].loc[top_features] / comparison_df['RF_Importance'].max()
f_stat_norm = comparison_df['F_Statistic'].loc[top_features] / comparison_df['F_Statistic'].max()

plt.bar(x - width, correlation_norm, width, label='Correlation with Target')
plt.bar(x, importance_norm, width, label='Random Forest Importance')
plt.bar(x + width, f_stat_norm, width, label='F-Statistic')

plt.xlabel('Features')
plt.ylabel('Normalized Score')
plt.title('Feature Ranking Comparison (Normalized)')
plt.xticks(x, top_features, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'feature_importance_vs_correlation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Independence vs. Correlation in ML
print("Example 4: Independence vs. Correlation in ML")
print("=========================================\n")

print("This example demonstrates the difference between correlation and independence")
print("Two random variables can be uncorrelated but dependent")
print("\n")

# Step 1: Generate data where variables are uncorrelated but dependent
print("Step 1: Generate Uncorrelated but Dependent Data")
print("--------------------------------------------\n")

# Generate data using a non-linear relationship: y = x² + noise
n_samples = 1000
x = np.random.uniform(-5, 5, n_samples)
y = x**2 + np.random.normal(0, 2, n_samples)

# Step 2: Calculate correlation coefficient
print("Step 2: Calculate Correlation Coefficient")
print("--------------------------------------\n")

correlation = np.corrcoef(x, y)[0, 1]
print(f"Pearson correlation between x and y: {correlation:.6f}")
print("Note: The correlation is close to zero, suggesting no linear relationship")
print("\n")

# Step 3: Test for independence
print("Step 3: Demonstrate Dependence Between Variables")
print("--------------------------------------------\n")

# Divide x into segments and calculate mean of y in each segment
x_segments = pd.qcut(x, 10, labels=False)
segment_means = [y[x_segments == i].mean() for i in range(10)]

print("Mean of y in each x segment:")
for i, mean in enumerate(segment_means):
    print(f"Segment {i+1}: {mean:.4f}")

print("\nIf x and y were independent, the mean of y would be approximately")
print(f"the same in each segment, but they clearly differ (overall mean: {y.mean():.4f})")
print("\n")

# Visualize the uncorrelated but dependent data
plt.figure(figsize=(12, 10))

# Scatter plot
plt.subplot(2, 2, 1)
plt.scatter(x, y, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y = x² + noise')
plt.title(f'Uncorrelated but Dependent Variables\nCorrelation: {correlation:.4f}')
plt.grid(True, alpha=0.3)

# Add a line of best fit (which should be flat due to zero correlation)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(np.sort(x), p(np.sort(x)), "r--", alpha=0.8)

# Segment means
plt.subplot(2, 2, 2)
unique_segments = np.unique(x_segments)
segment_x = [x[x_segments == i].mean() for i in range(len(unique_segments))]
plt.bar(range(len(segment_means)), segment_means, color='skyblue')
plt.axhline(y=y.mean(), color='r', linestyle='--', label=f'Overall Mean: {y.mean():.4f}')
plt.xlabel('x Segment')
plt.ylabel('Mean of y')
plt.title('Mean of y in Each x Segment')
plt.legend()

# Create a true independent comparison for contrast
plt.subplot(2, 2, 3)
z = np.random.normal(0, 1, n_samples)  # Independent of x
plt.scatter(x, z, alpha=0.5)
plt.xlabel('x')
plt.ylabel('z (truly independent)')
plt.title(f'Truly Independent Variables\nCorrelation: {np.corrcoef(x, z)[0, 1]:.4f}')
plt.grid(True, alpha=0.3)

# Segment means for truly independent case
plt.subplot(2, 2, 4)
independent_segment_means = [z[x_segments == i].mean() for i in range(10)]
plt.bar(range(len(independent_segment_means)), independent_segment_means, color='salmon')
plt.axhline(y=z.mean(), color='r', linestyle='--', label=f'Overall Mean: {z.mean():.4f}')
plt.xlabel('x Segment')
plt.ylabel('Mean of z')
plt.title('Mean of z in Each x Segment (truly independent)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'correlation_vs_independence.png'), dpi=100, bbox_inches='tight')
plt.close()

print("All covariance, correlation, and independence images created successfully.") 