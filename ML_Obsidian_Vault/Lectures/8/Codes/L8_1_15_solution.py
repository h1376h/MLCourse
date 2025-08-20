import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 15: FEATURE REDUNDANCY AND MULTICOLLINEARITY")
print("=" * 80)

# =============================================================================
# 1. What is feature redundancy and why is it problematic?
# =============================================================================
print("\n1. FEATURE REDUNDANCY: DEFINITION AND PROBLEMS")
print("-" * 50)

# Create a synthetic dataset with redundant features
np.random.seed(42)
n_samples = 1000

# Create base features
x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(0, 1, n_samples)

# Create redundant features (highly correlated with existing ones)
x3 = x1 + 0.1 * np.random.normal(0, 1, n_samples)  # Almost identical to x1
x4 = 2 * x1 + 0.2 * np.random.normal(0, 1, n_samples)  # Linear combination of x1
x5 = x1 + x2 + 0.1 * np.random.normal(0, 1, n_samples)  # Linear combination of x1, x2

# Create target variable
y = 3 * x1 + 2 * x2 + np.random.normal(0, 0.5, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'x5': x5,
    'y': y
})

print("Created synthetic dataset with redundant features:")
print(f"- x1: Independent feature (coefficient = 3)")
print(f"- x2: Independent feature (coefficient = 2)")
print(f"- x3: x1 + noise (redundant with x1)")
print(f"- x4: 2*x1 + noise (redundant with x1)")
print(f"- x5: x1 + x2 + noise (redundant combination)")
print(f"- y: 3*x1 + 2*x2 + noise")

# =============================================================================
# 2. How do you detect multicollinearity between features?
# =============================================================================
print("\n2. DETECTING MULTICOLLINEARITY")
print("-" * 30)

# Calculate correlation matrix
correlation_matrix = data.drop('y', axis=1).corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.3f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Identify high correlations
print("\nHigh Correlations (|r| > 0.8):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            print(f"  {correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: {corr_val:.3f}")

# =============================================================================
# 3. Calculate Variance Inflation Factor (VIF)
# =============================================================================
print("\n3. VARIANCE INFLATION FACTOR (VIF) CALCULATION")
print("-" * 45)

def calculate_vif(df):
    """Calculate VIF for all features in the dataframe"""
    features = df.drop('y', axis=1)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) 
                       for i in range(len(features.columns))]
    return vif_data

vif_results = calculate_vif(data)
print("\nVIF Results:")
print(vif_results)

# Visualize VIF
plt.figure(figsize=(10, 6))
bars = plt.bar(vif_results['Feature'], vif_results['VIF'], color='skyblue', edgecolor='black')
plt.axhline(y=5, color='red', linestyle='--', label='VIF Threshold = 5')
plt.axhline(y=10, color='orange', linestyle='--', label='VIF Threshold = 10')
plt.xlabel('Features')
plt.ylabel('VIF Value')
plt.title('Variance Inflation Factor (VIF) for Each Feature')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, vif_val in zip(bars, vif_results['VIF']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{vif_val:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'vif_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nVIF Interpretation:")
print("- VIF = 1: No multicollinearity")
print("- VIF 1-5: Moderate multicollinearity")
print("- VIF 5-10: High multicollinearity")
print("- VIF > 10: Very high multicollinearity")

# =============================================================================
# 4. Manual VIF Calculation for Specific Correlations
# =============================================================================
print("\n4. MANUAL VIF CALCULATION")
print("-" * 25)

def calculate_vif_from_correlation(correlation_with_others):
    """Calculate VIF from R-squared with other features"""
    r_squared = correlation_with_others ** 2
    vif = 1 / (1 - r_squared)
    return vif

# For correlation 0.9
corr_09 = 0.9
vif_09 = calculate_vif_from_correlation(corr_09)
print(f"If feature X has correlation 0.9 with other features:")
print(f"  R² = {corr_09**2:.3f}")
print(f"  VIF = 1/(1-R²) = 1/(1-{corr_09**2:.3f}) = {vif_09:.1f}")
print(f"  Should remove? {'YES' if vif_09 > 5 else 'NO'} (VIF > 5 threshold)")

# For correlation 0.8
corr_08 = 0.8
vif_08 = calculate_vif_from_correlation(corr_08)
print(f"\nIf feature X has correlation 0.8 with other features:")
print(f"  R² = {corr_08**2:.3f}")
print(f"  VIF = 1/(1-R²) = 1/(1-{corr_08**2:.3f}) = {vif_08:.1f}")
print(f"  Should remove? {'YES' if vif_08 > 5 else 'NO'} (VIF > 5 threshold)")

# Create visualization for VIF vs Correlation
correlations = np.arange(0, 0.99, 0.01)
vifs = [calculate_vif_from_correlation(c) for c in correlations]

plt.figure(figsize=(10, 6))
plt.plot(correlations, vifs, 'b-', linewidth=2, label='VIF vs Correlation')
plt.axhline(y=5, color='red', linestyle='--', label='VIF = 5 (threshold)')
plt.axhline(y=10, color='orange', linestyle='--', label='VIF = 10')
plt.axvline(x=0.8, color='green', linestyle=':', alpha=0.7, label='Correlation = 0.8')
plt.axvline(x=0.9, color='purple', linestyle=':', alpha=0.7, label='Correlation = 0.9')

# Mark specific points
plt.scatter([0.8, 0.9], [vif_08, vif_09], color=['green', 'purple'], s=100, zorder=5)
plt.annotate(f'r=0.8, VIF={vif_08:.1f}', xy=(0.8, vif_08), xytext=(0.6, 15),
            arrowprops=dict(arrowstyle='->', color='green'))
plt.annotate(f'r=0.9, VIF={vif_09:.1f}', xy=(0.9, vif_09), xytext=(0.7, 35),
            arrowprops=dict(arrowstyle='->', color='purple'))

plt.xlabel('Correlation with Other Features')
plt.ylabel('Variance Inflation Factor (VIF)')
plt.title('Relationship Between Correlation and VIF')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 50)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'vif_correlation_relationship.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. Impact on Model Performance and Interpretability
# =============================================================================
print("\n5. IMPACT ON MODEL PERFORMANCE")
print("-" * 30)

# Train models with different feature sets
X_all = data.drop('y', axis=1)
X_reduced = data[['x1', 'x2']]  # Only non-redundant features
y_true = data['y']

# Standardize features
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_reduced_scaled = scaler.fit_transform(X_reduced)

# Train models
model_all = LinearRegression()
model_reduced = LinearRegression()

model_all.fit(X_all_scaled, y_true)
model_reduced.fit(X_reduced_scaled, y_true)

# Make predictions
y_pred_all = model_all.predict(X_all_scaled)
y_pred_reduced = model_reduced.predict(X_reduced_scaled)

# Calculate metrics
mse_all = mean_squared_error(y_true, y_pred_all)
mse_reduced = mean_squared_error(y_true, y_pred_reduced)
r2_all = r2_score(y_true, y_pred_all)
r2_reduced = r2_score(y_true, y_pred_reduced)

print(f"Model with ALL features (including redundant):")
print(f"  MSE: {mse_all:.4f}")
print(f"  R²: {r2_all:.4f}")
print(f"  Coefficients: {model_all.coef_}")

print(f"\nModel with REDUCED features (no redundancy):")
print(f"  MSE: {mse_reduced:.4f}")
print(f"  R²: {r2_reduced:.4f}")
print(f"  Coefficients: {model_reduced.coef_}")

print(f"\nTrue coefficients should be approximately [3, 2]")

# Visualize coefficient comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# All features model
ax1.bar(range(len(model_all.coef_)), model_all.coef_, color='lightcoral', edgecolor='black')
ax1.set_xlabel('Feature Index')
ax1.set_ylabel('Coefficient Value')
ax1.set_title('Model Coefficients: All Features\n(Including Redundant)')
ax1.set_xticks(range(len(model_all.coef_)))
ax1.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax1.grid(True, alpha=0.3)

# Add coefficient values on bars
for i, coef in enumerate(model_all.coef_):
    ax1.text(i, coef + 0.05 if coef > 0 else coef - 0.1, f'{coef:.2f}', 
             ha='center', va='bottom' if coef > 0 else 'top')

# Reduced features model
ax2.bar(range(len(model_reduced.coef_)), model_reduced.coef_, color='lightblue', edgecolor='black')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Model Coefficients: Reduced Features\n(No Redundancy)')
ax2.set_xticks(range(len(model_reduced.coef_)))
ax2.set_xticklabels(['x1', 'x2'])
ax2.grid(True, alpha=0.3)

# Add coefficient values on bars
for i, coef in enumerate(model_reduced.coef_):
    ax2.text(i, coef + 0.05 if coef > 0 else coef - 0.1, f'{coef:.2f}', 
             ha='center', va='bottom' if coef > 0 else 'top')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'coefficient_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. Difference Between Redundancy and Irrelevance
# =============================================================================
print("\n6. REDUNDANCY vs IRRELEVANCE")
print("-" * 30)

# Create irrelevant features (no correlation with target)
np.random.seed(42)
x_irrelevant = np.random.normal(0, 1, n_samples)
data_with_irrelevant = data.copy()
data_with_irrelevant['x_irrelevant'] = x_irrelevant

# Calculate correlations with target
target_correlations = data_with_irrelevant.drop('y', axis=1).corrwith(data_with_irrelevant['y'])
print("\nCorrelation with Target Variable:")
for feature, corr in target_correlations.items():
    print(f"  {feature}: {corr:.3f}")

# Visualize the difference
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot redundant features (high correlation with each other)
axes[0, 0].scatter(data['x1'], data['x3'], alpha=0.6, color='red')
axes[0, 0].set_xlabel('x1')
axes[0, 0].set_ylabel('x3 (redundant)')
axes[0, 0].set_title(f'Redundant Features\nr = {data["x1"].corr(data["x3"]):.3f}')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(data['x1'], data['x4'], alpha=0.6, color='red')
axes[0, 1].set_xlabel('x1')
axes[0, 1].set_ylabel('x4 (redundant)')
axes[0, 1].set_title(f'Redundant Features\nr = {data["x1"].corr(data["x4"]):.3f}')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].scatter(data['x1'], data['x2'], alpha=0.6, color='blue')
axes[0, 2].set_xlabel('x1')
axes[0, 2].set_ylabel('x2 (independent)')
axes[0, 2].set_title(f'Independent Features\nr = {data["x1"].corr(data["x2"]):.3f}')
axes[0, 2].grid(True, alpha=0.3)

# Plot correlation with target
axes[1, 0].scatter(data['x1'], data['y'], alpha=0.6, color='green')
axes[1, 0].set_xlabel('x1')
axes[1, 0].set_ylabel('y (target)')
axes[1, 0].set_title(f'Relevant Feature\nr = {data["x1"].corr(data["y"]):.3f}')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(data['x3'], data['y'], alpha=0.6, color='orange')
axes[1, 1].set_xlabel('x3 (redundant)')
axes[1, 1].set_ylabel('y (target)')
axes[1, 1].set_title(f'Redundant but Relevant\nr = {data["x3"].corr(data["y"]):.3f}')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].scatter(data_with_irrelevant['x_irrelevant'], data['y'], alpha=0.6, color='gray')
axes[1, 2].set_xlabel('x_irrelevant')
axes[1, 2].set_ylabel('y (target)')
axes[1, 2].set_title(f'Irrelevant Feature\nr = {data_with_irrelevant["x_irrelevant"].corr(data["y"]):.3f}')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'redundancy_vs_irrelevance.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. Feature Selection Strategy
# =============================================================================
print("\n7. FEATURE SELECTION STRATEGY")
print("-" * 30)

def feature_selection_pipeline(df, vif_threshold=5, corr_threshold=0.95):
    """
    Implement a feature selection pipeline to remove redundant features
    """
    features = df.drop('y', axis=1)
    selected_features = list(features.columns)
    removed_features = []
    
    print(f"Starting with {len(selected_features)} features: {selected_features}")
    
    iteration = 1
    while True:
        print(f"\nIteration {iteration}:")
        
        # Calculate VIF for current features
        current_features = df[selected_features]
        vif_data = pd.DataFrame()
        vif_data["Feature"] = current_features.columns
        vif_data["VIF"] = [variance_inflation_factor(current_features.values, i) 
                          for i in range(len(current_features.columns))]
        
        # Find feature with highest VIF above threshold
        high_vif = vif_data[vif_data['VIF'] > vif_threshold]
        
        if len(high_vif) == 0:
            print(f"  All VIF values <= {vif_threshold}. Selection complete.")
            break
            
        # Remove feature with highest VIF
        feature_to_remove = high_vif.loc[high_vif['VIF'].idxmax(), 'Feature']
        max_vif = high_vif.loc[high_vif['VIF'].idxmax(), 'VIF']
        
        print(f"  Removing '{feature_to_remove}' (VIF = {max_vif:.2f})")
        selected_features.remove(feature_to_remove)
        removed_features.append(feature_to_remove)
        
        iteration += 1
        
        if len(selected_features) <= 1:
            break
    
    print(f"\nFinal selected features: {selected_features}")
    print(f"Removed features: {removed_features}")
    
    return selected_features, removed_features

selected_features, removed_features = feature_selection_pipeline(data)

# Compare performance with selected features
X_selected = data[selected_features]
X_selected_scaled = scaler.fit_transform(X_selected)

model_selected = LinearRegression()
model_selected.fit(X_selected_scaled, y_true)
y_pred_selected = model_selected.predict(X_selected_scaled)

mse_selected = mean_squared_error(y_true, y_pred_selected)
r2_selected = r2_score(y_true, y_pred_selected)

print(f"\nModel Performance Comparison:")
print(f"All features:      MSE = {mse_all:.4f}, R² = {r2_all:.4f}")
print(f"Selected features: MSE = {mse_selected:.4f}, R² = {r2_selected:.4f}")
print(f"Manual reduced:    MSE = {mse_reduced:.4f}, R² = {r2_reduced:.4f}")

# Create summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Performance comparison
models = ['All Features\n(5 features)', 'Selected Features\n(VIF-based)', 'Manual Reduced\n(2 features)']
mse_values = [mse_all, mse_selected, mse_reduced]
r2_values = [r2_all, r2_selected, r2_reduced]

ax1.bar(models, mse_values, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Model Performance: MSE')
ax1.grid(True, alpha=0.3)
for i, v in enumerate(mse_values):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

ax2.bar(models, r2_values, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Performance: R²')
ax2.grid(True, alpha=0.3)
for i, v in enumerate(r2_values):
    ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')

# VIF comparison before and after
features_before = ['x1', 'x2', 'x3', 'x4', 'x5']
vif_before = vif_results['VIF'].values
features_after = selected_features
if len(features_after) > 0:
    X_after = data[features_after]
    vif_after = [variance_inflation_factor(X_after.values, i) for i in range(len(features_after))]
else:
    vif_after = []

ax3.bar(features_before, vif_before, color='red', alpha=0.7, edgecolor='black', label='Before Selection')
ax3.axhline(y=5, color='black', linestyle='--', label='VIF Threshold = 5')
ax3.set_ylabel('VIF Value')
ax3.set_title('VIF Values: Before Feature Selection')
ax3.legend()
ax3.grid(True, alpha=0.3)
for i, v in enumerate(vif_before):
    ax3.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')

if len(features_after) > 0:
    ax4.bar(features_after, vif_after, color='green', alpha=0.7, edgecolor='black', label='After Selection')
    ax4.axhline(y=5, color='black', linestyle='--', label='VIF Threshold = 5')
    ax4.set_ylabel('VIF Value')
    ax4.set_title('VIF Values: After Feature Selection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    for i, v in enumerate(vif_after):
        ax4.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
else:
    ax4.text(0.5, 0.5, 'No features\nremaining', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('VIF Values: After Feature Selection')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. DETAILED STEP-BY-STEP MATHEMATICAL SOLUTIONS
# =============================================================================
print("\n8. DETAILED MATHEMATICAL SOLUTIONS (PEN-AND-PAPER STYLE)")
print("-" * 60)

print("\nA. VIF Calculation Step-by-Step:")
print("   Step 1: For feature X with correlation r = 0.9")
print("   Step 2: Calculate R² = r² = (0.9)² = 0.81")
print("   Step 3: VIF = 1/(1-R²) = 1/(1-0.81) = 1/0.19 = 5.26")
print("   Step 4: Since VIF = 5.26 > 5, remove the feature")

print("\n   For feature X with correlation r = 0.8:")
print("   Step 1: R² = r² = (0.8)² = 0.64")
print("   Step 2: VIF = 1/(1-R²) = 1/(1-0.64) = 1/0.36 = 2.78")
print("   Step 3: Since VIF = 2.78 < 5, retain the feature")

print("\nB. Correlation Matrix Analysis:")
print("   Step 1: Calculate pairwise correlations between all features")
print("   Step 2: Identify high correlations: |r| > 0.8")
print("   Step 3: For each high correlation pair:")
print("     - x1-x3: r = 0.995 (nearly identical)")
print("     - x1-x4: r = 0.995 (linear relationship)")
print("     - x3-x4: r = 0.990 (both derived from x1)")

print("\nC. Feature Selection Algorithm:")
print("   Step 1: Start with all features")
print("   Step 2: Calculate VIF for each feature")
print("   Step 3: Find feature with highest VIF > threshold")
print("   Step 4: Remove that feature")
print("   Step 5: Repeat until all VIF ≤ threshold")
print("   Step 6: Final set contains non-redundant features")

print("\nD. Model Performance Analysis:")
print("   Step 1: Train model with all features")
print("   Step 2: Train model with reduced features")
print("   Step 3: Compare MSE and R² scores")
print("   Step 4: Analyze coefficient stability")
print("   Step 5: Evaluate interpretability")

print("\nE. Redundancy vs Irrelevance Distinction:")
print("   Redundant features:")
print("     - High correlation with other features")
print("     - May have high correlation with target")
print("     - Provide overlapping information")
print("   Irrelevant features:")
print("     - Low correlation with other features")
print("     - Low correlation with target")
print("     - Provide no useful information")

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
