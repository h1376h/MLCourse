import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("Question 7: Overfitting and Generalization with Feature Selection")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic dataset with many features
print("\n1. Creating synthetic dataset...")
n_samples = 500
n_features_total = 100
n_informative = 10  # Only 10 features are actually informative

# Create regression dataset
X, y = make_regression(n_samples=n_samples, 
                      n_features=n_features_total, 
                      n_informative=n_informative,
                      noise=0.1,
                      random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset created: {n_samples} samples, {n_features_total} features")
print(f"Only {n_informative} features are informative (related to target)")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Function to evaluate model with different number of features
def evaluate_model_complexity(X_train, X_test, y_train, y_test, feature_counts):
    """Evaluate model performance with different numbers of features"""
    train_errors = []
    test_errors = []
    bias_squared = []
    variance = []
    
    for n_features in feature_counts:
        print(f"\nEvaluating with {n_features} features...")
        
        # Select top k features
        if n_features < X_train.shape[1]:
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
        else:
            X_train_selected = X_train
            X_test_selected = X_test
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        
        # Predictions
        train_pred = model.predict(X_train_selected)
        test_pred = model.predict(X_test_selected)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        train_errors.append(train_mse)
        test_errors.append(test_mse)
        
        # Estimate bias and variance using bootstrap
        n_bootstrap = 100
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_train_selected), size=len(X_train_selected), replace=True)
            X_boot = X_train_selected[indices]
            y_boot = y_train[indices]
            
            # Train model on bootstrap sample
            boot_model = LinearRegression()
            boot_model.fit(X_boot, y_boot)
            pred = boot_model.predict(X_test_selected)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias and variance
        mean_pred = np.mean(predictions, axis=0)
        bias_sq = np.mean((mean_pred - y_test) ** 2)
        var = np.mean(np.var(predictions, axis=0))
        
        bias_squared.append(bias_sq)
        variance.append(var)
        
        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Bias²: {bias_sq:.4f}")
        print(f"  Variance: {var:.4f}")
    
    return train_errors, test_errors, bias_squared, variance

# 4. Evaluate different numbers of features
feature_counts = [5, 10, 20, 30, 50, 100]
train_errors, test_errors, bias_squared, variance = evaluate_model_complexity(
    X_train, X_test, y_train, y_test, feature_counts)

# 5. Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: Training vs Test Error
plt.subplot(2, 3, 1)
plt.plot(feature_counts, train_errors, 'bo-', label='Training Error', linewidth=2, markersize=8)
plt.plot(feature_counts, test_errors, 'ro-', label='Test Error', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Test Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Bias-Variance Decomposition
plt.subplot(2, 3, 2)
plt.plot(feature_counts, bias_squared, 'go-', label='Bias²', linewidth=2, markersize=8)
plt.plot(feature_counts, variance, 'mo-', label='Variance', linewidth=2, markersize=8)
noise = 0.01  # Irreducible error (noise in data generation)
total_error = np.array(bias_squared) + np.array(variance) + noise
plt.plot(feature_counts, total_error, 'ko-', label='Total Error', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Error Components')
plt.title('Bias-Variance Decomposition')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 3: Model Complexity vs Generalization
plt.subplot(2, 3, 3)
generalization_gap = np.array(test_errors) - np.array(train_errors)
plt.plot(feature_counts, generalization_gap, 'co-', label='Generalization Gap', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Test Error - Train Error')
plt.title('Generalization Gap')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Feature Selection Impact
plt.subplot(2, 3, 4)
# Show which features are most important
selector_all = SelectKBest(score_func=f_regression, k=20)
selector_all.fit(X_train, y_train)
feature_scores = selector_all.scores_

top_20_scores = sorted(feature_scores, reverse=True)[:20]
plt.bar(range(len(top_20_scores)), top_20_scores)
plt.xlabel('Feature Rank')
plt.ylabel('F-Score')
plt.title('Top 20 Feature Importance Scores')
plt.grid(True, alpha=0.3)

# Plot 5: Learning Curves for Different Feature Counts
plt.subplot(2, 3, 5)
sample_sizes = np.linspace(0.1, 1.0, 10)
colors = ['red', 'blue', 'green', 'orange']
feature_subset = [10, 30, 50, 100]

for i, n_feat in enumerate(feature_subset):
    train_scores = []
    test_scores = []
    
    for size in sample_sizes:
        n_samples = int(size * len(X_train))
        X_sub = X_train[:n_samples]
        y_sub = y_train[:n_samples]
        
        if n_feat < X_train.shape[1]:
            selector = SelectKBest(score_func=f_regression, k=n_feat)
            X_sub_selected = selector.fit_transform(X_sub, y_sub)
            X_test_selected = selector.transform(X_test)
        else:
            X_sub_selected = X_sub
            X_test_selected = X_test
        
        model = LinearRegression()
        model.fit(X_sub_selected, y_sub)
        
        train_score = mean_squared_error(y_sub, model.predict(X_sub_selected))
        test_score = mean_squared_error(y_test, model.predict(X_test_selected))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    plt.plot(sample_sizes * len(X_train), test_scores, 
             color=colors[i], label=f'{n_feat} features', linewidth=2)

plt.xlabel('Training Set Size')
plt.ylabel('Test Error')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 6: Concrete Example Calculation
plt.subplot(2, 3, 6)
# Calculate specific example from problem
bias_100 = 0.1
variance_100 = 0.3
noise = 0.05
error_100 = bias_100**2 + variance_100 + noise

bias_20 = 0.2
variance_20 = 0.1
error_20 = bias_20**2 + variance_20 + noise

categories = ['Bias²', 'Variance', 'Noise', 'Total']
values_100 = [bias_100**2, variance_100, noise, error_100]
values_20 = [bias_20**2, variance_20, noise, error_20]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, values_100, width, label='100 features', alpha=0.8, color='red')
plt.bar(x + width/2, values_20, width, label='20 features', alpha=0.8, color='blue')

plt.xlabel('Error Components')
plt.ylabel('Error Value')
plt.title('Concrete Example: Error Decomposition')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, alpha=0.3)

# Add text annotations for values
for i, (v100, v20) in enumerate(zip(values_100, values_20)):
    plt.text(i - width/2, v100 + 0.01, f'{v100:.3f}', ha='center', va='bottom')
    plt.text(i + width/2, v20 + 0.01, f'{v20:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')

# 6. Create detailed overfitting demonstration
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Generate simple 1D example for clear visualization
np.random.seed(42)
X_1d = np.linspace(0, 1, 50).reshape(-1, 1)
y_1d = 2 * X_1d.ravel() + 0.5 * np.sin(10 * X_1d.ravel()) + 0.1 * np.random.randn(50)

X_1d_train, X_1d_test, y_1d_train, y_1d_test = train_test_split(X_1d, y_1d, test_size=0.3, random_state=42)

# Create polynomial features of different degrees
from sklearn.preprocessing import PolynomialFeatures

degrees = [1, 3, 9, 15]
titles = ['Underfitting (1 feature)', 'Good Fit (3 features)', 'Overfitting (9 features)', 'Severe Overfitting (15 features)']

X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

for i, (degree, title) in enumerate(zip(degrees, titles)):
    ax = axes[i//2, i%2]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_1d_train_poly = poly.fit_transform(X_1d_train)
    X_1d_test_poly = poly.transform(X_1d_test)
    X_plot_poly = poly.transform(X_plot)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_1d_train_poly, y_1d_train)
    
    # Predictions
    y_plot_pred = model.predict(X_plot_poly)
    y_train_pred = model.predict(X_1d_train_poly)
    y_test_pred = model.predict(X_1d_test_poly)
    
    # Calculate errors
    train_mse = mean_squared_error(y_1d_train, y_train_pred)
    test_mse = mean_squared_error(y_1d_test, y_test_pred)
    
    # Plot
    ax.scatter(X_1d_train, y_1d_train, alpha=0.6, color='blue', label='Training data')
    ax.scatter(X_1d_test, y_1d_test, alpha=0.6, color='red', label='Test data')
    ax.plot(X_plot, y_plot_pred, color='green', linewidth=2, label='Model prediction')
    
    ax.set_title(f'{title}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_demonstration.png'), dpi=300, bbox_inches='tight')

# 7. Create feature selection comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Before feature selection (100 features)
model_100 = LinearRegression()
model_100.fit(X_train, y_train)
train_pred_100 = model_100.predict(X_train)
test_pred_100 = model_100.predict(X_test)

train_mse_100 = mean_squared_error(y_train, train_pred_100)
test_mse_100 = mean_squared_error(y_test, test_pred_100)

# After feature selection (10 features)
selector_10 = SelectKBest(score_func=f_regression, k=10)
X_train_10 = selector_10.fit_transform(X_train, y_train)
X_test_10 = selector_10.transform(X_test)

model_10 = LinearRegression()
model_10.fit(X_train_10, y_train)
train_pred_10 = model_10.predict(X_train_10)
test_pred_10 = model_10.predict(X_test_10)

train_mse_10 = mean_squared_error(y_train, train_pred_10)
test_mse_10 = mean_squared_error(y_test, test_pred_10)

# Plot predictions vs actual
axes[0].scatter(y_train, train_pred_100, alpha=0.6, color='blue', label='Training')
axes[0].scatter(y_test, test_pred_100, alpha=0.6, color='red', label='Test')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(f'100 Features\nTrain MSE: {train_mse_100:.3f}\nTest MSE: {test_mse_100:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_train, train_pred_10, alpha=0.6, color='blue', label='Training')
axes[1].scatter(y_test, test_pred_10, alpha=0.6, color='red', label='Test')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'10 Features\nTrain MSE: {train_mse_10:.3f}\nTest MSE: {test_mse_10:.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Improvement comparison
metrics = ['Train MSE', 'Test MSE', 'Generalization Gap']
before = [train_mse_100, test_mse_100, test_mse_100 - train_mse_100]
after = [train_mse_10, test_mse_10, test_mse_10 - train_mse_10]

x = np.arange(len(metrics))
width = 0.35

axes[2].bar(x - width/2, before, width, label='100 features', alpha=0.8, color='red')
axes[2].bar(x + width/2, after, width, label='10 features', alpha=0.8, color='blue')

axes[2].set_xlabel('Metrics')
axes[2].set_ylabel('Error Value')
axes[2].set_title('Performance Comparison')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Add improvement percentages
for i, (b, a) in enumerate(zip(before, after)):
    improvement = (b - a) / b * 100
    axes[2].text(i, max(b, a) + 0.01, f'{improvement:.1f}% better', 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_comparison.png'), dpi=300, bbox_inches='tight')

# 8. Print detailed numerical results
print("\n" + "="*60)
print("DETAILED NUMERICAL RESULTS")
print("="*60)

print(f"\n1. OVERFITTING ANALYSIS:")
print(f"   With 100 features - Train MSE: {train_mse_100:.4f}, Test MSE: {test_mse_100:.4f}")
print(f"   With 10 features  - Train MSE: {train_mse_10:.4f}, Test MSE: {test_mse_10:.4f}")
print(f"   Generalization improvement: {((test_mse_100 - test_mse_10)/test_mse_100)*100:.1f}%")

print(f"\n2. BIAS-VARIANCE ANALYSIS:")
idx_100 = feature_counts.index(100)
idx_10 = feature_counts.index(10)
print(f"   100 features - Bias²: {bias_squared[idx_100]:.4f}, Variance: {variance[idx_100]:.4f}")
print(f"   10 features  - Bias²: {bias_squared[idx_10]:.4f}, Variance: {variance[idx_10]:.4f}")

print(f"\n3. CONCRETE EXAMPLE CALCULATION:")
print(f"   100 features: Bias² = {bias_100**2:.3f}, Variance = {variance_100:.3f}, Noise = {noise:.3f}")
print(f"   Total Error = {error_100:.3f}")
print(f"   20 features:  Bias² = {bias_20**2:.3f}, Variance = {variance_20:.3f}, Noise = {noise:.3f}")
print(f"   Total Error = {error_20:.3f}")
print(f"   Error reduction: {error_100 - error_20:.3f}")
print(f"   Improvement: {((error_100 - error_20)/error_100)*100:.1f}%")

print(f"\n4. MODEL COMPLEXITY INSIGHTS:")
print(f"   - As features increase from 10 to 100:")
print(f"     * Training error decreases: {train_errors[1]:.4f} → {train_errors[-1]:.4f}")
print(f"     * Test error increases: {test_errors[1]:.4f} → {test_errors[-1]:.4f}")
print(f"     * Variance increases: {variance[1]:.4f} → {variance[-1]:.4f}")

print(f"\nPlots saved to: {save_dir}")
print("\nAll visualizations and calculations completed!")
