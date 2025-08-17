import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_1_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 6: Bias-Variance Trade-off in Ensemble Methods")
print("=" * 60)

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 1000
X = np.random.uniform(-3, 3, (n_samples, 1))
true_function = 2 * X.flatten() + 0.5 * X.flatten()**2 + 0.1 * X.flatten()**3
noise = np.random.normal(0, 0.5, n_samples)
y = true_function + noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Generated {n_samples} samples with polynomial relationship + noise")
print(f"True function: y = 2x + 0.5x² + 0.1x³ + ε, where ε ~ N(0, 0.5)")

# Task 1: Characteristics of a typical "weak learner"
print("\n" + "="*60)
print("TASK 1: Characteristics of a typical 'weak learner'")
print("="*60)

# Demonstrate weak learners with different characteristics
weak_learners = {
    'Shallow Decision Tree (max_depth=2)': DecisionTreeRegressor(max_depth=2, random_state=42),
    'Shallow Decision Tree (max_depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
    'Linear Regression': LinearRegression()
}

print("Weak learners typically have:")
print("1. High bias (underfitting) - they cannot capture complex patterns")
print("2. Low variance - they are stable and consistent across different datasets")
print("3. Simple model structure - limited capacity to learn")

# Visualize weak learners
plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(weak_learners.items()):
    plt.subplot(1, 3, i+1)
    
    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Plot
    plt.scatter(X_test, y_test, alpha=0.6, s=20, color='lightblue', label='Test Data')
    plt.scatter(X_train, y_train, alpha=0.4, s=20, color='gray', label='Training Data')
    
    # Sort for smooth line
    X_sorted = np.sort(X_test.flatten())
    y_pred_sorted = model.predict(X_sorted.reshape(-1, 1))
    plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Model Prediction')
    
    # Plot true function
    true_y = 2 * X_sorted + 0.5 * X_sorted**2 + 0.1 * X_sorted**3
    plt.plot(X_sorted, true_y, 'g--', linewidth=2, label='True Function')
    
    plt.title(f'{name}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate bias and variance
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: MSE = {mse:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weak_learners_characteristics.png'), dpi=300, bbox_inches='tight')

# Task 2: Ensemble technique to decrease variance
print("\n" + "="*60)
print("TASK 2: Ensemble technique to decrease variance of unstable models")
print("="*60)

print("Bagging (Bootstrap Aggregating) is primarily used to decrease variance of unstable models.")
print("It works by training multiple models on different bootstrap samples and averaging their predictions.")

# Demonstrate bagging with high-variance models
high_variance_model = DecisionTreeRegressor(max_depth=10, random_state=42)
bagging_ensemble = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Fit models
high_variance_model.fit(X_train, y_train)
bagging_ensemble.fit(X_train, y_train)

# Make predictions
y_pred_single = high_variance_model.predict(X_test)
y_pred_bagging = bagging_ensemble.predict(X_test)

# Calculate metrics
mse_single = mean_squared_error(y_test, y_pred_single)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)

print(f"Single High-Variance Model MSE: {mse_single:.4f}")
print(f"Bagging Ensemble MSE: {mse_bagging:.4f}")
print(f"Variance reduction: {((mse_single - mse_bagging) / mse_single * 100):.1f}%")

# Visualize bagging effect
plt.figure(figsize=(15, 5))

# Single high-variance model
plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, alpha=0.6, s=20, color='lightblue', label='Test Data')
X_sorted = np.sort(X_test.flatten())
y_pred_sorted = high_variance_model.predict(X_sorted.reshape(-1, 1))
plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Single Model')
true_y = 2 * X_sorted + 0.5 * X_sorted**2 + 0.1 * X_sorted**3
plt.plot(X_sorted, true_y, 'g--', linewidth=2, label='True Function')
plt.title('Single High-Variance Model\n(Overfitting)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Bagging ensemble
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, alpha=0.6, s=20, color='lightblue', label='Test Data')
y_pred_sorted = bagging_ensemble.predict(X_sorted.reshape(-1, 1))
plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Bagging Ensemble')
plt.plot(X_sorted, true_y, 'g--', linewidth=2, label='True Function')
plt.title('Bagging Ensemble\n(Reduced Variance)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Variance comparison
plt.subplot(1, 3, 3)
models = ['Single\nHigh-Variance', 'Bagging\nEnsemble']
mses = [mse_single, mse_bagging]
colors = ['red', 'green']
bars = plt.bar(models, mses, color=colors, alpha=0.7)
plt.title('MSE Comparison\n(Lower = Better)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{mse:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bagging_variance_reduction.png'), dpi=300, bbox_inches='tight')

# Task 3: Ensemble technique to decrease bias
print("\n" + "="*60)
print("TASK 3: Ensemble technique to decrease bias of weak learners")
print("="*60)

print("Boosting is primarily used to decrease bias of weak learners.")
print("It works by sequentially training models, each focusing on the errors of previous models.")

# Demonstrate boosting with weak learners
weak_learner = DecisionTreeRegressor(max_depth=3, random_state=42)
boosting_ensemble = GradientBoostingRegressor(n_estimators=100, max_depth=3, 
                                             learning_rate=0.1, random_state=42)

# Fit models
weak_learner.fit(X_train, y_train)
boosting_ensemble.fit(X_train, y_train)

# Make predictions
y_pred_weak = weak_learner.predict(X_test)
y_pred_boosting = boosting_ensemble.predict(X_test)

# Calculate metrics
mse_weak = mean_squared_error(y_test, y_pred_weak)
mse_boosting = mean_squared_error(y_test, y_pred_boosting)

print(f"Weak Learner MSE: {mse_weak:.4f}")
print(f"Boosting Ensemble MSE: {mse_boosting:.4f}")
print(f"Bias reduction: {((mse_weak - mse_boosting) / mse_weak * 100):.1f}%")

# Visualize boosting effect
plt.figure(figsize=(15, 5))

# Weak learner
plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, alpha=0.6, s=20, color='lightblue', label='Test Data')
X_sorted = np.sort(X_test.flatten())
y_pred_sorted = weak_learner.predict(X_sorted.reshape(-1, 1))
plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Weak Learner')
true_y = 2 * X_sorted + 0.5 * X_sorted**2 + 0.1 * X_sorted**3
plt.plot(X_sorted, true_y, 'g--', linewidth=2, label='True Function')
plt.title('Weak Learner\n(High Bias)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Boosting ensemble
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, alpha=0.6, s=20, color='lightblue', label='Test Data')
y_pred_sorted = boosting_ensemble.predict(X_sorted.reshape(-1, 1))
plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Boosting Ensemble')
plt.plot(X_sorted, true_y, 'g--', linewidth=2, label='True Function')
plt.title('Boosting Ensemble\n(Reduced Bias)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Bias comparison
plt.subplot(1, 3, 3)
models = ['Weak\nLearner', 'Boosting\nEnsemble']
mses = [mse_weak, mse_boosting]
colors = ['red', 'green']
bars = plt.bar(models, mses, color=colors, alpha=0.7)
plt.title('MSE Comparison\n(Lower = Better)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{mse:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'boosting_bias_reduction.png'), dpi=300, bbox_inches='tight')

# Task 4: Why averaging reduces variance
print("\n" + "="*60)
print("TASK 4: Why averaging outputs reduces ensemble variance")
print("="*60)

print("Averaging multiple high-variance models reduces overall variance through:")
print("1. Error cancellation - positive and negative errors tend to cancel out")
print("2. Law of large numbers - more models lead to more stable predictions")
print("3. Independence assumption - different models make different errors")

# Demonstrate variance reduction through averaging
np.random.seed(42)
n_models = 100
n_points = 50

# Generate predictions from multiple high-variance models
X_demo = np.linspace(-2, 2, n_points)
true_values = 2 * X_demo + 0.5 * X_demo**2 + 0.1 * X_demo**3

# Simulate high-variance predictions
model_predictions = []
for i in range(n_models):
    # Each model has high variance (noisy predictions)
    noise = np.random.normal(0, 0.8, n_points)
    model_pred = true_values + noise
    model_predictions.append(model_pred)

model_predictions = np.array(model_predictions)

# Calculate variance at each point
variances = np.var(model_predictions, axis=0)
mean_predictions = np.mean(model_predictions, axis=0)

# Visualize variance reduction
plt.figure(figsize=(15, 10))

# Individual model predictions
plt.subplot(2, 2, 1)
for i in range(min(20, n_models)):  # Show first 20 models
    plt.plot(X_demo, model_predictions[i], 'b-', alpha=0.3, linewidth=0.5)
plt.plot(X_demo, true_values, 'r-', linewidth=3, label='True Function')
plt.plot(X_demo, mean_predictions, 'g--', linewidth=2, label='Ensemble Average')
plt.title('Individual Model Predictions\n(High Variance)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Variance at each point
plt.subplot(2, 2, 2)
plt.plot(X_demo, variances, 'b-', linewidth=2)
plt.fill_between(X_demo, variances, alpha=0.3, color='blue')
plt.title('Variance at Each Point\n(High in Complex Regions)')
plt.xlabel('$x$')
plt.ylabel('Variance')
plt.grid(True, alpha=0.3)

# Ensemble average vs individual
plt.subplot(2, 2, 3)
plt.scatter(X_demo, true_values, color='red', s=50, label='True Values', zorder=5)
plt.scatter(X_demo, mean_predictions, color='green', s=50, label='Ensemble Average', zorder=5)
plt.errorbar(X_demo, mean_predictions, yerr=np.sqrt(variances), fmt='none', 
             color='green', alpha=0.5, capsize=3)
plt.title('Ensemble Average vs True Values\n(With Standard Error Bars)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True, alpha=0.3)

# Variance reduction comparison
plt.subplot(2, 2, 4)
# Calculate variance of ensemble average vs individual models
individual_variance = np.mean(variances)
ensemble_variance = np.var(mean_predictions - true_values)

models = ['Individual\nModels', 'Ensemble\nAverage']
variances_comp = [individual_variance, ensemble_variance]
colors = ['red', 'green']
bars = plt.bar(models, variances_comp, color=colors, alpha=0.7)
plt.title('Variance Comparison\n(Lower = Better)')
plt.ylabel('Variance')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, var in zip(bars, variances_comp):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{var:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_reduction_through_averaging.png'), dpi=300, bbox_inches='tight')

# Mathematical demonstration of variance reduction
print("\nMathematical demonstration of variance reduction:")
print("If we have n independent models with variance σ²:")
print("Individual model variance: Var(Y_i) = σ²")
print("Ensemble variance: Var(Ȳ) = Var(ΣY_i/n) = σ²/n")
print(f"With {n_models} models, variance is reduced by factor of {n_models}")

# Show actual variance reduction
print(f"Actual variance reduction achieved:")
print(f"Individual models average variance: {individual_variance:.4f}")
print(f"Ensemble average variance: {ensemble_variance:.4f}")
print(f"Theoretical reduction factor: {n_models}")
print(f"Actual reduction factor: {individual_variance/ensemble_variance:.2f}")

# Summary information (printed to console instead of generating image)
print("\n" + "="*60)
print("ENSEMBLE METHODS SUMMARY")
print("="*60)
print("Weak Learners:")
print("• High Bias (underfitting) - cannot capture complex patterns")
print("• Low Variance (stable) - consistent across different datasets")
print("• Simple structure - limited capacity to learn")
print("• Limited capacity - restricted model complexity")
print()
print("Bagging (Bootstrap Aggregating):")
print("• Reduces Variance - stabilizes unstable models")
print("• Parallel training - models trained independently")
print("• Bootstrap sampling - different data subsets for each model")
print("• Averaging predictions - reduces prediction variance")
print()
print("Boosting:")
print("• Reduces Bias - improves weak learners")
print("• Sequential training - each model focuses on previous errors")
print("• Focus on errors - learns from misclassified examples")
print("• Weighted combination - gives more importance to better models")
print()
print("Variance Reduction by Averaging:")
print("• Error cancellation - positive/negative errors cancel out")
print("• Law of large numbers - stability increases with ensemble size")
print("• Independence assumption - different models make different errors")
print(f"• Theoretical reduction factor: 1/{n_models}")

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Question 6 Solutions:")
print("1. Weak learners have high bias, low variance, and simple structure")
print("2. Bagging reduces variance of unstable models through averaging")
print("3. Boosting reduces bias of weak learners through sequential learning")
print("4. Averaging reduces variance through error cancellation and law of large numbers")
