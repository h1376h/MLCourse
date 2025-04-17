import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_3_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"SCENARIO {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def plot_fit(x, y, model, title, filename):
    """Plot the data and the model's fit."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, edgecolor='k', facecolor='none', alpha=0.7, label='Data Points')
    x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, color='red', linewidth=2, label='Model Fit')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Target', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {file_path}")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename="learning_curve.png"):
    """Generate a simple plot of the test and training learning curve."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (e.g., R^2 or Neg MSE)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    # Convert Negative MSE scores to positive MSE
    train_scores = -train_scores
    test_scores = -test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(True)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training Error (MSE)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation Error (MSE)")

    plt.legend(loc="best")
    plt.ylabel("Mean Squared Error (MSE)") # Update label
    plt.yscale('log') # Often better to view errors on log scale
    plt.tight_layout()
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve figure saved to: {file_path}")
    return plt


def plot_complexity_error(X, y, max_degree=15, cv=5, filename="complexity_error.png"):
    """Plot training and validation error vs model complexity (polynomial degree)."""
    degrees = np.arange(1, max_degree + 1)
    train_errors, val_errors = [], []

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        # Use learning curve utility to get cross-validated scores
        # We are interested in the score on the full training data portion used in CV
        # and the average validation score. Need to use a consistent scoring metric.
        # Use neg_mean_squared_error as score, then convert to positive MSE
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring='neg_mean_squared_error', 
            train_sizes=np.array([0.8]), # Use a large portion for training score estimate
            n_jobs=None) 
            
        # Average over the single training size and the CV folds for test score
        # Convert Negative MSE to positive MSE
        train_errors.append(-np.mean(train_scores)) 
        val_errors.append(-np.mean(test_scores))

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', color="r", label="Training Error (MSE)")
    plt.plot(degrees, val_errors, 'o-', color="g", label="Validation Error (MSE)")
    
    # Find the degree with the minimum validation error
    best_degree = degrees[np.argmin(val_errors)]
    # min_val_error = np.min(val_errors)
    plt.axvline(best_degree, linestyle='--', color='gray', label=f'Optimal Degree ≈ {best_degree}')
    
    plt.title('Model Complexity vs. Error', fontsize=14)
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.xticks(degrees[::2]) # Show every other degree tick
    plt.legend(loc='best')
    plt.yscale('log') # Use log scale for potentially large error differences
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Complexity vs Error plot saved to: {file_path}")


def plot_high_variance_fits(n_fits=5, n_samples=10, n_features=20, filename="high_variance_fits.png"):
    """Illustrate high variance by fitting on slightly different small datasets."""
    plt.figure(figsize=(10, 6))
    X_base = np.random.randn(n_samples + n_fits - 1, n_features) # Base data pool
    true_coef = np.zeros(n_features)
    true_coef[:3] = [5, -2, 3] # Same true coefficients as scenario 1
    y_base = X_base @ true_coef + np.random.normal(0, 1, n_samples + n_fits - 1)

    # We need a single feature to plot against the target for visualization
    # Let's use the first feature as the x-axis for plotting purposes
    # This is purely illustrative as the model uses all 20 features
    x_plot_axis = np.linspace(X_base[:, 0].min() - 1, X_base[:, 0].max() + 1, 100) 
    # Create dummy data for prediction based on the plot axis (varying only the first feature)
    X_plot_dummy = np.zeros((100, n_features))
    X_plot_dummy[:, 0] = x_plot_axis

    for i in range(n_fits):
        # Take slightly different subsets of size n_samples
        X_subset = X_base[i:i+n_samples, :]
        y_subset = y_base[i:i+n_samples]
        
        model = LinearRegression()
        model.fit(X_subset, y_subset)
        
        # Predict using the dummy data where only the first feature varies
        y_plot = model.predict(X_plot_dummy)
        
        style = '-' if i == 0 else '--'
        alpha = 1.0 if i == 0 else 0.7
        plt.plot(x_plot_axis, y_plot, style, alpha=alpha, label=f'Fit {i+1}' if i < 3 else None) # Label only first few fits

        # Plot the actual subset points (using only the first feature for x-axis)
        if i == 0:
            plt.scatter(X_subset[:, 0], y_subset, edgecolor='k', facecolor='skyblue', alpha=0.9, label='Data Subset 1', s=60)
        elif i == 1:
             plt.scatter(X_subset[:, 0], y_subset, marker='x', color='red', alpha=0.9, label='Data Subset 2', s=60)


    plt.title('High Variance Illustration (Scenario 1: p > n)', fontsize=14)
    plt.xlabel('Feature 1 (for visualization only)', fontsize=12)
    plt.ylabel('Target y', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"High variance fits plot saved to: {file_path}")

# --- Scenario 1: High Dimensionality, Few Samples ---
print_step_header(1, "Linear Regression, 20 Features, 10 Samples")

n_samples, n_features = 10, 20
X = np.random.randn(n_samples, n_features)
# True coefficients (sparse, only a few non-zero)
true_coef = np.zeros(n_features)
true_coef[:3] = [5, -2, 3]
y = X @ true_coef + np.random.normal(0, 1, n_samples)

model1 = LinearRegression()
model1.fit(X, y)
train_score1 = model1.score(X, y)

print(f"Scenario 1 Setup: n_samples={n_samples}, n_features={n_features}")
print(f"Model: Linear Regression")
print(f"Training R^2 Score: {train_score1:.4f}")
print("Observation: With more features (20) than samples (10), the model can perfectly fit the training data (R^2=1.0).")
print("This is a classic sign of HIGH VARIANCE and OVERFITTING. The model memorizes the noise.")
print("Improvement Strategies: Get more data, use regularization (e.g., Ridge, Lasso), feature selection/reduction (PCA).")

# Plot learning curve (will show high gap between train and test)
# Need more data for a meaningful learning curve, but conceptually illustrating
X_large = np.random.randn(100, n_features)
y_large = X_large @ true_coef + np.random.randn(100) # Dummy larger data
plot_learning_curve(LinearRegression(), "Learning Curve (Scenario 1 - Overfitting Example)",
                    X_large, y_large, cv=5, filename="scenario1_learning_curve.png")
print("Note: Learning curve shows training error staying low while validation error is high and doesn't converge well.")
print("-" * 40)

# Plot high variance illustration for Scenario 1
plot_high_variance_fits(n_samples=n_samples, n_features=n_features, filename="scenario1_high_variance_fits.png")
print("Note: High variance plot shows how the model changes dramatically with small changes in the limited training data.")
print("-" * 40)


# --- Scenario 2: Simple Model (Shallow Tree) ---
print_step_header(2, "Shallow Decision Tree (Max Depth 2)")

# Generate non-linear data
n_samples_nl = 100
X_nl = np.sort(np.random.rand(n_samples_nl) * 10 - 5).reshape(-1, 1)
y_nl = np.sin(X_nl).ravel() + np.random.normal(0, 0.2, n_samples_nl)

model2 = DecisionTreeRegressor(max_depth=2)
model2.fit(X_nl, y_nl)
train_score2 = model2.score(X_nl, y_nl)

print(f"Scenario 2 Setup: n_samples={n_samples_nl}, modeling non-linear data with a shallow tree.")
print(f"Model: Decision Tree (max_depth=2)")
print(f"Training R^2 Score: {train_score2:.4f}")
print("Observation: A shallow tree is a simple model. If the underlying data relationship is complex (like this sine wave),")
print("the model might not capture it well, leading to HIGH BIAS and UNDERFITTING.")
print("Improvement Strategies: Increase tree depth, use a more complex model (e.g., deeper tree, random forest, gradient boosting).")

plot_fit(X_nl, y_nl, model2, "Shallow Decision Tree Fit (Scenario 2 - Underfitting)", "scenario2_fit.png")
plot_learning_curve(DecisionTreeRegressor(max_depth=2), "Learning Curve (Scenario 2 - Underfitting Example)",
                    X_nl, y_nl, cv=5, filename="scenario2_learning_curve.png")
print("Note: Learning curve shows both training and validation errors are high and converge, indicating underfitting.")
print("-" * 40)

# Generate the complexity vs error plot using the data from Scenario 2/4
print("Generating Complexity vs Error plot (using sine wave data)...")
plot_complexity_error(X_nl, y_nl, max_degree=15, cv=5, filename="scenario2_4_complexity_error.png")
print("Note: Complexity plot clearly shows high error for low complexity (underfit) and increasing validation error for high complexity (overfit).")
print("-" * 40)


# --- Scenario 3: Complex Model, Few Samples ---
print_step_header(3, "Complex Neural Network, 100 Samples")

n_samples = 100
X = np.random.rand(n_samples, 5) # 5 features
# Simple linear true relationship
true_coef_nn = np.array([2, -1, 0.5, 3, -2.5])
y = X @ true_coef_nn + np.random.normal(0, 0.5, n_samples)

# Very complex model for the data size and underlying relationship
model3 = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000, random_state=42, alpha=1e-5, early_stopping=False) # Small alpha -> less regularization
model3.fit(X, y)
train_score3 = model3.score(X, y)

print(f"Scenario 3 Setup: n_samples={n_samples}, simple linear data.")
print(f"Model: Complex Neural Network (3 hidden layers, 100 neurons each)")
print(f"Training R^2 Score: {train_score3:.4f}") # R2 score might not be the best metric here, but shows high fit
print("Observation: The neural network is very complex relative to the data size and the true linear relationship.")
print("It achieves a near-perfect training score, likely memorizing noise. This indicates HIGH VARIANCE and OVERFITTING.")
print("Improvement Strategies: Get more data, simplify the network (fewer layers/neurons), use regularization (increase alpha, add dropout), early stopping.")

# Note: Cannot easily plot fit for >1 feature. Learning curve is more informative.
# Use the same model structure for learning curve plotting
estimator_nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000, random_state=42, alpha=1e-5, warm_start=True, early_stopping=False)
plot_learning_curve(estimator_nn, "Learning Curve (Scenario 3 - Overfitting Example)", X, y, cv=5, filename="scenario3_learning_curve.png")
print("Note: Learning curve shows a large gap between low training error and high validation error, classic overfitting.")
print("-" * 40)


# --- Scenario 4: Simple Model, Non-linear Data ---
print_step_header(4, "Linear Regression for Non-linear Data")

# Reuse data from Scenario 2
model4 = LinearRegression()
model4.fit(X_nl, y_nl)
train_score4 = model4.score(X_nl, y_nl)

print(f"Scenario 4 Setup: Modeling non-linear (sine wave) data with a linear model.")
print(f"Model: Linear Regression")
print(f"Training R^2 Score: {train_score4:.4f}")
print("Observation: The linear model is too simple to capture the non-linear sine wave pattern.")
print("This results in poor fit even on the training data, indicating HIGH BIAS and UNDERFITTING.")
print("Improvement Strategies: Use polynomial features, basis expansion, or switch to a non-linear model (e.g., polynomial regression, decision tree, kernel methods, neural network).")

# Illustrate improvement with Polynomial Features
poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
poly_model.fit(X_nl, y_nl)

plot_fit(X_nl, y_nl, model4, "Linear Fit on Non-linear Data (Scenario 4 - Underfitting)", "scenario4_linear_fit.png")
plot_fit(X_nl, y_nl, poly_model, "Polynomial Fit on Non-linear Data (Improvement)", "scenario4_poly_fit.png")
plot_learning_curve(LinearRegression(), "Learning Curve (Scenario 4 - Underfitting Example)",
                    X_nl, y_nl, cv=5, filename="scenario4_learning_curve.png")
print("Note: Learning curve similar to Scenario 2, errors converge at a high value.")
print("-" * 40)

print("\nScript finished. Plots saved in:", save_dir) 