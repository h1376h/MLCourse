import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Question 5: Feature Scaling in Gradient Descent")
print("=" * 50)

# Step 1: Generate synthetic data with features of different scales
def generate_data(n_samples=100):
    """Generate synthetic data with features of different scales."""
    print("Step 1: Generating synthetic data with features of different scales")
    
    # Generate features with specified ranges
    x1 = np.random.uniform(0, 1, n_samples)  # x1 ranges from 0 to 1
    x2 = np.random.uniform(0, 10000, n_samples)  # x2 ranges from 0 to 10,000
    x3 = np.random.uniform(-100, 100, n_samples)  # x3 ranges from -100 to 100
    
    # Create a target with a known relationship to the features
    # y = 2*x1 + 0.0005*x2 + 0.05*x3 + noise
    y = 2 * x1 + 0.0005 * x2 + 0.05 * x3 + np.random.normal(0, 0.1, n_samples)
    
    # Create and display a dataframe with the first few samples
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })
    
    print("\nFirst few samples of the generated data:")
    print(df.head())
    
    # Display the feature statistics to highlight the different scales
    print("\nFeature statistics:")
    print(df.describe())
    
    # Create a visualization of the feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot histograms for each feature
    sns.histplot(x1, kde=True, ax=axes[0])
    axes[0].set_title('Distribution of x1 (0 to 1)')
    axes[0].set_xlabel('x1')
    
    sns.histplot(x2, kde=True, ax=axes[1])
    axes[1].set_title('Distribution of x2 (0 to 10,000)')
    axes[1].set_xlabel('x2')
    
    sns.histplot(x3, kde=True, ax=axes[2])
    axes[2].set_title('Distribution of x3 (-100 to 100)')
    axes[2].set_xlabel('x3')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_distributions.png"), dpi=300)
    plt.close()
    
    print("\nFeature distribution histograms saved.")
    print()
    
    return x1, x2, x3, y

# Generate the data
x1, x2, x3, y = generate_data()

# New function to add detailed step-by-step calculations
def detailed_mathematical_derivation():
    """Provide detailed step-by-step calculations for understanding gradient descent with different feature scales."""
    print("\nStep 1.5: Detailed Mathematical Derivation of Gradient Descent with Different Feature Scales")
    print("=" * 100)
    
    # Select a small subset of data for illustrative calculations
    sample_size = 3
    X_sample = np.column_stack((x1[:sample_size], x2[:sample_size], x3[:sample_size]))
    y_sample = y[:sample_size]
    
    print(f"Using a small sample of {sample_size} data points for illustration:")
    for i in range(sample_size):
        print(f"Data point {i+1}: x1 = {X_sample[i, 0]:.4f}, x2 = {X_sample[i, 1]:.2f}, x3 = {X_sample[i, 2]:.4f}, y = {y_sample[i]:.4f}")
    
    print("\n1. Linear Regression Model:")
    print("   y = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ε")
    print("   where θ₀, θ₁, θ₂, θ₃ are parameters to be learned")
    
    print("\n2. Cost Function (Mean Squared Error):")
    print("   J(θ) = (1/m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²")
    print("   where h_θ(x⁽ⁱ⁾) = θ₀ + θ₁x₁⁽ⁱ⁾ + θ₂x₂⁽ⁱ⁾ + θ₃x₃⁽ⁱ⁾")
    
    print("\n3. Gradient of the Cost Function:")
    print("   ∂J(θ)/∂θⱼ = (2/m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x_j⁽ⁱ⁾")
    print("   where x_0⁽ⁱ⁾ = 1 (for the intercept term)")
    
    # Initialize some parameters for demonstration
    theta_init = np.array([0.1, 0.1, 0.1, 0.1])
    learning_rate = 0.01
    
    print("\n4. Manual Calculation of One Gradient Descent Step (UNSCALED features):")
    print("   Initial parameters: θ₀ = 0.1, θ₁ = 0.1, θ₂ = 0.1, θ₃ = 0.1")
    print("   Learning rate α = 0.01")
    
    # Calculate predictions for each data point
    predictions = []
    for i in range(sample_size):
        h_theta = theta_init[0] + theta_init[1] * X_sample[i, 0] + theta_init[2] * X_sample[i, 1] + theta_init[3] * X_sample[i, 2]
        predictions.append(h_theta)
        print(f"   Prediction for data point {i+1}: h_θ(x⁽{i+1}⁾) = {theta_init[0]} + {theta_init[1]} * {X_sample[i, 0]:.4f} + {theta_init[2]} * {X_sample[i, 1]:.2f} + {theta_init[3]} * {X_sample[i, 2]:.4f} = {h_theta:.4f}")
    
    # Calculate errors
    errors = []
    for i in range(sample_size):
        error = predictions[i] - y_sample[i]
        errors.append(error)
        print(f"   Error for data point {i+1}: h_θ(x⁽{i+1}⁾) - y⁽{i+1}⁾ = {predictions[i]:.4f} - {y_sample[i]:.4f} = {error:.4f}")
    
    # Calculate gradients for each parameter
    gradients = np.zeros(4)
    
    # For θ₀ (intercept)
    gradients[0] = (2/sample_size) * sum(errors)
    gradient_calc_0 = " + ".join([f"{errors[i]:.4f} * 1" for i in range(sample_size)])
    print(f"   Gradient for θ₀: (2/{sample_size}) * ({gradient_calc_0}) = {gradients[0]:.4f}")
    
    # For θ₁ (coefficient of x₁)
    gradients[1] = (2/sample_size) * sum(errors[i] * X_sample[i, 0] for i in range(sample_size))
    gradient_calc_1 = " + ".join([f"{errors[i]:.4f} * {X_sample[i, 0]:.4f}" for i in range(sample_size)])
    print(f"   Gradient for θ₁: (2/{sample_size}) * ({gradient_calc_1}) = {gradients[1]:.4f}")
    
    # For θ₂ (coefficient of x₂)
    gradients[2] = (2/sample_size) * sum(errors[i] * X_sample[i, 1] for i in range(sample_size))
    gradient_calc_2 = " + ".join([f"{errors[i]:.4f} * {X_sample[i, 1]:.2f}" for i in range(sample_size)])
    print(f"   Gradient for θ₂: (2/{sample_size}) * ({gradient_calc_2}) = {gradients[2]:.4f}")
    
    # For θ₃ (coefficient of x₃)
    gradients[3] = (2/sample_size) * sum(errors[i] * X_sample[i, 2] for i in range(sample_size))
    gradient_calc_3 = " + ".join([f"{errors[i]:.4f} * {X_sample[i, 2]:.4f}" for i in range(sample_size)])
    print(f"   Gradient for θ₃: (2/{sample_size}) * ({gradient_calc_3}) = {gradients[3]:.4f}")
    
    # Update parameters
    theta_new = theta_init - learning_rate * gradients
    print("\n   Parameter Updates:")
    print(f"   θ₀ = {theta_init[0]} - {learning_rate} * {gradients[0]:.4f} = {theta_new[0]:.4f}")
    print(f"   θ₁ = {theta_init[1]} - {learning_rate} * {gradients[1]:.4f} = {theta_new[1]:.4f}")
    print(f"   θ₂ = {theta_init[2]} - {learning_rate} * {gradients[2]:.4f} = {theta_new[2]:.4f}")
    print(f"   θ₃ = {theta_init[3]} - {learning_rate} * {gradients[3]:.4f} = {theta_new[3]:.4f}")
    
    print("\n   IMPORTANT OBSERVATION: The gradient for θ₂ is much larger because x₂ has values in thousands!")
    print("   This causes θ₂ to take much larger steps, potentially leading to oscillation or divergence.")
    
    # Now perform the same calculation with normalized data
    X_sample_scaled = np.copy(X_sample)
    
    # Min-Max scaling (manually)
    x1_min, x1_max = np.min(x1), np.max(x1)
    x2_min, x2_max = np.min(x2), np.max(x2)
    x3_min, x3_max = np.min(x3), np.max(x3)
    
    for i in range(sample_size):
        X_sample_scaled[i, 0] = (X_sample[i, 0] - x1_min) / (x1_max - x1_min)
        X_sample_scaled[i, 1] = (X_sample[i, 1] - x2_min) / (x2_max - x2_min)
        X_sample_scaled[i, 2] = (X_sample[i, 2] - x3_min) / (x3_max - x3_min)
    
    print("\n5. Manual Calculation with MIN-MAX SCALED features:")
    print("   Initial parameters: θ₀ = 0.1, θ₁ = 0.1, θ₂ = 0.1, θ₃ = 0.1")
    
    print("\n   Scaled feature values:")
    for i in range(sample_size):
        print(f"   Data point {i+1}: x1_scaled = {X_sample_scaled[i, 0]:.4f}, x2_scaled = {X_sample_scaled[i, 1]:.4f}, x3_scaled = {X_sample_scaled[i, 2]:.4f}")
    
    # Calculate predictions for each data point with scaled features
    scaled_predictions = []
    for i in range(sample_size):
        h_theta = theta_init[0] + theta_init[1] * X_sample_scaled[i, 0] + theta_init[2] * X_sample_scaled[i, 1] + theta_init[3] * X_sample_scaled[i, 2]
        scaled_predictions.append(h_theta)
        print(f"   Prediction: h_θ(x⁽{i+1}⁾) = {theta_init[0]} + {theta_init[1]} * {X_sample_scaled[i, 0]:.4f} + {theta_init[2]} * {X_sample_scaled[i, 1]:.4f} + {theta_init[3]} * {X_sample_scaled[i, 2]:.4f} = {h_theta:.4f}")
    
    # Calculate errors with scaled features
    scaled_errors = []
    for i in range(sample_size):
        error = scaled_predictions[i] - y_sample[i]
        scaled_errors.append(error)
        print(f"   Error: h_θ(x⁽{i+1}⁾) - y⁽{i+1}⁾ = {scaled_predictions[i]:.4f} - {y_sample[i]:.4f} = {error:.4f}")
    
    # Calculate gradients for each parameter with scaled features
    scaled_gradients = np.zeros(4)
    
    # For θ₀ (intercept)
    scaled_gradients[0] = (2/sample_size) * sum(scaled_errors)
    scaled_gradient_calc_0 = " + ".join([f"{scaled_errors[i]:.4f} * 1" for i in range(sample_size)])
    print(f"   Gradient for θ₀: (2/{sample_size}) * ({scaled_gradient_calc_0}) = {scaled_gradients[0]:.4f}")
    
    # For θ₁ (coefficient of x₁)
    scaled_gradients[1] = (2/sample_size) * sum(scaled_errors[i] * X_sample_scaled[i, 0] for i in range(sample_size))
    scaled_gradient_calc_1 = " + ".join([f"{scaled_errors[i]:.4f} * {X_sample_scaled[i, 0]:.4f}" for i in range(sample_size)])
    print(f"   Gradient for θ₁: (2/{sample_size}) * ({scaled_gradient_calc_1}) = {scaled_gradients[1]:.4f}")
    
    # For θ₂ (coefficient of x₂)
    scaled_gradients[2] = (2/sample_size) * sum(scaled_errors[i] * X_sample_scaled[i, 1] for i in range(sample_size))
    scaled_gradient_calc_2 = " + ".join([f"{scaled_errors[i]:.4f} * {X_sample_scaled[i, 1]:.4f}" for i in range(sample_size)])
    print(f"   Gradient for θ₂: (2/{sample_size}) * ({scaled_gradient_calc_2}) = {scaled_gradients[2]:.4f}")
    
    # For θ₃ (coefficient of x₃)
    scaled_gradients[3] = (2/sample_size) * sum(scaled_errors[i] * X_sample_scaled[i, 2] for i in range(sample_size))
    scaled_gradient_calc_3 = " + ".join([f"{scaled_errors[i]:.4f} * {X_sample_scaled[i, 2]:.4f}" for i in range(sample_size)])
    print(f"   Gradient for θ₃: (2/{sample_size}) * ({scaled_gradient_calc_3}) = {scaled_gradients[3]:.4f}")
    
    # Update parameters
    scaled_theta_new = theta_init - learning_rate * scaled_gradients
    print("\n   Parameter Updates with scaled features:")
    print(f"   θ₀ = {theta_init[0]} - {learning_rate} * {scaled_gradients[0]:.4f} = {scaled_theta_new[0]:.4f}")
    print(f"   θ₁ = {theta_init[1]} - {learning_rate} * {scaled_gradients[1]:.4f} = {scaled_theta_new[1]:.4f}")
    print(f"   θ₂ = {theta_init[2]} - {learning_rate} * {scaled_gradients[2]:.4f} = {scaled_theta_new[2]:.4f}")
    print(f"   θ₃ = {theta_init[3]} - {learning_rate} * {scaled_gradients[3]:.4f} = {scaled_theta_new[3]:.4f}")
    
    print("\n   IMPORTANT OBSERVATION: Now all gradients are of similar magnitudes!")
    print("   This leads to more balanced parameter updates and better convergence properties.")
    
    print("\n6. Mathematical Relationship Between Original and Scaled Parameters:")
    print("   Assuming we have learned parameters θ_scaled for scaled features, how do we relate them to original features?")
    
    print("\n   For Min-Max Scaling:")
    print("   x_scaled = (x - x_min) / (x_max - x_min)")
    print("   This implies: x = x_min + (x_max - x_min) * x_scaled")
    
    print("\n   Substituting into the linear model:")
    print("   y = θ₀ + θ₁*x₁ + θ₂*x₂ + θ₃*x₃")
    print("   y = θ₀ + θ₁*[x₁_min + (x₁_max - x₁_min)*x₁_scaled] + θ₂*[x₂_min + (x₂_max - x₂_min)*x₂_scaled] + θ₃*[x₃_min + (x₃_max - x₃_min)*x₃_scaled]")
    print("   y = θ₀ + θ₁*x₁_min + θ₁*(x₁_max - x₁_min)*x₁_scaled + θ₂*x₂_min + θ₂*(x₂_max - x₂_min)*x₂_scaled + θ₃*x₃_min + θ₃*(x₃_max - x₃_min)*x₃_scaled")
    
    print("\n   Rearranging:")
    print("   y = [θ₀ + θ₁*x₁_min + θ₂*x₂_min + θ₃*x₃_min] + [θ₁*(x₁_max - x₁_min)]*x₁_scaled + [θ₂*(x₂_max - x₂_min)]*x₂_scaled + [θ₃*(x₃_max - x₃_min)]*x₃_scaled")
    
    print("\n   Therefore:")
    print("   θ₀_scaled = θ₀ + θ₁*x₁_min + θ₂*x₂_min + θ₃*x₃_min")
    print("   θ₁_scaled = θ₁*(x₁_max - x₁_min)")
    print("   θ₂_scaled = θ₂*(x₂_max - x₂_min)")
    print("   θ₃_scaled = θ₃*(x₃_max - x₃_min)")
    
    print("\n   Solving for original parameters:")
    print("   θ₁ = θ₁_scaled / (x₁_max - x₁_min)")
    print("   θ₂ = θ₂_scaled / (x₂_max - x₂_min)")
    print("   θ₃ = θ₃_scaled / (x₃_max - x₃_min)")
    
    print("\n   Similarly for Standardization:")
    print("   x_scaled = (x - μ) / σ")
    print("   This implies: x = μ + σ * x_scaled")
    
    print("\n   Following similar derivation:")
    print("   θ₁ = θ₁_scaled / σ₁")
    print("   θ₂ = θ₂_scaled / σ₂")
    print("   θ₃ = θ₃_scaled / σ₃")
    
    print("=" * 100)

# Call the detailed mathematical derivation function
detailed_mathematical_derivation()

# Step 2: Implement gradient descent without feature scaling
def gradient_descent_no_scaling(X, y, learning_rate=0.01, n_iterations=1000):
    """Implement batch gradient descent without feature scaling."""
    print("Step 2: Implementing gradient descent WITHOUT feature scaling")
    
    # Add intercept term to X
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    n_features = X_b.shape[1]
    
    # Initialize weights
    theta = np.random.randn(n_features, 1)
    
    # Store cost history and parameter history
    cost_history = np.zeros(n_iterations)
    theta_history = np.zeros((n_iterations, n_features))
    
    # Gradient descent iterations
    for i in range(n_iterations):
        # Compute predictions
        predictions = X_b.dot(theta)
        
        # Compute errors
        errors = predictions - y.reshape(-1, 1)
        
        # Compute gradients
        gradients = 2/m * X_b.T.dot(errors)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Store history
        cost_history[i] = np.mean(errors ** 2)
        theta_history[i] = theta.flatten()
        
        # Print progress every 100 iterations
        if i % 100 == 0 or i == n_iterations - 1:
            print(f"Iteration {i}: Cost = {cost_history[i]:.4f}")
    
    print("\nFinal parameter values without scaling:")
    print(f"theta_0 (intercept): {theta[0, 0]:.8f}")
    print(f"theta_1 (x1): {theta[1, 0]:.8f}")
    print(f"theta_2 (x2): {theta[2, 0]:.8f}")
    print(f"theta_3 (x3): {theta[3, 0]:.8f}")
    print()
    
    return theta, cost_history, theta_history

# Step 3: Implement feature scaling techniques
def scale_features(x1, x2, x3):
    """Implement Min-Max scaling and Standardization for features."""
    print("Step 3: Implementing feature scaling techniques")
    
    # Original features
    X_original = np.column_stack((x1, x2, x3))
    
    # 1. Min-Max Scaling
    print("\nMin-Max Scaling:")
    min_max_scaler = MinMaxScaler()
    X_min_max = min_max_scaler.fit_transform(X_original)
    
    # Get the min and max values used for scaling
    mins = min_max_scaler.data_min_
    maxs = min_max_scaler.data_max_
    
    print("Original ranges:")
    for i, (feat_name, feat_min, feat_max) in enumerate(zip(['x1', 'x2', 'x3'], mins, maxs)):
        print(f"{feat_name}: [{feat_min:.2f}, {feat_max:.2f}]")
    
    print("\nAfter Min-Max Scaling:")
    print("All features range: [0, 1]")
    
    # Show how the transformation works for each feature
    print("\nMin-Max scaling transformation:")
    print("x_scaled = (x - x_min) / (x_max - x_min)")
    for i, (feat_name, feat_min, feat_max) in enumerate(zip(['x1', 'x2', 'x3'], mins, maxs)):
        print(f"For {feat_name}: x_scaled = (x - {feat_min:.2f}) / ({feat_max:.2f} - {feat_min:.2f})")
    
    # 2. Standardization (Z-score normalization)
    print("\nStandardization (Z-score normalization):")
    standard_scaler = StandardScaler()
    X_standardized = standard_scaler.fit_transform(X_original)
    
    # Get the mean and standard deviation values used for scaling
    means = standard_scaler.mean_
    stds = standard_scaler.scale_
    
    print("Original statistics:")
    for i, (feat_name, feat_mean, feat_std) in enumerate(zip(['x1', 'x2', 'x3'], means, stds)):
        print(f"{feat_name}: mean = {feat_mean:.2f}, std = {feat_std:.2f}")
    
    print("\nAfter Standardization:")
    print("All features have: mean = 0, std = 1")
    
    # Show how the transformation works for each feature
    print("\nStandardization transformation:")
    print("x_scaled = (x - μ) / σ")
    for i, (feat_name, feat_mean, feat_std) in enumerate(zip(['x1', 'x2', 'x3'], means, stds)):
        print(f"For {feat_name}: x_scaled = (x - {feat_mean:.2f}) / {feat_std:.2f}")
    
    # Create a visualization comparing original and scaled features
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Original features (first column)
    sns.histplot(x1, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Original x1')
    axes[0, 0].set_xlabel('x1')
    
    sns.histplot(x2, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Original x2')
    axes[1, 0].set_xlabel('x2')
    
    sns.histplot(x3, kde=True, ax=axes[2, 0])
    axes[2, 0].set_title('Original x3')
    axes[2, 0].set_xlabel('x3')
    
    # Min-Max scaled features (second column)
    sns.histplot(X_min_max[:, 0], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Min-Max Scaled x1')
    axes[0, 1].set_xlabel('x1 (scaled)')
    
    sns.histplot(X_min_max[:, 1], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Min-Max Scaled x2')
    axes[1, 1].set_xlabel('x2 (scaled)')
    
    sns.histplot(X_min_max[:, 2], kde=True, ax=axes[2, 1])
    axes[2, 1].set_title('Min-Max Scaled x3')
    axes[2, 1].set_xlabel('x3 (scaled)')
    
    # Standardized features (third column)
    sns.histplot(X_standardized[:, 0], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Standardized x1')
    axes[0, 2].set_xlabel('x1 (standardized)')
    
    sns.histplot(X_standardized[:, 1], kde=True, ax=axes[1, 2])
    axes[1, 2].set_title('Standardized x2')
    axes[1, 2].set_xlabel('x2 (standardized)')
    
    sns.histplot(X_standardized[:, 2], kde=True, ax=axes[2, 2])
    axes[2, 2].set_title('Standardized x3')
    axes[2, 2].set_xlabel('x3 (standardized)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scaled_features_comparison.png"), dpi=300)
    plt.close()
    
    print("\nFeature scaling visualizations saved.")
    print()
    
    return X_original, X_min_max, X_standardized, (mins, maxs), (means, stds)

# Scale the features
X_original, X_min_max, X_standardized, (mins, maxs), (means, stds) = scale_features(x1, x2, x3)

# Step 4: Run gradient descent on original and scaled features
print("Step 4: Running gradient descent with original and scaled features")

# Run gradient descent without scaling
print("\nGradient Descent WITHOUT feature scaling:")
theta_no_scaling, cost_history_no_scaling, theta_history_no_scaling = gradient_descent_no_scaling(
    X_original, y, learning_rate=0.01, n_iterations=1000
)

# Run gradient descent with Min-Max scaling
print("\nGradient Descent WITH Min-Max scaling:")
theta_min_max, cost_history_min_max, theta_history_min_max = gradient_descent_no_scaling(
    X_min_max, y, learning_rate=0.01, n_iterations=1000
)

# Run gradient descent with Standardization
print("\nGradient Descent WITH Standardization:")
theta_standardized, cost_history_standardized, theta_history_standardized = gradient_descent_no_scaling(
    X_standardized, y, learning_rate=0.01, n_iterations=1000
)

# Step 5: Visualize the convergence of gradient descent
def visualize_convergence():
    """Visualize how feature scaling affects gradient descent convergence."""
    print("Step 5: Visualizing the convergence of gradient descent")
    
    # Plot the cost history for all three cases
    plt.figure(figsize=(12, 6))
    plt.plot(cost_history_no_scaling, label='No Scaling', linewidth=2)
    plt.plot(cost_history_min_max, label='Min-Max Scaling', linewidth=2)
    plt.plot(cost_history_standardized, label='Standardization', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Gradient Descent Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale to better see the differences
    plt.savefig(os.path.join(save_dir, "convergence_comparison.png"), dpi=300)
    plt.close()
    
    # Plot the learning process in parameter space (for selected parameters)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot parameter trajectories for no scaling
    axes[0].plot(theta_history_no_scaling[:, 1], label='$\\theta_1$ (x1)', linewidth=2)
    axes[0].plot(theta_history_no_scaling[:, 2], label='$\\theta_2$ (x2)', linewidth=2)
    axes[0].plot(theta_history_no_scaling[:, 3], label='$\\theta_3$ (x3)', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Parameter Value')
    axes[0].set_title('Parameter Convergence - No Scaling')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot parameter trajectories for Min-Max scaling
    axes[1].plot(theta_history_min_max[:, 1], label='$\\theta_1$ (x1)', linewidth=2)
    axes[1].plot(theta_history_min_max[:, 2], label='$\\theta_2$ (x2)', linewidth=2)
    axes[1].plot(theta_history_min_max[:, 3], label='$\\theta_3$ (x3)', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Parameter Value')
    axes[1].set_title('Parameter Convergence - Min-Max Scaling')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot parameter trajectories for Standardization
    axes[2].plot(theta_history_standardized[:, 1], label='$\\theta_1$ (x1)', linewidth=2)
    axes[2].plot(theta_history_standardized[:, 2], label='$\\theta_2$ (x2)', linewidth=2)
    axes[2].plot(theta_history_standardized[:, 3], label='$\\theta_3$ (x3)', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Parameter Value')
    axes[2].set_title('Parameter Convergence - Standardization')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parameter_convergence.png"), dpi=300)
    plt.close()
    
    # Zoom in on the parameter space to better visualize the different scales
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Parameter path for θ₁ (x1)
    axes[0].plot(theta_history_no_scaling[:, 1], label='No Scaling', linewidth=2)
    axes[0].plot(theta_history_min_max[:, 1], label='Min-Max Scaling', linewidth=2)
    axes[0].plot(theta_history_standardized[:, 1], label='Standardization', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('$\\theta_1$ (x1) Value')
    axes[0].set_title('Convergence of $\\theta_1$ (coefficient for x1)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Parameter path for θ₂ (x2)
    axes[1].plot(theta_history_no_scaling[:, 2], label='No Scaling', linewidth=2)
    axes[1].plot(theta_history_min_max[:, 2], label='Min-Max Scaling', linewidth=2)
    axes[1].plot(theta_history_standardized[:, 2], label='Standardization', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('$\\theta_2$ (x2) Value')
    axes[1].set_title('Convergence of $\\theta_2$ (coefficient for x2)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Parameter path for θ₃ (x3)
    axes[2].plot(theta_history_no_scaling[:, 3], label='No Scaling', linewidth=2)
    axes[2].plot(theta_history_min_max[:, 3], label='Min-Max Scaling', linewidth=2)
    axes[2].plot(theta_history_standardized[:, 3], label='Standardization', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('$\\theta_3$ (x3) Value')
    axes[2].set_title('Convergence of $\\theta_3$ (coefficient for x3)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parameter_convergence_by_feature.png"), dpi=300)
    plt.close()
    
    print("\nConvergence visualizations saved.")
    print()

visualize_convergence()

# Step 6: Explain the effect on parameter interpretation
def explain_parameter_interpretation():
    """Explain how feature scaling affects the interpretation of parameters."""
    print("Step 6: Explaining the effect on parameter interpretation")
    
    # Compute the true parameters in the original scale
    print("\nRecomputing the parameters for interpretation:")
    
    # The original underlying model is: y = 2*x1 + 0.0005*x2 + 0.05*x3 + noise
    print("\nOriginal (true) model parameters:")
    print("theta_1 (x1): 2.0")
    print("theta_2 (x2): 0.0005")
    print("theta_3 (x3): 0.05")
    
    # For Min-Max scaling, transform parameters back to original scale
    print("\nMin-Max scaled parameters transformed to original scale:")
    
    # Original scale θ₁ = Min-Max scale θ₁ / (x1_max - x1_min)
    # Original scale θ₂ = Min-Max scale θ₂ / (x2_max - x2_min)
    # Original scale θ₃ = Min-Max scale θ₃ / (x3_max - x3_min)
    
    mm_theta1_original = theta_min_max[1, 0] / (maxs[0] - mins[0])
    mm_theta2_original = theta_min_max[2, 0] / (maxs[1] - mins[1])
    mm_theta3_original = theta_min_max[3, 0] / (maxs[2] - mins[2])
    
    print(f"theta_1 (x1): {mm_theta1_original:.4f}")
    print(f"theta_2 (x2): {mm_theta2_original:.8f}")
    print(f"theta_3 (x3): {mm_theta3_original:.4f}")
    
    # For Standardized scaling, transform parameters back to original scale
    print("\nStandardized parameters transformed to original scale:")
    
    # Original scale θ₁ = Standardized scale θ₁ / std(x1)
    # Original scale θ₂ = Standardized scale θ₂ / std(x2)
    # Original scale θ₃ = Standardized scale θ₃ / std(x3)
    
    std_theta1_original = theta_standardized[1, 0] / stds[0]
    std_theta2_original = theta_standardized[2, 0] / stds[1]
    std_theta3_original = theta_standardized[3, 0] / stds[2]
    
    print(f"theta_1 (x1): {std_theta1_original:.4f}")
    print(f"theta_2 (x2): {std_theta2_original:.8f}")
    print(f"theta_3 (x3): {std_theta3_original:.4f}")
    
    # Relative importance visualization
    print("\nCreating visualization of feature importance with and without scaling...")
    
    # For unscaled features
    unscaled_coeffs = np.abs([theta_no_scaling[1, 0], theta_no_scaling[2, 0], theta_no_scaling[3, 0]])
    
    # For min-max scaled features (keep scaled coefficients to show relative importance)
    minmax_coeffs = np.abs([theta_min_max[1, 0], theta_min_max[2, 0], theta_min_max[3, 0]])
    
    # For standardized features (keep scaled coefficients to show relative importance)
    std_coeffs = np.abs([theta_standardized[1, 0], theta_standardized[2, 0], theta_standardized[3, 0]])
    
    # Plot comparative bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    feature_names = ['x1', 'x2', 'x3']
    
    # Plot unscaled coefficients
    axes[0].bar(feature_names, unscaled_coeffs)
    axes[0].set_title('Feature Importance (Unscaled)')
    axes[0].set_ylabel('Absolute Coefficient Value')
    
    # Plot min-max scaled coefficients
    axes[1].bar(feature_names, minmax_coeffs)
    axes[1].set_title('Feature Importance (Min-Max Scaled)')
    axes[1].set_ylabel('Absolute Coefficient Value')
    
    # Plot standardized coefficients
    axes[2].bar(feature_names, std_coeffs)
    axes[2].set_title('Feature Importance (Standardized)')
    axes[2].set_ylabel('Absolute Coefficient Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance_comparison.png"), dpi=300)
    plt.close()
    
    print("\nFeature importance visualization saved.")
    print()

explain_parameter_interpretation()

# Summarize the findings
print("\nSummary of Findings:")
print("1. Without feature scaling, gradient descent can converge very slowly or fail to converge")
print("   due to the vastly different magnitudes of features (x1: 0-1, x2: 0-10000, x3: -100-100).")
print()
print("2. Feature scaling techniques like Min-Max scaling and Standardization help gradient")
print("   descent converge faster and more reliably by bringing all features to similar scales.")
print()
print("3. Min-Max scaling transforms features to the range [0, 1], while Standardization")
print("   transforms them to have mean 0 and standard deviation 1.")
print()
print("4. When features are scaled, the interpretation of learned parameters changes:")
print("   - Without scaling: parameters directly represent the effect of one unit change in original features")
print("   - With scaling: parameters represent the effect of changes in scaled features")
print("   - Parameters for scaled features can be transformed back to the original scale for interpretation")
print()
print("5. Scaling helps reveal the true relative importance of features, which can be hidden")
print("   when the features have vastly different scales.")
print()
print(f"Visualization images saved to: {save_dir}")
print("Generated images:")
print("- feature_distributions.png: Histograms of original feature distributions")
print("- scaled_features_comparison.png: Comparison of original vs. scaled features")
print("- convergence_comparison.png: Convergence of cost function for different scaling methods")
print("- parameter_convergence.png: Convergence of all parameters for different scaling methods")
print("- parameter_convergence_by_feature.png: Convergence of individual parameters across methods")
print("- feature_importance_comparison.png: Comparison of feature importance with different scaling") 