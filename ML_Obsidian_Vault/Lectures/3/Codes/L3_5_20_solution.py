import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# -------------------- Step 1: Introduction to Online Learning --------------------
print("Question 20: Online Learning with Stochastic Gradient Descent for a Recommendation System")
print("=" * 80)
print("\nStep 1: Understanding Online Learning vs Batch Learning")
print("-" * 80)
print("""Online learning is a machine learning paradigm where the model is updated incrementally
as new data becomes available, one instance or small batch at a time. This is particularly
suitable for the e-commerce recommendation scenario because:

1. Data arrives continuously as users interact with the platform
2. The model needs to adapt quickly to new user behaviors
3. Storing and reprocessing all historical data would be inefficient
4. Real-time recommendations require up-to-date models

In contrast, batch learning processes all available data at once to train a model, requiring
retraining from scratch when new data arrives.""")

# -------------------- Step 2: Implementing Linear Regression with SGD --------------------
print("\n\nStep 2: Stochastic Gradient Descent Update Rule for Linear Regression")
print("-" * 80)
print("""For linear regression with squared error loss, the stochastic gradient descent (SGD)
update rule for a single data point (x, y) is:

w ← w - η ∇L(w)
w ← w - η (w^T x - y)x

Where:
- w is the weight vector (model parameters)
- η (eta) is the learning rate
- (w^T x - y) is the prediction error for the current data point
- x is the feature vector for the current data point
- ∇L(w) is the gradient of the loss function with respect to w

This update rule enables online learning because:
1. It requires only a single data point to update the model
2. It doesn't need to store or reprocess historical data
3. It gradually improves the model with each new observation
4. It has a constant time and space complexity regardless of dataset size""")

# -------------------- Step 3: Setting up synthetic data for demonstration --------------------
print("\n\nStep 3: Setting up a synthetic e-commerce recommendation scenario")
print("-" * 80)

# Generate synthetic data for an e-commerce recommendation system
def generate_initial_data(n_samples=1000, n_features=5):
    # Each feature represents some user behavior metric
    X = np.random.randn(n_samples, n_features)
    
    # True weights representing the importance of each feature
    true_weights = np.array([0.5, -0.2, 0.3, -0.1, 0.7])
    
    # Add intercept term
    true_bias = 2.0
    
    # Generate target values (purchase amounts) with some noise
    y = X.dot(true_weights) + true_bias + np.random.randn(n_samples) * 0.5
    
    # Make sure all purchase amounts are positive
    y = np.maximum(y, 0.0)
    
    return X, y, true_weights, true_bias

# Generate new data point that might arrive in real-time
def generate_new_data_point():
    X_new = np.random.randn(1, 5)
    true_weights = np.array([0.5, -0.2, 0.3, -0.1, 0.7])
    true_bias = 2.0
    y_new = X_new.dot(true_weights) + true_bias + np.random.randn() * 0.5
    y_new = max(y_new[0], 0.0)
    return X_new, y_new

# Generate initial dataset
X, y, true_weights, true_bias = generate_initial_data()

print(f"Generated synthetic data with {len(X)} samples and {X.shape[1]} features")
print(f"True weights: {true_weights}")
print(f"True bias: {true_bias}")

# Display first few samples
sample_data = pd.DataFrame(
    np.column_stack([X[:5], y[:5]]), 
    columns=[f"Feature {i+1}" for i in range(X.shape[1])] + ["Purchase Amount"]
)
print("\nFirst few samples (showing 5 features and purchase amount):")
print(sample_data)

# -------------------- Step 4: Implementing and visualizing different learning methods --------------------
print("\n\nStep 4: Implementing Batch Learning with Normal Equations")
print("-" * 80)

def normal_equation_solution(X, y):
    """Solve linear regression using normal equations."""
    start_time = time.time()
    
    # Add bias term (intercept) to X
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Calculate solution using normal equation
    XTX = X_with_bias.T.dot(X_with_bias)
    XTy = X_with_bias.T.dot(y)
    
    # This would be the bottleneck for large datasets
    w = np.linalg.inv(XTX).dot(XTy)
    
    elapsed_time = time.time() - start_time
    return w[0], w[1:], elapsed_time

# Get baseline solution using normal equations
bias_ne, weights_ne, time_ne = normal_equation_solution(X, y)

print(f"Normal Equation Solution (Batch Learning):")
print(f"Weights: {weights_ne}")
print(f"Bias: {bias_ne}")
print(f"Computation time: {time_ne:.6f} seconds")

print("\n\nStep 5: Implementing Online Learning with SGD")
print("-" * 80)

def sgd_train(X, y, learning_rate=0.01, n_epochs=1):
    """Train linear regression using SGD."""
    n_samples, n_features = X.shape
    
    # Initialize weights randomly
    weights = np.random.randn(n_features) * 0.01
    bias = 0.0
    
    start_time = time.time()
    
    # Training history for visualization
    weights_history = [weights.copy()]
    bias_history = [bias]
    loss_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        
        # Process each training example
        for i in range(n_samples):
            # Forward pass (prediction)
            y_pred = np.dot(X_shuffled[i], weights) + bias
            
            # Compute error
            error = y_pred - y_shuffled[i]
            
            # Gradient of the loss function
            grad_weights = error * X_shuffled[i]
            grad_bias = error
            
            # Update weights and bias
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias
            
            # Accumulate loss
            epoch_loss += 0.5 * error**2
            
            # Store every 100th update for visualization
            if i % 100 == 0:
                weights_history.append(weights.copy())
                bias_history.append(bias)
        
        # Average loss for this epoch
        avg_epoch_loss = epoch_loss / n_samples
        loss_history.append(avg_epoch_loss)
        
        # Print progress for the first and last epochs
        if epoch == 0 or epoch == n_epochs-1:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.6f}")
    
    elapsed_time = time.time() - start_time
    
    return weights, bias, elapsed_time, weights_history, bias_history, loss_history

# Train model using SGD
weights_sgd, bias_sgd, time_sgd, weights_history, bias_history, loss_history = sgd_train(
    X, y, learning_rate=0.01, n_epochs=5
)

print(f"\nSGD Solution (after 5 epochs):")
print(f"Weights: {weights_sgd}")
print(f"Bias: {bias_sgd}")
print(f"Computation time: {time_sgd:.6f} seconds")

# -------------------- Step 6: Demonstrating Online Learning with New Data --------------------
print("\n\nStep 6: Updating the model with a new data point (Online Learning)")
print("-" * 80)

# Generate a new data point
X_new, y_new = generate_new_data_point()

print(f"New data point:")
print(f"Features: {X_new[0]}")
print(f"Purchase amount: {y_new}")

print("\nUpdating model using SGD for a single new data point:")

def update_model_with_new_data_point(weights, bias, X_new, y_new, learning_rate=0.01):
    """Update the model with a single new data point using SGD."""
    # Make a copy of the current model parameters
    weights_before = weights.copy()
    bias_before = bias
    
    # Forward pass (prediction)
    y_pred = np.dot(X_new[0], weights) + bias
    
    # Compute error
    error = y_pred - y_new
    
    # Gradient of the loss function
    grad_weights = error * X_new[0]
    grad_bias = error
    
    # Update weights and bias
    weights_after = weights - learning_rate * grad_weights
    bias_after = bias - learning_rate * grad_bias
    
    return weights_after, bias_after, weights_before, bias_before, y_pred, error

# Update model with new data point
weights_updated, bias_updated, weights_before, bias_before, y_pred, error = update_model_with_new_data_point(
    weights_sgd, bias_sgd, X_new, y_new, learning_rate=0.01
)

print(f"Prediction before update: {y_pred:.4f}")
print(f"Actual value: {y_new:.4f}")
print(f"Prediction error: {error:.4f}")

print("\nModel parameters before update:")
print(f"Weights: {weights_before}")
print(f"Bias: {bias_before}")

print("\nModel parameters after update:")
print(f"Weights: {weights_updated}")
print(f"Bias: {bias_updated}")

print("\nParameter changes:")
print(f"Weight changes: {weights_updated - weights_before}")
print(f"Bias change: {bias_updated - bias_before}")

# -------------------- Step 7: Comparing computational requirements --------------------
print("\n\nStep 7: Comparing computational and memory requirements")
print("-" * 80)

# Let's compare normal equations vs SGD for different dataset sizes
dataset_sizes = [1000, 2000, 5000, 10000, 20000]
sgd_times = []
ne_times = []
sgd_memory = []
ne_memory = []

for size in dataset_sizes:
    # Generate larger dataset
    X_large, y_large, _, _ = generate_initial_data(n_samples=size)
    
    # Time for SGD (single epoch)
    _, _, time_sgd, _, _, _ = sgd_train(X_large, y_large, learning_rate=0.01, n_epochs=1)
    sgd_times.append(time_sgd)
    
    # Memory for SGD: only need to store the model parameters and current data point
    sgd_mem = X_large.shape[1] * 8 * 2 + 16  # weights + gradients + few scalars (in bytes)
    sgd_memory.append(sgd_mem / 1024)  # Convert to KB
    
    try:
        # Time for normal equations
        _, _, time_ne = normal_equation_solution(X_large, y_large)
        ne_times.append(time_ne)
        
        # Memory for normal equations: need to store X^T*X and intermediate matrices
        ne_mem = X_large.shape[0] * X_large.shape[1] * 8 + X_large.shape[1]**2 * 8 * 2
        ne_memory.append(ne_mem / 1024)  # Convert to KB
    except:
        # In case of memory error for large matrices
        ne_times.append(None)
        ne_memory.append(None)

# Plot time comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(dataset_sizes, sgd_times, 'o-', label='SGD (1 epoch)')
plt.plot(dataset_sizes, ne_times, 'o-', label='Normal Equations')
plt.xlabel('Dataset Size (samples)')
plt.ylabel('Computation Time (seconds)')
plt.title('Time Complexity Comparison')
plt.legend()
plt.grid(True)

# Plot memory comparison
plt.subplot(1, 2, 2)
plt.plot(dataset_sizes, sgd_memory, 'o-', label='SGD')
plt.plot(dataset_sizes, ne_memory, 'o-', label='Normal Equations')
plt.xlabel('Dataset Size (samples)')
plt.ylabel('Memory Usage (KB)')
plt.title('Memory Requirement Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "computational_comparison.png"), dpi=300)
plt.close()

print("Generated comparison of computational and memory requirements between SGD and Normal Equations.")
print("- SGD has a time complexity of O(n) per epoch, where n is the number of samples")
print("- Normal equations have a time complexity of O(n*d^2 + d^3), where d is the number of features")
print("- SGD has a constant memory requirement O(d)")
print("- Normal equations require memory of O(n*d + d^2)")
print("\nFor large datasets or streaming data, SGD is clearly more efficient, making it suitable for online learning.")

# -------------------- Step 8: Implementing adaptive learning rate methods --------------------
print("\n\nStep 8: Addressing issues with Simple SGD - Adaptive Learning Rates")
print("-" * 80)

def sgd_with_adaptive_rates(X, y, n_epochs=5):
    """Train with different SGD variants for comparison."""
    n_samples, n_features = X.shape
    
    # Initialize weights randomly (same for all methods)
    initial_weights = np.random.randn(n_features) * 0.01
    initial_bias = 0.0
    
    # Parameters for adaptive methods
    learning_rate = 0.01
    momentum = 0.9
    rho = 0.9  # For RMSprop
    epsilon = 1e-8  # Small constant for numerical stability
    
    # Dictionary to store results
    results = {
        "standard_sgd": {"loss_history": [], "final_weights": None, "final_bias": None},
        "momentum_sgd": {"loss_history": [], "final_weights": None, "final_bias": None},
        "rmsprop_sgd": {"loss_history": [], "final_weights": None, "final_bias": None},
    }
    
    # Method implementations
    def train_standard_sgd():
        weights = initial_weights.copy()
        bias = initial_bias
        loss_history = []
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for i in range(n_samples):
                idx = indices[i]
                y_pred = np.dot(X[idx], weights) + bias
                error = y_pred - y[idx]
                
                # Standard SGD update
                weights -= learning_rate * error * X[idx]
                bias -= learning_rate * error
                
                epoch_loss += 0.5 * error**2
            
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)
            
        return weights, bias, loss_history
    
    def train_momentum_sgd():
        weights = initial_weights.copy()
        bias = initial_bias
        loss_history = []
        
        # Initialize velocity
        velocity_weights = np.zeros_like(weights)
        velocity_bias = 0.0
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for i in range(n_samples):
                idx = indices[i]
                y_pred = np.dot(X[idx], weights) + bias
                error = y_pred - y[idx]
                
                # Compute gradients
                grad_weights = error * X[idx]
                grad_bias = error
                
                # Update velocity (momentum)
                velocity_weights = momentum * velocity_weights + learning_rate * grad_weights
                velocity_bias = momentum * velocity_bias + learning_rate * grad_bias
                
                # Update parameters using velocity
                weights -= velocity_weights
                bias -= velocity_bias
                
                epoch_loss += 0.5 * error**2
            
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)
            
        return weights, bias, loss_history
    
    def train_rmsprop():
        weights = initial_weights.copy()
        bias = initial_bias
        loss_history = []
        
        # Initialize accumulated squared gradients
        square_weights = np.zeros_like(weights)
        square_bias = 0.0
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for i in range(n_samples):
                idx = indices[i]
                y_pred = np.dot(X[idx], weights) + bias
                error = y_pred - y[idx]
                
                # Compute gradients
                grad_weights = error * X[idx]
                grad_bias = error
                
                # Update accumulated squared gradients
                square_weights = rho * square_weights + (1 - rho) * grad_weights**2
                square_bias = rho * square_bias + (1 - rho) * grad_bias**2
                
                # Compute adaptive learning rates
                adaptive_lr_weights = learning_rate / (np.sqrt(square_weights) + epsilon)
                adaptive_lr_bias = learning_rate / (np.sqrt(square_bias) + epsilon)
                
                # Update parameters
                weights -= adaptive_lr_weights * grad_weights
                bias -= adaptive_lr_bias * grad_bias
                
                epoch_loss += 0.5 * error**2
            
            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)
            
        return weights, bias, loss_history
    
    # Run all methods
    print("Training with standard SGD...")
    results["standard_sgd"]["final_weights"], results["standard_sgd"]["final_bias"], results["standard_sgd"]["loss_history"] = train_standard_sgd()
    
    print("Training with momentum SGD...")
    results["momentum_sgd"]["final_weights"], results["momentum_sgd"]["final_bias"], results["momentum_sgd"]["loss_history"] = train_momentum_sgd()
    
    print("Training with RMSprop (adaptive learning rates)...")
    results["rmsprop_sgd"]["final_weights"], results["rmsprop_sgd"]["final_bias"], results["rmsprop_sgd"]["loss_history"] = train_rmsprop()
    
    return results

# Run comparison of SGD variants
sgd_results = sgd_with_adaptive_rates(X, y, n_epochs=15)

# Plot comparison of loss for different SGD variants
plt.figure(figsize=(10, 6))
plt.plot(sgd_results["standard_sgd"]["loss_history"], 'o-', label='Standard SGD')
plt.plot(sgd_results["momentum_sgd"]["loss_history"], 's-', label='SGD with Momentum')
plt.plot(sgd_results["rmsprop_sgd"]["loss_history"], '^-', label='RMSprop (Adaptive LR)')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Comparison of SGD Variants')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sgd_variants_comparison.png"), dpi=300)
plt.close()

print("\nSGD Variants Comparison:")
print("1. Standard SGD: Simple update rule using fixed learning rate")
print("2. Momentum SGD: Accelerates training by accumulating previous gradients")
print("3. RMSprop: Adjusts learning rates adaptively based on recent gradient magnitudes")

print("\nPotential issues with simple SGD for online learning:")
print("1. Sensitivity to learning rate - too high causes divergence, too low causes slow convergence")
print("2. Difficulty handling features with different scales")
print("3. Susceptibility to getting stuck in local minima or saddle points")
print("4. Noisy updates that can lead to instability, especially with outliers")
print("5. Slow convergence when the loss surface has high curvature in some directions")

print("\nAdvantages of adaptive learning rate methods:")
print("1. Automatically adjust learning rates for each parameter")
print("2. Better handle different feature scales without manual tuning")
print("3. Progress faster on shallow dimensions and slower on steep dimensions")
print("4. More stable training process even with non-stationary data distributions")
print("5. Often converge faster and to better solutions")

# -------------------- Step 9: Visualization of model evolution during online learning --------------------
print("\n\nStep 9: Visualizing model evolution during online learning")
print("-" * 80)

# Create a sequence of new data points to simulate online learning
def simulate_online_learning(initial_weights, initial_bias, n_new_points=20):
    """Simulate online learning with a stream of new data points."""
    weights = initial_weights.copy()
    bias = initial_bias
    
    # Store history for visualization
    weight_history = [weights.copy()]
    bias_history = [bias]
    error_history = []
    
    # Learning rate
    learning_rate = 0.01
    
    # Generate stream of new data points
    for i in range(n_new_points):
        # Generate new data point
        X_new, y_new = generate_new_data_point()
        
        # Make prediction with current model
        y_pred = np.dot(X_new[0], weights) + bias
        error = y_pred - y_new
        error_history.append(error)
        
        # Update model with SGD
        weights -= learning_rate * error * X_new[0]
        bias -= learning_rate * error
        
        # Store history
        weight_history.append(weights.copy())
        bias_history.append(bias)
    
    return weight_history, bias_history, error_history

# Run online learning simulation
weight_evolution, bias_evolution, error_evolution = simulate_online_learning(
    sgd_results["rmsprop_sgd"]["final_weights"], 
    sgd_results["rmsprop_sgd"]["final_bias"],
    n_new_points=50
)

# Create visualization of model evolution
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2)

# Plot weight evolution
ax1 = plt.subplot(gs[0, :])
for i in range(len(weight_evolution[0])):
    weights_feature_i = [w[i] for w in weight_evolution]
    ax1.plot(weights_feature_i, label=f'Weight {i+1}')
ax1.set_xlabel('Update Step')
ax1.set_ylabel('Weight Value')
ax1.set_title('Evolution of Weights during Online Learning')
ax1.legend()
ax1.grid(True)

# Plot bias evolution
ax2 = plt.subplot(gs[1, 0])
ax2.plot(bias_evolution)
ax2.set_xlabel('Update Step')
ax2.set_ylabel('Bias Value')
ax2.set_title('Evolution of Bias during Online Learning')
ax2.grid(True)

# Plot error evolution
ax3 = plt.subplot(gs[1, 1])
ax3.plot(error_evolution)
ax3.set_xlabel('Update Step')
ax3.set_ylabel('Prediction Error')
ax3.set_title('Prediction Error during Online Learning')
ax3.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "online_learning_evolution.png"), dpi=300)
plt.close()

print("Generated visualization of model evolution during online learning.")
print("The plots show how the model parameters (weights and bias) change with each new data point")
print("and how the prediction error evolves over time.")

# -------------------- Step 10: Summary of Online Learning with SGD --------------------
print("\n\nStep 10: Summary of Findings")
print("-" * 80)
print("""Summary of Online Learning with SGD for Recommendation Systems:

1. Online Learning with SGD is well-suited for recommendation systems because:
   - It can process continuous streams of user interaction data
   - It adapts quickly to changing user preferences
   - It maintains a constant memory footprint regardless of data volume
   - It provides immediate model updates without complete retraining

2. Mathematical update rule for SGD in linear regression:
   - w ← w - η (w^T x - y)x
   - For each new data point, the model parameters are adjusted in the opposite direction
     of the gradient, proportional to the prediction error

3. Computational and memory advantages over batch methods:
   - Time complexity: O(d) per update vs O(n*d^2 + d^3) for normal equations
   - Memory complexity: O(d) vs O(n*d + d^2) for normal equations
   - Scales linearly with feature count, not with data volume
   - Enables processing of virtually unlimited data streams

4. Challenges and improvements:
   - Basic SGD can be sensitive to learning rate, feature scaling, and noisy data
   - Adaptive methods like momentum and RMSprop improve convergence and stability
   - Adaptive learning rates automatically adjust to different feature scales and
     data distributions, making them ideal for non-stationary recommendation environments

5. The visualizations demonstrate:
   - How model parameters evolve during online learning
   - Convergence behavior of different SGD variants
   - Computational efficiency of online learning vs batch methods""")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- computational_comparison.png: Comparison of computational and memory requirements")
print("- sgd_variants_comparison.png: Comparison of different SGD variants")
print("- online_learning_evolution.png: Evolution of model parameters during online learning") 