import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib.patches import FancyArrowPatch

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = False  # We'll use mathtext instead of full LaTeX
plt.rcParams['mathtext.default'] = 'regular'

def print_heading(text, char="="):
    """Print a heading with the specified character as separator."""
    print(f"\n{char * 80}")
    print(f"{text}")
    print(f"{char * 80}\n")

# Step 1: Generate synthetic data for linear regression
def generate_data(n_samples=20, n_features=2, noise=0.5):
    """Generate synthetic data for linear regression."""
    print_heading("Step 1: Generate Synthetic Data", "=")
    
    # Generate feature values
    X = np.random.randn(n_samples, n_features)
    
    # Add bias term (intercept) as first column
    X = np.hstack((np.ones((n_samples, 1)), X))
    
    # True weights (including bias)
    true_w = np.array([2.0, 1.5, -0.5])
    
    # Generate target values with some noise
    y = X @ true_w + noise * np.random.randn(n_samples)
    
    print(f"Generated {n_samples} samples with {n_features} features:")
    print(f"- X shape: {X.shape}")
    print(f"- y shape: {y.shape}")
    print(f"- True weights: {true_w}")
    
    # Show some sample data
    sample_df = pd.DataFrame({
        'Bias (x_0)': X[:5, 0],
        'x_1': X[:5, 1],
        'x_2': X[:5, 2],
        'y': y[:5]
    })
    print("\nSample data (first 5 rows):")
    print(sample_df)
    
    return X, y, true_w

# Step 2: Implement LMS algorithm
def lms_algorithm(X, y, learning_rate=0.1, max_epochs=1):
    """
    Implement the Least Mean Squares (LMS) algorithm for online learning.
    Returns weight history and predictions for visualization.
    """
    print_heading("Step 2: Implement LMS Algorithm", "=")
    
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    w = np.zeros(n_features)
    
    # Store weight history for visualization
    weight_history = [w.copy()]
    
    # Store predictions for each example
    all_predictions = []
    
    # Store loss history
    loss_history = []
    
    # Initialize cumulative squared error
    cumulative_squared_error = 0
    
    print(f"Initial weights: {w}")
    print(f"Learning rate (α): {learning_rate}")
    print("\nStarting online learning with LMS algorithm...")
    print("\n{:<5} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Step", "Example", "Prediction", "Target", "Error", "Update", "New Weights"))
    
    # For each epoch
    for epoch in range(max_epochs):
        # For each training example
        for i in range(n_samples):
            # Get current example
            x_i = X[i]
            y_i = y[i]
            
            # Make prediction
            prediction = np.dot(w, x_i)
            
            # Calculate error
            error = y_i - prediction
            
            # Update cumulative squared error
            cumulative_squared_error += error**2
            
            # Store current weights
            current_w = w.copy()
            
            # Calculate weight update (LMS rule)
            update = learning_rate * error * x_i
            
            # Update weights
            w = w + update
            
            # Store weights
            weight_history.append(w.copy())
            
            # Store prediction
            all_predictions.append(prediction)
            
            # Calculate current MSE
            current_mse = cumulative_squared_error / (i + 1 + epoch * n_samples)
            loss_history.append(current_mse)
            
            # Print step details
            print("{:<5} {:<15} {:<15.3f} {:<15.3f} {:<15.3f} {:<15} {:<15}".format(
                i + 1 + epoch * n_samples, 
                f"({', '.join([f'{v:.2f}' for v in x_i])})", 
                prediction, 
                y_i, 
                error,
                f"({', '.join([f'{v:.3f}' for v in update])})",
                f"({', '.join([f'{v:.3f}' for v in w])})"
            ))
    
    # Calculate final predictions for all examples
    final_predictions = np.dot(X, w)
    
    # Calculate final MSE
    final_mse = np.mean((y - final_predictions) ** 2)
    
    print(f"\nFinal weights after {n_samples} examples: {w}")
    print(f"Final Mean Squared Error: {final_mse:.4f}")
    
    return w, weight_history, all_predictions, loss_history

# Step 3: Implement batch gradient descent for comparison
def batch_gradient_descent(X, y, learning_rate=0.01, max_epochs=20):
    """
    Implement batch gradient descent for comparison with online learning.
    """
    print_heading("Step 3: Batch Gradient Descent for Comparison", "=")
    
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    w = np.zeros(n_features)
    
    # Store weight history
    weight_history = [w.copy()]
    
    # Store loss history
    loss_history = []
    
    print(f"Initial weights: {w}")
    print(f"Learning rate (α): {learning_rate}")
    print(f"Maximum epochs: {max_epochs}")
    
    print("\n{:<5} {:<15} {:<20} {:<20}".format(
        "Epoch", "MSE", "Weight Update", "New Weights"))
    
    for epoch in range(max_epochs):
        # Make predictions
        predictions = np.dot(X, w)
        
        # Calculate errors
        errors = y - predictions
        
        # Calculate mean squared error
        mse = np.mean(errors ** 2)
        loss_history.append(mse)
        
        # Calculate gradient (average over all examples)
        gradient = -2/n_samples * np.dot(X.T, errors)
        
        # Calculate update
        update = -learning_rate * gradient
        
        # Update weights
        w = w + update
        
        # Store weights
        weight_history.append(w.copy())
        
        # Print epoch details
        print("{:<5} {:<15.4f} {:<20} {:<20}".format(
            epoch + 1, 
            mse,
            f"({', '.join([f'{v:.3f}' for v in update])})",
            f"({', '.join([f'{v:.3f}' for v in w])})"
        ))
    
    # Calculate final predictions
    final_predictions = np.dot(X, w)
    
    # Calculate final MSE
    final_mse = np.mean((y - final_predictions) ** 2)
    
    print(f"\nFinal weights after {max_epochs} epochs: {w}")
    print(f"Final Mean Squared Error: {final_mse:.4f}")
    
    return w, weight_history, loss_history

# Step 4: Mini-batch gradient descent for comparison
def mini_batch_gradient_descent(X, y, batch_size=5, learning_rate=0.05, max_epochs=10):
    """
    Implement mini-batch gradient descent for comparison.
    """
    print_heading("Step 4: Mini-Batch Gradient Descent for Comparison", "=")
    
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    w = np.zeros(n_features)
    
    # Store weight history
    weight_history = [w.copy()]
    
    # Store loss history
    loss_history = []
    all_mse = []
    
    print(f"Initial weights: {w}")
    print(f"Learning rate (α): {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Maximum epochs: {max_epochs}")
    
    print("\n{:<5} {:<10} {:<15} {:<20} {:<20}".format(
        "Epoch", "Batch", "MSE", "Weight Update", "New Weights"))
    
    for epoch in range(max_epochs):
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_mse = []
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            # Get current batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Make predictions
            predictions = np.dot(X_batch, w)
            
            # Calculate errors
            errors = y_batch - predictions
            
            # Calculate mean squared error
            mse = np.mean(errors ** 2)
            epoch_mse.append(mse)
            all_mse.append(mse)
            
            # Calculate gradient (average over batch)
            gradient = -2/len(X_batch) * np.dot(X_batch.T, errors)
            
            # Calculate update
            update = -learning_rate * gradient
            
            # Update weights
            w = w + update
            
            # Store weights
            weight_history.append(w.copy())
            
            # Print batch details
            print("{:<5} {:<10} {:<15.4f} {:<20} {:<20}".format(
                epoch + 1, 
                f"{i//batch_size + 1}/{(n_samples+batch_size-1)//batch_size}",
                mse,
                f"({', '.join([f'{v:.3f}' for v in update])})",
                f"({', '.join([f'{v:.3f}' for v in w])})"
            ))
        
        # Calculate average MSE for the epoch
        avg_mse = np.mean(epoch_mse)
        loss_history.append(avg_mse)
        
        print(f"Epoch {epoch+1} average MSE: {avg_mse:.4f}")
    
    # Calculate final predictions
    final_predictions = np.dot(X, w)
    
    # Calculate final MSE
    final_mse = np.mean((y - final_predictions) ** 2)
    
    print(f"\nFinal weights after {max_epochs} epochs: {w}")
    print(f"Final Mean Squared Error: {final_mse:.4f}")
    
    return w, weight_history, loss_history, all_mse

# Step 6: Learning rate sensitivity analysis for LMS
def learning_rate_sensitivity_analysis(X, y, true_w):
    """
    Analyze the sensitivity of LMS algorithm to different learning rates.
    """
    print_heading("Step 6: Learning Rate Sensitivity Analysis", "=")
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    max_epochs = 3  # Run multiple epochs to see convergence
    
    results = []
    
    print("Running LMS algorithm with different learning rates...")
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        
        # Initialize weights to zeros
        w = np.zeros(X.shape[1])
        weight_history = [w.copy()]
        mse_history = []
        
        # Run LMS for multiple epochs
        for epoch in range(max_epochs):
            cumulative_squared_error = 0
            
            for i in range(len(X)):
                # Get current example
                x_i = X[i]
                y_i = y[i]
                
                # Make prediction
                prediction = np.dot(w, x_i)
                
                # Calculate error
                error = y_i - prediction
                
                # Update cumulative squared error
                cumulative_squared_error += error**2
                
                # Update weights
                update = lr * error * x_i
                w = w + update
                
                # Store weights
                weight_history.append(w.copy())
                
                # Calculate current MSE
                current_mse = cumulative_squared_error / (i + 1 + epoch * len(X))
                mse_history.append(current_mse)
        
        # Calculate distance from true weights at each step
        weight_distances = [np.linalg.norm(w_hist - true_w) for w_hist in weight_history]
        
        # Calculate final predictions and MSE
        final_predictions = np.dot(X, w)
        final_mse = np.mean((y - final_predictions) ** 2)
        
        results.append({
            'learning_rate': lr,
            'final_weights': w,
            'final_mse': final_mse,
            'weight_distances': weight_distances,
            'mse_history': mse_history
        })
        
        print(f"Final MSE: {final_mse:.4f}, Final weights: {w}")
    
    # Create visualization for learning rate sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Weight distance from true weights over time
    for result in results:
        lr = result['learning_rate']
        distances = result['weight_distances']
        iterations = range(len(distances))
        ax1.plot(iterations, distances, label=f"α = {lr}")
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance from True Weights')
    ax1.set_title('Convergence Speed with Different Learning Rates')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Final MSE for each learning rate
    learning_rates_vals = [result['learning_rate'] for result in results]
    final_mses = [result['final_mse'] for result in results]
    
    ax2.plot(learning_rates_vals, final_mses, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Learning Rate (α)')
    ax2.set_ylabel('Final MSE')
    ax2.set_title('Final MSE vs. Learning Rate')
    ax2.set_xscale('log')
    ax2.grid(True)
    
    # Add annotations for optimal learning rate
    best_idx = np.argmin(final_mses)
    best_lr = learning_rates_vals[best_idx]
    best_mse = final_mses[best_idx]
    
    ax2.annotate(f'Best: α = {best_lr}\nMSE = {best_mse:.4f}',
                xy=(best_lr, best_mse),
                xytext=(best_lr * 1.5, best_mse * 1.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate_sensitivity.png'), dpi=300)
    plt.close()
    
    # Return information about the best learning rate
    best_result = results[best_idx]
    
    print(f"\nBest learning rate found: {best_lr}")
    print(f"Best final MSE: {best_mse:.4f}")
    print(f"Best final weights: {best_result['final_weights']}")
    
    return best_result, results

# Step 5: Visualize the results
def visualize_results(X, y, true_w, lms_w, lms_weight_history, lms_loss_history,
                     batch_w, batch_weight_history, batch_loss_history,
                     mini_batch_w, mini_batch_weight_history, mini_batch_loss_history, mini_batch_all_mse):
    """Visualize the results of the different algorithms."""
    print_heading("Step 5: Visualizing Results", "=")
    
    # 1. Visualize the evolution of weights for all methods
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot LMS weights
    for i in range(3):
        feature_name = ["bias", "$x_1$", "$x_2$"][i]
        weights = [w[i] for w in lms_weight_history]
        ax.plot(weights, label=f"LMS - $w_{i}$ ({feature_name})", linestyle='-')
    
    # Mark true weights
    for i in range(3):
        feature_name = ["bias", "$x_1$", "$x_2$"][i]
        ax.axhline(y=true_w[i], color=f"C{i}", linestyle='--', 
                   label=f"True $w_{i}$ ({feature_name}): {true_w[i]:.2f}")
    
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Weight Value')
    ax.set_title('Evolution of Weights in LMS Algorithm')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lms_weight_evolution.png'), dpi=300)
    plt.close()
    
    # 2. Compare weight trajectories (Batch vs. LMS) - in 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract weights
    lms_w0 = [w[0] for w in lms_weight_history]
    lms_w1 = [w[1] for w in lms_weight_history]
    lms_w2 = [w[2] for w in lms_weight_history]
    
    batch_w0 = [w[0] for w in batch_weight_history]
    batch_w1 = [w[1] for w in batch_weight_history]
    batch_w2 = [w[2] for w in batch_weight_history]
    
    mini_batch_w0 = [w[0] for w in mini_batch_weight_history]
    mini_batch_w1 = [w[1] for w in mini_batch_weight_history]
    mini_batch_w2 = [w[2] for w in mini_batch_weight_history]
    
    # Plot trajectories
    ax.plot(lms_w0, lms_w1, lms_w2, 'r-', linewidth=2, label='LMS (Online)')
    ax.plot(batch_w0, batch_w1, batch_w2, 'b-', linewidth=2, label='Batch GD')
    ax.plot(mini_batch_w0[:50], mini_batch_w1[:50], mini_batch_w2[:50], 'g-', linewidth=2, label='Mini-Batch GD')
    
    # Plot starting points
    ax.scatter([lms_weight_history[0][0]], [lms_weight_history[0][1]], [lms_weight_history[0][2]], 
               c='red', s=100, marker='o', label='Starting Point')
    
    # Plot true weights
    ax.scatter([true_w[0]], [true_w[1]], [true_w[2]], c='black', s=200, marker='*', label='True Weights')
    
    # Plot final weights
    ax.scatter([lms_w[-1]], [lms_w1[-1]], [lms_w2[-1]], c='red', s=100, marker='x', label='LMS Final')
    ax.scatter([batch_w0[-1]], [batch_w1[-1]], [batch_w2[-1]], c='blue', s=100, marker='x', label='Batch Final')
    ax.scatter([mini_batch_w0[-1]], [mini_batch_w1[-1]], [mini_batch_w2[-1]], c='green', s=100, marker='x', label='Mini-Batch Final')
    
    # Add labels and title
    ax.set_xlabel('$w_0$ (Bias)')
    ax.set_ylabel('$w_1$')
    ax.set_zlabel('$w_2$')
    ax.set_title('Weight Trajectories in 3D Space')
    
    # Add legend
    ax.legend()
    
    plt.savefig(os.path.join(save_dir, 'weight_trajectories_3d.png'), dpi=300)
    plt.close()
    
    # 3. Compare learning curves (MSE over time)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For LMS, we need to create the x-axis representing the number of examples seen
    lms_examples = np.arange(1, len(lms_loss_history) + 1)
    ax.plot(lms_examples, lms_loss_history, 'r-', label='LMS (Online) Learning')
    
    # For batch, the x-axis represents epochs
    batch_epochs = np.arange(1, len(batch_loss_history) + 1)
    ax.plot(batch_epochs * len(y), batch_loss_history, 'b-', label='Batch GD')
    
    # For mini-batch, plot full MSE history
    ax.plot(np.arange(1, len(mini_batch_all_mse) + 1), mini_batch_all_mse, 'g-', label='Mini-Batch GD')
    
    ax.set_xlabel('Number of Examples Processed')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Learning Curves: MSE vs. Examples Processed')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300)
    plt.close()
    
    # 4. Visualize predictions vs. actual values (LMS)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Make predictions with all models
    lms_predictions = X @ lms_w
    batch_predictions = X @ batch_w
    mini_batch_predictions = X @ mini_batch_w
    
    # Create scatter plot for actual values
    ax.scatter(np.arange(len(y)), y, label='Actual Values', color='black', s=50)
    
    # Create scatter plot for predicted values
    ax.scatter(np.arange(len(y)), lms_predictions, label='LMS Predictions', color='red', marker='x', s=50)
    ax.scatter(np.arange(len(y)), batch_predictions, label='Batch GD Predictions', color='blue', marker='+', s=50)
    ax.scatter(np.arange(len(y)), mini_batch_predictions, label='Mini-Batch GD Predictions', color='green', marker='*', s=50)
    
    # Connect actual and predicted values
    for i in range(len(y)):
        ax.plot([i, i], [y[i], lms_predictions[i]], 'r--', alpha=0.3)
        ax.plot([i, i], [y[i], batch_predictions[i]], 'b--', alpha=0.3)
        ax.plot([i, i], [y[i], mini_batch_predictions[i]], 'g--', alpha=0.3)
    
    ax.set_xlabel('Example Index')
    ax.set_ylabel('Value')
    ax.set_title('Actual vs. Predicted Values')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 5. Create a conceptual visualization of LMS update rule
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw coordinate system
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Example weights and update
    w_old = np.array([1.0, 1.0])
    error = 2.0
    x_example = np.array([0.5, 1.0])
    learning_rate = 0.1
    update = learning_rate * error * x_example
    w_new = w_old + update
    
    # Plot old weights
    ax.scatter(w_old[0], w_old[1], color='blue', s=100, label='$w(t)$')
    
    # Plot new weights
    ax.scatter(w_new[0], w_new[1], color='red', s=100, label='$w(t+1)$')
    
    # Draw update arrow
    arrow = FancyArrowPatch((w_old[0], w_old[1]), (w_new[0], w_new[1]), 
                            arrowstyle='->',
                            mutation_scale=15,
                            linewidth=2,
                            color='green')
    ax.add_patch(arrow)
    
    # Add text for update formula
    formula = r"$\mathbf{w}(t+1) = \mathbf{w}(t) + \alpha \cdot (y - \hat{y}) \cdot \mathbf{x}$"
    ax.text(0.5, 1.7, formula, fontsize=18, ha='center')
    
    # Add text for current values
    ax.text(0.5, 1.5, f"$\\alpha = {learning_rate}$, error = ${error}$, $\\mathbf{{x}} = [{x_example[0]}, {x_example[1]}]$", fontsize=14, ha='center')
    ax.text(0.5, 1.3, f"update = ${learning_rate} \\cdot {error} \\cdot [{x_example[0]}, {x_example[1]}] = [{update[0]}, {update[1]}]$", fontsize=14, ha='center')
    
    # Set plot limits and labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_title('LMS Update Rule Visualization')
    
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lms_update_rule.png'), dpi=300)
    plt.close()
    
    # 6. Visualize a comparison of learning algorithms
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a table-like visualization
    algorithms = ['LMS (Online)', 'Batch Gradient Descent', 'Mini-Batch Gradient Descent']
    features = [
        'Update Frequency', 
        'Memory Requirements', 
        'Computation per Update',
        'Convergence Speed',
        'Final Accuracy',
        'Sensitivity to Learning Rate',
        'Ability to Handle New Data'
    ]
    
    # Scores for each algorithm (out of 5)
    scores = np.array([
        [5, 1, 3],  # Update Frequency
        [5, 1, 3],  # Memory Requirements
        [5, 1, 3],  # Computation per Update
        [3, 1, 2],  # Convergence Speed
        [3, 5, 4],  # Final Accuracy
        [1, 5, 3],  # Sensitivity to Learning Rate
        [5, 1, 3]   # Ability to Handle New Data
    ])
    
    # Normalize scores to [0, 1]
    normalized_scores = scores / 5.0
    
    # Create a heatmap
    im = ax.imshow(normalized_scores, cmap='viridis')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(algorithms)
    ax.set_yticklabels(features)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Score (Normalized)', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(algorithms)):
            text = ax.text(j, i, scores[i, j],
                           ha="center", va="center", color="white" if normalized_scores[i, j] < 0.7 else "black")
    
    ax.set_title("Comparison of Learning Algorithms")
    fig.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300)
    plt.close()
    
    print(f"All visualizations saved to: {save_dir}")

# Main function
def main():
    # Generate data
    X, y, true_w = generate_data(n_samples=20, n_features=2, noise=0.5)
    
    # Run LMS algorithm
    lms_w, lms_weight_history, lms_predictions, lms_loss_history = lms_algorithm(X, y, learning_rate=0.1)
    
    # Run batch gradient descent
    batch_w, batch_weight_history, batch_loss_history = batch_gradient_descent(X, y, learning_rate=0.01, max_epochs=20)
    
    # Run mini-batch gradient descent
    mini_batch_w, mini_batch_weight_history, mini_batch_loss_history, mini_batch_all_mse = mini_batch_gradient_descent(
        X, y, batch_size=5, learning_rate=0.05, max_epochs=10)
    
    # Visualize results
    visualize_results(X, y, true_w, lms_w, lms_weight_history, lms_loss_history,
                    batch_w, batch_weight_history, batch_loss_history,
                    mini_batch_w, mini_batch_weight_history, mini_batch_loss_history, mini_batch_all_mse)
    
    # Run learning rate sensitivity analysis
    best_result, all_results = learning_rate_sensitivity_analysis(X, y, true_w)
    
    # Print theoretical insights
    print_heading("Theoretical Insights", "=")
    print("1. LMS Update Rule: w(t+1) = w(t) + α * (y - ŷ) * x")
    print("2. LMS is equivalent to stochastic gradient descent for mean squared error loss")
    print("3. LMS advantages for online learning:")
    print("   - Immediate updates with each new example")
    print("   - Low memory requirements (no need to store full dataset)")
    print("   - Can adapt to changing data distributions")
    print("4. Trade-offs between updating immediately vs. batch updates:")
    print("   - Online (LMS): More noise, faster adaptation, lower memory")
    print("   - Batch: More stable, potentially better final solution, higher memory")
    print("5. Learning rate sensitivity:")
    print(f"   - Best learning rate from analysis: {best_result['learning_rate']}")
    print(f"   - Higher learning rates tend to make updates more noisy but potentially faster")
    print(f"   - Lower learning rates provide smoother convergence but require more iterations")
    
    print("\nImages generated:")
    for img in os.listdir(save_dir):
        print(f"- {img}")

if __name__ == "__main__":
    main() 