import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Enable LaTeX rendering for better text display
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Given data from the problem
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 1, 3, 2])
y = np.array([5, 4, 9, 8])

# Problem parameters
initial_weights = np.array([0, 0, 0])  # w0, w1, w2
learning_rate = 0.1

# Print the raw data in tabular format
print("Raw data from the problem:")
data_table = pd.DataFrame({
    '$x_1$': x1,
    '$x_2$': x2,
    'y': y
})
print(data_table)
print()

# Step 1: Explain the SGD update rule
def explain_sgd_update_rule():
    """Explain the Stochastic Gradient Descent update rule."""
    print("Step 1: Stochastic Gradient Descent Update Rule")
    print("----------------------------------------------")
    print("For linear regression with the squared error loss function:")
    print("  J(w) = (1/2) * (h(x; w) - y)²")
    print("  where h(x; w) = w₀ + w₁x₁ + w₂x₂")
    print()
    print("The SGD update rule for each parameter wⱼ is:")
    print("  wⱼ = wⱼ - α * ∂J/∂wⱼ")
    print("  where:")
    print("    - α is the learning rate (α = 0.1 in this problem)")
    print("    - ∂J/∂wⱼ is the partial derivative of the loss with respect to wⱼ")
    print()
    print("For a single training example (x, y), the gradient is:")
    print("  ∂J/∂w₀ = (h(x; w) - y) * 1")
    print("  ∂J/∂w₁ = (h(x; w) - y) * x₁")
    print("  ∂J/∂w₂ = (h(x; w) - y) * x₂")
    print()
    print("So the full update rules are:")
    print("  w₀ = w₀ - α * (h(x; w) - y)")
    print("  w₁ = w₁ - α * (h(x; w) - y) * x₁")
    print("  w₂ = w₂ - α * (h(x; w) - y) * x₂")
    print()
    
    # Create a visualization of the SGD concept
    plt.figure(figsize=(10, 6))
    
    # Create example gradient path (just for visualization)
    w0_values = np.linspace(-1, 2, 20)
    w1_values = np.linspace(-1, 2, 20)
    W0, W1 = np.meshgrid(w0_values, w1_values)
    
    # Create a simplified cost function surface (for illustration)
    Z = 3 * (W0 - 1)**2 + (W1 - 1)**2 + 0.5
    
    # Example SGD path (random walk with general direction)
    np.random.seed(42)
    sgd_w0 = [-0.8]
    sgd_w1 = [-0.6]
    for i in range(8):
        sgd_w0.append(sgd_w0[-1] + 0.3 + 0.1 * np.random.randn())
        sgd_w1.append(sgd_w1[-1] + 0.2 + 0.1 * np.random.randn())
    
    # Plot the SGD concept
    plt.contourf(W0, W1, Z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Cost $J(w)$')
    plt.plot(sgd_w0, sgd_w1, 'r-o', linewidth=2, markersize=8, label='SGD Path')
    plt.scatter(1, 1, s=100, c='white', edgecolor='black', marker='*', label='Optimum')
    
    plt.title('Stochastic Gradient Descent Concept Visualization')
    plt.xlabel('$w_0$')
    plt.ylabel('$w_1$')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sgd_concept.png"), dpi=300)
    plt.close()

explain_sgd_update_rule()

# Step 2: Calculate the prediction, error, and gradient for the first example
def calculate_first_example_gradient():
    """Calculate the gradient for the first training example."""
    print("Step 2: Calculating the Gradient for the First Training Example")
    print("----------------------------------------------------------")
    
    # Get the first example
    first_x1 = x1[0]
    first_x2 = x2[0]
    first_y = y[0]
    
    # Initial weights
    w0, w1, w2 = initial_weights
    
    print(f"First training example: x₁ = {first_x1}, x₂ = {first_x2}, y = {first_y}")
    print(f"Initial weights: w₀ = {w0}, w₁ = {w1}, w₂ = {w2}")
    print()
    
    # Calculate the prediction using the model
    h_x = w0 + w1 * first_x1 + w2 * first_x2
    print(f"Prediction h(x; w) = w₀ + w₁x₁ + w₂x₂ = {w0} + {w1}*{first_x1} + {w2}*{first_x2} = {h_x}")
    
    # Calculate the error/loss
    error = h_x - first_y
    squared_error = 0.5 * (error ** 2)
    print(f"Error: h(x; w) - y = {h_x} - {first_y} = {error}")
    print(f"Squared error loss: (1/2) * (h(x; w) - y)² = 0.5 * ({error})² = {squared_error}")
    print()
    
    # Calculate the gradients
    grad_w0 = error  # The derivative of the squared error with respect to w0
    grad_w1 = error * first_x1  # The derivative with respect to w1
    grad_w2 = error * first_x2  # The derivative with respect to w2
    
    print("Calculating the gradients:")
    print(f"∂J/∂w₀ = (h(x; w) - y) = {error}")
    print(f"∂J/∂w₁ = (h(x; w) - y) * x₁ = {error} * {first_x1} = {grad_w1}")
    print(f"∂J/∂w₂ = (h(x; w) - y) * x₂ = {error} * {first_x2} = {grad_w2}")
    print()
    
    return error, [grad_w0, grad_w1, grad_w2], first_x1, first_x2, first_y

error, gradients, first_x1, first_x2, first_y = calculate_first_example_gradient()

# Step 3: Update the weights using the calculated gradient
def update_weights():
    """Perform one parameter update using the gradient from the first example."""
    print("Step 3: Updating the Parameters using SGD")
    print("-------------------------------------")
    
    # Unpack the gradients
    grad_w0, grad_w1, grad_w2 = gradients
    
    # Initial weights
    w0, w1, w2 = initial_weights
    
    # Update the weights using the SGD rule
    new_w0 = w0 - learning_rate * grad_w0
    new_w1 = w1 - learning_rate * grad_w1
    new_w2 = w2 - learning_rate * grad_w2
    
    print("Using the SGD update rule: wⱼ = wⱼ - α * ∂J/∂wⱼ")
    print(f"w₀ = {w0} - {learning_rate} * {grad_w0} = {new_w0}")
    print(f"w₁ = {w1} - {learning_rate} * {grad_w1} = {new_w1}")
    print(f"w₂ = {w2} - {learning_rate} * {grad_w2} = {new_w2}")
    print()
    
    print("Updated weights after one SGD step:")
    print(f"w₀ = {new_w0}, w₁ = {new_w1}, w₂ = {new_w2}")
    print()
    
    # Calculate the new prediction
    new_h_x = new_w0 + new_w1 * first_x1 + new_w2 * first_x2
    print(f"New prediction with updated weights: h(x; w) = {new_w0} + {new_w1}*{first_x1} + {new_w2}*{first_x2} = {new_h_x}")
    
    # Calculate the new error/loss
    new_error = new_h_x - first_y
    new_squared_error = 0.5 * (new_error ** 2)
    print(f"New error: h(x; w) - y = {new_h_x} - {first_y} = {new_error}")
    print(f"New squared error loss: (1/2) * (h(x; w) - y)² = 0.5 * ({new_error})² = {new_squared_error}")
    print()
    
    # Calculate initial squared error for comparison
    h_x = w0 + w1 * first_x1 + w2 * first_x2
    error = h_x - first_y
    squared_error = 0.5 * (error ** 2)
    
    # Show improvement
    improvement = squared_error - new_squared_error
    print(f"Improvement in squared error: {squared_error} - {new_squared_error} = {improvement}")
    print()
    
    return [new_w0, new_w1, new_w2], new_h_x, new_error, new_squared_error

new_weights, new_prediction, new_error, new_squared_error = update_weights()

# Step 4: Visualize the first update step
def visualize_update():
    """Create visualizations to show the effect of the first SGD update."""
    print("Step 4: Visualizing the SGD Update")
    print("--------------------------------")
    
    # Calculate predictions before and after update
    w0, w1, w2 = initial_weights
    prediction_before = w0 + w1 * x1 + w2 * x2
    
    new_w0, new_w1, new_w2 = new_weights
    prediction_after = new_w0 + new_w1 * x1 + new_w2 * x2
    
    # Prepare data for visualization
    errors_before = prediction_before - y
    squared_errors_before = 0.5 * (errors_before ** 2)
    errors_after = prediction_after - y
    squared_errors_after = 0.5 * (errors_after ** 2)
    
    # Create figure for visualizing the first example update
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Visualize model before and after update (2D)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x1, y, color='blue', s=100, label='Actual data')
    ax1.scatter(first_x1, first_y, color='red', s=150, edgecolor='black', label='First example')
    
    # For simplicity, we'll create a 1D plot showing x1 vs y, holding x2 fixed at its mean
    x1_range = np.linspace(min(x1) - 0.5, max(x1) + 0.5, 100)
    x2_mean = np.mean(x2)
    
    # Lines for before and after update
    y_before = w0 + w1 * x1_range + w2 * x2_mean
    y_after = new_w0 + new_w1 * x1_range + new_w2 * x2_mean
    
    ax1.plot(x1_range, y_before, 'r--', label='Before update (initial model)')
    ax1.plot(x1_range, y_after, 'g-', label='After update (updated model)')
    
    # Show the prediction improvement for the first example
    first_pred_before = w0 + w1 * first_x1 + w2 * first_x2
    first_pred_after = new_w0 + new_w1 * first_x1 + new_w2 * first_x2
    
    ax1.plot([first_x1, first_x1], [first_y, first_pred_before], 'r-', linewidth=2)
    ax1.plot([first_x1, first_x1], [first_y, first_pred_after], 'g-', linewidth=2)
    ax1.scatter(first_x1, first_pred_before, color='red', s=100, marker='x', label='Prediction before')
    ax1.scatter(first_x1, first_pred_after, color='green', s=100, marker='x', label='Prediction after')
    
    ax1.set_xlabel('Feature $x_1$')
    ax1.set_ylabel('Target $y$')
    ax1.set_title('Effect of First SGD Update ($x_2$ fixed at mean)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Visualize the error before and after update
    ax2 = fig.add_subplot(gs[0, 1])
    
    bar_positions = np.arange(4)
    width = 0.35
    
    bars1 = ax2.bar(bar_positions - width/2, squared_errors_before, width, label='Before update', color='red', alpha=0.7)
    bars2 = ax2.bar(bar_positions + width/2, squared_errors_after, width, label='After update', color='green', alpha=0.7)
    
    ax2.set_ylabel('Squared Error')
    ax2.set_title('Squared Error Comparison')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels([f'Example {i+1}' for i in range(4)])
    ax2.legend()
    
    # Add text showing error values for the first example
    ax2.text(0 - width/2, squared_errors_before[0] * 1.1, f'{squared_errors_before[0]:.3f}', 
             ha='center', va='bottom', color='black')
    ax2.text(0 + width/2, squared_errors_after[0] * 1.1, f'{squared_errors_after[0]:.3f}', 
             ha='center', va='bottom', color='black')
    
    # Plot 3: 3D visualization of the model change
    ax3 = fig.add_subplot(gs[1, :], projection='3d')
    
    # Create meshgrid for 3D surface
    x1_range = np.linspace(min(x1) - 0.5, max(x1) + 0.5, 20)
    x2_range = np.linspace(min(x2) - 0.5, max(x2) + 0.5, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Calculate z values for surfaces
    Z_before = w0 + w1 * X1 + w2 * X2
    Z_after = new_w0 + new_w1 * X1 + new_w2 * X2
    
    # Plot the surfaces
    surf1 = ax3.plot_surface(X1, X2, Z_before, cmap='Reds', alpha=0.5, label='Before update')
    surf2 = ax3.plot_surface(X1, X2, Z_after, cmap='Greens', alpha=0.5, label='After update')
    
    # Add data points
    ax3.scatter(x1, x2, y, c='blue', s=100, marker='o', label='Data points')
    ax3.scatter(first_x1, first_x2, first_y, c='red', s=150, edgecolor='black', marker='o', label='First example')
    
    # Add first prediction points
    ax3.scatter(first_x1, first_x2, first_pred_before, c='red', s=100, marker='x', label='Prediction before')
    ax3.scatter(first_x1, first_x2, first_pred_after, c='green', s=100, marker='x', label='Prediction after')
    
    # Connect predictions to actual point to show error
    ax3.plot([first_x1, first_x1], [first_x2, first_x2], [first_y, first_pred_before], 'r-', linewidth=2)
    ax3.plot([first_x1, first_x1], [first_x2, first_x2], [first_y, first_pred_after], 'g-', linewidth=2)
    
    # Set labels and title
    ax3.set_xlabel('Feature $x_1$')
    ax3.set_ylabel('Feature $x_2$')
    ax3.set_zlabel('Target $y$')
    ax3.set_title('3D Visualization of SGD Update')
    
    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Before update'),
        Patch(facecolor='green', alpha=0.5, label='After update'),
        Line2D([0], [0], marker='o', color='blue', label='Data points',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='red', label='First example',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='x', color='red', label='Prediction before',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='x', color='green', label='Prediction after',
               markerfacecolor='green', markersize=10)
    ]
    ax3.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sgd_update_visualization.png"), dpi=300)
    plt.close()

visualize_update()

# Step 5: Explain the difference between SGD and Batch Gradient Descent
def explain_sgd_vs_bgd():
    """Explain the key differences between SGD and Batch Gradient Descent."""
    print("Step 5: Explaining Differences Between SGD and Batch Gradient Descent")
    print("------------------------------------------------------------------")
    print("Key differences between Stochastic Gradient Descent (SGD) and Batch Gradient Descent:")
    print()
    print("1. Update Frequency:")
    print("   - SGD: Updates parameters after each training example")
    print("   - Batch GD: Updates parameters after processing the entire dataset")
    print()
    print("2. Gradient Calculation:")
    print("   - SGD: Computes gradient using a single training example")
    print("   - Batch GD: Computes gradient by averaging over all training examples")
    print()
    print("3. Computational Efficiency:")
    print("   - SGD: Faster per update, especially for large datasets")
    print("   - Batch GD: More computationally intensive per update")
    print()
    print("4. Convergence Behavior:")
    print("   - SGD: Noisy updates, may oscillate around minimum")
    print("   - Batch GD: Smoother descent path, more stable updates")
    print()
    print("5. Memory Requirements:")
    print("   - SGD: Lower memory requirements")
    print("   - Batch GD: Requires loading all data for each update")
    print()
    print("6. Escape Local Minima:")
    print("   - SGD: Noise in updates may help escape local minima")
    print("   - Batch GD: May get stuck in local minima")
    print()
    print("7. Learning Rate Sensitivity:")
    print("   - SGD: Typically requires smaller learning rates or scheduling")
    print("   - Batch GD: Can often use larger learning rates")
    print()
    
    # Create a visualization comparing SGD and Batch GD
    plt.figure(figsize=(12, 8))
    
    # Create example gradient paths
    w0_values = np.linspace(-1, 2, 30)
    w1_values = np.linspace(-1, 2, 30)
    W0, W1 = np.meshgrid(w0_values, w1_values)
    
    # Cost function surface (for illustration)
    Z = 3 * (W0 - 1)**2 + (W1 - 1)**2 + 0.5
    
    # Example SGD path (more zigzag)
    np.random.seed(42)
    sgd_w0 = [-0.8]
    sgd_w1 = [-0.6]
    for i in range(25):
        sgd_w0.append(sgd_w0[-1] + 0.07 + 0.08 * np.random.randn())
        sgd_w1.append(sgd_w1[-1] + 0.05 + 0.08 * np.random.randn())
    
    # Example Batch GD path (smoother)
    batch_w0 = [-0.8]
    batch_w1 = [-0.6]
    for i in range(10):
        batch_w0.append(batch_w0[-1] + 0.2)
        batch_w1.append(batch_w1[-1] + 0.18)
    
    # Plot the contour and paths
    plt.contourf(W0, W1, Z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Cost $J(w)$')
    plt.plot(sgd_w0, sgd_w1, 'r-o', linewidth=2, markersize=8, alpha=0.7, label='SGD Path (Noisy)')
    plt.plot(batch_w0, batch_w1, 'b-o', linewidth=3, markersize=10, alpha=0.7, label='Batch GD Path (Smooth)')
    plt.scatter(1, 1, s=150, c='white', edgecolor='black', marker='*', label='Global Minimum')
    
    plt.title('SGD vs Batch Gradient Descent Comparison')
    plt.xlabel('$w_0$')
    plt.ylabel('$w_1$')
    plt.grid(True)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sgd_vs_bgd.png"), dpi=300)
    plt.close()

explain_sgd_vs_bgd()

# NEW: Add a visualization showing multiple iterations of SGD
def visualize_sgd_iterations():
    """Create a visualization that shows how SGD converges over multiple iterations."""
    print("Step 6: Visualizing SGD Convergence Over Multiple Iterations")
    print("----------------------------------------------------------")
    
    # Number of iterations to simulate
    num_iterations = 10
    
    # Initialize arrays to track weights and loss over iterations
    weights_history = []
    loss_history = []
    
    # Start with initial weights
    current_weights = np.array(initial_weights)
    weights_history.append(current_weights.copy())
    
    # Calculate initial loss
    predictions = np.zeros(len(y))
    for i in range(len(y)):
        predictions[i] = current_weights[0] + current_weights[1] * x1[i] + current_weights[2] * x2[i]
    errors = predictions - y
    mse = np.mean(errors**2)
    loss_history.append(mse)
    
    # Simulate SGD iterations
    for iteration in range(num_iterations):
        # Loop through each example in order (for simplicity)
        for i in range(len(x1)):
            # Calculate prediction for this example
            prediction = current_weights[0] + current_weights[1] * x1[i] + current_weights[2] * x2[i]
            
            # Calculate error
            error = prediction - y[i]
            
            # Calculate gradients for this example
            grad_w0 = error
            grad_w1 = error * x1[i]
            grad_w2 = error * x2[i]
            
            # Update weights
            current_weights[0] -= learning_rate * grad_w0
            current_weights[1] -= learning_rate * grad_w1
            current_weights[2] -= learning_rate * grad_w2
        
        # Save weights after this iteration
        weights_history.append(current_weights.copy())
        
        # Calculate loss after this iteration
        predictions = np.zeros(len(y))
        for i in range(len(y)):
            predictions[i] = current_weights[0] + current_weights[1] * x1[i] + current_weights[2] * x2[i]
        errors = predictions - y
        mse = np.mean(errors**2)
        loss_history.append(mse)
    
    # Print final weights and loss
    print(f"Weights after {num_iterations} iterations: w_0 = {current_weights[0]:.4f}, w_1 = {current_weights[1]:.4f}, w_2 = {current_weights[2]:.4f}")
    print(f"Final MSE loss: {loss_history[-1]:.4f}")
    print()
    
    # Create a visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot 1: Learning curve (loss over iterations)
    ax1 = fig.add_subplot(gs[0, 0])
    iterations = np.arange(len(loss_history))
    ax1.plot(iterations, loss_history, 'b-o', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('SGD Learning Curve')
    ax1.grid(True)
    
    # Plot 2: Weight trajectories
    ax2 = fig.add_subplot(gs[0, 1:])
    w0_values = [w[0] for w in weights_history]
    w1_values = [w[1] for w in weights_history]
    w2_values = [w[2] for w in weights_history]
    
    ax2.plot(iterations, w0_values, 'r-o', linewidth=2, label='$w_0$')
    ax2.plot(iterations, w1_values, 'g-o', linewidth=2, label='$w_1$')
    ax2.plot(iterations, w2_values, 'b-o', linewidth=2, label='$w_2$')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Weight Value')
    ax2.set_title('Weight Trajectories')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Fit improvement over iterations
    ax3 = fig.add_subplot(gs[1, :])
    
    # Select a few iterations to display
    iterations_to_show = [0, 1, 3, num_iterations]
    colors = ['red', 'orange', 'green', 'blue']
    
    # For simplicity, show predictions for different iterations in a 2D plot
    # We'll create a grid of x1 values and use the mean of x2 for simplicity
    x1_grid = np.linspace(min(x1) - 0.5, max(x1) + 0.5, 100)
    x2_mean = np.mean(x2)
    
    # Plot the data points
    ax3.scatter(x1, y, color='blue', s=120, edgecolor='black', label='Data points')
    
    # Plot predictions for selected iterations
    for idx, it in enumerate(iterations_to_show):
        w0, w1, w2 = weights_history[it]
        y_pred = w0 + w1 * x1_grid + w2 * x2_mean
        ax3.plot(x1_grid, y_pred, color=colors[idx], linewidth=2, 
                 label=f'Iteration {it}', linestyle=['-', '--', '-.', ':'][idx])
    
    ax3.set_xlabel('Feature $x_1$')
    ax3.set_ylabel('Target $y$')
    ax3.set_title('Model Fit Improvement Over Iterations ($x_2$ fixed at mean)')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sgd_iterations.png"), dpi=300)
    plt.close()

visualize_sgd_iterations()

# Summary of the solution
print("\nQuestion 2 Solution Summary:")
print("1. SGD Update Rule: For each parameter wⱼ, wⱼ = wⱼ - α * ∂J/∂wⱼ")
print("2. Gradient for first example:")
print(f"   - ∂J/∂w₀ = {gradients[0]}")
print(f"   - ∂J/∂w₁ = {gradients[1]}")
print(f"   - ∂J/∂w₂ = {gradients[2]}")
print("3. Updated weights after one SGD step:")
print(f"   - w₀ = {new_weights[0]}")
print(f"   - w₁ = {new_weights[1]}")
print(f"   - w₂ = {new_weights[2]}")
print("4. Key differences between SGD and Batch GD: Update frequency, gradient calculation,")
print("   computational efficiency, convergence behavior, memory requirements, etc.")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- sgd_concept.png: Visualization of the SGD concept")
print("- sgd_update_visualization.png: Detailed visualization of the first SGD update")
print("- sgd_vs_bgd.png: Comparison between SGD and Batch Gradient Descent")
print("- sgd_iterations.png: Convergence of SGD over multiple iterations") 