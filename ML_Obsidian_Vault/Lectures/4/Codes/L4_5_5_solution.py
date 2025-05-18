import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Task 1: Compare perceptron update rule with gradient descent for logistic regression
def perceptron_update(w, x, y, eta):
    """
    Perceptron update rule: w_{t+1} = w_t + η*y*x if misclassified
    
    Args:
        w: Current weights
        x: Feature vector
        y: True label (±1)
        eta: Learning rate
    
    Returns:
        Updated weights
    """
    # Check if point is misclassified
    if y * np.dot(w, x) <= 0:
        return w + eta * y * x
    else:
        return w  # No update if correctly classified

def logistic_regression_gradient_update(w, x, y, eta):
    """
    Gradient descent update for logistic regression: w_{t+1} = w_t + η*(y - sigmoid(w^T*x))*x
    
    Args:
        w: Current weights
        x: Feature vector
        y: True label (0 or 1)
        eta: Learning rate
    
    Returns:
        Updated weights
    """
    # Convert y from {-1, 1} to {0, 1} for logistic regression
    y_binary = (y + 1) / 2
    
    # Compute sigmoid of w^T*x
    sigmoid = 1 / (1 + np.exp(-np.dot(w, x)))
    
    # Update rule
    return w + eta * (y_binary - sigmoid) * x

# Task 2: Visualize why perceptron might converge faster for linearly separable data
def generate_linearly_separable_data(n_samples=100, noise=0.1, separable=True):
    """
    Generate a dataset that can be either linearly separable or not based on parameter
    """
    if separable:
        # For separable data, we'll create points with a clear margin
        # Generate random points
        X = np.random.randn(n_samples, 2)
        
        # Create a separating line
        true_w = np.array([0.0, -1.0, 1.0])  # [bias, w1, w2]
        
        # Add bias term
        X_with_bias = np.column_stack((np.ones(n_samples), X))
        
        # Compute raw values
        raw_values = np.dot(X_with_bias, true_w)
        
        # Create labels based on decision boundary with margin
        y = np.sign(raw_values)
        
        # For points too close to the boundary, push them away to create margin
        margin = 0.5
        too_close = np.abs(raw_values) < margin
        
        for i in range(n_samples):
            if too_close[i]:
                direction = y[i]  # +1 or -1
                
                # Make adjustments to features to push away from boundary
                if direction > 0:
                    X[i, 1] += margin  # Push up points above boundary
                else:
                    X[i, 1] -= margin  # Push down points below boundary
        
        # Recalculate X_with_bias with updated X
        X_with_bias = np.column_stack((np.ones(n_samples), X))
        
        # Verify labels are correct after adjustment
        raw_values = np.dot(X_with_bias, true_w)
        y = np.sign(raw_values)
    
    else:
        # For non-separable data, we'll create overlapping clusters
        X = np.random.randn(n_samples, 2)
        
        # Create a separating line that will be imperfect
        true_w = np.array([0.0, -1.0, 1.0])  # [bias, w1, w2]
        
        # Add bias term
        X_with_bias = np.column_stack((np.ones(n_samples), X))
        
        # Compute raw values
        raw_values = np.dot(X_with_bias, true_w)
        
        # Create initial labels
        y = np.sign(raw_values)
        
        # Deliberately flip some labels near the boundary to make it non-separable
        margin_zone = np.abs(raw_values) < 1.0
        flip_candidates = np.where(margin_zone)[0]
        
        if len(flip_candidates) > 0:
            # Select ~20% of the points in the margin zone to flip
            flip_count = max(1, int(0.2 * len(flip_candidates)))
            flip_indices = np.random.choice(flip_candidates, size=flip_count, replace=False)
            y[flip_indices] = -y[flip_indices]
    
    # Ensure we have both positive and negative examples
    if len(np.unique(y)) < 2:
        # If we only have one class, create at least one example of the other class
        if all(y > 0):
            y[0] = -1
        else:
            y[0] = 1
            
    # Convert to ±1
    y = np.where(y >= 0, 1, -1)
    
    return X_with_bias, y, true_w

def train_models_and_compare(X, y, n_epochs=100):
    """
    Train perceptron and logistic regression models, track convergence and weights
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Initialize weights
    w_perceptron = np.zeros(n_features)
    w_logistic = np.zeros(n_features)
    
    # Learning rate
    eta = 0.01
    
    # Track misclassifications and loss
    perceptron_errors = []
    logistic_loss = []
    
    # Store weights at each epoch
    perceptron_weights = [w_perceptron.copy()]
    logistic_weights = [w_logistic.copy()]
    
    # Convert y for logistic regression
    y_binary = (y + 1) / 2
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        y_binary_shuffled = y_binary[indices]
        
        # Track errors for this epoch
        perceptron_epoch_errors = 0
        logistic_epoch_loss = 0
        
        for i in range(n_samples):
            # Perceptron update
            pred = np.sign(np.dot(w_perceptron, X_shuffled[i]))
            if pred != y_shuffled[i]:
                perceptron_epoch_errors += 1
                w_perceptron = perceptron_update(w_perceptron, X_shuffled[i], y_shuffled[i], eta)
            
            # Logistic regression update
            sigmoid = 1 / (1 + np.exp(-np.dot(w_logistic, X_shuffled[i])))
            log_loss = -y_binary_shuffled[i] * np.log(sigmoid + 1e-10) - (1 - y_binary_shuffled[i]) * np.log(1 - sigmoid + 1e-10)
            logistic_epoch_loss += log_loss
            
            w_logistic = logistic_regression_gradient_update(w_logistic, X_shuffled[i], y_shuffled[i], eta)
        
        perceptron_errors.append(perceptron_epoch_errors)
        logistic_loss.append(logistic_epoch_loss / n_samples)
        
        # Save weights at this epoch - use normalized weights to avoid numerical issues
        p_norm = np.linalg.norm(w_perceptron)
        l_norm = np.linalg.norm(w_logistic)
        
        # Only normalize if norm is non-zero to avoid division by zero
        if p_norm > 1e-10:
            perceptron_weights.append((w_perceptron / p_norm).copy())
        else:
            perceptron_weights.append(w_perceptron.copy())
            
        if l_norm > 1e-10:
            logistic_weights.append((w_logistic / l_norm).copy())
        else:
            logistic_weights.append(w_logistic.copy())
    
    return w_perceptron, w_logistic, perceptron_errors, logistic_loss, perceptron_weights, logistic_weights

def _draw_subplot_boundary(ax, X_features, y_data, w_model, true_w, model_name, cmap, xx, yy, mesh_points, x_min, x_max, y_min, y_max, epoch_num, is_perceptron):
    """
    Helper function to draw one subplot for decision boundary visualization.
    """
    ax.cla() # Clear the axis for the new frame

    # Predict based on model type - both perceptron and logistic regression 
    # use the sign of w^T*x for decision boundaries
    Z_model = np.array([1 if np.dot(w_model, x_mp) > 0 else -1 for x_mp in mesh_points])
    Z_model = Z_model.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z_model, cmap=cmap, alpha=0.3, shading='auto')

    # Plot data points
    ax.scatter(X_features[y_data == 1, 0], X_features[y_data == 1, 1], c='blue', marker='o', s=50, edgecolor='k', label='Class +1')
    ax.scatter(X_features[y_data == -1, 0], X_features[y_data == -1, 1], c='red', marker='x', s=50, label='Class -1')

    x_boundary_plot_range = np.array([x_min, x_max])

    # Model boundary - determine by solving w[0] + w[1]*x1 + w[2]*x2 = 0 for x2
    line_color = 'b' if is_perceptron else 'r'
    if abs(w_model[2]) > 1e-10:  # Check if w[2] is effectively non-zero
        y_model = -(w_model[0] + w_model[1] * x_boundary_plot_range) / w_model[2]
        ax.plot(x_boundary_plot_range, np.clip(y_model, y_min, y_max), color=line_color, linestyle='-', linewidth=2, label=model_name)
    elif abs(w_model[1]) > 1e-10:  # If w[2] is zero but w[1] is not
        x_intercept_model = -w_model[0] / w_model[1]
        if x_min <= x_intercept_model <= x_max:
            ax.axvline(x=x_intercept_model, color=line_color, linestyle='-', linewidth=2, label=model_name)

    # True boundary
    if abs(true_w[2]) > 1e-10:
        y_true = -(true_w[0] + true_w[1] * x_boundary_plot_range) / true_w[2]
        ax.plot(x_boundary_plot_range, np.clip(y_true, y_min, y_max), 'g--', linewidth=2, label='True Boundary')
    elif abs(true_w[1]) > 1e-10:
        x_intercept_true = -true_w[0] / true_w[1]
        if x_min <= x_intercept_true <= x_max:
            ax.axvline(x=x_intercept_true, color='g', linestyle='--', linewidth=2, label='True Boundary')

    ax.set_title(f'{model_name} (Epoch {epoch_num})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper left')


def plot_decision_boundaries(X, y, w_perceptron, w_logistic, true_w, iteration, static_image_dir=None):
    """
    Plot and save a static image of decision boundaries for a specific iteration.
    """
    X_features = X[:, 1:]
    h = 0.02
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1 # Adjusted margin slightly
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1 # Adjusted margin slightly
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _draw_subplot_boundary(ax1, X_features, y, w_perceptron, true_w, "Perceptron", cmap, xx, yy, mesh_points, x_min, x_max, y_min, y_max, iteration, is_perceptron=True)
    _draw_subplot_boundary(ax2, X_features, y, w_logistic, true_w, "Logistic Regression", cmap, xx, yy, mesh_points, x_min, x_max, y_min, y_max, iteration, is_perceptron=False)
    
    plt.tight_layout()
    
    if static_image_dir:
        os.makedirs(static_image_dir, exist_ok=True)
        plt.savefig(os.path.join(static_image_dir, f'decision_boundaries_epoch_{iteration}.png'), dpi=300)
    
    plt.close(fig)

def plot_convergence(perceptron_errors, logistic_loss, is_separable=True):
    """
    Plot convergence rates for perceptron and logistic regression
    """
    plt.figure(figsize=(12, 5))
    
    # Plot perceptron errors
    plt.subplot(1, 2, 1)
    plt.plot(perceptron_errors, 'b-', linewidth=2, label='Perceptron Errors')
    plt.title('Perceptron Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Misclassifications')
    plt.grid(True)
    plt.legend()
    
    # Plot logistic loss
    plt.subplot(1, 2, 2)
    plt.plot(logistic_loss, 'r-', linewidth=2, label='Logistic Loss')
    plt.title('Logistic Regression Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save with appropriate name indicating separability
    if is_separable:
        plt.savefig(os.path.join(save_dir, 'convergence_comparison_separable.png'), dpi=300)
    else:
        plt.savefig(os.path.join(save_dir, 'convergence_comparison_non_separable.png'), dpi=300)
    
    plt.close()

def visualize_dataset(X, y, true_w, title="Dataset Visualization", filename="dataset_visualization.png"):
    """
    Create a simple visualization of the dataset with the true decision boundary
    """
    plt.figure(figsize=(8, 6))
    
    # Extract features (skip bias term)
    X_features = X[:, 1:]
    
    # Plot the points
    plt.scatter(X_features[y == 1, 0], X_features[y == 1, 1], 
                c='blue', marker='o', s=50, edgecolor='k', label='Class +1')
    plt.scatter(X_features[y == -1, 0], X_features[y == -1, 1], 
                c='red', marker='x', s=50, label='Class -1')
    
    # Compute limits for the plot
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
    
    # Draw the true decision boundary
    x_boundary = np.linspace(x_min, x_max, 100)
    if abs(true_w[2]) > 1e-10:  # Check if third coefficient is non-zero
        # Solve w0 + w1*x + w2*y = 0 for y
        y_boundary = -(true_w[0] + true_w[1] * x_boundary) / true_w[2]
        plt.plot(x_boundary, y_boundary, 'g--', linewidth=2, label='True Decision Boundary')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Check if the data is actually linearly separable
    # One way to check: see if any points are on the wrong side of the true boundary
    is_separable = True
    for i in range(len(X)):
        decision_value = np.dot(true_w, X[i])
        if (decision_value > 0 and y[i] < 0) or (decision_value < 0 and y[i] > 0):
            is_separable = False
            break
    
    # Add text annotation about separability
    if "Non-Linearly" not in title:
        if is_separable:
            plt.text(0.05, 0.05, "Dataset is linearly separable", 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        else:
            plt.text(0.05, 0.05, "Dataset is NOT linearly separable", 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    else:
        if is_separable:
            plt.text(0.05, 0.05, "Warning: Dataset appears to be separable!", 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'))
        else:
            plt.text(0.05, 0.05, "Confirmed: Dataset is not linearly separable", 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def visualize_specific_example():
    """
    Visualize the perceptron update for the specific example in task 3
    """
    # Given problem data
    x = np.array([1, 2, 1])  # Including bias term
    y = 1
    w = np.array([0, 1, -1])
    eta = 0.5
    
    # Calculate the updated weights
    w_updated = perceptron_update(w, x, y, eta)
    
    print(f"Original weights: w = {w}")
    print(f"Misclassified point: x = {x}, y = {y}")
    print(f"Learning rate: η = {eta}")
    
    # Check if misclassified
    decision_value = np.dot(w, x)
    print(f"Decision value: w·x = {decision_value}")
    
    if decision_value <= 0:
        print("Point is misclassified (y·w·x ≤ 0)")
        print(f"Update: w_new = w + η·y·x = {w} + {eta}·{y}·{x} = {w_updated}")
    else:
        print("Point is correctly classified (y·w·x > 0)")
        print("No update needed: w_new = w")
    
    # Create a 3D visualization of the weight update
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Original decision boundary: w[0] + w[1]*x + w[2]*y = 0
    # Solve for y: y = -(w[0] + w[1]*x) / w[2]
    
    # Create a meshgrid for visualization
    x1 = np.linspace(-2, 4, 100)
    x2 = np.linspace(-2, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate the original decision plane
    if w[2] != 0:  # Make sure w[2] is not zero
        Z_original = -(w[0] + w[1] * X1) / w[2]
    else:
        Z_original = np.zeros_like(X1)
    
    # Calculate the updated decision plane
    if w_updated[2] != 0:  # Make sure w_updated[2] is not zero
        Z_updated = -(w_updated[0] + w_updated[1] * X1) / w_updated[2]
    else:
        Z_updated = np.zeros_like(X1)
    
    # Plot original decision boundary
    surf = ax.plot_surface(X1, X2, Z_original, color='blue', alpha=0.3, label='Original Decision Boundary')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    
    # Plot updated decision boundary
    surf = ax.plot_surface(X1, X2, Z_updated, color='red', alpha=0.3, label='Updated Decision Boundary')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    
    # Plot the point
    ax.scatter([x[1]], [x[2]], [0], color='green', s=120, marker='o', label=f'Point ({x[1]}, {x[2]}), y={y}')
    
    # Plot the weight vector as an arrow
    ax.quiver(0, 0, 0, w[1], w[2], 0, color='blue', linewidth=2, label=f'Original w = [{w[0]}, {w[1]}, {w[2]}]')
    ax.quiver(0, 0, 0, w_updated[1], w_updated[2], 0, color='red', linewidth=2, label=f'Updated w = [{w_updated[0]}, {w_updated[1]}, {w_updated[2]}]')
    
    # Calculate the update vector (only for visualization)
    update_vector = eta * y * x
    ax.quiver(w[1], w[2], 0, update_vector[1], update_vector[2], 0, color='green', linewidth=2, 
             label=f'Update: η·y·x = {eta}·{y}·[{x[0]}, {x[1]}, {x[2]}]')
    
    # Set labels and title
    ax.set_xlabel('$w_1$ (weight for $x_1$)')
    ax.set_ylabel('$w_2$ (weight for $x_2$)')
    ax.set_zlabel('$z$')
    ax.set_title('Perceptron Weight Update and Decision Boundary Change')
    
    # Add a legend
    ax.legend(loc='upper left')
    
    # Set viewpoint for better visualization
    ax.view_init(elev=30, azim=-60)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'specific_example_update.png'), dpi=300)
    plt.close()
    
    # 2D visualization of before and after
    plt.figure(figsize=(12, 5))
    
    # Before update
    plt.subplot(1, 2, 1)
    x1_vals = np.linspace(-1, 3, 100)
    # w[0] + w[1]*x1 + w[2]*x2 = 0 => x2 = -(w[0] + w[1]*x1) / w[2]
    if w[2] != 0:
        x2_vals = -(w[0] + w[1] * x1_vals) / w[2]
        plt.plot(x1_vals, x2_vals, 'b-', linewidth=2, label='Decision Boundary')
    else:
        plt.axvline(x=-w[0]/w[1], color='b', linewidth=2, label='Decision Boundary')
    
    plt.scatter(x[1], x[2], color='g', s=120, marker='o', edgecolor='k', label=f'Point ({x[1]}, {x[2]}), y={y}')
    plt.quiver(0, 0, w[1], w[2], angles='xy', scale_units='xy', scale=1, color='b', linewidth=2, 
               label=f'w = [{w[0]}, {w[1]}, {w[2]}]')
    
    # Indicate the region for y = 1 and y = -1
    plt.fill_between(x1_vals, x2_vals, 4, color='lightblue', alpha=0.3, label='Predicted: y = -1')
    plt.fill_between(x1_vals, -2, x2_vals, color='lightcoral', alpha=0.3, label='Predicted: y = +1')
    
    if decision_value <= 0:
        plt.text(2.5, 3.5, "MISCLASSIFIED", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add text annotations for regions with improved visibility
    plt.text(2.5, 2.5, "Predicted: y = -1", color='blue', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.5, 0.5, "Predicted: y = +1", color='red', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlim(-1, 3)
    plt.ylim(-2, 4)
    plt.grid(True)
    plt.title('Before Update')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    
    # After update
    plt.subplot(1, 2, 2)
    # w_updated[0] + w_updated[1]*x1 + w_updated[2]*x2 = 0 => x2 = -(w_updated[0] + w_updated[1]*x1) / w_updated[2]
    if w_updated[2] != 0:
        x2_vals_updated = -(w_updated[0] + w_updated[1] * x1_vals) / w_updated[2]
        plt.plot(x1_vals, x2_vals_updated, 'r-', linewidth=2, label='Updated Decision Boundary')
    else:
        plt.axvline(x=-w_updated[0]/w_updated[1], color='r', linewidth=2, label='Updated Decision Boundary')
    
    if w[2] != 0:
        plt.plot(x1_vals, x2_vals, 'b--', linewidth=2, label='Original Decision Boundary')
    else:
        plt.axvline(x=-w[0]/w[1], color='b', linestyle='--', linewidth=2, label='Original Decision Boundary')
    
    plt.scatter(x[1], x[2], color='g', s=120, marker='o', edgecolor='k', label=f'Point ({x[1]}, {x[2]}), y={y}')
    plt.quiver(0, 0, w_updated[1], w_updated[2], angles='xy', scale_units='xy', scale=1, color='r', linewidth=2, 
               label=f'w_new = [{w_updated[0]}, {w_updated[1]}, {w_updated[2]}]')
    
    # Indicate the region for y = 1 and y = -1 after update (swap colors to match correct predictions)
    # Since w_updated = w (no change in this case), we're using the same boundary
    plt.fill_between(x1_vals, x2_vals_updated, 4, color='lightcoral', alpha=0.3, label='Predicted: y = +1')
    plt.fill_between(x1_vals, -2, x2_vals_updated, color='lightblue', alpha=0.3, label='Predicted: y = -1')
    
    # Check if the point is now correctly classified
    decision_value_updated = np.dot(w_updated, x)
    if decision_value_updated > 0:
        plt.text(2.5, 3.5, "CORRECTLY CLASSIFIED", color='green', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    else:
        plt.text(2.5, 3.5, "STILL MISCLASSIFIED", color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add text annotations for regions with improved visibility
    plt.text(2.5, 2.5, "Predicted: y = +1", color='red', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.5, 0.5, "Predicted: y = -1", color='blue', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlim(-1, 3)
    plt.ylim(-2, 4)
    plt.grid(True)
    plt.title('After Update')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'specific_example_2d.png'), dpi=300)
    plt.close()

def visualize_update_non_reduction():
    """
    Visualize a case where perceptron update doesn't reduce the total number of misclassifications
    """
    # Create a scenario where fixing one point misclassifies another
    X = np.array([
        [1, 1, 1],  # First point with bias term
        [1, -1, 2],  # Second point with bias term
    ])
    y = np.array([1, 1])  # Both points have label +1
    
    # Choose initial weights that misclassify the first point but correctly classify the second
    w_initial = np.array([-2, 1, -1])
    
    # Check classifications
    classifications_before = [np.sign(np.dot(w_initial, x)) for x in X]
    print(f"Initial weights: w = {w_initial}")
    print(f"Classifications before update: {classifications_before}")
    print(f"True labels: {y}")
    print(f"First point is {'correctly' if classifications_before[0] == y[0] else 'incorrectly'} classified.")
    print(f"Second point is {'correctly' if classifications_before[1] == y[1] else 'incorrectly'} classified.")
    
    # Update weights based on the first point (which is misclassified)
    eta = 1.0
    w_updated = perceptron_update(w_initial, X[0], y[0], eta)
    print(f"Updated weights: w_new = {w_updated}")
    
    # Check classifications after update
    classifications_after = [np.sign(np.dot(w_updated, x)) for x in X]
    print(f"Classifications after update: {classifications_after}")
    print(f"First point is {'correctly' if classifications_after[0] == y[0] else 'incorrectly'} classified.")
    print(f"Second point is {'correctly' if classifications_after[1] == y[1] else 'incorrectly'} classified.")
    
    # Find the number of misclassifications before and after
    misclassifications_before = sum(1 for i in range(len(y)) if classifications_before[i] != y[i])
    misclassifications_after = sum(1 for i in range(len(y)) if classifications_after[i] != y[i])
    print(f"Number of misclassifications before: {misclassifications_before}")
    print(f"Number of misclassifications after: {misclassifications_after}")
    
    # Visualize the decision boundary and the points before and after update
    plt.figure(figsize=(12, 5))
    
    # Original decision boundary
    plt.subplot(1, 2, 1)
    
    # Define the plotting range
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Calculate decision boundary
    # w[0] + w[1]*x1 + w[2]*x2 = 0 => x2 = -(w[0] + w[1]*x1) / w[2]
    if w_initial[2] != 0:
        boundary = -(w_initial[0] + w_initial[1] * x1_range) / w_initial[2]
        plt.plot(x1_range, boundary, 'b-', linewidth=2, label='Decision Boundary')
        
        # Shade regions
        plt.fill_between(x1_range, boundary, 3, color='lightblue', alpha=0.3, label='Predicted: -1')
        plt.fill_between(x1_range, -3, boundary, color='lightcoral', alpha=0.3, label='Predicted: +1')
    else:
        plt.axvline(x=-w_initial[0]/w_initial[1], color='b', linewidth=2, label='Decision Boundary')
    
    # Plot the points
    markers = ['o', 's']  # Different markers for each point for clarity
    for i in range(len(X)):
        color = 'green' if classifications_before[i] == y[i] else 'red'
        plt.scatter(X[i, 1], X[i, 2], color=color, s=120, marker=markers[i], edgecolor='k',
                   label=f'Point {i+1} (y={y[i]}, pred={classifications_before[i]})')
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.title('Before Update: Decision Boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper right')
    
    # Updated decision boundary
    plt.subplot(1, 2, 2)
    
    if w_updated[2] != 0:
        boundary_updated = -(w_updated[0] + w_updated[1] * x1_range) / w_updated[2]
        plt.plot(x1_range, boundary_updated, 'r-', linewidth=2, label='Updated Decision Boundary')
        
        # Shade regions
        plt.fill_between(x1_range, boundary_updated, 3, color='lightblue', alpha=0.3, label='Predicted: -1')
        plt.fill_between(x1_range, -3, boundary_updated, color='lightcoral', alpha=0.3, label='Predicted: +1')
    else:
        plt.axvline(x=-w_updated[0]/w_updated[1], color='r', linewidth=2, label='Updated Decision Boundary')
    
    # Plot the original boundary for comparison
    if w_initial[2] != 0:
        plt.plot(x1_range, boundary, 'b--', linewidth=2, label='Original Decision Boundary')
    else:
        plt.axvline(x=-w_initial[0]/w_initial[1], color='b', linestyle='--', linewidth=2, label='Original Decision Boundary')
    
    # Plot the points
    for i in range(len(X)):
        color = 'green' if classifications_after[i] == y[i] else 'red'
        plt.scatter(X[i, 1], X[i, 2], color=color, s=120, marker=markers[i], edgecolor='k',
                  label=f'Point {i+1} (y={y[i]}, pred={classifications_after[i]})')
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.title('After Update: Decision Boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'non_reduction_example.png'), dpi=300)
    plt.close()

def main():
    # Generate data - first with separable data
    X_separable, y_separable, true_weights_separable = generate_linearly_separable_data(
        n_samples=100, noise=0.1, separable=True)
    
    # Visualize the separable dataset
    visualize_dataset(X_separable, y_separable, true_weights_separable, 
                     title="Linearly Separable Dataset", 
                     filename="separable_dataset.png")
    
    # Train models on separable data
    num_epochs_training = 50  # Example: number of epochs
    final_w_p, final_w_l, errors_p, loss_l, history_p_w, history_l_w = \
        train_models_and_compare(X_separable, y_separable, n_epochs=num_epochs_training)
    
    # Plot convergence for separable data
    plot_convergence(errors_p, loss_l, is_separable=True)
    
    # Plot final decision boundaries for separable data
    plot_decision_boundaries(X_separable, y_separable, final_w_p, final_w_l, true_weights_separable, 
                           num_epochs_training, static_image_dir=save_dir)
    
    # Rename the separable file correctly
    orig_path = os.path.join(save_dir, f'decision_boundaries_epoch_{num_epochs_training}.png')
    sep_path = os.path.join(save_dir, f'decision_boundaries_epoch_{num_epochs_training}_separable.png')
    if os.path.exists(orig_path):
        os.rename(orig_path, sep_path)
    
    # Now try with non-separable data
    X_non_separable, y_non_separable, true_weights_non_separable = generate_linearly_separable_data(
        n_samples=100, noise=0.1, separable=False)
    
    # Visualize the non-separable dataset
    visualize_dataset(X_non_separable, y_non_separable, true_weights_non_separable, 
                     title="Non-Linearly Separable Dataset", 
                     filename="non_separable_dataset.png")
    
    # Train models on non-separable data
    final_w_p_ns, final_w_l_ns, errors_p_ns, loss_l_ns, history_p_w_ns, history_l_w_ns = \
        train_models_and_compare(X_non_separable, y_non_separable, n_epochs=num_epochs_training)
    
    # Plot convergence for non-separable data
    plot_convergence(errors_p_ns, loss_l_ns, is_separable=False)
    
    # Plot final decision boundaries for non-separable data
    plot_decision_boundaries(X_non_separable, y_non_separable, final_w_p_ns, final_w_l_ns, 
                          true_weights_non_separable, num_epochs_training, static_image_dir=save_dir)
    
    # Rename the file to indicate non-separable data
    orig_path = os.path.join(save_dir, f'decision_boundaries_epoch_{num_epochs_training}.png')
    non_sep_path = os.path.join(save_dir, f'decision_boundaries_epoch_{num_epochs_training}_non_separable.png')
    if os.path.exists(orig_path):
        os.rename(orig_path, non_sep_path)
    
    # Visualize specific example for Task 3
    visualize_specific_example()

    # Visualize non-reduction case for Task 4
    visualize_update_non_reduction()

    print(f"All visualizations saved in: {save_dir}")

if __name__ == '__main__':
    main()