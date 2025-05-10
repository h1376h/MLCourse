import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

def create_output_dir():
    """Create directory to save figures using consistent directory structure"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L5_2_Quiz_5")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_dataset():
    """Load the tumor classification dataset"""
    data = {
        'Age': [15, 65, 30, 90, 44, 20, 50, 36],
        'Tumor_Size': [20, 30, 50, 20, 35, 70, 40, 25],
        'Malignant': [0, 0, 1, 1, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)

def sigmoid(z):
    """
    Compute the sigmoid function for input z with numerical stability
    
    This implementation handles large negative or positive inputs to avoid overflow.
    """
    # Clip z values to avoid overflow in exp
    z_safe = np.clip(z, -500, 500)
    
    # For large positive inputs, sigmoid approaches 1
    # For large negative inputs, sigmoid approaches 0
    # For other inputs, compute sigmoid directly
    result = np.zeros_like(z, dtype=float)
    
    # Handle large negative inputs
    mask_neg = z <= -100
    result[mask_neg] = 0
    
    # Handle large positive inputs
    mask_pos = z >= 100
    result[mask_pos] = 1
    
    # Handle normal range inputs
    mask_mid = ~(mask_neg | mask_pos)
    result[mask_mid] = 1 / (1 + np.exp(-z_safe[mask_mid]))
    
    return result

def compute_cost(X, y, theta):
    """
    Compute the logistic regression cost function (negative log-likelihood)
    
    Parameters:
    X (ndarray): Feature matrix with intercept (m x (n+1))
    y (ndarray): Target vector (m x 1)
    theta (ndarray): Parameter vector ((n+1) x 1)
    
    Returns:
    float: The cost value (negative log-likelihood)
    """
    m = len(y)
    h = sigmoid(X @ theta)
    # Use small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1-epsilon)
    # Standard logistic regression cost function with negative sign and division by m
    cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    return cost

def compute_gradient(X, y, theta):
    """
    Compute the gradient of the logistic regression cost function
    
    Parameters:
    X (ndarray): Feature matrix with intercept (m x (n+1))
    y (ndarray): Target vector (m x 1)
    theta (ndarray): Parameter vector ((n+1) x 1)
    
    Returns:
    ndarray: The gradient vector ((n+1) x 1)
    """
    m = len(y)
    h = sigmoid(X @ theta)
    # For standard cost function (negative log-likelihood), gradient is X^T * (h - y) / m
    grad = X.T @ (h - y) / m
    return grad

def plot_sigmoid_function(save_dir):
    """Plot the sigmoid function"""
    plt.figure(figsize=(10, 6))
    z = np.linspace(-10, 10, 100)
    sig = sigmoid(z)
    plt.plot(z, sig, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('z')
    plt.ylabel('g(z)')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'sigmoid_function_alt.png'), dpi=300, bbox_inches='tight')
    plt.close()

def perform_gradient_descent(X, y, initial_theta, learning_rate, num_iterations):
    """
    Perform gradient descent to optimize the parameters
    
    Parameters:
    X (ndarray): Feature matrix with intercept
    y (ndarray): Target vector
    initial_theta (ndarray): Initial parameter values
    learning_rate (float): Step size for parameter updates
    num_iterations (int): Number of iterations to perform
    
    Returns:
    tuple: (optimized_theta, cost_history)
    """
    theta = initial_theta.copy()
    cost_history = [compute_cost(X, y, theta)]
    theta_history = [theta.copy()]
    
    for i in range(num_iterations):
        gradients = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
        theta_history.append(theta.copy())
        
    return theta, theta_history, cost_history

def perform_sgd(X, y, initial_theta, learning_rate, num_iterations, seed_offset=0):
    """
    Perform stochastic gradient descent to optimize the parameters
    
    Parameters:
    X (ndarray): Feature matrix with intercept
    y (ndarray): Target vector
    initial_theta (ndarray): Initial parameter values
    learning_rate (float): Step size for parameter updates
    num_iterations (int): Number of iterations to perform
    seed_offset (int): Offset to add to random seed for reproducibility
    
    Returns:
    tuple: (optimized_theta, cost_history)
    """
    m = len(y)
    theta = initial_theta.copy()
    cost_history = [compute_cost(X, y, theta)]
    theta_history = [theta.copy()]
    
    for i in range(num_iterations):
        # Randomly select a training example
        np.random.seed(RANDOM_SEED + seed_offset + i)
        random_index = np.random.randint(0, m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        
        # Calculate hypothesis and error
        zi = xi @ theta
        hi = sigmoid(zi)
        error = hi - yi
        
        # Calculate gradient (SGD uses just the gradient for one example)
        sgd_grad = error[0] * xi[0]
        theta = theta - learning_rate * sgd_grad
        
        # Calculate new cost on full dataset
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta.copy())
        
    return theta, theta_history, cost_history

def plot_cost_history(cost_history, save_path):
    """Plot the cost function over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, 'b-', linewidth=2, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Cost Function over Gradient Descent Iterations')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_decision_boundary(X, y, theta, new_point, save_path):
    """Plot the decision boundary with given parameters"""
    # Get min/max for feature ranges
    x1_min, x1_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    x2_min, x2_max = X[:, 1].min() - 10, X[:, 1].max() + 10

    # Create a meshgrid for visualization
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    grid_with_intercept = np.c_[np.ones(grid.shape[0]), grid]

    # Calculate predictions on the grid
    Z = sigmoid(grid_with_intercept @ theta)
    Z = Z.reshape(xx1.shape)

    # Plot the decision boundary and dataset
    plt.figure(figsize=(12, 8))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(['blue', 'red']))
    plt.contour(xx1, xx2, Z, [0.5], linewidths=2, colors='black')

    # Plot the decision boundary as a line
    x1_line = np.array([x1_min, x1_max])
    x2_line = -(theta[0] + theta[1] * x1_line) / theta[2]
    plt.plot(x1_line, x2_line, 'k--', linewidth=2)

    # Add annotation for the decision boundary equation
    slope = -theta[1]/theta[2]
    intercept = -theta[0]/theta[2]
    plt.annotate(f'Tumor_Size = {intercept:.2f} {slope:.2f}*Age',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=100, edgecolors='k', label='Benign')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', s=100, linewidth=2, label='Malignant')

    # Mark the new point
    if new_point is not None:
        plt.scatter(new_point[0], new_point[1], c='green', marker='*', s=200, label='New Patient')

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel('Age (years)')
    plt.ylabel('Tumor Size (mm)')
    plt.title('Logistic Regression Decision Boundary for Tumor Classification')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def predict_new_patient(theta, patient_features):
    """Make prediction for a new patient"""
    # Add intercept term
    patient_with_intercept = np.append(1, patient_features)
    
    # Calculate z = θᵀx
    z = patient_with_intercept @ theta
    
    # Calculate probability using sigmoid function
    probability = sigmoid(z)
    
    # Make classification decision
    prediction = 1 if probability >= 0.5 else 0
    
    return z, probability, prediction

def plot_learning_rate_effect(X, y, initial_theta, save_path, learning_rates=None, num_iterations=20):
    """Visualize the effect of different learning rates"""
    if learning_rates is None:
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        _, _, cost_history = perform_gradient_descent(
            X, y, initial_theta, lr, num_iterations
        )
        plt.plot(range(len(cost_history)), cost_history, marker='o', linewidth=2, label=f'α = {lr}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_new_patient_prediction(new_z, save_path):
    """Plot the sigmoid function with the new patient's z value marked"""
    plt.figure(figsize=(10, 6))
    # Plot the sigmoid function
    z_range = np.linspace(-10, 10, 1000)
    sig_values = sigmoid(z_range)
    plt.plot(z_range, sig_values, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.axvline(x=0, color='g', linestyle='--', label='Decision Boundary (z=0)')

    # Mark our calculated z value
    plt.scatter([new_z], [sigmoid(new_z)], color='red', s=100, zorder=5, 
                label=f'New Patient (z={new_z:.2f})')

    plt.xlabel('z = θᵀx')
    plt.ylabel('P(y=1|x) = g(z)')
    plt.title('Sigmoid Function and New Patient Prediction')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_visualization(X, y, theta, new_patient, save_path):
    """Create a visualization of probability contours and 3D surface"""
    plt.figure(figsize=(12, 5))

    # Create a meshgrid of age and tumor size values
    age_range = np.linspace(10, 100, 100)
    tumor_range = np.linspace(10, 80, 100)
    age_grid, tumor_grid = np.meshgrid(age_range, tumor_range)

    # Calculate the probability for each point in the grid
    grid_points = np.c_[np.ones(age_grid.ravel().shape), age_grid.ravel(), tumor_grid.ravel()]
    z_grid = grid_points @ theta
    prob_grid = sigmoid(z_grid).reshape(age_grid.shape)

    # Create a contour plot of probabilities
    plt.subplot(1, 2, 1)
    contour = plt.contourf(age_grid, tumor_grid, prob_grid, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='P(malignant)')
    plt.contour(age_grid, tumor_grid, prob_grid, levels=[0.5], 
                colors='red', linestyles='dashed', linewidths=2)
    plt.xlabel('Age (years)')
    plt.ylabel('Tumor Size (mm)')
    plt.title('Probability of Malignancy')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', 
                label='Benign', edgecolors='k')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', 
                label='Malignant', s=80)
    plt.scatter(new_patient[0], new_patient[1], c='green', marker='*', 
                s=200, label='New Patient')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Add a 3D visualization
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(age_grid, tumor_grid, prob_grid, cmap='viridis', alpha=0.8)
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='P(malignant)')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Tumor Size (mm)')
    ax.set_zlabel('Probability')
    ax.set_title('3D Probability Surface')
    ax.view_init(30, 45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_surface(X, y, theta, new_patient, new_probability, save_path):
    """Create a 3D probability surface with training points"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh for feature space
    age_range = np.linspace(10, 100, 50)
    size_range = np.linspace(10, 80, 50)
    age_grid, size_grid = np.meshgrid(age_range, size_range)
    
    # Calculate probability at each point
    grid_points = np.c_[np.ones(age_grid.size), age_grid.ravel(), size_grid.ravel()]
    z_values = grid_points @ theta
    prob_grid = sigmoid(z_values).reshape(age_grid.shape)

    # Plot probability surface
    surf = ax.plot_surface(age_grid, size_grid, prob_grid, cmap='coolwarm', alpha=0.8)

    # Add the training data points
    for i in range(len(y)):
        if y[i] == 0:  # Benign
            ax.scatter(X[i, 0], X[i, 1], 0, c='blue', marker='o', s=100, 
                      label='Benign' if i == 0 else "")
        else:  # Malignant
            ax.scatter(X[i, 0], X[i, 1], 1, c='red', marker='x', s=100, 
                      label='Malignant' if i == 0 else "")

    # Add the decision boundary (0.5 probability contour)
    ax.contour(age_grid, size_grid, prob_grid, [0.5], colors='k', linestyles='dashed')

    # Add the new patient point
    ax.scatter(new_patient[0], new_patient[1], new_probability, c='green', marker='*', 
              s=200, label='New Patient')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Tumor Size (mm)')
    ax.set_zlabel('Probability of Malignancy')
    ax.set_title('Probability Surface for Tumor Classification')
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cost_function_surface(X, y, theta0_fixed, save_path):
    """Create a 3D visualization of the cost function surface"""
    # Create a simplified version of the cost function surface (for θ1 and θ2, fixing θ0)
    theta1_range = np.linspace(-2, 2, 50)
    theta2_range = np.linspace(-2, 2, 50)
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    cost_grid = np.zeros_like(theta1_grid)

    for i in range(len(theta1_range)):
        for j in range(len(theta2_range)):
            theta_test = np.array([theta0_fixed, theta1_grid[i, j], theta2_grid[i, j]])
            cost_grid[i, j] = compute_cost(X, y, theta_test)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(theta1_grid, theta2_grid, cost_grid, cmap='viridis', alpha=0.8)

    ax.set_xlabel('θ₁ (Age coefficient)')
    ax.set_ylabel('θ₂ (Tumor Size coefficient)')
    ax.set_zlabel('Cost J(θ)')
    ax.set_title('Logistic Regression Cost Function Surface (θ₀ fixed)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_decision_regions_with_confidence(X, y, theta, save_path):
    """Plot decision regions with confidence levels"""
    # Get min/max for feature ranges
    x1_min, x1_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    x2_min, x2_max = X[:, 1].min() - 10, X[:, 1].max() + 10

    # Create a meshgrid for visualization
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    grid_with_intercept = np.c_[np.ones(grid.shape[0]), grid]

    # Calculate predictions and confidence on the grid
    Z_prob = sigmoid(grid_with_intercept @ theta)
    Z_prob = Z_prob.reshape(xx1.shape)
    
    # Calculate confidence (distance from 0.5)
    Z_conf = np.abs(Z_prob - 0.5)
    Z_conf = Z_conf.reshape(xx1.shape)

    # Plot the decision regions and confidence
    plt.figure(figsize=(12, 8))
    
    # Plot confidence levels
    levels = [0.1, 0.2, 0.3, 0.4, 0.45]
    confidence_contour = plt.contourf(xx1, xx2, Z_conf, levels=levels, 
                                     alpha=0.6, cmap='Blues_r')
    plt.colorbar(confidence_contour, label='Uncertainty (closer to 0 = higher uncertainty)')
    
    # Plot decision boundary
    plt.contour(xx1, xx2, Z_prob, [0.5], linewidths=3, colors='black', 
                linestyles='solid', label='Decision Boundary')
    
    # Add confidence region contours
    plt.contour(xx1, xx2, Z_prob, [0.1, 0.25, 0.75, 0.9], 
                colors=['red', 'orange', 'orange', 'red'], 
                linestyles='dashed', linewidths=2)
    
    # Add annotation for confidence levels
    plt.annotate('High certainty\nbenign', xy=(20, 15), xytext=(10, 15),
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.2))
    
    plt.annotate('High certainty\nmalignant', xy=(70, 70), xytext=(80, 70),
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.2))
    
    plt.annotate('Uncertain\nregion', xy=(50, 37), xytext=(50, 37),
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=100, 
                edgecolors='k', label='Benign')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', s=100, 
                linewidth=2, label='Malignant')

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel('Age (years)')
    plt.ylabel('Tumor Size (mm)')
    plt.title('Decision Regions with Confidence Levels')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_visualization(X, y, save_path):
    """Create a scatter plot visualization of the dataset"""
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot with different markers and colors for each class
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=100, 
                edgecolors='k', label='Benign (y=0)')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', s=100, 
                linewidth=2, label='Malignant (y=1)')
    
    # Add labels and title
    plt.xlabel('Age (years)')
    plt.ylabel('Tumor Size (mm)')
    plt.title('Tumor Classification Dataset')
    
    # Add a legend and grid
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    save_dir = create_output_dir()
    
    # Load dataset
    df = load_dataset()
    print("\n" + "="*80)
    print("DATASET:")
    print("="*80)
    print(df)
    print("\n")
    
    # Extract features and target
    X = df[['Age', 'Tumor_Size']].values
    y = df['Malignant'].values
    m = len(y)  # number of training examples
    
    # Print details about the extracted data
    print(f"Number of examples (m): {m}")
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Benign examples (y=0): {np.sum(y == 0)}")
    print(f"Malignant examples (y=1): {np.sum(y == 1)}")
    print("\n")
    
    # Add intercept term
    X_with_intercept = np.c_[np.ones((m, 1)), X]
    print("Feature matrix with intercept term:")
    print(X_with_intercept)
    print("\n")
    
    # Plot sigmoid function
    print("="*80)
    print("SIGMOID FUNCTION")
    print("="*80)
    print("The sigmoid function is defined as: g(z) = 1 / (1 + e^(-z))")
    print("Properties of the sigmoid function:")
    print("  - When z = 0: g(0) = 1/(1+e^0) = 1/2 = 0.5")
    print("  - As z → +∞: g(z) → 1")
    print("  - As z → -∞: g(z) → 0")
    print("\n")
    
    plot_sigmoid_function(save_dir)
    
    # Visualize the dataset
    plot_dataset_visualization(
        X, y,
        os.path.join(save_dir, 'dataset_visualization_alt.png')
    )
    
    # Task 1: Calculate the initial cost
    print("="*80)
    print("TASK 1: INITIAL COST CALCULATION")
    print("="*80)
    initial_theta = np.zeros(3)
    print(f"Initial parameters: θ = {initial_theta}")
    
    # Step-by-step calculation of initial cost
    print("\nStep-by-step calculation of initial cost J(θ):")
    print("-"*60)
    
    # Create a DataFrame to show the calculation
    calculation_df = pd.DataFrame({
        'x₁=age': df['Age'],
        'x₂=size': df['Tumor_Size'],
        'y': df['Malignant'],
    })
    
    # Calculate hypothesis values h(x) for each example
    h_values = np.zeros(m)
    for i in range(m):
        z_i = X_with_intercept[i] @ initial_theta
        h_i = sigmoid(z_i)
        h_values[i] = h_i
        print(f"Example {i+1}: z_{i+1} = θᵀx_{i+1} = {initial_theta} @ {X_with_intercept[i]} = {z_i}")
        print(f"             h(x_{i+1}) = g(z_{i+1}) = g({z_i}) = {h_i}")
    
    calculation_df['h(x)'] = h_values
    
    # Calculate cost components
    y_log_h = np.zeros(m)
    one_minus_y_log_one_minus_h = np.zeros(m)
    
    for i in range(m):
        if y[i] == 1:
            y_log_h[i] = y[i] * np.log(h_values[i])
            print(f"Example {i+1} (y_{i+1}=1): y*log(h(x)) = {y[i]}*log({h_values[i]:.4f}) = {y_log_h[i]:.5f}")
        else:
            one_minus_y_log_one_minus_h[i] = (1-y[i]) * np.log(1-h_values[i])
            print(f"Example {i+1} (y_{i+1}=0): (1-y)*log(1-h(x)) = {1-y[i]}*log(1-{h_values[i]:.4f}) = {one_minus_y_log_one_minus_h[i]:.5f}")
    
    # Update DataFrame for display
    calculation_df['y*log(h(x))'] = y_log_h
    calculation_df['(1-y)*log(1-h(x))'] = one_minus_y_log_one_minus_h
    
    # Fill empty values for better display
    calculation_df['y*log(h(x))'] = calculation_df.apply(
        lambda row: f"{row['y*log(h(x))']:.5f}" if row['y'] == 1 else "", axis=1
    )
    calculation_df['(1-y)*log(1-h(x))'] = calculation_df.apply(
        lambda row: f"{row['(1-y)*log(1-h(x))']:.5f}" if row['y'] == 0 else "", axis=1
    )
    
    # Display the calculation table
    print("\nCost function calculation table:")
    print(calculation_df.to_string(index=False))
    
    # Calculate the total cost
    total_cost = np.sum(y_log_h) + np.sum(one_minus_y_log_one_minus_h)
    print(f"\nSum of cost terms (raw log-likelihood): {total_cost:.5f}")
    
    # Show standard negative log-likelihood cost
    std_cost = -total_cost / m
    print(f"Standard cost (negative log-likelihood): {std_cost:.5f}")
    
    # Update the double-check with compute_cost function
    initial_cost = compute_cost(X_with_intercept, y, initial_theta)
    print(f"\nVerification (standard negative log-likelihood): J(θ) = {initial_cost:.5f}")
    print("\n")
    
    # Task 2: Gradient Descent Iterations
    print("="*80)
    print("TASK 2: GRADIENT DESCENT ITERATIONS")
    print("="*80)
    
    learning_rate = 0.01
    num_iterations = 2
    
    print(f"Learning rate α: {learning_rate}")
    print(f"Initial parameters θ: {initial_theta}")
    print(f"Initial cost J(θ): {initial_cost:.4f}")
    
    theta, theta_history, cost_history = perform_gradient_descent(
        X_with_intercept, y, initial_theta, learning_rate, num_iterations
    )
    
    # Plot cost over iterations
    plot_cost_history(
        cost_history, 
        os.path.join(save_dir, 'gradient_descent_cost_alt.png')
    )
    
    # Task 3: Stochastic Gradient Descent
    print("="*80)
    print("TASK 3: STOCHASTIC GRADIENT DESCENT")
    print("="*80)
    
    sgd_learning_rate = 0.1
    sgd_iterations = 2
    
    print(f"Learning rate α: {sgd_learning_rate}")
    print(f"Initial parameters θ: {initial_theta}")
    print(f"Initial cost J(θ): {initial_cost:.4f}")
    
    sgd_theta, sgd_theta_history, sgd_cost_history = perform_sgd(
        X_with_intercept, y, initial_theta, sgd_learning_rate, sgd_iterations
    )
    
    # Task 4: Decision Boundary Explanation
    print("="*80)
    print("TASK 4: DECISION BOUNDARY EXPLANATION")
    print("="*80)
    print("The decision boundary equation θᵀx = 0 represents the set of points where P(y=1|x) = 0.5.")
    print("For the logistic regression model, P(y=1|x) = 1/(1+e^(-θᵀx)).")
    print("When θᵀx = 0, P(y=1|x) = 1/(1+e^0) = 1/2 = 0.5.")
    print("Geometrically, this creates a boundary in the feature space that separates the")
    print("regions where the model predicts class 0 (below 0.5 probability) and class 1 (above 0.5 probability).")
    print("For a model with two features, this boundary is a line. With more features, it becomes a hyperplane.")
    print("\n")
    
    # Task 5: Decision Boundary with Final Parameters
    print("="*80)
    print("TASK 5: DECISION BOUNDARY WITH FINAL PARAMETERS")
    print("="*80)
    final_theta = np.array([-136.95, 1.1, 2.2])
    print(f"Final optimized parameters: θ = {final_theta}")
    
    # Calculate the decision boundary equation: θ0 + θ1*x1 + θ2*x2 = 0
    print("\nStep-by-step derivation of decision boundary equation:")
    print("1. The decision boundary is defined by the equation: θ₀ + θ₁*Age + θ₂*Tumor_Size = 0")
    print(f"2. Substituting our parameters: {final_theta[0]:.2f} + {final_theta[1]:.2f}*Age + {final_theta[2]:.2f}*Tumor_Size = 0")
    print(f"3. Solving for Tumor_Size: {final_theta[2]:.2f}*Tumor_Size = -({final_theta[0]:.2f} + {final_theta[1]:.2f}*Age)")
    print(f"4. Dividing both sides by {final_theta[2]:.2f}: Tumor_Size = -({final_theta[0]:.2f} + {final_theta[1]:.2f}*Age)/{final_theta[2]:.2f}")
    
    slope = -final_theta[1]/final_theta[2]
    intercept = -final_theta[0]/final_theta[2]
    print(f"5. Simplifying: Tumor_Size = {intercept:.2f} {slope:.2f}*Age")
    print("\n")
    
    # Task 6: Predict for a new patient
    print("="*80)
    print("TASK 6: PREDICTION FOR NEW PATIENT")
    print("="*80)
    new_patient_age = 50
    new_patient_tumor_size = 30
    new_patient_features = np.array([new_patient_age, new_patient_tumor_size])
    print(f"New patient data: [Age, Tumor_Size] = {new_patient_features}")
    
    # Make prediction
    new_z, new_probability, new_prediction = predict_new_patient(
        final_theta, new_patient_features
    )
    
    # Plot the sigmoid function with new patient's z value
    plot_new_patient_prediction(
        new_z,
        os.path.join(save_dir, 'new_patient_prediction_alt.png')
    )
    
    # Plot probability visualization (contour and 3D)
    plot_probability_visualization(
        X, y, final_theta,
        [new_patient_age, new_patient_tumor_size],
        os.path.join(save_dir, 'probability_visualization_alt.png')
    )
    
    # Plot 3D probability surface with training points
    plot_probability_surface(
        X, y, final_theta,
        [new_patient_age, new_patient_tumor_size], 
        new_probability,
        os.path.join(save_dir, 'probability_surface_alt.png')
    )
    
    # Plot decision boundary with new patient
    plot_decision_boundary(
        X, y, final_theta, 
        [new_patient_age, new_patient_tumor_size],
        os.path.join(save_dir, 'decision_boundary_alt.png')
    )
    
    # Add the new visualization showing decision regions with confidence levels
    plot_decision_regions_with_confidence(
        X, y, final_theta,
        os.path.join(save_dir, 'decision_regions_confidence_alt.png')
    )
    
    # Task 7: Interpretation of coefficients
    print("="*80)
    print("TASK 7: INTERPRETATION OF COEFFICIENTS")
    print("="*80)
    print(f"θ₁ (Age coefficient) = {final_theta[1]:.2f}")
    print(f"θ₂ (Tumor Size coefficient) = {final_theta[2]:.2f}")
    
    print("\nInterpretation in terms of log-odds:")
    print(f"- For each additional year of age, the log-odds of malignancy increase by {final_theta[1]:.2f},")
    print("  holding tumor size constant.")
    print(f"- For each additional mm in tumor size, the log-odds of malignancy increase by {final_theta[2]:.2f},")
    print("  holding age constant.")
    
    print("\nInterpretation in terms of odds ratios:")
    print(f"- The odds ratio for a 1-year increase in age is e^{final_theta[1]:.2f} = {np.exp(final_theta[1]):.2f}.")
    print("  This means the odds of malignancy are multiplied by this factor for each year of age.")
    print(f"- The odds ratio for a 1-mm increase in tumor size is e^{final_theta[2]:.2f} = {np.exp(final_theta[2]):.2f}.")
    print("  This means the odds of malignancy are multiplied by this factor for each mm in tumor size.")
    
    print("\nRelative importance:")
    print(f"Since θ₂ ({final_theta[2]:.2f}) > θ₁ ({final_theta[1]:.2f}), tumor size has a stronger effect")
    print(f"on the probability of malignancy than age. The effect of tumor size is approximately")
    print(f"{final_theta[2]/final_theta[1]:.2f} times stronger than the effect of age.")
    print("\n")
    
    # Task 8: Effect of learning rate
    print("="*80)
    print("TASK 8: EFFECT OF LEARNING RATE")
    print("="*80)
    print("Increasing the learning rate:")
    print("- Advantages:")
    print("  * Faster convergence if the rate is well-tuned")
    print("  * Requires fewer iterations to reach the optimum")
    print("  * Can escape local minima more easily")
    print("- Disadvantages:")
    print("  * Risk of overshooting the minimum and divergence if too large")
    print("  * May oscillate around the minimum without reaching it")
    print("  * Can cause numerical instability")
    print("\n")
    
    print("Decreasing the learning rate:")
    print("- Advantages:")
    print("  * More stable and reliable convergence")
    print("  * Less sensitive to noise in the data")
    print("  * Better precision near the optimum")
    print("- Disadvantages:")
    print("  * Slower progress, requiring more iterations")
    print("  * May get stuck in local minima or plateau regions")
    print("  * Very small rates may make progress imperceptibly slow")
    print("\n")
    
    # Update learning_rate_effect parameters to ensure convergence 
    # (use smaller learning rates to avoid oscillation)
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    plot_learning_rate_effect(
        X_with_intercept, y, initial_theta,
        os.path.join(save_dir, 'learning_rate_effect_alt.png'),
        learning_rates=learning_rates,
        num_iterations=50
    )
    
    # Plot cost function surface
    plot_cost_function_surface(
        X_with_intercept, y, -10,  # Fix theta0 at -10 for visualization
        os.path.join(save_dir, 'cost_function_surface_alt.png')
    )
    
    print("\nAll visualizations and calculations completed. Images saved to:", save_dir)

if __name__ == "__main__":
    main() 