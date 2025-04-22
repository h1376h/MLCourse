import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import os

# Create output directory for images if it doesn't exist
os.makedirs('../Images/L2_1_Quiz_33', exist_ok=True)

def print_step_header(step_number, step_title):
    """Print a formatted header for each step in the solution."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}")

def print_substep(substep_title):
    """Print a formatted header for sub-steps."""
    print(f"\n{'-' * 40}")
    print(f"{substep_title}")
    print(f"{'-' * 40}")

def save_figure(fig, filename):
    """Save a figure to the output directory."""
    filepath = f'../Images/L2_1_Quiz_33/{filename}'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    return filepath

# Given data
# Data is organized as [x1, x2] for each class
class1_data = np.array([[2.1, -2.5], 
                         [1.1, -3.1], 
                         [1.4, -2.1]])

class2_data = np.array([[3.3, -1.8], 
                         [4.4, 6.5], 
                         [3.4, 5.8]])

class3_data = np.array([[4.5, 7.2], 
                         [4.1, 5.65], 
                         [-1.3, -2.3], 
                         [-3.2, -4.5], 
                         [-3.2, -4.5], 
                         [-2.1, -3.3]])

# New point to classify
new_point = np.array([2.0, 1.0])

# Prior probabilities
priors = {
    'omega1': 0.4,
    'omega2': 0.35,
    'omega3': 0.25
}

def plot_data(data_dict, new_point=None, title="Data Visualization"):
    """Plot the data points from each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, (class_name, data) in enumerate(data_dict.items()):
        ax.scatter(data[:, 0], data[:, 1], c=colors[i], marker=markers[i], 
                   s=100, label=f'{class_name}')
    
    # Plot the new point if provided
    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], c='black', marker='*', 
                   s=200, label='New Point')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)
    
    return fig, ax

def plot_decision_boundaries(data_dict, mean_vectors, new_point, prediction):
    """
    Plot decision boundaries for the discriminant functions.
    This uses a grid-based approach to evaluate the discriminant at each point.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract all x and y values to determine plot boundaries
    all_x1 = []
    all_x2 = []
    for class_name, data in data_dict.items():
        all_x1.extend(data[:, 0])
        all_x2.extend(data[:, 1])
    
    # Add margins to the boundaries
    margin = 2
    x1_min, x1_max = min(all_x1) - margin, max(all_x1) + margin
    x2_min, x2_max = min(all_x2) - margin, max(all_x2) + margin
    
    # Create a meshgrid
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1_min, x1_max, 500),
        np.linspace(x2_min, x2_max, 500)
    )
    
    # Calculate the discriminant function value for each point in the grid
    discriminant_values = np.zeros((3, x1_grid.shape[0], x1_grid.shape[1]))
    
    for i, class_name in enumerate(['omega1', 'omega2', 'omega3']):
        # Calculate discriminant for each point
        for j in range(x1_grid.shape[0]):
            for k in range(x1_grid.shape[1]):
                grid_point = np.array([x1_grid[j, k], x2_grid[j, k]])
                # Using the definition g_k(x) = -1/2 * ||x - mu_k||^2 + ln P(omega_k)
                discriminant_values[i, j, k] = (-0.5 * np.sum((grid_point - mean_vectors[class_name])**2) + 
                                              np.log(priors[class_name]))
    
    # Get the class with highest discriminant value at each point
    decision_regions = np.argmax(discriminant_values, axis=0)
    
    # Create a colormap
    cmap = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    
    # Plot the decision regions
    ax.pcolormesh(x1_grid, x2_grid, decision_regions, cmap=cmap, alpha=0.3)
    
    # Plot the data points
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    class_names = list(data_dict.keys())
    
    for i, (class_name, data) in enumerate(data_dict.items()):
        ax.scatter(data[:, 0], data[:, 1], c=colors[i], marker=markers[i], 
                   s=100, label=f'{class_name}')
    
    # Plot the new point
    ax.scatter(new_point[0], new_point[1], c='black', marker='*', 
               s=200, label=f'New Point (Class {prediction})')
    
    # Plot the mean vectors
    for i, (class_name, mean) in enumerate(mean_vectors.items()):
        ax.scatter(mean[0], mean[1], c=colors[i], marker='X', 
                   s=200, edgecolors='black', label=f'Mean {class_name}')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Decision Boundaries and Classification Result', fontsize=14)
    
    return fig, ax

def plot_discriminant_functions(discriminant_values, new_point_x1=None, new_point_x2=None):
    """Plot the discriminant functions across x1 and x2 space."""
    # Generate a range of x1 and x2 values
    x1_range = np.linspace(-4, 6, 100)
    x2_range = np.linspace(-6, 8, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Initialize discriminant function values
    G1 = np.zeros_like(X1)
    G2 = np.zeros_like(X1)
    G3 = np.zeros_like(X1)
    
    # Calculate discriminant function values at each point
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x = np.array([X1[j, i], X2[j, i]])
            # Using the definition g_k(x) = -1/2 * ||x - mu_k||^2 + ln P(omega_k)
            G1[j, i] = (-0.5 * np.sum((x - discriminant_values['omega1']['mean'])**2) + 
                       np.log(priors['omega1']))
            G2[j, i] = (-0.5 * np.sum((x - discriminant_values['omega2']['mean'])**2) + 
                       np.log(priors['omega2']))
            G3[j, i] = (-0.5 * np.sum((x - discriminant_values['omega3']['mean'])**2) + 
                       np.log(priors['omega3']))
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the discriminant functions
    surf1 = ax.plot_surface(X1, X2, G1, cmap='Blues', alpha=0.7, label='g1')
    surf2 = ax.plot_surface(X1, X2, G2, cmap='Greens', alpha=0.7, label='g2')
    surf3 = ax.plot_surface(X1, X2, G3, cmap='Reds', alpha=0.7, label='g3')
    
    # Add colors to surfaces
    surf1._facecolors2d = plt.cm.Blues(0.7)
    surf2._facecolors2d = plt.cm.Greens(0.7)
    surf3._facecolors2d = plt.cm.Reds(0.7)
    
    # Plot the new point
    if new_point_x1 is not None and new_point_x2 is not None:
        # Calculate the discriminant values at the new point
        x = np.array([new_point_x1, new_point_x2])
        g1_val = (-0.5 * np.sum((x - discriminant_values['omega1']['mean'])**2) + 
                 np.log(priors['omega1']))
        g2_val = (-0.5 * np.sum((x - discriminant_values['omega2']['mean'])**2) + 
                 np.log(priors['omega2']))
        g3_val = (-0.5 * np.sum((x - discriminant_values['omega3']['mean'])**2) + 
                 np.log(priors['omega3']))
        
        # Plot the point on each surface
        ax.scatter([new_point_x1], [new_point_x2], [g1_val], c='blue', s=100, marker='*')
        ax.scatter([new_point_x1], [new_point_x2], [g2_val], c='green', s=100, marker='*')
        ax.scatter([new_point_x1], [new_point_x2], [g3_val], c='red', s=100, marker='*')
        
        # Plot vertical lines to show the discriminant values
        ax.plot([new_point_x1, new_point_x1], [new_point_x2, new_point_x2], 
                [min(G1.min(), G2.min(), G3.min()), g1_val], 'b--')
        ax.plot([new_point_x1, new_point_x1], [new_point_x2, new_point_x2], 
                [min(G1.min(), G2.min(), G3.min()), g2_val], 'g--')
        ax.plot([new_point_x1, new_point_x1], [new_point_x2, new_point_x2], 
                [min(G1.min(), G2.min(), G3.min()), g3_val], 'r--')
    
    # Customize the plot
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_zlabel('g(X)', fontsize=12)
    ax.set_title('Discriminant Functions', fontsize=14)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4),
        Line2D([0], [0], color='green', lw=4),
        Line2D([0], [0], color='red', lw=4)
    ]
    ax.legend(custom_lines, ['g1(X)', 'g2(X)', 'g3(X)'])
    
    # Adjust viewing angle
    ax.view_init(30, 220)
    
    return fig, ax

def main():
    # STEP 1: Organize the data and calculate overall mean vector
    print_step_header(1, "Organizing Data and Computing Mean Vectors")
    
    # Create a dictionary to store the data for each class
    data_dict = {
        'omega1': class1_data,
        'omega2': class2_data,
        'omega3': class3_data
    }
    
    # Visualize the data
    print_substep("Visualizing the data")
    fig, ax = plot_data(data_dict, new_point=new_point, title="Dataset Visualization")
    save_figure(fig, "step1a_data_visualization.png")
    plt.close(fig)
    
    # Calculate the overall mean vector
    print_substep("Computing the overall mean vector")
    all_data = np.vstack([class1_data, class2_data, class3_data])
    
    # Show detailed calculation for overall mean
    print("Detailed calculation for the overall mean vector:")
    print(f"Data points from all classes:")
    for i, point in enumerate(all_data):
        print(f"X_{i+1} = [{point[0]}, {point[1]}]")
    
    print(f"\nTotal number of data points: N = {len(all_data)}")
    print(f"Sum of all x1 values: {np.sum(all_data[:, 0])}")
    print(f"Sum of all x2 values: {np.sum(all_data[:, 1])}")
    
    overall_mean = np.mean(all_data, axis=0)
    print(f"\nOverall mean vector μ = [Sum(x1)/N, Sum(x2)/N] = [{np.sum(all_data[:, 0])}/{len(all_data)}, {np.sum(all_data[:, 1])}/{len(all_data)}] = {overall_mean}")
    
    # Calculate class-specific mean vectors
    print_substep("Computing class-specific mean vectors")
    mean_vectors = {}
    
    for class_name, data in data_dict.items():
        print(f"\nDetailed calculation for the mean vector of {class_name}:")
        print(f"Data points in {class_name}:")
        for i, point in enumerate(data):
            print(f"X_{i+1}^({class_name[-1]}) = [{point[0]}, {point[1]}]")
        
        print(f"\nNumber of data points in {class_name}: N_{class_name[-1]} = {len(data)}")
        print(f"Sum of x1 values in {class_name}: {np.sum(data[:, 0])}")
        print(f"Sum of x2 values in {class_name}: {np.sum(data[:, 1])}")
        
        mean_vectors[class_name] = np.mean(data, axis=0)
        print(f"\nMean vector for {class_name} (μ_{class_name[-1]}) = [Sum(x1)/N_{class_name[-1]}, Sum(x2)/N_{class_name[-1]}] = [{np.sum(data[:, 0])}/{len(data)}, {np.sum(data[:, 1])}/{len(data)}] = {mean_vectors[class_name]}")
    
    # Plot the mean vectors
    print_substep("Visualizing mean vectors")
    fig, ax = plot_data(data_dict, title="Dataset with Mean Vectors")
    
    # Add mean vectors to the plot
    colors = ['blue', 'green', 'red']
    for i, (class_name, mean) in enumerate(mean_vectors.items()):
        ax.scatter(mean[0], mean[1], c=colors[i], marker='X', 
                   s=200, edgecolors='black', label=f'Mean {class_name}')
    
    # Add overall mean
    ax.scatter(overall_mean[0], overall_mean[1], c='black', marker='X', 
               s=200, label='Overall Mean')
    
    ax.legend()
    save_figure(fig, "step1b_mean_vectors.png")
    plt.close(fig)
    
    # STEP 2: Calculate the within-class scatter matrix for class omega1
    print_step_header(2, "Computing Within-Class Scatter Matrix for Class omega1")
    
    within_class_scatter_omega1 = np.zeros((2, 2))
    
    # Calculate the within-class scatter matrix for omega1
    print_substep("Computing S_w^(1)")
    print(f"Within-class scatter matrix formula: S_w^(1) = Sum_i[(X_i^(1) - μ_1)(X_i^(1) - μ_1)^T]")
    print(f"Where:\n- X_i^(1) is each data point in class omega1\n- μ_1 is the mean vector for class omega1: {mean_vectors['omega1']}")
    
    for i, data_point in enumerate(class1_data):
        # Compute (x - μ_1)
        diff = data_point - mean_vectors['omega1']
        # Compute (x - μ_1)(x - μ_1)^T
        outer_product = np.outer(diff, diff)
        
        print(f"\nCalculation for data point X_{i+1}^(1) = [{data_point[0]}, {data_point[1]}]:")
        print(f"Difference from mean: X_{i+1}^(1) - μ_1 = [{data_point[0]}, {data_point[1]}] - [{mean_vectors['omega1'][0]}, {mean_vectors['omega1'][1]}] = {diff}")
        print(f"Detailed outer product calculation:")
        print(f"[{diff[0]}, {diff[1]}] × [{diff[0]}, {diff[1]}]^T =")
        print(f"[{diff[0]} × {diff[0]}, {diff[0]} × {diff[1]}]")
        print(f"[{diff[1]} × {diff[0]}, {diff[1]} × {diff[1]}] =")
        print(f"[{diff[0] * diff[0]}, {diff[0] * diff[1]}]")
        print(f"[{diff[1] * diff[0]}, {diff[1] * diff[1]}] =")
        print(f"Outer product: \n{outer_product}")
        
        # Add to the scatter matrix
        within_class_scatter_omega1 += outer_product
        print(f"Running sum of scatter matrix after this point: \n{within_class_scatter_omega1}")
    
    print(f"\nFinal within-class scatter matrix for omega1: \n{within_class_scatter_omega1}")
    print(f"S_w^(1) = [" +
          f"[{within_class_scatter_omega1[0, 0]:.8f}, {within_class_scatter_omega1[0, 1]:.8f}], " +
          f"[{within_class_scatter_omega1[1, 0]:.8f}, {within_class_scatter_omega1[1, 1]:.8f}]]")
    
    # STEP 3: Compute discriminant functions
    print_step_header(3, "Computing Discriminant Functions")
    
    # We'll use the identity matrix as the covariance matrix
    identity_matrix = np.eye(2)
    print("Assuming equal covariance matrices for all classes:")
    print(f"Σ_1 = Σ_2 = Σ_3 = I = [[1, 0], [0, 1]]")
    
    # Store discriminant function information
    discriminant_info = {
        'omega1': {'mean': mean_vectors['omega1'], 'prior': priors['omega1']},
        'omega2': {'mean': mean_vectors['omega2'], 'prior': priors['omega2']},
        'omega3': {'mean': mean_vectors['omega3'], 'prior': priors['omega3']}
    }
    
    print_substep("Defining discriminant functions")
    print("For normal distributions with equal covariance matrices, the discriminant function is:")
    print("g_k(X) = -0.5(X - μ_k)^T Σ^(-1) (X - μ_k) + ln(P(ω_k))")
    print("With Σ = I, this simplifies to:")
    print("g_k(X) = -0.5(X - μ_k)^T(X - μ_k) + ln(P(ω_k))")
    print("g_k(X) = -0.5||X - μ_k||^2 + ln(P(ω_k))")
    print("Where ||X - μ_k||^2 is the squared Euclidean distance between the point X and the class mean μ_k")
    
    for class_name, info in discriminant_info.items():
        print(f"\nDiscriminant function for {class_name}:")
        print(f"g_{class_name[-1]}(X) = -0.5(X - μ_{class_name[-1]})^T(X - μ_{class_name[-1]}) + ln(P({class_name}))")
        print(f"g_{class_name[-1]}(X) = -0.5||X - [{info['mean'][0]:.2f}, {info['mean'][1]:.2f}]||^2 + ln({info['prior']:.2f})")
        
        # Expand the discriminant function algebraically
        print(f"Expanding the discriminant function:")
        mean_x1, mean_x2 = info['mean'][0], info['mean'][1]
        print(f"g_{class_name[-1]}(X) = -0.5[(x1 - {mean_x1:.2f})^2 + (x2 - {mean_x2:.2f})^2] + {np.log(info['prior']):.4f}")
        
    # Visualize the discriminant functions
    print_substep("Visualizing discriminant functions")
    fig, ax = plot_discriminant_functions(discriminant_info, new_point[0], new_point[1])
    save_figure(fig, "step3a_discriminant_functions.png")
    plt.close(fig)
    
    # STEP 4: Classify the new point
    print_step_header(4, "Classifying the New Point")
    
    # Calculate discriminant function values for the new point
    g_values = {}
    
    print_substep(f"Evaluating discriminant functions for X = [{new_point[0]}, {new_point[1]}]")
    print(f"For the new point X = [{new_point[0]}, {new_point[1]}], we evaluate each discriminant function:")
    
    for class_name, info in discriminant_info.items():
        print(f"\nCalculating g_{class_name[-1]}(X) for class {class_name}:")
        
        # Calculate (x - μ)
        diff = new_point - info['mean']
        
        # Show the difference calculation
        print(f"X - μ_{class_name[-1]} = [{new_point[0]}, {new_point[1]}] - [{info['mean'][0]}, {info['mean'][1]}] = {diff}")
        
        # Calculate (x - μ)^T(x - μ) = ||x - μ||^2
        squared_distance = np.sum(diff**2)
        
        # Show the squared distance calculation
        print(f"||X - μ_{class_name[-1]}||^2 = ({diff[0]})^2 + ({diff[1]})^2 = {diff[0]**2} + {diff[1]**2} = {squared_distance}")
        
        # Calculate the discriminant function value
        g_value = -0.5 * squared_distance + np.log(info['prior'])
        g_values[class_name] = g_value
        
        # Show the discriminant function calculation
        print(f"g_{class_name[-1]}(X) = -0.5 * {squared_distance:.4f} + ln({info['prior']:.2f})")
        print(f"g_{class_name[-1]}(X) = {-0.5 * squared_distance:.4f} + {np.log(info['prior']):.4f} = {g_value:.4f}")
    
    # Find the class with the highest discriminant value
    print_substep("Determining the class with the highest discriminant value")
    prediction = max(g_values, key=g_values.get)
    
    print("Comparing the discriminant values:")
    for class_name, g_value in g_values.items():
        print(f"g_{class_name[-1]}(X) = {g_value:.4f}")
    
    print(f"\nSince g_{prediction[-1]}(X) has the highest value among all discriminant functions,")
    print(f"the new point is classified as {prediction} with g-value: {g_values[prediction]:.4f}")
    
    # Visualize the decision boundary and classification result
    print_substep("Visualizing decision boundary and classification result")
    fig, ax = plot_decision_boundaries(data_dict, mean_vectors, new_point, prediction)
    save_figure(fig, "step4a_decision_boundary.png")
    plt.close(fig)
    
    print("\nAll calculations complete!")

if __name__ == "__main__":
    main() 