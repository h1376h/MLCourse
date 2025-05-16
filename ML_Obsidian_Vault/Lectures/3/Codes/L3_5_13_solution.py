import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)  # For reproducibility

def statement1_normal_equations_complexity():
    """
    Statement 1: The computational complexity of solving linear regression using normal equations
    is O(n^3), where n is the number of features.
    
    This function demonstrates the time complexity of normal equations by measuring execution time
    as the number of features increases.
    """
    print("\n==== Statement 1: Computational Complexity of Normal Equations ====")
    print("Statement: The computational complexity of solving linear regression using normal equations is O(n^3), where n is the number of features.")
    
    # List of feature dimensions to test
    feature_dims = [10, 50, 100, 200, 400, 800]
    
    # Number of samples (fixed)
    n_samples = 1000
    
    # Store execution times for different operations
    times_total = []  # Total execution time
    times_xtx = []    # X^T * X time
    times_inv = []    # Matrix inversion time
    times_xty = []    # X^T * y time
    times_solution = []  # Final multiplication time
    
    # Measure time for each feature dimension
    for n_features in feature_dims:
        print(f"Testing with {n_features} features...")
        
        # Generate synthetic regression data
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)
        
        # Time the normal equations calculation with breakdown
        start_time_total = time.time()
        
        # Step 1: X^T * X
        start_time = time.time()
        XTX = X.T @ X
        end_time = time.time()
        times_xtx.append(end_time - start_time)
        
        # Step 2: Matrix inversion (X^T X)^(-1)
        start_time = time.time()
        XTX_inv = np.linalg.inv(XTX)  # This is O(n^3)
        end_time = time.time()
        times_inv.append(end_time - start_time)
        
        # Step 3: X^T * y
        start_time = time.time()
        XTy = X.T @ y
        end_time = time.time()
        times_xty.append(end_time - start_time)
        
        # Step 4: Final solution
        start_time = time.time()
        theta = XTX_inv @ XTy
        end_time = time.time()
        times_solution.append(end_time - start_time)
        
        end_time_total = time.time()
        execution_time = end_time_total - start_time_total
        times_total.append(execution_time)
        
        print(f"Execution time: {execution_time:.6f} seconds")
    
    # Create a visualization of the complexity
    plt.figure(figsize=(10, 6))
    
    # Plot execution time vs. number of features
    plt.plot(feature_dims, times_total, 'o-', linewidth=2, markersize=10, label='Total time')
    
    # Plot a cubic function (n^3) for comparison
    # Normalize to match the scale of measured times
    n_values = np.array(feature_dims)
    cubic = n_values**3
    scaling_factor = times_total[-1] / cubic[-1]
    plt.plot(n_values, scaling_factor * cubic, '--', 
             label='O(n³) reference', alpha=0.7)
    
    plt.xlabel('Number of features (n)')
    plt.ylabel('Execution time (seconds)')
    plt.title('Execution Time of Normal Equations vs. Number of Features')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement1_complexity.png'), dpi=300, bbox_inches='tight')
    
    # Create a new figure to show the breakdown of operations
    plt.figure(figsize=(12, 8))
    
    # Plot time for each operation
    plt.plot(feature_dims, times_xtx, 'o-', linewidth=2, label='X^T * X (O(n²m))')
    plt.plot(feature_dims, times_inv, 'o-', linewidth=2, label='Matrix inversion (O(n³))')
    plt.plot(feature_dims, times_xty, 'o-', linewidth=2, label='X^T * y (O(nm))')
    plt.plot(feature_dims, times_solution, 'o-', linewidth=2, label='Final solution (O(n²))')
    
    # Plot reference complexity functions
    n_squared = n_values**2
    scaling_n2 = max(times_xtx) / n_squared[-1]
    scaling_n3 = max(times_inv) / cubic[-1]
    
    plt.plot(n_values, scaling_n2 * n_squared, '--', alpha=0.5, label='O(n²) reference')
    plt.plot(n_values, scaling_n3 * cubic, '--', alpha=0.5, label='O(n³) reference')
    
    plt.xlabel('Number of features (n)')
    plt.ylabel('Execution time (seconds)')
    plt.title('Breakdown of Normal Equations Operations by Computational Complexity')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Use log scale to better visualize differences
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement1_complexity_breakdown.png'), dpi=300, bbox_inches='tight')
    
    # Create a bar chart to compare average percentage of time spent on each operation
    plt.figure(figsize=(10, 6))
    
    # Calculate percentages
    percentages = [
        np.mean([t/total for t, total in zip(times_xtx, times_total)]) * 100,
        np.mean([t/total for t, total in zip(times_inv, times_total)]) * 100,
        np.mean([t/total for t, total in zip(times_xty, times_total)]) * 100,
        np.mean([t/total for t, total in zip(times_solution, times_total)]) * 100
    ]
    
    # Create bar chart
    operations = ['X^T * X\n(O(n²m))', 'Matrix inversion\n(O(n³))', 'X^T * y\n(O(nm))', 'Final solution\n(O(n²))']
    colors = ['skyblue', 'salmon', 'lightgreen', 'lightyellow']
    
    plt.bar(operations, percentages, color=colors)
    plt.axhline(y=np.max(percentages), color='red', linestyle='--', alpha=0.7, 
                label=f'Dominant operation: {operations[np.argmax(percentages)]}')
    
    plt.ylabel('Percentage of total time (%)')
    plt.title('Average Percentage of Time Spent on Each Operation')
    plt.ylim(0, np.max(percentages) * 1.2)  # Add some headroom
    plt.grid(True, axis='y')
    plt.legend()
    
    # Add percentage labels on top of bars
    for i, v in enumerate(percentages):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement1_time_distribution.png'), dpi=300, bbox_inches='tight')
    
    # Determine if the statement is true or false
    conclusion = """
    CONCLUSION: TRUE
    
    The computational complexity of solving linear regression using normal equations is indeed O(n³), 
    where n is the number of features. This is primarily due to the matrix inversion operation (X^T X)^(-1),
    which has a time complexity of O(n³) for an n×n matrix.
    
    The plot shows how the execution time increases with the number of features, closely following
    a cubic growth pattern, which confirms the O(n³) complexity. This can become computationally
    expensive when dealing with datasets that have a large number of features.
    """
    
    print(conclusion)
    return conclusion

def statement2_sgd_gradient_computation():
    """
    Statement 2: Stochastic gradient descent uses all training examples to compute 
    the gradient in each iteration.
    
    This function implements both batch gradient descent and stochastic gradient descent
    to demonstrate the difference in how they compute gradients.
    """
    print("\n==== Statement 2: Stochastic Gradient Descent and Gradient Computation ====")
    print("Statement: Stochastic gradient descent uses all training examples to compute the gradient in each iteration.")
    
    # Generate synthetic regression data
    n_samples, n_features = 1000, 5
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Learning rate and number of iterations
    learning_rate = 0.01
    n_iterations = 100
    
    # Implement batch gradient descent (uses all examples)
    def batch_gradient_descent(X, y, learning_rate, n_iterations):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        examples_used_history = []
        
        for iteration in range(n_iterations):
            # Compute predictions
            predictions = X @ theta
            
            # Compute errors
            errors = predictions - y
            
            # Compute gradient using all examples
            gradient = (1/m) * X.T @ errors
            
            # Update parameters
            theta = theta - learning_rate * gradient
            
            # Compute cost (MSE)
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            
            # Record number of examples used
            examples_used_history.append(m)
        
        return theta, cost_history, examples_used_history
    
    # Implement stochastic gradient descent (uses one example at a time)
    def stochastic_gradient_descent(X, y, learning_rate, n_iterations):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        examples_used_history = []
        
        for iteration in range(n_iterations):
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process one example at a time
            for i in range(m):
                # Get a single example
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                # Compute prediction
                prediction = x_i @ theta
                
                # Compute error
                error = prediction - y_i
                
                # Compute gradient using a single example
                gradient = x_i.T @ error
                
                # Update parameters
                theta = theta - learning_rate * gradient
                
                # Record number of examples used (just 1)
                examples_used_history.append(1)
                
                # Only compute cost periodically to save time
                if i % 50 == 0:
                    # Compute current cost on all data
                    predictions_all = X @ theta
                    errors_all = predictions_all - y
                    cost = (1/(2*m)) * np.sum(errors_all**2)
                    cost_history.append(cost)
        
        return theta, cost_history, examples_used_history
    
    # Run both algorithms
    print("Running batch gradient descent...")
    theta_bgd, cost_history_bgd, examples_bgd = batch_gradient_descent(
        X_train_scaled, y_train, learning_rate, n_iterations)
    
    print("Running stochastic gradient descent...")
    theta_sgd, cost_history_sgd, examples_sgd = stochastic_gradient_descent(
        X_train_scaled, y_train, learning_rate, n_iterations)
    
    # Create visualizations to demonstrate the difference
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Examples used per gradient computation
    plt.subplot(2, 1, 1)
    plt.plot(examples_bgd[:50], label='Batch GD (all examples)')
    plt.plot(examples_sgd[:50], label='SGD (single example)')
    plt.xlabel('Update step')
    plt.ylabel('Number of examples used')
    plt.title('Number of Examples Used per Gradient Computation')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Convergence comparison
    plt.subplot(2, 1, 2)
    
    # For batch GD, we can directly plot iteration vs cost
    plt.plot(cost_history_bgd, label='Batch GD')
    
    # For SGD, we need to select costs at regular intervals
    # since we only computed them periodically
    sgd_iterations = np.arange(len(cost_history_sgd))
    plt.plot(sgd_iterations * (len(examples_sgd) / len(cost_history_sgd)), 
             cost_history_sgd, label='SGD')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost Convergence: Batch GD vs. SGD')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement2_sgd.png'), dpi=300, bbox_inches='tight')
    
    # Additional figure showing the key differences with illustrations
    plt.figure(figsize=(12, 6))
    
    # Create a visual representation
    m, n = 5, 2  # Small example for visualization
    
    # Plot for Batch Gradient Descent
    plt.subplot(1, 2, 1)
    for i in range(m):
        plt.scatter(i, 0, c='blue', s=100)
    plt.title('Batch Gradient Descent')
    plt.text(m/2, -0.5, 'Uses ALL examples\nfor EACH update', ha='center')
    plt.xlim(-1, m)
    plt.ylim(-1, 1)
    plt.xticks(range(m), [f'Ex {i+1}' for i in range(m)])
    plt.yticks([])
    plt.grid(False)
    
    # Plot for Stochastic Gradient Descent
    plt.subplot(1, 2, 2)
    plt.scatter(0, 0, c='red', s=100)
    for i in range(1, m):
        plt.scatter(i, 0, c='gray', s=100, alpha=0.3)
    plt.title('Stochastic Gradient Descent')
    plt.text(m/2, -0.5, 'Uses ONE example\nper update', ha='center')
    plt.xlim(-1, m)
    plt.ylim(-1, 1)
    plt.xticks(range(m), [f'Ex {i+1}' for i in range(m)])
    plt.yticks([])
    plt.grid(False)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement2_sgd_illustration.png'), dpi=300, bbox_inches='tight')
    
    # Determine if the statement is true or false
    conclusion = """
    CONCLUSION: FALSE
    
    Stochastic Gradient Descent (SGD) does NOT use all training examples to compute the gradient in each iteration.
    Instead, SGD uses only ONE randomly selected training example to compute the gradient and update the parameters
    in each iteration.
    
    This is in contrast to Batch Gradient Descent, which uses ALL training examples to compute the gradient
    in each iteration. The key characteristics of SGD are:
    
    1. It processes one example at a time
    2. It performs parameter updates more frequently
    3. It follows a noisier path toward convergence
    4. It can escape shallow local minima due to the noise in updates
    5. It is computationally more efficient per iteration but may require more iterations overall
    
    The visualizations clearly show that SGD uses only a single example per update, while batch gradient
    descent uses all examples for each update.
    """
    
    print(conclusion)
    return conclusion 

def statement3_mini_batch_gradient_descent():
    """
    Statement 3: Mini-batch gradient descent combines the advantages of both batch and stochastic
    gradient descent.
    
    This function implements and compares batch, stochastic, and mini-batch gradient descent.
    """
    print("\n==== Statement 3: Mini-Batch Gradient Descent Advantages ====")
    print("Statement: Mini-batch gradient descent combines the advantages of both batch and stochastic gradient descent.")
    
    # Generate synthetic regression data
    n_samples, n_features = 1000, 5
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Learning rate and number of iterations
    learning_rate = 0.01
    n_iterations = 100
    mini_batch_size = 32
    
    # Implement batch gradient descent
    def batch_gradient_descent(X, y, learning_rate, n_iterations):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        time_history = []
        theta_history = []  # Track parameter history for visualization
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # Compute predictions
            predictions = X @ theta
            
            # Compute errors
            errors = predictions - y
            
            # Compute gradient using all examples
            gradient = (1/m) * X.T @ errors
            
            # Update parameters
            theta = theta - learning_rate * gradient
            theta_history.append(theta.copy())
            
            # Compute cost (MSE)
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            
            # Record time elapsed
            time_history.append(time.time() - start_time)
        
        return theta, cost_history, time_history, theta_history
    
    # Implement stochastic gradient descent
    def stochastic_gradient_descent(X, y, learning_rate, n_iterations):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        time_history = []
        theta_history = []  # Track parameter history for visualization
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # Compute cost at the beginning of each epoch
            predictions = X @ theta
            errors = predictions - y
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            time_history.append(time.time() - start_time)
            theta_history.append(theta.copy())
            
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process one example at a time
            for i in range(m):
                # Get a single example
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                # Compute prediction
                prediction = x_i @ theta
                
                # Compute error
                error = prediction - y_i
                
                # Compute gradient using a single example
                gradient = x_i.T @ error
                
                # Update parameters
                theta = theta - learning_rate * gradient
        
        return theta, cost_history, time_history, theta_history
    
    # Implement mini-batch gradient descent
    def mini_batch_gradient_descent(X, y, learning_rate, n_iterations, batch_size):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        time_history = []
        theta_history = []  # Track parameter history for visualization
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # Compute cost at the beginning of each epoch
            predictions = X @ theta
            errors = predictions - y
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            time_history.append(time.time() - start_time)
            theta_history.append(theta.copy())
            
            # Shuffle data for each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, m, batch_size):
                # Get a mini-batch
                end = min(i + batch_size, m)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Compute predictions
                predictions_batch = X_batch @ theta
                
                # Compute errors
                errors_batch = predictions_batch - y_batch
                
                # Compute gradient using mini-batch
                gradient = (1/len(X_batch)) * X_batch.T @ errors_batch
                
                # Update parameters
                theta = theta - learning_rate * gradient
        
        return theta, cost_history, time_history, theta_history
    
    # Run all three algorithms
    print("Running batch gradient descent...")
    theta_bgd, cost_history_bgd, time_history_bgd, theta_history_bgd = batch_gradient_descent(
        X_train_scaled, y_train, learning_rate, n_iterations)
    
    print("Running stochastic gradient descent...")
    theta_sgd, cost_history_sgd, time_history_sgd, theta_history_sgd = stochastic_gradient_descent(
        X_train_scaled, y_train, learning_rate, n_iterations)
    
    print("Running mini-batch gradient descent...")
    theta_mbgd, cost_history_mbgd, time_history_mbgd, theta_history_mbgd = mini_batch_gradient_descent(
        X_train_scaled, y_train, learning_rate, n_iterations, mini_batch_size)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cost vs Iterations
    axes[0, 0].plot(range(n_iterations), cost_history_bgd, label='Batch GD')
    axes[0, 0].plot(range(n_iterations), cost_history_sgd, label='SGD')
    axes[0, 0].plot(range(n_iterations), cost_history_mbgd, label='Mini-Batch GD')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('Cost (MSE)')
    axes[0, 0].set_title('Cost vs. Iterations')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Cost vs Time
    axes[0, 1].plot(time_history_bgd, cost_history_bgd, label='Batch GD')
    axes[0, 1].plot(time_history_sgd, cost_history_sgd, label='SGD')
    axes[0, 1].plot(time_history_mbgd, cost_history_mbgd, label='Mini-Batch GD')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Cost (MSE)')
    axes[0, 1].set_title('Cost vs. Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Illustration of batch sizes
    batch_sizes = [X_train_scaled.shape[0], 1, mini_batch_size]
    labels = ['Batch GD', 'SGD', 'Mini-Batch GD']
    colors = ['blue', 'red', 'green']
    
    # Create a simplified dataset for visualization
    sample_size = 100
    
    for i, (size, label, color) in enumerate(zip(batch_sizes, labels, colors)):
        # Create scatter points representing data
        x_positions = np.arange(sample_size)
        
        # Create batches visual
        n_batches = int(np.ceil(sample_size / size))
        
        for b in range(n_batches):
            start_idx = b * size
            end_idx = min(start_idx + size, sample_size)
            axes[1, 0].scatter(x_positions[start_idx:end_idx], 
                              [i] * (end_idx - start_idx),
                              c=color, alpha=0.7, s=50)
            
            # Draw rectangle around batch
            width = end_idx - start_idx
            rect = plt.Rectangle((start_idx - 0.5, i - 0.4), width, 0.8, 
                                 fill=False, edgecolor=color, linewidth=2)
            axes[1, 0].add_patch(rect)
    
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_yticklabels(labels)
    axes[1, 0].set_xlabel('Data points')
    axes[1, 0].set_title('Batch Size Comparison')
    axes[1, 0].set_xlim(-1, sample_size)
    axes[1, 0].set_ylim(-0.5, 2.5)
    
    # Plot 4: Advantages comparison - with radar chart instead of bar chart
    characteristics = ['Speed per iteration', 'Memory efficiency', 
                      'Convergence stability', 'Computational efficiency',
                      'Escape local minima']
    
    # Scores for each algorithm (0-5 scale)
    batch_scores = [2, 1, 5, 1, 1]
    sgd_scores = [5, 5, 1, 5, 4]
    mbgd_scores = [4, 4, 3, 4, 3]
    
    x_pos = np.arange(len(characteristics))
    width = 0.25
    
    axes[1, 1].bar(x_pos - width, batch_scores, width, label='Batch GD', color='blue')
    axes[1, 1].bar(x_pos, sgd_scores, width, label='SGD', color='red')
    axes[1, 1].bar(x_pos + width, mbgd_scores, width, label='Mini-Batch GD', color='green')
    
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(characteristics, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Score (0-5)')
    axes[1, 1].set_title('Comparison of Algorithm Characteristics')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 5.5)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement3_mini_batch.png'), dpi=300, bbox_inches='tight')
    
    # Create a radar chart for comparing the algorithms (more informative than a table)
    plt.figure(figsize=(10, 8))
    
    # Compute angles for radar chart
    N = len(characteristics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the characteristics scores
    batch_scores_radar = batch_scores + batch_scores[:1]  # Close the loop
    sgd_scores_radar = sgd_scores + sgd_scores[:1]  # Close the loop
    mbgd_scores_radar = mbgd_scores + mbgd_scores[:1]  # Close the loop
    
    # Set up the radar chart
    ax = plt.subplot(111, polar=True)
    
    # Plot each algorithm
    ax.plot(angles, batch_scores_radar, 'o-', linewidth=2, label='Batch GD', color='blue')
    ax.plot(angles, sgd_scores_radar, 'o-', linewidth=2, label='SGD', color='red')
    ax.plot(angles, mbgd_scores_radar, 'o-', linewidth=2, label='Mini-Batch GD', color='green')
    
    # Fill in the areas
    ax.fill(angles, batch_scores_radar, alpha=0.1, color='blue')
    ax.fill(angles, sgd_scores_radar, alpha=0.1, color='red')
    ax.fill(angles, mbgd_scores_radar, alpha=0.1, color='green')
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(characteristics)
    
    # Set y-ticks
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.set_ylim(0, 5)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.title('Gradient Descent Variants Comparison', size=15)
    
    # Save the radar chart
    plt.savefig(os.path.join(save_dir, 'statement3_radar_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create 3D visualization of parameter space trajectory
    # We'll use the first 3 dimensions for visualization
    if X_train_scaled.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories in parameter space
        theta_history_bgd_array = np.array(theta_history_bgd)
        theta_history_sgd_array = np.array(theta_history_sgd)
        theta_history_mbgd_array = np.array(theta_history_mbgd)
        
        # Plot only the first 3 parameters 
        ax.plot(theta_history_bgd_array[:, 0], theta_history_bgd_array[:, 1], theta_history_bgd_array[:, 2], 
                'o-', label='Batch GD', color='blue', markersize=4, alpha=0.7)
        ax.plot(theta_history_sgd_array[:, 0], theta_history_sgd_array[:, 1], theta_history_sgd_array[:, 2], 
                'o-', label='SGD', color='red', markersize=4, alpha=0.7)
        ax.plot(theta_history_mbgd_array[:, 0], theta_history_mbgd_array[:, 1], theta_history_mbgd_array[:, 2], 
                'o-', label='Mini-Batch GD', color='green', markersize=4, alpha=0.7)
        
        # Mark starting and ending points
        ax.scatter(theta_history_bgd_array[0, 0], theta_history_bgd_array[0, 1], theta_history_bgd_array[0, 2], 
                  color='black', s=100, marker='o', label='Start')
        ax.scatter(theta_history_bgd_array[-1, 0], theta_history_bgd_array[-1, 1], theta_history_bgd_array[-1, 2], 
                  color='cyan', s=100, marker='*', label='BGD End')
        ax.scatter(theta_history_sgd_array[-1, 0], theta_history_sgd_array[-1, 1], theta_history_sgd_array[-1, 2], 
                  color='magenta', s=100, marker='*', label='SGD End')  
        ax.scatter(theta_history_mbgd_array[-1, 0], theta_history_mbgd_array[-1, 1], theta_history_mbgd_array[-1, 2], 
                  color='yellow', s=100, marker='*', label='MBGD End')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_zlabel('Parameter 3')
        ax.set_title('Parameter Space Trajectories')
        ax.legend()
        
        plt.savefig(os.path.join(save_dir, 'statement3_parameter_trajectories.png'), dpi=300, bbox_inches='tight')
    
    # Print the table data to include in markdown
    comparison_data = [
        ['Characteristics', 'Batch GD', 'SGD', 'Mini-Batch GD'],
        ['Batch Size', 'All examples', 'Single example', f'Small batch (e.g., {mini_batch_size})'],
        ['Computational Cost/Iteration', 'High', 'Low', 'Medium'],
        ['Memory Efficiency', 'Low', 'High', 'Medium-High'],
        ['Convergence Speed', 'Slow', 'Fast but noisy', 'Medium-Fast'],
        ['Stability', 'Very stable', 'Very noisy', 'Moderately stable'],
        ['Parallelization', 'Good', 'Poor', 'Very Good'],
        ['Local Minima Escape', 'Poor', 'Good', 'Medium']
    ]
    
    # Print table in a format that can be easily copied into markdown
    print("\nTable for Markdown:")
    print("| Characteristics | Batch GD | SGD | Mini-Batch GD |")
    print("|----------------|----------|-----|---------------|")
    for row in comparison_data[1:]:  # Skip header row
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    
    # Determine if the statement is true or false
    conclusion = """
    CONCLUSION: TRUE
    
    Mini-batch gradient descent does indeed combine the advantages of both batch and stochastic 
    gradient descent. It finds a balance between the two approaches:
    
    1. Like batch GD, mini-batch GD has more stable convergence than SGD
    2. Like SGD, mini-batch GD is computationally efficient and has low memory requirements
    3. Mini-batch GD allows for effective parallelization on GPUs, which is not possible with SGD
    4. Mini-batch GD can escape some local minima (like SGD) while maintaining more stability (like batch GD)
    5. Mini-batch GD reduces the variance in parameter updates compared to SGD
    
    The figures show how mini-batch GD positions itself between batch GD and SGD in terms of:
    - Convergence behavior
    - Computational efficiency
    - Memory usage
    - Stability
    
    By processing small batches of data (typically 32, 64, or 128 examples), mini-batch GD 
    achieves a good trade-off between computation speed and convergence stability.
    """
    
    print(conclusion)
    return conclusion

def statement4_learning_rate_effect():
    """
    Statement 4: A learning rate that is too small in gradient descent will always result in
    divergence (i.e., the parameters moving away from the optimum).
    
    This function demonstrates the effect of different learning rates on gradient descent.
    """
    print("\n==== Statement 4: Effect of Learning Rate Size ====")
    print("Statement: A learning rate that is too small in gradient descent will always result in divergence (i.e., the parameters moving away from the optimum).")
    
    # Create a simple quadratic function for demonstration
    def f(x):
        return x**2
    
    def gradient_f(x):
        return 2*x
    
    # Generate a range of x values for plotting
    x_range = np.linspace(-5, 5, 100)
    y_range = f(x_range)
    
    # Define different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.1]
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    labels = ['Very small (0.001)', 'Small (0.01)', 'Medium (0.1)', 
              'Large (0.5)', 'Too large (1.1)']
    
    # Initialize figure
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Function and gradient paths
    plt.subplot(2, 1, 1)
    plt.plot(x_range, y_range, 'k-', linewidth=2)
    
    # Start from the same initial position
    x_init = 4
    
    # Run gradient descent with different learning rates
    n_iterations = 15
    paths = []
    
    for i, lr in enumerate(learning_rates):
        x = x_init
        path = [x]
        
        for _ in range(n_iterations):
            gradient = gradient_f(x)
            x = x - lr * gradient
            path.append(x)
        
        paths.append(path)
        
        # Plot the path
        plt.plot(path, [f(x) for x in path], 'o-', color=colors[i], 
                 label=f'LR: {labels[i]}', linewidth=2, markersize=6)
    
    # Mark the optimum
    plt.plot(0, 0, 'r*', markersize=15, label='Optimum')
    
    plt.xlabel('x')
    plt.ylabel('f(x) = x²')
    plt.title('Effect of Learning Rate on Gradient Descent Path')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Convergence over iterations
    plt.subplot(2, 1, 2)
    
    for i, path in enumerate(paths):
        values = [abs(x) for x in path]  # Distance from optimum (0)
        plt.plot(range(len(path)), values, 'o-', color=colors[i], 
                 label=f'LR: {labels[i]}', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Distance from optimum |x|')
    plt.title('Convergence Behavior with Different Learning Rates')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement4_learning_rate.png'), dpi=300, bbox_inches='tight')
    
    # Create a second figure to visualize convergence/divergence regions
    plt.figure(figsize=(10, 8))
    
    # Create a custom function with multiple local minima
    def g(x):
        return 0.1 * x**4 - 0.5 * x**3 - 0.5 * x**2 + 0.5 * x + 1
    
    def gradient_g(x):
        return 0.4 * x**3 - 1.5 * x**2 - x + 0.5
    
    # Generate x and y values
    x_range = np.linspace(-2.5, 2.5, 500)
    y_range = g(x_range)
    
    # Find minima positions (for visualization)
    from scipy.optimize import minimize
    min1 = minimize(g, -1.5).x[0]
    min2 = minimize(g, 1.5).x[0]
    
    # Plot function
    plt.plot(x_range, y_range, 'k-', linewidth=2)
    
    # Mark minima
    plt.plot(min1, g(min1), 'r*', markersize=15, label='Local Minimum 1')
    plt.plot(min2, g(min2), 'g*', markersize=15, label='Local Minimum 2')
    
    # Run GD with different learning rates from multiple starting points
    start_points = [-2, -1, 0, 1, 2]
    lrs = [0.01, 0.05, 0.1, 0.5, 2.0]
    
    # For demonstration of small learning rate convergence
    for start in start_points:
        x = start
        path = [x]
        lr = 0.01  # Small learning rate
        
        for _ in range(50):
            gradient = gradient_g(x)
            x = x - lr * gradient
            path.append(x)
        
        plt.plot(path, [g(x) for x in path], 'b-', alpha=0.5, linewidth=1)
        plt.plot(path[-1], g(path[-1]), 'bo', markersize=5)
    
    # For demonstration of large learning rate divergence
    for start in start_points:
        x = start
        path = [x]
        lr = 2.0  # Large learning rate
        
        for _ in range(10):
            gradient = gradient_g(x)
            x = x - lr * gradient
            
            # If diverging beyond our graph limits, stop
            if abs(x) > 5:
                path.append(np.sign(x) * 5)  # Clamp to edge of graph
                break
                
            path.append(x)
        
        plt.plot(path, [g(x) if abs(x) <= 5 else g(np.sign(x) * 2.5) for x in path], 
                 'r-', alpha=0.5, linewidth=1)
        plt.plot(path[-1], g(path[-1]) if abs(path[-1]) <= 5 else g(np.sign(path[-1]) * 2.5), 
                 'ro', markersize=5)
    
    # Add text explanations
    plt.text(-1.5, 3.5, "Small learning rate:\nSlow but stable convergence", 
             fontsize=12, bbox=dict(facecolor='blue', alpha=0.1))
    plt.text(0.5, 2.5, "Large learning rate:\nCan cause divergence", 
             fontsize=12, bbox=dict(facecolor='red', alpha=0.1))
    
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.title('Small vs. Large Learning Rates on a Complex Function')
    plt.grid(True)
    plt.ylim(-2, 5)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement4_convergence_divergence.png'), dpi=300, bbox_inches='tight')
    
    # Create a 2D contour plot to visualize the optimization landscape
    plt.figure(figsize=(12, 10))
    
    # Create a 2D function for contour plot
    def h(x, y):
        return 0.1 * (x**2 + y**2) + np.exp(-5 * ((x-1)**2 + (y-1)**2)) - 0.8 * np.exp(-5 * ((x+1)**2 + (y+1)**2))
    
    def grad_h(x, y):
        dx = 0.2 * x - 10 * (x-1) * np.exp(-5 * ((x-1)**2 + (y-1)**2)) + 8 * (x+1) * np.exp(-5 * ((x+1)**2 + (y+1)**2)) 
        dy = 0.2 * y - 10 * (y-1) * np.exp(-5 * ((x-1)**2 + (y-1)**2)) + 8 * (y+1) * np.exp(-5 * ((x+1)**2 + (y+1)**2))
        return dx, dy
    
    # Create a grid of points
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = h(X, Y)
    
    # Create the contour plot
    plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Function Value')
    
    # Add contour lines
    contours = plt.contour(X, Y, Z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Start from different points with different learning rates
    start_points = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    learning_rates_2d = [0.05, 0.5, 2.0]
    colors_2d = ['blue', 'green', 'red']
    labels_2d = ['Small LR (0.05)', 'Medium LR (0.5)', 'Large LR (2.0)']
    
    # For each starting point and learning rate
    for start in start_points:
        for i, lr in enumerate(learning_rates_2d):
            x, y = start
            path_x = [x]
            path_y = [y]
            
            # Run gradient descent for 15 iterations
            for _ in range(15):
                dx, dy = grad_h(x, y)
                x = x - lr * dx
                y = y - lr * dy
                
                # If diverging too far, stop
                if abs(x) > 5 or abs(y) > 5:
                    break
                    
                path_x.append(x)
                path_y.append(y)
            
            # Plot the path with arrow markers
            plt.plot(path_x, path_y, 'o-', color=colors_2d[i], alpha=0.7, linewidth=1.5, markersize=4)
            
            # Add arrow to show direction
            if len(path_x) >= 2:
                plt.arrow(path_x[-2], path_y[-2], path_x[-1]-path_x[-2], path_y[-1]-path_y[-2], 
                         head_width=0.1, head_length=0.2, fc=colors_2d[i], ec=colors_2d[i], alpha=0.7)
    
    # Add a legend
    for i, label in enumerate(labels_2d):
        plt.plot([], [], 'o-', color=colors_2d[i], label=label)
    
    # Plot minima
    from scipy.optimize import minimize
    result = minimize(lambda p: h(p[0], p[1]), [0, 0], method='BFGS')
    min_x, min_y = result.x
    plt.plot(min_x, min_y, 'r*', markersize=15, label='Global Minimum')
    
    # Add title and labels
    plt.title('Gradient Descent on 2D Function with Different Learning Rates')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'statement4_contour_optimization.png'), dpi=300, bbox_inches='tight')
    
    # Determine if the statement is true or false
    conclusion = """
    CONCLUSION: FALSE
    
    A learning rate that is too SMALL will NOT result in divergence. In fact, the opposite is true:
    
    1. Small learning rates generally lead to stable, albeit slow, convergence toward the optimum
    2. It's large learning rates that can cause divergence by making the algorithm overshoot the minimum
    
    As demonstrated in the visualizations:
    
    - Very small learning rates (0.001): Extremely slow convergence but guaranteed stability
    - Small learning rates (0.01): Slow but steady convergence
    - Medium learning rates (0.1): Good balance between speed and stability
    - Large learning rates (0.5): Faster convergence but may oscillate around the minimum
    - Too large learning rates (>1.0): Can cause divergence by overshooting the minimum
    
    The trade-off with small learning rates is not divergence but inefficiency - gradient descent
    will take many more iterations to reach the optimum, potentially making training very slow.
    However, it will still eventually converge (assuming the cost function is convex or the
    algorithm reaches a local minimum in non-convex cases).
    """
    
    print(conclusion)
    return conclusion

def statement5_feature_scaling_normal_equations():
    """
    Statement 5: Feature scaling is generally unnecessary when using the normal equations method
    to solve linear regression.
    
    This function demonstrates whether feature scaling affects the solution when using normal equations.
    """
    print("\n==== Statement 5: Feature Scaling and Normal Equations ====")
    print("Statement: Feature scaling is generally unnecessary when using the normal equations method to solve linear regression.")
    
    # Generate synthetic regression data with features on different scales
    np.random.seed(42)
    n_samples, n_features = 1000, 3
    
    # Create features with very different scales
    X = np.zeros((n_samples, n_features))
    X[:, 0] = np.random.normal(0, 1, n_samples)  # Standard scale
    X[:, 1] = np.random.normal(0, 100, n_samples)  # 100x larger scale
    X[:, 2] = np.random.normal(0, 0.01, n_samples)  # 100x smaller scale
    
    # Create true parameters
    w_true = np.array([0.5, 0.005, 50])  # Different scales to match feature scales
    
    # Generate target variable with noise
    y = X @ w_true + np.random.normal(0, 0.1, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Function to solve using normal equations
    def normal_equations(X, y):
        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Compute parameters using normal equations
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return theta
    
    # Function to calculate RMSE
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Solve using normal equations without scaling
    print("Solving with normal equations without feature scaling...")
    theta_no_scaling = normal_equations(X_train, y_train)
    
    # Make predictions on test set
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y_pred_no_scaling = X_test_b @ theta_no_scaling
    rmse_no_scaling = rmse(y_test, y_pred_no_scaling)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Solve using normal equations with scaling
    print("Solving with normal equations with feature scaling...")
    theta_with_scaling = normal_equations(X_train_scaled, y_train)
    
    # Make predictions on test set with scaled features
    X_test_scaled_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
    y_pred_with_scaling = X_test_scaled_b @ theta_with_scaling
    rmse_with_scaling = rmse(y_test, y_pred_with_scaling)
    
    # Compare results
    print(f"RMSE without scaling: {rmse_no_scaling:.6f}")
    print(f"RMSE with scaling: {rmse_with_scaling:.6f}")
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Feature distributions before and after scaling
    for i in range(n_features):
        plt.subplot(3, 2, i*2+1)
        plt.hist(X_train[:, i], bins=30, alpha=0.7)
        plt.title(f'Feature {i+1} Before Scaling')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.subplot(3, 2, i*2+2)
        plt.hist(X_train_scaled[:, i], bins=30, alpha=0.7)
        plt.title(f'Feature {i+1} After Scaling')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statement5_feature_distributions.png'), dpi=300, bbox_inches='tight')
    
    # Second figure: Comparison of model parameters and performance
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Model parameters
    plt.subplot(2, 1, 1)
    
    # For unscaled features, we need the bias term and coefficients
    intercept_no_scaling = theta_no_scaling[0]
    coef_no_scaling = theta_no_scaling[1:]
    
    # For scaled features, we need to transform coefficients back to original scale
    intercept_with_scaling = theta_with_scaling[0]
    coef_with_scaling_original_scale = np.zeros_like(coef_no_scaling)
    
    # Transform the coefficients back to the original scale
    for i in range(n_features):
        coef_with_scaling_original_scale[i] = (theta_with_scaling[i+1] / 
                                               scaler.scale_[i])
    
    # Adjust intercept for the transformation
    intercept_adjusted = (intercept_with_scaling - 
                         np.sum(coef_with_scaling_original_scale * scaler.mean_))
    
    # Original true parameters (need to add an intercept term)
    original_params = np.insert(w_true, 0, 0)  # Add 0 for intercept
    
    # Estimated parameters
    no_scaling_params = theta_no_scaling
    with_scaling_params = np.insert(coef_with_scaling_original_scale, 0, intercept_adjusted)
    
    # To make parameter comparison clearer, focus only on coefficients
    param_labels = ['Intercept', 'Coef 1', 'Coef 2', 'Coef 3']
    x_pos = np.arange(len(param_labels))
    
    width = 0.25
    plt.bar(x_pos - width, original_params, width, 
            label='True Parameters', alpha=0.7)
    plt.bar(x_pos, no_scaling_params, width, 
            label='No Scaling Estimate', alpha=0.7)
    plt.bar(x_pos + width, with_scaling_params, width, 
            label='With Scaling Estimate', alpha=0.7)
    
    plt.ylabel('Parameter Value')
    plt.xticks(x_pos, param_labels)
    plt.title('Model Parameters Comparison')
    plt.legend()
    
    # Plot 2: RMSE comparison
    plt.subplot(2, 1, 2)
    
    rmse_values = [rmse_no_scaling, rmse_with_scaling]
    plt.bar(['Without Scaling', 'With Scaling'], rmse_values, color=['blue', 'green'])
    plt.ylabel('RMSE')
    plt.title('Test RMSE Comparison')
    
    # Add text annotation about numerical stability
    plt.figtext(0.5, 0.01, 
                "Note: While feature scaling doesn't change the mathematical solution,\n"
                "it can improve numerical stability, especially with ill-conditioned matrices.",
                ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.1))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'statement5_normal_equations_scaling.png'), dpi=300, bbox_inches='tight')
    
    # Create a figure showing the condition number and numerical stability
    plt.figure(figsize=(10, 6))
    
    # Calculate condition numbers
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_train_scaled_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    
    cond_no_scaling = np.linalg.cond(X_train_b.T @ X_train_b)
    cond_with_scaling = np.linalg.cond(X_train_scaled_b.T @ X_train_scaled_b)
    
    print(f"Condition number without scaling: {cond_no_scaling:.2e}")
    print(f"Condition number with scaling: {cond_with_scaling:.2e}")
    
    # Plot condition numbers
    plt.bar(['Without Scaling', 'With Scaling'], 
            [cond_no_scaling, cond_with_scaling],
            color=['red', 'green'], log=True)
    
    plt.ylabel('Condition Number (log scale)')
    plt.title('Effect of Scaling on Matrix Condition Number')
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
                "Lower condition numbers indicate better numerical stability.\n"
                "Scaling can significantly improve numerical stability even though\n"
                "the mathematical solution remains the same.",
                ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.1))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'statement5_condition_number.png'), dpi=300, bbox_inches='tight')
    
    # Determine if the statement is true or false
    conclusion = """
    CONCLUSION: TRUE
    
    Feature scaling is generally unnecessary when using the normal equations method to solve linear regression,
    because:
    
    1. The mathematical solution remains the same whether features are scaled or not
    2. Both scaled and unscaled solutions yield the same predictions when properly transformed back
    3. The normal equations directly solve for the optimal parameters without iterative optimization
    
    However, there is an important caveat:
    
    While feature scaling doesn't affect the mathematical solution, it can significantly improve
    numerical stability, especially when dealing with ill-conditioned matrices (when features have
    very different scales). In the visualization, we can see that:
    
    - The condition number of the matrix without scaling is much higher
    - A high condition number can lead to numerical precision issues
    - In extreme cases, numerical instability might affect the solution accuracy
    
    So while the statement is technically true from a mathematical perspective, in practical
    implementations, scaling can sometimes help with numerical stability issues, especially
    with features on vastly different scales or when using limited numerical precision.
    
    This stands in contrast to gradient-based methods (like gradient descent), where scaling
    is essential for ensuring proper convergence.
    """
    
    print(conclusion)
    return conclusion

def main():
    """Run all statement evaluations and collect conclusions."""
    
    # Create a title
    print("=" * 80)
    print("QUESTION 13: EVALUATING STATEMENTS ABOUT LINEAR REGRESSION")
    print("=" * 80)
    
    # Run each statement evaluation
    conclusions = {}
    
    print("\nEvaluating Statement 1...")
    conclusions[1] = statement1_normal_equations_complexity()
    
    print("\nEvaluating Statement 2...")
    conclusions[2] = statement2_sgd_gradient_computation()
    
    print("\nEvaluating Statement 3...")
    conclusions[3] = statement3_mini_batch_gradient_descent()
    
    print("\nEvaluating Statement 4...")
    conclusions[4] = statement4_learning_rate_effect()
    
    print("\nEvaluating Statement 5...")
    conclusions[5] = statement5_feature_scaling_normal_equations()
    
    # Print summary of results
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    for i in range(1, 6):
        # Extract just the TRUE/FALSE part from the conclusion
        result = "TRUE" if "TRUE" in conclusions[i].split("\n")[1] else "FALSE"
        print(f"Statement {i}: {result}")
    
    print("\nDetailed explanations and visualizations have been saved to:")
    print(f"{save_dir}")

if __name__ == "__main__":
    main()