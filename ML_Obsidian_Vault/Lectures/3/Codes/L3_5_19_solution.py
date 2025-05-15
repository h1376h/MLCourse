import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 19: Normal Equations vs. Gradient Descent for Linear Regression")
print("-" * 80)
print("Problem Characteristics:")
print("- Training set: n = 10,000 examples")
print("- Features: d = 1,000 after one-hot encoding")
print("- Matrix X^T X is non-singular")
print("- Computational resources are limited")
print()

# 1. Closed-form solution using normal equations
def normal_equations_solution():
    """Demonstrate the closed-form solution for linear regression using normal equations."""
    print("Task 1: Closed-form solution for linear regression using normal equations")
    print("The normal equation is:")
    print("    w = (X^T X)^(-1) X^T y")
    print()
    print("Where:")
    print("- w is the weight vector (d×1)")
    print("- X is the design matrix (n×d)")
    print("- y is the target vector (n×1)")
    print()
    
    # Create visualization for normal equations
    plt.figure(figsize=(10, 6))
    plt.title("Normal Equations Method Process", fontsize=16)
    
    # Create a flow diagram-like visualization
    steps = ["Design Matrix\nX (n×d)", 
             "Compute\nX^T X (d×d)", 
             "Invert Matrix\n(X^T X)^(-1) (d×d)", 
             "Compute\nX^T y (d×1)", 
             "Compute\nw = (X^T X)^(-1) X^T y"]
    
    # Horizontal positions
    x_pos = np.linspace(0, 1, len(steps))
    # Vertical position (all the same)
    y_pos = np.ones_like(x_pos) * 0.5
    
    # Plot nodes
    plt.scatter(x_pos, y_pos, s=3000, c='skyblue', alpha=0.6, zorder=1)
    
    # Add text
    for i, step in enumerate(steps):
        plt.text(x_pos[i], y_pos[i], step, ha='center', va='center', fontsize=12)
    
    # Connect nodes with arrows
    for i in range(len(steps)-1):
        plt.arrow(x_pos[i] + 0.08, y_pos[i], x_pos[i+1] - x_pos[i] - 0.16, 0,
                 head_width=0.02, head_length=0.02, fc='k', ec='k', zorder=0)
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(0.3, 0.7)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "normal_equations_process.png"), dpi=300)
    plt.close()
    
    # Simulate normal equations with timing for small and large datasets
    print("Simulating normal equations computation time:")
    
    data_sizes = [(100, 10), (1000, 100), (10000, 1000)]
    times = []
    
    for n, d in data_sizes:
        start_time = time.time()
        
        # Generate random data
        X = np.random.randn(n, d)
        y = np.random.randn(n, 1)
        
        # Compute normal equations (without actually computing the weights)
        XTX = X.T @ X
        XTy = X.T @ y
        
        # Only perform matrix inversion if the matrix is small (for larger matrices it's too slow)
        if d <= 100:
            XTX_inv = np.linalg.inv(XTX)
            # Compute weights: w = (X^T X)^(-1) X^T y
            w = XTX_inv @ XTy
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        print(f"n={n}, d={d}: {elapsed_time:.4f} seconds")
        if d > 100:
            print("  (Matrix inversion skipped for large matrices)")
    
    return times

ne_times = normal_equations_solution()

# 2. Gradient descent update rule
def gradient_descent_solution():
    """Demonstrate gradient descent for linear regression."""
    print("\nTask 2: Update rule for batch gradient descent in linear regression")
    print("The batch gradient descent update rule is:")
    print("    w := w - α ∇J(w)")
    print("    w := w - α (1/n) X^T (X w - y)")
    print()
    print("Where:")
    print("- w is the weight vector (d×1)")
    print("- α is the learning rate")
    print("- X is the design matrix (n×d)")
    print("- y is the target vector (n×1)")
    print("- J(w) is the cost function (mean squared error)")
    print()
    
    # Visualize gradient descent process
    plt.figure(figsize=(10, 6))
    plt.title("Gradient Descent Method Process", fontsize=16)
    
    # Create a flow diagram with iteration
    steps = ["Initialize\nw randomly", 
             "Compute\nXw (n×1)", 
             "Compute\nXw - y (n×1)", 
             "Compute\nX^T(Xw - y) (d×1)", 
             "Update\nw := w - α∇J(w)"]
    
    # Circular arrangement for iteration visualization
    x_pos = np.linspace(0, 1, len(steps))
    y_pos = np.ones_like(x_pos) * 0.5
    
    # Plot nodes
    plt.scatter(x_pos, y_pos, s=3000, c='lightgreen', alpha=0.6, zorder=1)
    
    # Add text
    for i, step in enumerate(steps):
        plt.text(x_pos[i], y_pos[i], step, ha='center', va='center', fontsize=12)
    
    # Connect nodes with arrows
    for i in range(len(steps)-1):
        plt.arrow(x_pos[i] + 0.08, y_pos[i], x_pos[i+1] - x_pos[i] - 0.16, 0,
                 head_width=0.02, head_length=0.02, fc='k', ec='k', zorder=0)
    
    # Show iteration (connect last to second node)
    plt.arrow(x_pos[-1], y_pos[-1] - 0.15, -0.9, -0.3,
             head_width=0.02, head_length=0.02, fc='k', ec='k', zorder=0, 
             linestyle='dashed', color='red')
    plt.text(x_pos[-1] - 0.45, y_pos[-1] - 0.3, "Iterate until convergence", 
             ha='center', va='center', color='red', fontsize=10)
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(0.1, 0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_descent_process.png"), dpi=300)
    plt.close()
    
    # Simulate gradient descent with timing for small and large datasets
    print("Simulating gradient descent computation time:")
    
    data_sizes = [(100, 10), (1000, 100), (10000, 1000)]
    times = []
    
    for n, d in data_sizes:
        start_time = time.time()
        
        # Generate random data
        X = np.random.randn(n, d)
        y = np.random.randn(n, 1)
        
        # Initialize weights
        w = np.zeros((d, 1))
        
        # Learning rate
        alpha = 0.01
        
        # Just do a few iterations for timing purposes
        num_iterations = 5
        for _ in range(num_iterations):
            # Compute gradient: (1/n) * X^T (X w - y)
            gradient = (1/n) * X.T @ (X @ w - y)
            
            # Update weights
            w = w - alpha * gradient
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        print(f"n={n}, d={d}: {elapsed_time:.4f} seconds (for {num_iterations} iterations)")
    
    return times

gd_times = gradient_descent_solution()

# 3. Compare computational complexity
def compare_complexity():
    """Compare the computational complexity of normal equations and gradient descent."""
    print("\nTask 3: Computational complexity comparison")
    print("Normal Equations:")
    print("- Computing X^T X: O(n*d^2)")
    print("- Inverting X^T X: O(d^3)")
    print("- Computing X^T y: O(n*d)")
    print("- Computing (X^T X)^(-1) X^T y: O(d^2)")
    print("- Total: O(n*d^2 + d^3)")
    print()
    print("Gradient Descent (per iteration):")
    print("- Computing X*w: O(n*d)")
    print("- Computing (X*w - y): O(n)")
    print("- Computing X^T(X*w - y): O(n*d)")
    print("- Updating weights: O(d)")
    print("- Total per iteration: O(n*d)")
    print("- If k iterations are needed: O(k*n*d)")
    print()
    
    # Create comparison table
    methods = ["Normal Equations", "Gradient Descent"]
    time_complexity = ["O(n*d^2 + d^3)", "O(k*n*d)"]
    space_complexity = ["O(d^2) for X^T X", "O(d) for weights"]
    pros = ["Direct solution, no iterations or hyperparameters", 
            "Works well with large datasets, scales linearly with n"]
    cons = ["Scales poorly with features (d^3)", 
            "Requires iterations and hyperparameter tuning"]
    
    comparison_data = {
        "Method": methods,
        "Time Complexity": time_complexity,
        "Space Complexity": space_complexity,
        "Advantages": pros,
        "Disadvantages": cons
    }
    
    print("Comparison Table:")
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    print()
    
    # Visualize complexity comparison
    plt.figure(figsize=(12, 7))
    plt.title("Time Complexity Comparison", fontsize=16)
    
    # Log-log plot showing complexity growth
    d_values = np.logspace(1, 4, 100)  # d from 10 to 10,000
    n_values = [10000]  # Fixed n = 10,000
    
    for n in n_values:
        # Normal equation complexity: n*d^2 + d^3
        ne_complexity = n * d_values**2 + d_values**3
        
        # Gradient descent complexity: k*n*d (assuming k=100 iterations)
        k = 100
        gd_complexity = k * n * d_values
        
        plt.loglog(d_values, ne_complexity, 'b-', label=f"Normal Equations: O(n*d^2 + d^3), n={n}")
        plt.loglog(d_values, gd_complexity, 'g-', label=f"Gradient Descent: O(k*n*d), k={k}, n={n}")
    
    # Highlight regions with n=10,000, d=1,000 (given case)
    plt.axvline(x=1000, color='r', linestyle='--', alpha=0.5, label="d=1,000 (given case)")
    
    # Highlight crossover point
    crossover_idx = np.argmin(np.abs(ne_complexity - gd_complexity))
    crossover_d = d_values[crossover_idx]
    plt.scatter([crossover_d], [ne_complexity[crossover_idx]], color='red', s=100, zorder=5)
    plt.annotate(f'Crossover point\nd ≈ {crossover_d:.0f}', 
                 xy=(crossover_d, ne_complexity[crossover_idx]),
                 xytext=(crossover_d*0.5, ne_complexity[crossover_idx]*0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12)
    
    plt.xlabel('Number of Features (d)', fontsize=14)
    plt.ylabel('Computational Complexity (operations)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "complexity_comparison.png"), dpi=300)
    plt.close()
    
    # Create bar chart comparing execution times from previous simulations
    data_sizes = [(100, 10), (1000, 100), (10000, 1000)]
    x_pos = np.arange(len(data_sizes))
    labels = [f"n={n}, d={d}" for n, d in data_sizes]
    
    # Create a DataFrame for plotting
    exec_times_df = pd.DataFrame({
        'Dataset': labels,
        'Normal Equations': ne_times,
        'Gradient Descent': gd_times
    })
    
    # Convert the DataFrame to long format for seaborn
    exec_times_long = pd.melt(exec_times_df, id_vars=['Dataset'], 
                             value_vars=['Normal Equations', 'Gradient Descent'],
                             var_name='Method', value_name='Time (s)')
    
    plt.figure(figsize=(12, 7))
    plt.title("Execution Time Comparison", fontsize=16)
    
    # Use seaborn for a grouped bar chart
    sns.barplot(x='Dataset', y='Time (s)', hue='Method', data=exec_times_long)
    
    plt.xlabel('Dataset Size', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.yscale('log')  # Log scale for better visibility
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "execution_time_comparison.png"), dpi=300)
    plt.close()

compare_complexity()

# 4. Recommendation based on problem characteristics
def recommendation_initial_case():
    """Provide recommendation for the initial case: n=10,000, d=1,000."""
    print("\nTask 4: Recommendation for the initial case (n=10,000, d=1,000)")
    print("Analysis:")
    print("- Normal Equations complexity: O(n*d^2 + d^3) = O(10,000*1,000^2 + 1,000^3)")
    print("  = O(10^10 + 10^9) ≈ O(10^10) operations")
    print("- Gradient Descent complexity (assuming k=100 iterations): O(k*n*d) = O(100*10,000*1,000)")
    print("  = O(10^9) operations")
    print()
    print("Memory Requirements:")
    print("- Normal Equations: Need to store X^T X (d×d matrix) = 1,000×1,000 = 1 million elements")
    print("- Gradient Descent: Only need to store weights and gradients: O(d) = 1,000 elements")
    print()
    print("Recommendation: Gradient Descent")
    print("Reasons:")
    print("1. Computational Efficiency: For n=10,000 and d=1,000, normal equations require")
    print("   significantly more computation (O(10^10) vs O(10^9) for gradient descent).")
    print("2. Memory Requirements: X^T X matrix would be 1,000×1,000, requiring ~8MB of memory")
    print("   (assuming 8 bytes per double), which is manageable but larger than necessary.")
    print("3. Limited Computational Resources: The problem specifies limited resources, making")
    print("   gradient descent's lower memory footprint and potentially parallelizable computation")
    print("   more appropriate.")
    print()
    
    # Create a visualization for the recommendation
    plt.figure(figsize=(12, 7))
    plt.title("Decision Factors for n=10,000, d=1,000", fontsize=16)
    
    # Data for radar chart
    categories = ['Computational\nEfficiency', 'Memory\nEfficiency', 
                 'Implementation\nSimplicity', 'Adaptability to\nLarge Datasets', 
                 'No Hyperparameter\nTuning']
    
    # Convert to radians
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Scores for each method (on a scale of 0-5)
    ne_scores = [2, 2, 5, 1, 5]  # Normal Equations
    gd_scores = [4, 5, 3, 5, 2]  # Gradient Descent
    
    # Complete the loop
    ne_scores += ne_scores[:1]
    gd_scores += gd_scores[:1]
    
    # Set up the plot
    ax = plt.subplot(111, polar=True)
    
    # Plot each method
    ax.plot(angles, ne_scores, 'b-', linewidth=2, label='Normal Equations')
    ax.fill(angles, ne_scores, 'b', alpha=0.1)
    
    ax.plot(angles, gd_scores, 'g-', linewidth=2, label='Gradient Descent')
    ax.fill(angles, gd_scores, 'g', alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Set y-ticks
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], fontsize=10)
    plt.ylim(0, 5)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.grid(True)
    
    # Add an explanatory textbox
    plt.figtext(0.5, 0.01, "For n=10,000 and d=1,000 with limited resources, Gradient Descent is recommended\n"
               "due to better computational and memory efficiency despite requiring hyperparameter tuning.",
               ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.savefig(os.path.join(save_dir, "recommendation_case1.png"), dpi=300)
    plt.close()

recommendation_initial_case()

# 5. Recommendation for the changed scenario
def recommendation_changed_case():
    """Provide recommendation for the changed scenario: n=10 million, d=100."""
    print("\nTask 5: Recommendation for changed scenario (n=10 million, d=100)")
    print("Analysis:")
    print("- Normal Equations complexity: O(n*d^2 + d^3) = O(10^7*100^2 + 100^3)")
    print("  = O(10^11 + 10^6) ≈ O(10^11) operations")
    print("- Gradient Descent complexity (assuming k=100 iterations): O(k*n*d) = O(100*10^7*100)")
    print("  = O(10^11) operations")
    print()
    print("Memory Requirements:")
    print("- Normal Equations: Need to store X^T X (d×d matrix) = 100×100 = 10,000 elements")
    print("- Gradient Descent: Only need to store weights and gradients: O(d) = 100 elements")
    print()
    print("Additional Considerations:")
    print("- With n=10 million, the design matrix X would be extremely large (10^7 × 100)")
    print("- Normal equations require computing X^T X, which means reading all data into memory")
    print("- Gradient descent can process data in batches (stochastic or mini-batch gradient descent)")
    print()
    print("Recommendation: Gradient Descent (specifically, mini-batch or stochastic gradient descent)")
    print("Reasons:")
    print("1. Data Size: With 10 million examples, processing all data at once for normal equations")
    print("   would be impractical and likely exceed memory limits.")
    print("2. Batch Processing: Gradient descent variants allow processing data in smaller batches,")
    print("   making it feasible to handle very large datasets without loading all data into memory.")
    print("3. Parallelization: Mini-batch gradient descent can be parallelized across multiple CPUs or GPUs,")
    print("   significantly speeding up computation for large datasets.")
    print("4. Similar Asymptotic Complexity: While the theoretical complexity is similar for both methods,")
    print("   the practical implementation advantages of gradient descent make it clearly superior for this case.")
    print()
    
    # Visualization for the changed scenario
    plt.figure(figsize=(12, 6))
    plt.title("Comparing Methods for n=10,000,000, d=100", fontsize=16)
    
    # Create a diagram showing advantages/disadvantages
    feature_names = ["Computational\nEfficiency", "Memory\nEfficiency", 
                    "Batch\nProcessing", "Implementation\nSimplicity", 
                    "Works for\nVery Large n"]
    
    ne_scores = [2, 3, 1, 5, 1]  # Normal Equations
    gd_scores = [4, 5, 5, 3, 5]  # Gradient Descent
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, ne_scores, width, label='Normal Equations', color='skyblue')
    rects2 = ax.bar(x + width/2, gd_scores, width, label='Gradient Descent', color='lightgreen')
    
    ax.set_ylim(0, 5.5)
    ax.set_ylabel('Score (1-5)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=12)
    ax.legend(fontsize=14)
    
    # Add a horizontal line indicating the threshold for acceptable performance
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.3)
    plt.text(len(feature_names)-1, 3.1, "Acceptable Threshold", color='r', fontsize=10)
    
    # Add value labels on the bars
    def autolabel(rects):
        """Attach a text label above each bar showing its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Add an explanatory text box
    plt.figtext(0.5, 0.01, "With 10 million examples and 100 features, Gradient Descent (especially mini-batch or stochastic variants)\n"
               "is strongly recommended due to its ability to process data in batches and handle very large datasets.",
               ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "recommendation_case2.png"), dpi=300)
    plt.close()
    
    # Additional visualization showing batch processing advantage
    plt.figure(figsize=(12, 6))
    plt.title("Memory Usage with Increasing Dataset Size", fontsize=16)
    
    # Create data for memory usage
    n_values = np.logspace(3, 7, 100)  # n from 1,000 to 10 million
    d = 100  # Fixed at 100 features
    
    # Memory usage for normal equations (need to store X)
    ne_memory = n_values * d * 8  # bytes (8 bytes per double)
    
    # Memory usage for mini-batch gradient descent (batch size 1,000)
    batch_size = 1000
    gd_memory = np.minimum(n_values, batch_size) * d * 8
    
    # Convert to GB for readability
    ne_memory_gb = ne_memory / (1024**3)
    gd_memory_gb = gd_memory / (1024**3)
    
    plt.loglog(n_values, ne_memory_gb, 'b-', label='Normal Equations', linewidth=2)
    plt.loglog(n_values, gd_memory_gb, 'g-', label='Mini-Batch Gradient Descent', linewidth=2)
    
    # Add reference lines for different memory capacities
    plt.axhline(y=4, color='r', linestyle='--', alpha=0.5, label='4 GB RAM')
    plt.axhline(y=16, color='orange', linestyle='--', alpha=0.5, label='16 GB RAM')
    plt.axhline(y=64, color='purple', linestyle='--', alpha=0.5, label='64 GB RAM')
    
    # Highlight the n=10 million point
    plt.axvline(x=10**7, color='r', linestyle='--', alpha=0.5, label='n=10 million')
    
    plt.xlabel('Number of Examples (n)', fontsize=14)
    plt.ylabel('Memory Required (GB)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "memory_comparison.png"), dpi=300)
    plt.close()

recommendation_changed_case()

# Summary
print("\nSummary of Findings:")
print("1. Normal Equations provide a direct closed-form solution: w = (X^T X)^(-1) X^T y")
print("2. Gradient Descent uses iterative updates: w := w - α (1/n) X^T (X w - y)")
print("3. Computational complexity:")
print("   - Normal Equations: O(n*d^2 + d^3)")
print("   - Gradient Descent: O(k*n*d) for k iterations")
print("4. For n=10,000 and d=1,000: Gradient Descent is recommended due to lower computational")
print("   complexity and memory requirements")
print("5. For n=10 million and d=100: Gradient Descent (specifically mini-batch or stochastic variants)")
print("   is strongly recommended due to the ability to process data in batches and avoid memory limitations")
print()

print(f"Images saved to: {save_dir}")
print("Generated images:")
print("- normal_equations_process.png: Visualization of normal equations solution process")
print("- gradient_descent_process.png: Visualization of gradient descent process")
print("- complexity_comparison.png: Time complexity comparison of both methods")
print("- execution_time_comparison.png: Empirical execution time comparison")
print("- recommendation_case1.png: Decision factors for the n=10,000, d=1,000 case")
print("- recommendation_case2.png: Decision factors for the n=10 million, d=100 case")
print("- memory_comparison.png: Memory usage comparison for large datasets") 