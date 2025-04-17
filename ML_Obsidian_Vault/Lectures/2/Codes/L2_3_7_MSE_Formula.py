import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_7")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- An estimator with bias b(θ) and variance v(θ)")
print("- Goal: Express the Mean Squared Error (MSE) in terms of bias and variance")
print()
print("This involves understanding the bias-variance tradeoff in estimator selection.")
print()

# Step 2: Define the MSE formula
print_step_header(2, "The MSE Formula")

print("The Mean Squared Error (MSE) of an estimator θ̂ is defined as:")
print("MSE(θ̂) = E[(θ̂ - θ)²]")
print()
print("Where:")
print("- θ̂ is the estimator")
print("- θ is the true parameter value")
print("- E[·] denotes the expected value")
print()
print("The MSE measures the expected squared deviation of the estimator from the true parameter value.")
print()

# Step 3: Deriving the MSE formula in terms of bias and variance
print_step_header(3, "Deriving the MSE Formula")

print("To express MSE in terms of bias and variance, we can decompose it as follows:")
print()
print("MSE(θ̂) = E[(θ̂ - θ)²]")
print("       = E[(θ̂ - E[θ̂] + E[θ̂] - θ)²]")
print("       = E[((θ̂ - E[θ̂]) + (E[θ̂] - θ))²]")
print()
print("Expanding the square:")
print("MSE(θ̂) = E[(θ̂ - E[θ̂])² + 2(θ̂ - E[θ̂])(E[θ̂] - θ) + (E[θ̂] - θ)²]")
print()
print("Now, let's analyze each term:")
print("- E[(θ̂ - E[θ̂])²] is the variance of θ̂, which is v(θ)")
print("- E[2(θ̂ - E[θ̂])(E[θ̂] - θ)] = 2(E[θ̂] - θ)·E[θ̂ - E[θ̂]] = 0 (since E[θ̂ - E[θ̂]] = 0)")
print("- (E[θ̂] - θ)² is the squared bias, which is (b(θ))²")
print()
print("Therefore:")
print("MSE(θ̂) = v(θ) + (b(θ))²")
print()
print("This is the MSE formula in terms of bias and variance.")

# Step 4: Visual representation of MSE decomposition
print_step_header(4, "Visual Representation of MSE Decomposition")

# Function to create the visual demonstration
def create_mse_visualization():
    fig = plt.figure(figsize=(12, 8))
    
    # Create a distribution of the estimator
    theta_true = 5  # True parameter value
    bias = 1        # Bias of the estimator
    variance = 2    # Variance of the estimator
    
    # Generate estimator values based on bias and variance
    np.random.seed(42)
    estimator_values = np.random.normal(theta_true + bias, np.sqrt(variance), 1000)
    
    # Set up plot
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # Main plot: distribution of estimator values
    ax1 = plt.subplot(gs[0, 0])
    
    # Plot the distribution of estimator values
    kde_x = np.linspace(0, 10, 1000)
    kde = stats.gaussian_kde(estimator_values)
    ax1.plot(kde_x, kde(kde_x), 'b-', lw=2, label='Distribution of θ̂')
    
    # Add vertical lines for true value and expected estimator
    ax1.axvline(x=theta_true, color='r', linestyle='-', lw=2, label='True θ')
    ax1.axvline(x=theta_true + bias, color='g', linestyle='--', lw=2, label='E[θ̂]')
    
    # Add histogram of estimator values
    ax1.hist(estimator_values, bins=30, alpha=0.3, density=True, color='blue')
    
    # Add annotations
    ax1.annotate('Bias = E[θ̂] - θ', 
                 xy=((theta_true + theta_true + bias)/2, 0.1),
                 xytext=(theta_true + 0.5, 0.15),
                 arrowprops=dict(facecolor='black', arrowstyle='<->'),
                 fontsize=12, ha='center')
    
    ax1.annotate('Variance = E[(θ̂ - E[θ̂])²]', 
                 xy=(theta_true + bias, 0.05),
                 xytext=(theta_true + bias + 1, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, ha='left')
    
    # Add MSE formula
    ax1.text(0.05, 0.95, 'MSE(θ̂) = Variance + Bias²', transform=ax1.transAxes, 
             fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.text(0.05, 0.87, f'MSE(θ̂) = {variance} + {bias}² = {variance + bias**2}', 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    
    # Set labels and title
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Estimator Values', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Right plot: MSE components visualization
    ax2 = plt.subplot(gs[0, 1])
    
    # Create a bar chart of MSE components
    components = ['Variance', 'Bias²', 'MSE']
    values = [variance, bias**2, variance + bias**2]
    colors = ['blue', 'green', 'red']
    
    ax2.bar(components, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('MSE Components', fontsize=14)
    
    # Bottom plot: MSE as function of bias and variance
    ax3 = plt.subplot(gs[1, :])
    
    # Create a 3D-like plot of MSE as a function of bias and variance
    bias_range = np.linspace(-2, 2, 50)
    var_range = np.linspace(0.1, 3, 50)
    
    B, V = np.meshgrid(bias_range, var_range)
    MSE = V + B**2
    
    # Create a contour plot
    contour = ax3.contourf(B, V, MSE, 20, cmap='viridis')
    
    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label('MSE Value', fontsize=10)
    
    # Mark our example point
    ax3.plot(bias, variance, 'ro', markersize=8)
    ax3.annotate(f'Our Example\nBias={bias}, Variance={variance}', 
                xy=(bias, variance), xytext=(bias+0.5, variance+0.5),
                arrowprops=dict(facecolor='white', shrink=0.05),
                fontsize=10, color='white')
    
    # Set labels and title
    ax3.set_xlabel('Bias', fontsize=12)
    ax3.set_ylabel('Variance', fontsize=12)
    ax3.set_title('MSE as a Function of Bias and Variance', fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(save_dir, "mse_decomposition.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the visualization
create_mse_visualization()

# Step 5: Illustrate the bias-variance tradeoff
print_step_header(5, "The Bias-Variance Tradeoff")

def create_bias_variance_tradeoff_plot():
    plt.figure(figsize=(10, 6))
    
    # Define values for complexity
    complexity = np.linspace(0.1, 10, 100)
    
    # Define bias and variance as functions of complexity
    bias = 5 / complexity
    variance = 0.1 * complexity
    mse = bias + variance
    
    # Plot the curves
    plt.plot(complexity, bias, 'b-', linewidth=2, label='Bias²')
    plt.plot(complexity, variance, 'g-', linewidth=2, label='Variance')
    plt.plot(complexity, mse, 'r-', linewidth=2, label='MSE = Bias² + Variance')
    
    # Add minimum MSE point
    min_mse_idx = np.argmin(mse)
    min_complexity = complexity[min_mse_idx]
    min_mse = mse[min_mse_idx]
    
    plt.scatter([min_complexity], [min_mse], color='r', s=100, zorder=3)
    plt.annotate('Optimal\nComplexity', 
                xy=(min_complexity, min_mse), 
                xytext=(min_complexity + 2, min_mse - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12)
    
    # Add regions
    plt.axvspan(0, min_complexity, alpha=0.2, color='blue', label='High Bias (Underfitting)')
    plt.axvspan(min_complexity, 10, alpha=0.2, color='green', label='High Variance (Overfitting)')
    
    # Add labels and title
    plt.xlabel('Model Complexity', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper center')
    
    # Save the figure
    file_path = os.path.join(save_dir, "bias_variance_tradeoff.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the tradeoff plot
create_bias_variance_tradeoff_plot()

# Step 6: Real-world examples
print_step_header(6, "Real-World Examples")

print("Example 1: Comparing Two Estimators")
print("Estimator A: Has low bias but high variance")
print("Estimator B: Has higher bias but lower variance")
print()
print("Let's say:")
print("- Estimator A: bias²=0.25, variance=4.0, MSE=4.25")
print("- Estimator B: bias²=1.0, variance=2.0, MSE=3.0")
print()
print("Despite Estimator A having lower bias, Estimator B has lower MSE due to its significantly lower variance.")
print()

def create_estimator_comparison_plot():
    plt.figure(figsize=(10, 6))
    
    # Define parameters for two estimators
    estimators = ['Estimator A', 'Estimator B']
    bias_squared = [0.25, 1.0]
    variance = [4.0, 2.0]
    mse = [4.25, 3.0]
    
    # Set width of bars
    barWidth = 0.25
    
    # Set position of bars on X axis
    r1 = np.arange(len(estimators))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, bias_squared, width=barWidth, edgecolor='white', label='Bias²')
    plt.bar(r2, variance, width=barWidth, edgecolor='white', label='Variance')
    plt.bar(r3, mse, width=barWidth, edgecolor='white', label='MSE')
    
    # Add labels and title
    plt.xlabel('Estimator', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of Two Estimators', fontsize=14)
    plt.xticks([r + barWidth for r in range(len(estimators))], estimators)
    plt.legend()
    
    # Add annotations
    for i, (b, v, m) in enumerate(zip(bias_squared, variance, mse)):
        plt.annotate(f"Bias²={b}", (r1[i], b), textcoords="offset points", 
                     xytext=(0,5), ha='center', fontsize=9)
        plt.annotate(f"Var={v}", (r2[i], v), textcoords="offset points", 
                     xytext=(0,5), ha='center', fontsize=9)
        plt.annotate(f"MSE={m}", (r3[i], m), textcoords="offset points", 
                     xytext=(0,5), ha='center', fontsize=9)
    
    # Save the figure
    file_path = os.path.join(save_dir, "estimator_comparison.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the estimator comparison plot
create_estimator_comparison_plot()

# Step 7: Conclusion
print_step_header(7, "Conclusion and Answer")

print("The Mean Squared Error (MSE) of an estimator θ̂ with bias b(θ) and variance v(θ) is:")
print()
print("MSE(θ̂) = v(θ) + (b(θ))²")
print()
print("This formula demonstrates the bias-variance tradeoff in statistical estimation:")
print("1. The MSE is the sum of the variance and the squared bias")
print("2. Reducing bias often increases variance, and vice versa")
print("3. The optimal estimator minimizes the MSE, balancing bias and variance")
print()
print("This is a fundamental concept in statistical learning theory and model selection.") 