import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Problem Setup")

print("Given:")
print("Estimator A: θ̂ₐ with bias b_A(θ) = 0.1θ and variance Var(θ̂ₐ) = 0.5")
print("Estimator B: θ̂ᵦ with bias b_B(θ) = 0 and variance Var(θ̂ᵦ) = 0.8")
print("\nTasks:")
print("1. Calculate the MSE for each estimator when θ = 2")
print("2. Determine which estimator is preferred when θ = 2")
print("3. Find the range of θ for which estimator A has lower MSE than estimator B")
print("4. Discuss the bias-variance tradeoff in this context")

# Step 2: Calculate MSE for both estimators when θ = 2
print_step_header(2, "Calculating MSE When θ = 2")

# Define functions for bias, variance, and MSE
def bias_A(theta):
    return 0.1 * theta

def bias_B(theta):
    return 0

def var_A(theta):
    return 0.5

def var_B(theta):
    return 0.8

def mse_A(theta):
    return bias_A(theta)**2 + var_A(theta)

def mse_B(theta):
    return bias_B(theta)**2 + var_B(theta)

# Calculate MSE for θ = 2
theta_value = 2
mse_A_value = mse_A(theta_value)
mse_B_value = mse_B(theta_value)

print(f"For θ = {theta_value}:")
print(f"Estimator A:")
print(f"  Bias = {bias_A(theta_value)}")
print(f"  Variance = {var_A(theta_value)}")
print(f"  MSE = Bias² + Variance = {bias_A(theta_value)}² + {var_A(theta_value)} = {mse_A_value}")
print(f"\nEstimator B:")
print(f"  Bias = {bias_B(theta_value)}")
print(f"  Variance = {var_B(theta_value)}")
print(f"  MSE = Bias² + Variance = {bias_B(theta_value)}² + {var_B(theta_value)} = {mse_B_value}")

if mse_A_value < mse_B_value:
    print(f"\nFor θ = {theta_value}, Estimator A has lower MSE ({mse_A_value} < {mse_B_value}).")
    print("Therefore, Estimator A is preferred.")
else:
    print(f"\nFor θ = {theta_value}, Estimator B has lower MSE ({mse_B_value} < {mse_A_value}).")
    print("Therefore, Estimator B is preferred.")

# Visualize the comparison for θ = 2
plt.figure(figsize=(12, 6))
labels = ['Estimator A', 'Estimator B']
biases = [bias_A(theta_value)**2, bias_B(theta_value)**2]
variances = [var_A(theta_value), var_B(theta_value)]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bias_bars = ax.bar(x - width/2, biases, width, label='Squared Bias')
variance_bars = ax.bar(x + width/2, variances, width, label='Variance')

# Add text for MSE values
for i in range(len(labels)):
    mse = biases[i] + variances[i]
    ax.text(i, mse + 0.05, f'MSE = {mse:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Comparison of Squared Bias and Variance for θ = 2', fontsize=14)
ax.set_ylabel('Value', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend()

# Add a horizontal line for the minimum MSE
min_mse = min(mse_A_value, mse_B_value)
ax.axhline(y=min_mse, color='r', linestyle='--', alpha=0.5, label=f'Min MSE = {min_mse:.2f}')

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mse_comparison_theta2.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Find when estimator A has lower MSE
print_step_header(3, "Finding When Estimator A Has Lower MSE Than Estimator B")

print("We need to find the values of θ where MSE_A(θ) < MSE_B(θ):")
print("MSE_A(θ) = bias_A(θ)² + var_A(θ) = (0.1θ)² + 0.5 = 0.01θ² + 0.5")
print("MSE_B(θ) = bias_B(θ)² + var_B(θ) = 0² + 0.8 = 0.8")
print("\nSolving MSE_A(θ) < MSE_B(θ):")
print("0.01θ² + 0.5 < 0.8")
print("0.01θ² < 0.3")
print("θ² < 30")
print("θ < √30 ≈ 5.48")
print("\nTherefore, Estimator A has lower MSE than Estimator B when |θ| < 5.48.")

# Calculate the crossover point analytically
from math import sqrt
crossover_point = sqrt((var_B(0) - var_A(0))/0.01)
print(f"The exact crossover point is θ = {crossover_point:.4f}")

# Visualize MSE as a function of θ
theta_range = np.linspace(0, 10, 1000)
mse_A_values = [mse_A(theta) for theta in theta_range]
mse_B_values = [mse_B(theta) for theta in theta_range]

plt.figure(figsize=(10, 6))
plt.plot(theta_range, mse_A_values, 'b-', linewidth=2, label='MSE of Estimator A')
plt.plot(theta_range, mse_B_values, 'g-', linewidth=2, label='MSE of Estimator B')

# Mark the crossover point
plt.axvline(x=crossover_point, color='r', linestyle='--', label=f'Crossover at θ = {crossover_point:.2f}')
plt.axhline(y=mse_A(crossover_point), color='r', linestyle=':', alpha=0.5)

# Highlight the regions
plt.fill_between(theta_range, mse_A_values, mse_B_values, 
                 where=(theta_range < crossover_point), 
                 color='blue', alpha=0.1, interpolate=True, 
                 label='Region where A is better')
plt.fill_between(theta_range, mse_A_values, mse_B_values, 
                 where=(theta_range >= crossover_point), 
                 color='green', alpha=0.1, interpolate=True, 
                 label='Region where B is better')

plt.title('MSE Comparison as a Function of θ', fontsize=14)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.xlim(0, 10)
plt.ylim(0, 1.5)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mse_comparison_range.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Visualize the Bias-Variance Tradeoff
print_step_header(4, "Understanding the Bias-Variance Tradeoff")

print("The bias-variance tradeoff is a fundamental concept in statistics and machine learning.")
print("It refers to the balance between two sources of error:")
print("1. Bias: How far the expected predictions are from the true values")
print("2. Variance: How much the predictions vary across different samples")
print("\nIn this problem:")
print("- Estimator A has higher bias but lower variance")
print("- Estimator B has zero bias but higher variance")
print("\nThe MSE combines both sources of error: MSE = Bias² + Variance")
print("\nObservations:")
print("1. For small values of θ, the squared bias term in Estimator A is small, so its lower variance gives it the advantage")
print("2. For large values of θ, the squared bias term grows quadratically, eventually making Estimator B preferable")
print("3. The crossover point (θ ≈ 5.48) represents where these two sources of error balance out")

# Create a visualization of the bias-variance tradeoff
theta_points = np.linspace(0, 10, 100)
bias_A_squared = [bias_A(theta)**2 for theta in theta_points]
var_A_values = [var_A(theta) for theta in theta_points]
bias_B_squared = [bias_B(theta)**2 for theta in theta_points]
var_B_values = [var_B(theta) for theta in theta_points]

plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2)

# Plot 1: Components of MSE for Estimator A
ax1 = plt.subplot(gs[0, 0])
ax1.plot(theta_points, bias_A_squared, 'r-', linewidth=2, label='Squared Bias')
ax1.plot(theta_points, var_A_values, 'b-', linewidth=2, label='Variance')
ax1.plot(theta_points, np.add(bias_A_squared, var_A_values), 'g-', linewidth=2, label='MSE')
ax1.set_title('Components of MSE for Estimator A', fontsize=12)
ax1.set_xlabel('θ', fontsize=10)
ax1.set_ylabel('Value', fontsize=10)
ax1.grid(True)
ax1.legend()

# Plot 2: Components of MSE for Estimator B
ax2 = plt.subplot(gs[0, 1])
ax2.plot(theta_points, bias_B_squared, 'r-', linewidth=2, label='Squared Bias')
ax2.plot(theta_points, var_B_values, 'b-', linewidth=2, label='Variance')
ax2.plot(theta_points, np.add(bias_B_squared, var_B_values), 'g-', linewidth=2, label='MSE')
ax2.set_title('Components of MSE for Estimator B', fontsize=12)
ax2.set_xlabel('θ', fontsize=10)
ax2.set_ylabel('Value', fontsize=10)
ax2.grid(True)
ax2.legend()

# Plot 3: Comparison of MSEs
ax3 = plt.subplot(gs[1, 0])
ax3.plot(theta_points, [mse_A(theta) for theta in theta_points], 'b-', linewidth=2, label='MSE of A')
ax3.plot(theta_points, [mse_B(theta) for theta in theta_points], 'g-', linewidth=2, label='MSE of B')
ax3.axvline(x=crossover_point, color='r', linestyle='--', label=f'Crossover at θ = {crossover_point:.2f}')
ax3.axvline(x=theta_value, color='k', linestyle=':', label=f'θ = {theta_value}')
ax3.set_title('MSE Comparison', fontsize=12)
ax3.set_xlabel('θ', fontsize=10)
ax3.set_ylabel('MSE', fontsize=10)
ax3.grid(True)
ax3.legend()

# Plot 4: Bias-Variance Tradeoff Illustration
ax4 = plt.subplot(gs[1, 1])
# Create some example data to illustrate the tradeoff
complexity = np.linspace(1, 10, 100)
bias = 10 / complexity
variance = 0.1 * complexity
mse = bias + variance

ax4.plot(complexity, bias, 'r-', linewidth=2, label='Bias')
ax4.plot(complexity, variance, 'b-', linewidth=2, label='Variance')
ax4.plot(complexity, mse, 'g-', linewidth=2, label='Total Error')
ax4.set_title('Conceptual Bias-Variance Tradeoff', fontsize=12)
ax4.set_xlabel('Model Complexity', fontsize=10)
ax4.set_ylabel('Error', fontsize=10)
ax4.grid(True)
ax4.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bias_variance_tradeoff.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Summary
print_step_header(5, "Summary of Results")

# Create a table-like visualization for MSE comparison across different θ values
theta_examples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mse_A_examples = [mse_A(theta) for theta in theta_examples]
mse_B_examples = [mse_B(theta) for theta in theta_examples]
preferred = ["A" if mse_A(theta) < mse_B(theta) else "B" for theta in theta_examples]

# Create a visualization of the table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table_data = []
for i in range(len(theta_examples)):
    table_data.append([
        theta_examples[i], 
        f"{bias_A(theta_examples[i]):.2f}",
        f"{var_A(theta_examples[i]):.2f}",
        f"{mse_A_examples[i]:.2f}",
        f"{bias_B(theta_examples[i]):.2f}",
        f"{var_B(theta_examples[i]):.2f}",
        f"{mse_B_examples[i]:.2f}",
        preferred[i]
    ])

table = ax.table(
    cellText=table_data, 
    colLabels=['θ', 'Bias A', 'Var A', 'MSE A', 'Bias B', 'Var B', 'MSE B', 'Preferred'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color the cells based on which estimator is preferred
for i in range(len(theta_examples)):
    if preferred[i] == "A":
        table[(i+1, 7)].set_facecolor('lightblue')
    else:
        table[(i+1, 7)].set_facecolor('lightgreen')

plt.title('Comparison of Estimators Across Different θ Values', fontsize=14)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "estimator_comparison_table.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Final summary
print("\nFinal summary:")
print("1. For θ = 2:")
print(f"   - MSE of Estimator A = {mse_A_value:.4f}")
print(f"   - MSE of Estimator B = {mse_B_value:.4f}")
print(f"   - {'Estimator A' if mse_A_value < mse_B_value else 'Estimator B'} is preferred")
print(f"\n2. Estimator A has lower MSE when |θ| < {crossover_point:.4f}")
print(f"   - When θ is small, the bias term (0.1θ)² in Estimator A is outweighed by its lower variance")
print(f"   - When θ is large, the bias term grows quadratically, making the unbiased Estimator B preferable")
print("\n3. Bias-Variance Tradeoff:")
print("   - Estimator A: Higher bias but lower variance")
print("   - Estimator B: Zero bias but higher variance")
print("   - The optimal choice depends on the true parameter value θ")
print("   - This demonstrates that an estimator with some bias can outperform an unbiased estimator if it has sufficiently lower variance") 