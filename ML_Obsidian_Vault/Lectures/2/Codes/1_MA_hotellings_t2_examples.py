import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

print("\n=== HOTELLING'S T² TEST EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Analysis")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Helper function to plot confidence ellipses
def plot_confidence_ellipse(ax, x, y, cov, n, alpha=0.05, color='blue', label=None):
    """
    Plot confidence ellipse at specified alpha level
    
    Parameters:
    -----------
    ax : matplotlib axes
        The axes to plot on
    x, y : float
        Center of the ellipse
    cov : 2x2 array
        Covariance matrix
    n : int
        Sample size
    alpha : float
        Significance level (default: 0.05)
    color : str
        Color of the ellipse
    label : str
        Label for the ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using t² statistic for the ellipse
    t_squared = stats.f.ppf(1-alpha, 2, n-2) * 2 * (n-1) / (n-2)
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Calculate angles and lengths of ellipse axes
    theta = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(t_squared * eigenvals / n)
    
    # Create ellipse
    ellipse = Ellipse(
        xy=(x, y),
        width=width,
        height=height,
        angle=theta,
        facecolor='none',
        edgecolor=color,
        label=label
    )
    
    ax.add_patch(ellipse)
    return ellipse

# Example 1: Manufacturing Process Evaluation
print("Example 1: Manufacturing Process Evaluation")

# Problem data
mean_vector = np.array([52, 78, 105])
cov_matrix = np.array([
    [4, 2, 1],
    [2, 9, 3],
    [1, 3, 16]
])
mu0 = np.array([50, 80, 100])
n = 25
alpha = 0.05  # 5% significance level
p = 3  # Number of variables

print("A manufacturing company has implemented a new process for producing electronic components.")
print("They want to determine if the new process has changed the overall quality of the components.")
print("They measure three key properties (resistance, weight, and strength) for a sample of 25 components.")
print("\nThe sample mean vector is:")
print(f"μ̄ = [{mean_vector[0]}, {mean_vector[1]}, {mean_vector[2]}]")
print("\nThe sample covariance matrix is:")
print(f"⎡ {cov_matrix[0, 0]} {cov_matrix[0, 1]} {cov_matrix[0, 2]} ⎤")
print(f"⎢ {cov_matrix[1, 0]} {cov_matrix[1, 1]} {cov_matrix[1, 2]} ⎥")
print(f"⎣ {cov_matrix[2, 0]} {cov_matrix[2, 1]} {cov_matrix[2, 2]} ⎦")
print("\nThe historical (known) mean vector for the old process is:")
print(f"μ₀ = [{mu0[0]}, {mu0[1]}, {mu0[2]}]")

# Step 1: Define the hypotheses
print("\nStep 1: Define the Hypotheses")
print("H₀: μ = μ₀ (No change in mean vector)")
print("H₁: μ ≠ μ₀ (Change in mean vector)")

# Step 2: Calculate Hotelling's T² Statistic
print("\nStep 2: Calculate Hotelling's T² Statistic")
print("The Hotelling's T² statistic is defined as:")
print("T² = n · (x̄ - μ₀)ᵀ · S⁻¹ · (x̄ - μ₀)")

# Calculate the difference vector
diff_vector = mean_vector - mu0
print("\nFirst, calculate the difference vector:")
print(f"x̄ - μ₀ = [{mean_vector[0]}, {mean_vector[1]}, {mean_vector[2]}] - [{mu0[0]}, {mu0[1]}, {mu0[2]}] = [{diff_vector[0]}, {diff_vector[1]}, {diff_vector[2]}]")

# Calculate inverse of covariance matrix
cov_inv = np.linalg.inv(cov_matrix)

# Format the inverse matrix with precision
cov_inv_formatted = np.round(cov_inv, 3)
print("\nThe inverse of the sample covariance matrix is:")
print(f"S⁻¹ = ⎡ {cov_inv_formatted[0, 0]} {cov_inv_formatted[0, 1]} {cov_inv_formatted[0, 2]} ⎤")
print(f"     ⎢ {cov_inv_formatted[1, 0]} {cov_inv_formatted[1, 1]} {cov_inv_formatted[1, 2]} ⎥")
print(f"     ⎣ {cov_inv_formatted[2, 0]} {cov_inv_formatted[2, 1]} {cov_inv_formatted[2, 2]} ⎦")

# Calculate the partial results
partial_result = np.dot(diff_vector, cov_inv)
print("\nComputing the T² statistic:")
print(f"First, compute the vector-matrix product (x̄ - μ₀)ᵀ · S⁻¹:")
print(f"[{diff_vector[0]}, {diff_vector[1]}, {diff_vector[2]}] · S⁻¹ = [{partial_result[0]:.3f}, {partial_result[1]:.3f}, {partial_result[2]:.3f}]")

# Calculate each component of the result vector
print("\nWhere:")
for i in range(p):
    component = 0
    component_calc = ""
    for j in range(p):
        term = diff_vector[j] * cov_inv[j, i]
        component += term
        term_sign = "+" if term >= 0 else ""
        if j < p-1:
            component_calc += f"{diff_vector[j]} × {cov_inv_formatted[j, i]} {term_sign} "
        else:
            component_calc += f"{diff_vector[j]} × {cov_inv_formatted[j, i]}"
    print(f"Component {i+1} = {component_calc} = {component:.3f}")

# Calculate T² statistic
t_squared = n * np.dot(partial_result, diff_vector)
print(f"\nNow, compute the final product and multiply by n = {n}:")
t_squared_calc = f"T² = {n} · ("
for i in range(p):
    t_squared_calc += f"{partial_result[i]:.3f} × {diff_vector[i]}"
    if i < p-1:
        t_squared_calc += " + "
t_squared_calc += ")"

# Calculate the intermediate sum
intermediate_sum = 0
intermediate_calc = ""
for i in range(p):
    term = partial_result[i] * diff_vector[i]
    intermediate_sum += term
    if i < p-1:
        intermediate_calc += f"{partial_result[i]:.3f} × {diff_vector[i]} + "
    else:
        intermediate_calc += f"{partial_result[i]:.3f} × {diff_vector[i]}"
print(f"T² = {n} · ({intermediate_calc}) = {n} · {intermediate_sum:.3f} = {t_squared:.3f}")

# Step 3: Convert to F-Statistic
print("\nStep 3: Convert to F-Statistic")
print(f"For Hotelling's T² test with p = {p} variables and n = {n} observations, we convert to an F-statistic:")
print("F = (n - p) / (p(n - 1)) · T²")

f_statistic = (n - p) / (p * (n - 1)) * t_squared
print(f"F = ({n} - {p}) / ({p} · ({n} - 1)) · {t_squared:.3f} = {n-p} / {p*(n-1)} · {t_squared:.3f} = {f_statistic:.2f}")

# Step 4: Determine the Critical Value
critical_f = stats.f.ppf(1 - alpha, p, n - p)
print("\nStep 4: Determine the Critical Value")
print(f"The critical value from the F-distribution with {p} and {n-p} degrees of freedom at α = {alpha} is:")
print(f"F_crit = {critical_f:.2f}")

# Step 5: Make Decision
print("\nStep 5: Make Decision")
if f_statistic > critical_f:
    decision = "reject"
else:
    decision = "fail to reject"
print(f"Since the calculated F-value ({f_statistic:.2f}) {'exceeds' if f_statistic > critical_f else 'does not exceed'} "
      f"the critical value ({critical_f:.2f}), we {decision} the null hypothesis.")

# Calculate p-value
p_value = 1 - stats.f.cdf(f_statistic, p, n - p)
print(f"\nThe p-value for this test is {p_value:.6f}, which is {'less' if p_value < alpha else 'greater'} than α = {alpha}.")

# Interpretation
print("\nInterpretation:")
if decision == "reject":
    print("There is significant evidence to conclude that the new manufacturing process has changed the overall properties of the electronic components.")
    print("The simultaneous test of all three properties shows that the changes cannot be attributed to random variation.")
else:
    print("There is insufficient evidence to conclude that the new manufacturing process has changed the overall properties of the electronic components.")

print("\nThe advantage of Hotelling's T² test over multiple univariate t-tests is that it accounts for the correlation structure")
print("among the measured properties and controls the overall Type I error rate.")

# Create the visualization for Example 1
print("\nGenerating visualizations for Example 1...")

# Visualize in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate random samples from multivariate normal for visualization
np.random.seed(42)
samples = np.random.multivariate_normal(mean_vector, cov_matrix, 100)

# Plot the samples
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='blue', alpha=0.6, label='New Process Samples')
ax.scatter([mean_vector[0]], [mean_vector[1]], [mean_vector[2]], c='red', s=100, marker='x', label='Sample Mean')
ax.scatter([mu0[0]], [mu0[1]], [mu0[2]], c='green', s=100, marker='o', label='Historical Mean')

# Add labels
ax.set_xlabel('Resistance')
ax.set_ylabel('Weight')
ax.set_zlabel('Strength')
ax.set_title("3D Visualization of Manufacturing Process Data")

# Add test results as text annotation
result_text = f"Hotelling's T² = {t_squared:.2f}\nF-statistic = {f_statistic:.2f}\np-value = {p_value:.4f}"
ax.text2D(0.05, 0.95, result_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'hotellings_t2_3d_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create 2D visualizations (pairwise projections)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Pairs to visualize
pairs = [(0, 1), (0, 2), (1, 2)]
pair_labels = [('Resistance', 'Weight'), ('Resistance', 'Strength'), ('Weight', 'Strength')]

for i, ((idx1, idx2), (label1, label2)) in enumerate(zip(pairs, pair_labels)):
    # Extract the appropriate 2D samples
    samples_2d = samples[:, [idx1, idx2]]
    mean_2d = mean_vector[[idx1, idx2]]
    mu0_2d = mu0[[idx1, idx2]]
    cov_2d = cov_matrix[np.ix_([idx1, idx2], [idx1, idx2])]
    
    # Plot samples
    axs[i].scatter(samples_2d[:, 0], samples_2d[:, 1], c='blue', alpha=0.4, label='Samples')
    axs[i].scatter(mean_2d[0], mean_2d[1], c='red', s=100, marker='x', label='Sample Mean')
    axs[i].scatter(mu0_2d[0], mu0_2d[1], c='green', s=100, marker='o', label='Historical Mean')
    
    # Plot 95% confidence ellipse
    plot_confidence_ellipse(axs[i], mean_2d[0], mean_2d[1], cov_2d, n, alpha=0.05, color='red', 
                           label='95% Confidence Region')
    
    # Add vector from historical mean to sample mean
    axs[i].arrow(mu0_2d[0], mu0_2d[1], mean_2d[0] - mu0_2d[0], mean_2d[1] - mu0_2d[1], 
                color='purple', width=0.3, head_width=1.5, length_includes_head=True,
                label='Difference Vector')
    
    # Set labels
    axs[i].set_xlabel(label1)
    axs[i].set_ylabel(label2)
    axs[i].set_title(f'{label1} vs {label2}')
    
    # Adjust limits with padding
    padding = 2
    axs[i].set_xlim(min(samples_2d[:, 0].min(), mu0_2d[0]) - padding, 
                   max(samples_2d[:, 0].max(), mu0_2d[0]) + padding)
    axs[i].set_ylim(min(samples_2d[:, 1].min(), mu0_2d[1]) - padding, 
                   max(samples_2d[:, 1].max(), mu0_2d[1]) + padding)
    
    # Add grid
    axs[i].grid(True, linestyle='--', alpha=0.6)

# Add overall legend to the last subplot
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(images_dir, 'hotellings_t2_2d_projections.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Comparison with Multiple Individual T-Tests
print("\n\nExample 2: Comparison with Multiple Individual T-Tests")
print("Using the same manufacturing data from Example 1, we'll compare the results of Hotelling's T² test")
print("with those that would be obtained from conducting three separate univariate t-tests.")

# Step 1: Calculate Individual T-Statistics
print("\nStep 1: Calculate Individual T-Statistics")
print("For each variable, we compute a univariate t-statistic:")

t_stats = []
for i, (property_name, sample_mean, pop_mean, var) in enumerate(zip(
    ['Resistance', 'Weight', 'Strength'], 
    mean_vector, mu0, np.diag(cov_matrix))):
    
    t_stat = (sample_mean - pop_mean) / np.sqrt(var/n)
    t_stats.append(t_stat)
    
    print(f"\nFor {property_name}:")
    print(f"t = (x̄ - μ₀) / (s/√n) = ({sample_mean} - {pop_mean}) / √({var}/{n}) = {sample_mean - pop_mean} / {np.sqrt(var/n):.2f} = {t_stat:.2f}")

# Step 2: Apply the Bonferroni Correction
print("\nStep 2: Apply the Bonferroni Correction")
print(f"When conducting multiple tests, we need to adjust the significance level to control the family-wise error rate.")
print(f"Using the Bonferroni correction, the adjusted significance level for each test would be α' = {alpha}/{p} = {alpha/p:.4f}.")

# Calculate critical values
alpha_adjusted = alpha / p
t_critical = stats.t.ppf(1 - alpha_adjusted/2, n-1)  # Using two-tailed test
print(f"The critical t-value with {n-1} degrees of freedom at α' = {alpha_adjusted:.4f} is approximately {t_critical:.2f}.")

# Step 3: Compare Results
print("\nStep 3: Compare Results")
print("\n| Property   | T-statistic | Critical Value | Decision   |")
print("|------------|-------------|----------------|------------|")

decisions = []
for i, (property_name, t_stat) in enumerate(zip(['Resistance', 'Weight', 'Strength'], t_stats)):
    decision = "Reject H₀" if abs(t_stat) > t_critical else "Fail to reject H₀"
    decisions.append(decision)
    print(f"| {property_name:<10} | {t_stat:11.2f} | {t_critical:14.2f} | {decision:<10} |")

# Step 4: Compare with Hotelling's T² Results
print("\nStep 4: Compare with Hotelling's T² Results")
print(f"Hotelling's T² test resulted in F = {f_statistic:.2f}, which {'exceeds' if f_statistic > critical_f else 'does not exceed'} the critical value of {critical_f:.2f},")
print(f"leading to {'rejection' if decision == 'reject' else 'non-rejection'} of the null hypothesis that all means are simultaneously equal to their target values.")

# Step 5: Analyze the Advantages of Hotelling's T² Test
print("\nStep 5: Analyze the Advantages of Hotelling's T² Test")
print("\n1. Control of Type I Error: The Bonferroni correction is often too conservative, especially as the number")
print("   of variables increases. Hotelling's T² maintains the correct significance level without being overly conservative.")
print("\n2. Accounts for Correlations: The individual t-tests treat each property as independent, but we can see from")
print("   the covariance matrix that they are correlated. Hotelling's T² accounts for these correlations.")
print("\n3. Power Advantage: In general, Hotelling's T² has greater statistical power than multiple individual tests,")
print("   particularly when the variables are correlated.")
print("\n4. Simplicity: A single test provides a clear answer about whether the overall process has changed,")
print("   rather than dealing with potentially conflicting results from multiple tests.")

# Interpretation
print("\nInterpretation:")
print("While both approaches lead to the same conclusion in this case, the Hotelling's T² test provides a more elegant")
print("and statistically sound framework for analyzing multivariate data. It properly accounts for the correlation")
print("structure and maintains appropriate control over the Type I error rate without sacrificing power.")
print("\nIn practice, one might first perform the Hotelling's T² test to determine if there is an overall change,")
print("and then follow up with individual t-tests to identify which specific properties have changed and by how much.")

# Create a visualization comparing the methods
print("\nGenerating visualization comparing Hotelling's T² test with multiple t-tests...")

# Create a comparison visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Bar labels
properties = ['Resistance', 'Weight', 'Strength']
x = np.arange(len(properties))
width = 0.35

# Absolute t-statistics
abs_t_stats = np.abs(t_stats)

# Plot t-statistics and critical values
rects1 = ax.bar(x - width/2, abs_t_stats, width, label='|t-statistic|', color='skyblue')
rects2 = ax.bar(x + width/2, np.repeat(t_critical, len(properties)), width, 
                label='Critical t-value (with Bonferroni)', color='lightcoral')

# Add a horizontal line for the equivalent Hotelling's T² threshold
# Converting F to an equivalent t for visualization
equivalent_t = np.sqrt(p * f_statistic)
ax.axhline(y=equivalent_t, linestyle='--', color='purple', label="Equivalent Hotelling's T² threshold")

# Customize plot
ax.set_xlabel('Properties')
ax.set_ylabel('Absolute Value of Test Statistic')
ax.set_title('Comparison of Multiple t-tests vs. Hotelling\'s T² Test')
ax.set_xticks(x)
ax.set_xticklabels(properties)
ax.legend()

# Add text annotations
for rect, t_stat, decision in zip(rects1, abs_t_stats, decisions):
    height = rect.get_height()
    ax.annotate(f'{t_stat:.2f}\n{decision}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Add test results as text annotation
result_text = (f"Hotelling's T² = {t_squared:.2f}\n"
              f"F-statistic = {f_statistic:.2f}\n"
              f"p-value = {p_value:.4f}\n"
              f"Decision: {'Reject H₀' if decision == 'reject' else 'Fail to reject H₀'}")
ax.text(0.02, 0.98, result_text, transform=ax.transAxes, 
        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'hotellings_t2_vs_multiple_ttests.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create a visualization of the F-distribution with the test statistic
print("\nGenerating F-distribution visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Create x values for the F-distribution
x = np.linspace(0, 20, 1000)
# Calculate F-distribution PDF values
y = stats.f.pdf(x, p, n-p)

# Plot the F-distribution
ax.plot(x, y, 'b-', lw=2, label=f'F({p}, {n-p}) distribution')

# Add vertical lines for the F-statistic and critical value
ax.axvline(f_statistic, color='red', linestyle='-', lw=2, label=f'F-statistic = {f_statistic:.2f}')
ax.axvline(critical_f, color='green', linestyle='--', lw=2, label=f'Critical value = {critical_f:.2f}')

# Fill the rejection region
x_fill = np.linspace(critical_f, max(x), 100)
y_fill = stats.f.pdf(x_fill, p, n-p)
ax.fill_between(x_fill, y_fill, alpha=0.3, color='red', label='Rejection region')

# Fill the p-value area
if f_statistic > critical_f:
    x_pvalue = np.linspace(f_statistic, max(x), 100)
    y_pvalue = stats.f.pdf(x_pvalue, p, n-p)
    ax.fill_between(x_pvalue, y_pvalue, alpha=0.5, color='purple', label=f'p-value = {p_value:.6f}')

# Add labels and title
ax.set_xlabel('F-value')
ax.set_ylabel('Probability Density')
ax.set_title(f'F-Distribution with {p} and {n-p} Degrees of Freedom')

# Add legend
ax.legend()

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'hotellings_t2_f_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll Hotelling's T² test example images have been created successfully in the Images/Multivariate_Analysis directory.") 