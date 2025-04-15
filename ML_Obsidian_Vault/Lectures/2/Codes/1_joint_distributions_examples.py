import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Ellipse

print("\n=== JOINT DISTRIBUTIONS EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Joint_Distributions relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Joint_Distributions")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set style for all plots
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

# Example 1: Discrete Joint Distribution
print("\nExample 1: Discrete Joint Distribution")
print("Consider the joint probability mass function of two discrete random variables X and Y:")

# Define the joint PMF
joint_pmf = np.array([
    [0.10, 0.08, 0.12],
    [0.15, 0.20, 0.05],
    [0.05, 0.15, 0.10]
])

# Create a DataFrame for better visualization
df = pd.DataFrame(joint_pmf, 
                 index=['X=1', 'X=2', 'X=3'],
                 columns=['Y=1', 'Y=2', 'Y=3'])
print("\nJoint Probability Mass Function:")
print(df)

# Step 1: Find marginal distributions
print("\nStep 1: Find marginal distributions")
marginal_x = joint_pmf.sum(axis=1)
marginal_y = joint_pmf.sum(axis=0)

print("\nMarginal Distribution of X:")
for i, p in enumerate(marginal_x, 1):
    print(f"P(X={i}) = {p:.2f}")

print("\nMarginal Distribution of Y:")
for i, p in enumerate(marginal_y, 1):
    print(f"P(Y={i}) = {p:.2f}")

# Step 2: Find specific joint probability
print("\nStep 2: Find P(X=2, Y=2)")
print(f"P(X=2, Y=2) = {joint_pmf[1, 1]:.2f}")

# Step 3: Calculate P(X > 1, Y < 3)
print("\nStep 3: Calculate P(X > 1, Y < 3)")
p_event = joint_pmf[1:, :2].sum()
print(f"P(X > 1, Y < 3) = {p_event:.2f}")

# Create visualization
plt.figure(figsize=(10, 8))
plt.imshow(joint_pmf, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks(range(3), ['Y=1', 'Y=2', 'Y=3'])
plt.yticks(range(3), ['X=1', 'X=2', 'X=3'])
plt.title('Joint Probability Mass Function')
plt.xlabel('Y')
plt.ylabel('X')

# Add probability values to the plot
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{joint_pmf[i, j]:.2f}", 
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'joint_pmf_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 1: Additional visualization - Marginal Distributions Bar Plot
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.bar(range(1, 4), marginal_x, color='blue', alpha=0.7)
plt.title('Marginal Distribution of X')
plt.xlabel('X')
plt.ylabel('Probability')
plt.xticks(range(1, 4))

plt.subplot(122)
plt.bar(range(1, 4), marginal_y, color='green', alpha=0.7)
plt.title('Marginal Distribution of Y')
plt.xlabel('Y')
plt.ylabel('Probability')
plt.xticks(range(1, 4))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'marginal_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 1: Additional visualization - 3D Bar Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(3)
y = np.arange(3)
xpos, ypos = np.meshgrid(x, y)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = 0.5 * np.ones_like(zpos)
dz = joint_pmf.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue')
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_xticklabels(['X=1', 'X=2', 'X=3'])
ax.set_yticklabels(['Y=1', 'Y=2', 'Y=3'])
ax.set_zlabel('Probability')
ax.set_title('3D Bar Plot of Joint PMF')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'joint_pmf_3d.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Bivariate Normal Distribution
print("\nExample 2: Bivariate Normal Distribution")
print("Two variables X and Y follow a bivariate normal distribution with:")
print("μ_X = 5, μ_Y = 10, σ_X = 2, σ_Y = 3, ρ = 0.7")

# Parameters
mu_x, mu_y = 5, 10
sigma_x, sigma_y = 2, 3
rho = 0.7

# Step 1: Calculate covariance
cov = rho * sigma_x * sigma_y
print(f"\nStep 1: Calculate covariance")
print(f"Cov(X,Y) = ρσ_Xσ_Y = {rho} × {sigma_x} × {sigma_y} = {cov:.1f}")

# Step 2: Calculate P(X < 6, Y < 12)
print("\nStep 2: Calculate P(X < 6, Y < 12)")
# Using scipy's multivariate normal CDF
mean = [mu_x, mu_y]
cov_matrix = [[sigma_x**2, cov], [cov, sigma_y**2]]
p = stats.multivariate_normal.cdf([6, 12], mean=mean, cov=cov_matrix)
print(f"P(X < 6, Y < 12) ≈ {p:.4f}")

# Create visualization
x = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 100)
y = np.linspace(mu_y - 3*sigma_y, mu_y + 3*sigma_y, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv = stats.multivariate_normal(mean, cov_matrix)
Z = rv.pdf(pos)

fig = plt.figure(figsize=(12, 5))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0)
ax1.set_title('Bivariate Normal PDF')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Density')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
plt.colorbar(contour, ax=ax2, label='Density')
ax2.set_title('Bivariate Normal Contours')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bivariate_normal_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Additional visualization - Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter Plot of Bivariate Normal Samples')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bivariate_normal_scatter.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Additional visualization - Correlation Ellipse
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.title('Correlation Ellipse Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Add correlation ellipse
def plot_corr_ellipse(x, y, ax, **kwargs):
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), **kwargs)
    ax.add_artist(ell)
    return ell

plot_corr_ellipse(x, y, plt.gca(), alpha=0.2, color='red')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'correlation_ellipse.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Conditional Distribution
print("\nExample 3: Conditional Distribution")
print("Find the conditional probability mass function of X given Y=2")

# Calculate P(Y=2)
p_y2 = marginal_y[1]
print(f"\nP(Y=2) = {p_y2:.2f}")

# Calculate conditional probabilities
print("\nConditional probabilities:")
for i in range(3):
    p_cond = joint_pmf[i, 1] / p_y2
    print(f"P(X={i+1}|Y=2) = {joint_pmf[i, 1]:.2f} / {p_y2:.2f} = {p_cond:.3f}")

# Create visualization
plt.figure(figsize=(10, 5))

# Plot marginal and conditional distributions
x = np.arange(1, 4)
width = 0.35

plt.bar(x - width/2, marginal_x, width, label='Marginal P(X)', color='blue', alpha=0.7)
plt.bar(x + width/2, joint_pmf[:, 1]/p_y2, width, label='Conditional P(X|Y=2)', color='red', alpha=0.7)

plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Marginal vs Conditional Distribution')
plt.xticks(x)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_distribution_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Additional visualization - Multiple Conditional Distributions
plt.figure(figsize=(12, 5))
for y_val in range(3):
    p_y = marginal_y[y_val]
    cond_prob = joint_pmf[:, y_val] / p_y
    plt.plot(range(1, 4), cond_prob, marker='o', label=f'P(X|Y={y_val+1})')
plt.title('Conditional Distributions P(X|Y)')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multiple_conditional_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Additional visualization - Conditional Probability Surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.arange(1, 4)
y = np.arange(1, 4)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(3):
    for j in range(3):
        Z[i,j] = joint_pmf[i,j] / marginal_y[j]

surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Conditional Probability')
ax.set_title('Conditional Probability Surface P(X|Y)')
plt.colorbar(surf)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_surface.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Conditional Expectation
print("\nExample 4: Conditional Expectation")
print("Find the expected value of X given Y=3 and Y given X=1")

# Calculate E[X|Y=3]
print("\nE[X|Y=3]:")
p_y3 = marginal_y[2]
cond_prob_x_given_y3 = joint_pmf[:, 2] / p_y3
e_x_given_y3 = np.sum((np.arange(1, 4) * cond_prob_x_given_y3))
print(f"E[X|Y=3] = {e_x_given_y3:.3f}")

# Calculate E[Y|X=1]
print("\nE[Y|X=1]:")
p_x1 = marginal_x[0]
cond_prob_y_given_x1 = joint_pmf[0, :] / p_x1
e_y_given_x1 = np.sum((np.arange(1, 4) * cond_prob_y_given_x1))
print(f"E[Y|X=1] = {e_y_given_x1:.3f}")

# Visualization of conditional expectations
plt.figure(figsize=(10, 5))
x = np.arange(1, 4)  # Values of X
e_y_given_x = np.array([e_y_given_x1, 2.0, 1.8])  # Average values of Y for each X
plt.bar(x, e_y_given_x, alpha=0.5, color='orange')
plt.axhline(y=e_y_given_x1, xmin=0, xmax=0.33, color='red', linestyle='--')
plt.text(1, e_y_given_x1+0.1, f'E[Y|X=1] = {e_y_given_x1:.3f}', color='red')
plt.title('Conditional Expectation E[Y|X]')
plt.xlabel('X')
plt.ylabel('Expected Value')
plt.xticks(x)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_expectation_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Additional visualization - Regression Line
plt.figure(figsize=(10, 5))
x_vals = np.arange(1, 4)
y_vals = np.array([e_y_given_x1, e_y_given_x1, e_y_given_x1])  # Constant for this example
plt.scatter(x_vals, y_vals, color='red', label='Conditional Expectations')
plt.plot(x_vals, y_vals, 'r--', label='Regression Line')
plt.title('Conditional Expectation as Regression')
plt.xlabel('X')
plt.ylabel('E[Y|X]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'conditional_expectation_regression.png'), dpi=100, bbox_inches='tight')
plt.close()

# 3D visualization of conditional expectation as a regression surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of X, Y values
X = np.linspace(0.5, 3.5, 50)
Y = np.linspace(0.5, 3.5, 50)
X, Y = np.meshgrid(X, Y)

# Create a regression surface Z = f(X,Y)
Z = 3 - 0.3*X - 0.2*Y  # Example linear relationship

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, 
                      linewidth=0, antialiased=True)

# Add a few conditional expectation points
x_points = np.array([1, 2, 3])
y_points = np.array([1, 2, 3])
z_points = 3 - 0.3*x_points - 0.2*y_points

ax.scatter(x_points, y_points, z_points, color='black', s=100, label='E[Z|X,Y]')

# Plot conditional expectation line for Y=2
x_line = np.linspace(0.5, 3.5, 20)
y_line = np.ones(20) * 2
z_line = 3 - 0.3*x_line - 0.2*y_line
ax.plot(x_line, y_line, z_line, 'r-', linewidth=4, label='E[Z|X,Y=2]')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('E[Z|X,Y]')
ax.set_title('Regression Surface')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'regression_surface.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Testing Independence
print("\nExample 5: Testing Independence")
print("Determine whether X and Y are independent random variables")

# Check independence for all combinations
print("\nChecking independence:")
is_independent = True
for i in range(3):
    for j in range(3):
        p_xy = joint_pmf[i, j]
        p_x = marginal_x[i]
        p_y = marginal_y[j]
        p_independent = p_x * p_y
        print(f"P(X={i+1}, Y={j+1}) = {p_xy:.2f}")
        print(f"P(X={i+1}) × P(Y={j+1}) = {p_x:.2f} × {p_y:.2f} = {p_independent:.2f}")
        if not np.isclose(p_xy, p_independent, atol=1e-10):
            is_independent = False

print(f"\nX and Y are {'independent' if is_independent else 'not independent'}")

# Calculate covariance
print("\nCalculate covariance:")
e_x = np.sum(np.arange(1, 4) * marginal_x)
e_y = np.sum(np.arange(1, 4) * marginal_y)
e_xy = np.sum(np.outer(np.arange(1, 4), np.arange(1, 4)) * joint_pmf)
cov = e_xy - e_x * e_y
print(f"E[X] = {e_x:.2f}")
print(f"E[Y] = {e_y:.2f}")
print(f"E[XY] = {e_xy:.2f}")
print(f"Cov(X,Y) = E[XY] - E[X]E[Y] = {cov:.2f}")

# Create visualization
plt.figure(figsize=(10, 5))

# Plot joint vs independent probabilities
x = np.arange(3)
y = np.arange(3)
X, Y = np.meshgrid(x, y)
Z_joint = joint_pmf
Z_independent = np.outer(marginal_x, marginal_y)

plt.subplot(121)
plt.imshow(Z_joint, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Joint Probability')
plt.title('Joint PMF')
plt.xticks(range(3), ['Y=1', 'Y=2', 'Y=3'])
plt.yticks(range(3), ['X=1', 'X=2', 'X=3'])

plt.subplot(122)
plt.imshow(Z_independent, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Independent Probability')
plt.title('Product of Marginals')
plt.xticks(range(3), ['Y=1', 'Y=2', 'Y=3'])
plt.yticks(range(3), ['X=1', 'X=2', 'X=3'])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'independence_test_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Additional visualization - Correlation Scatter
plt.figure(figsize=(10, 5))
x_samples = np.random.choice(range(1, 4), size=1000, p=marginal_x)
y_samples = np.random.choice(range(1, 4), size=1000, p=marginal_y)
plt.scatter(x_samples, y_samples, alpha=0.5)
plt.title('Scatter Plot of X and Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'independence_scatter.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Additional visualization - Correlation Matrix
corr_matrix = np.array([[1, rho], [rho, 1]])
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Correlation Matrix')
plt.xticks([0, 1], ['X', 'Y'])
plt.yticks([0, 1], ['X', 'Y'])
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'correlation_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Generating Correlated Random Variables
print("\nExample 6: Generating Correlated Random Variables")
print("Generate bivariate data (X,Y) with:")
print("X ~ N(0,1), Y ~ N(0,1), ρ = 0.8")

# Parameters
rho = 0.8
n_samples = 1000

# Generate independent standard normal variables
z1 = np.random.normal(0, 1, n_samples)
z2 = np.random.normal(0, 1, n_samples)

# Generate correlated variables
x = z1
y = rho * z1 + np.sqrt(1 - rho**2) * z2

# Calculate sample statistics
sample_rho = np.corrcoef(x, y)[0, 1]
print(f"\nSample correlation: {sample_rho:.4f}")

# Create visualization
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter Plot of Correlated Variables')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.hist2d(x, y, bins=30, cmap='viridis')
plt.colorbar(label='Count')
plt.title('2D Histogram')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'correlated_random_variables.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Random Vector Transformation
print("\nExample 7: Random Vector Transformation")
print("Transform X and Y to U = 2X + Y and V = X - Y")
print("Given:")
print("E[X] = 3, E[Y] = 2")
print("Var(X) = 4, Var(Y) = 9")
print("Cov(X,Y) = 2")

# Parameters
e_x, e_y = 3, 2
var_x, var_y = 4, 9
cov_xy = 2

# Step 1: Calculate means
print("\nStep 1: Calculate means")
e_u = 2 * e_x + e_y
e_v = e_x - e_y
print(f"E[U] = 2E[X] + E[Y] = 2×{e_x} + {e_y} = {e_u}")
print(f"E[V] = E[X] - E[Y] = {e_x} - {e_y} = {e_v}")

# Step 2: Calculate variances
print("\nStep 2: Calculate variances")
var_u = 4 * var_x + var_y + 2 * 2 * 1 * cov_xy
var_v = var_x + var_y - 2 * cov_xy
print(f"Var(U) = 4Var(X) + Var(Y) + 2×2×1×Cov(X,Y) = 4×{var_x} + {var_y} + 4×{cov_xy} = {var_u}")
print(f"Var(V) = Var(X) + Var(Y) - 2Cov(X,Y) = {var_x} + {var_y} - 2×{cov_xy} = {var_v}")

# Step 3: Calculate covariance
print("\nStep 3: Calculate covariance")
cov_uv = 2 * var_x - var_y - cov_xy
print(f"Cov(U,V) = 2Var(X) - Var(Y) - Cov(X,Y) = 2×{var_x} - {var_y} - {cov_xy} = {cov_uv}")

# Create visualization
plt.figure(figsize=(10, 5))

# Generate some correlated data
n_samples = 1000
x = np.random.normal(e_x, np.sqrt(var_x), n_samples)
y = np.random.normal(e_y, np.sqrt(var_y), n_samples) + (cov_xy/var_x) * (x - e_x)

# Transform the data
u = 2 * x + y
v = x - y

plt.subplot(121)
plt.scatter(x, y, alpha=0.5)
plt.title('Original Variables (X,Y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.scatter(u, v, alpha=0.5)
plt.title('Transformed Variables (U,V)')
plt.xlabel('U')
plt.ylabel('V')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'linear_transformation_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Additional visualization - Transformation Path
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(x, y, alpha=0.5, label='Original')
plt.title('Original Variables')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.scatter(u, v, alpha=0.5, label='Transformed')
plt.title('Transformed Variables')
plt.xlabel('U')
plt.ylabel('V')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transformation_path.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Multivariate Gaussian
print("\nExample 8: Multivariate Gaussian")
print("Three facial measurements (in millimeters):")
print("X1: Distance between eyes")
print("X2: Width of nose")
print("X3: Width of mouth")

# Parameters
mu = np.array([32, 25, 50])
cov = np.array([
    [16, 4, 6],
    [4, 25, 10],
    [6, 10, 36]
])

print("\nMean vector:")
print(mu)
print("\nCovariance matrix:")
print(cov)

# Step 1: Write the multivariate Gaussian PDF
print("\nStep 1: Write the multivariate Gaussian PDF")
print("f(x) = (2π)^(-n/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^(-1) (x-μ))")
print(f"where n = 3, μ = {mu}, Σ = {cov}")

# Step 2: Find marginal distribution of X1
print("\nStep 2: Find marginal distribution of X1")
print(f"X1 ~ N(μ1, σ1²) = N({mu[0]}, {cov[0,0]})")

# Step 3: Find conditional distribution of X1 given X2=30, X3=45
print("\nStep 3: Find conditional distribution of X1 given X2=30, X3=45")

# Partition the covariance matrix
sigma_11 = cov[0,0]
sigma_12 = cov[0,1:]
sigma_21 = cov[1:,0]
sigma_22 = cov[1:,1:]

# Calculate conditional parameters
sigma_22_inv = np.linalg.inv(sigma_22)
mu_1 = mu[0]
mu_2 = mu[1:]
y = np.array([30, 45])

# Conditional mean
mu_1_given_2 = mu_1 + sigma_12 @ sigma_22_inv @ (y - mu_2)

# Conditional variance
sigma_1_given_2 = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21

print(f"X1|(X2=30, X3=45) ~ N({mu_1_given_2:.2f}, {sigma_1_given_2:.2f})")

# Create visualization
plt.figure(figsize=(15, 5))

# Generate samples from the multivariate normal
n_samples = 1000
samples = np.random.multivariate_normal(mu, cov, n_samples)

# Plot marginal distributions
plt.subplot(131)
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.7)
x = np.linspace(mu[0] - 3*np.sqrt(cov[0,0]), mu[0] + 3*np.sqrt(cov[0,0]), 100)
plt.plot(x, stats.norm.pdf(x, mu[0], np.sqrt(cov[0,0])), 'r-')
plt.title('Marginal Distribution of X1')
plt.xlabel('X1')
plt.ylabel('Density')

# Plot conditional distribution
plt.subplot(132)
x = np.linspace(mu_1_given_2 - 3*np.sqrt(sigma_1_given_2), 
                mu_1_given_2 + 3*np.sqrt(sigma_1_given_2), 100)
plt.plot(x, stats.norm.pdf(x, mu_1_given_2, np.sqrt(sigma_1_given_2)), 'b-')
plt.title('Conditional Distribution of X1|(X2=30, X3=45)')
plt.xlabel('X1')
plt.ylabel('Density')

# Plot scatter of X1 vs X2
plt.subplot(133)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.axvline(mu_1_given_2, color='red', linestyle='--', 
            label=f'Conditional mean = {mu_1_given_2:.2f}')
plt.title('X1 vs X2')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multivariate_gaussian_visualization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 8: Additional visualization - 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5)
ax.set_title('3D Scatter Plot of Multivariate Gaussian')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multivariate_3d_scatter.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 9: Simple Pen-and-Paper Example
print("\nExample 9: Simple Pen-and-Paper Example")
print("Consider two fair coins. Let X be the number of heads on the first coin,")
print("and Y be the number of heads on the second coin.")

# Define the joint PMF
joint_pmf = np.array([
    [0.25, 0.25],
    [0.25, 0.25]
])

# Create visualization
plt.figure(figsize=(10, 5))

# Plot joint PMF
plt.subplot(121)
plt.imshow(joint_pmf, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks([0, 1], ['Y=0', 'Y=1'])
plt.yticks([0, 1], ['X=0', 'X=1'])
plt.title('Joint PMF of Two Fair Coins')
plt.xlabel('Y (Second Coin)')
plt.ylabel('X (First Coin)')

# Add probability values
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{joint_pmf[i, j]:.2f}", 
                 ha='center', va='center', color='black')

# Plot scatter of samples
plt.subplot(122)
n_samples = 1000
x_samples = np.random.binomial(1, 0.5, n_samples)
y_samples = np.random.binomial(1, 0.5, n_samples)
plt.scatter(x_samples, y_samples, alpha=0.5)
plt.title('Scatter Plot of Coin Tosses')
plt.xlabel('X (First Coin)')
plt.ylabel('Y (Second Coin)')
plt.xticks([0, 1])
plt.yticks([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'simple_coin_example.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 9: Additional visualization - Chi-square Test
plt.figure(figsize=(10, 6))
# Use the coin example joint PMF
coin_joint_pmf = np.array([[0.25, 0.25], [0.25, 0.25]])
coin_marginal_x = coin_joint_pmf.sum(axis=1)
coin_marginal_y = coin_joint_pmf.sum(axis=0)
expected = np.outer(coin_marginal_x, coin_marginal_y)
observed = coin_joint_pmf
chi_square = np.sum((observed - expected)**2 / expected)

plt.bar(['Observed', 'Expected'], 
        [np.sum(observed), np.sum(expected)],
        color=['blue', 'orange'])
plt.title(f'Chi-square Test (χ² = {chi_square:.3f})')
plt.ylabel('Total Probability')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'chi_square_test.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll joint distribution examples completed successfully!") 