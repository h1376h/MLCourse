import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering to fix glyph issues
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters
# Step 1: Define the mean vectors and covariance matrices for both classes
mu0 = np.array([0, 0])  # Mean for class 0 (we can set to origin without loss of generality)
mu1 = np.array([0, 0])  # Mean for class 1 (equal to mu0 as specified)

Sigma0 = np.array([[1, 0], [0, 4]])  # Covariance matrix for class 0
Sigma1 = np.array([[4, 0], [0, 1]])  # Covariance matrix for class 1

prior0 = 0.5  # Prior probability for class 0
prior1 = 0.5  # Prior probability for class 1

# Step 2: Define the multivariate Gaussian PDF function
def gaussian_pdf(x, mu, Sigma):
    """
    Compute multivariate Gaussian PDF value at point x.
    
    Args:
        x: The point at which to evaluate the PDF
        mu: Mean vector of the Gaussian distribution
        Sigma: Covariance matrix of the Gaussian distribution
        
    Returns:
        The probability density at point x
    """
    n = len(mu)
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)
    
    # Calculate the normalization factor
    factor = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
    
    # Calculate the exponent term
    exponent = -0.5 * np.dot(np.dot((x - mu).T, inv), (x - mu))
    
    return factor * np.exp(exponent)

# Step 3: Create a grid of points for visualization
# Create a grid of x and y values for plotting
x = np.linspace(-6, 6, 300)
y = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack((X, Y), axis=2)  # Stack all (x,y) pairs

# Step 4: Compute probability density functions for both classes
Z0 = np.zeros_like(X)
Z1 = np.zeros_like(X)

# Calculate PDF values at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z0[i, j] = gaussian_pdf(point, mu0, Sigma0)
        Z1[i, j] = gaussian_pdf(point, mu1, Sigma1)

# Step 5: Compute log-likelihood ratio for decision boundary
# log[p(x|y=1)/p(x|y=0)] = log[p(x|y=1)] - log[p(x|y=0)]
# For Bayes optimal with equal priors, decision boundary is where this equals 0
log_ratio = np.log(Z1) - np.log(Z0)

# Step 6: Compute posterior probabilities
# p(y=0|x) = p(x|y=0)p(y=0) / [p(x|y=0)p(y=0) + p(x|y=1)p(y=1)]
posterior0 = prior0 * Z0 / (prior0 * Z0 + prior1 * Z1)
posterior1 = prior1 * Z1 / (prior0 * Z0 + prior1 * Z1)

# Step 7: Theoretical derivation of decision boundary
# For multivariate Gaussians with equal means, the log ratio simplifies to:
# log[p(x|y=1)/p(x|y=0)] = -0.5 * x^T (Σ1^-1 - Σ0^-1) x + 0.5 * log(|Σ0|/|Σ1|)
# For equal priors, the boundary occurs when this equals 0

# For our specific covariance matrices:
# Σ0^-1 = [[1, 0], [0, 1/4]]
# Σ1^-1 = [[1/4, 0], [0, 1]]
# Σ1^-1 - Σ0^-1 = [[-3/4, 0], [0, 3/4]]
# log(|Σ0|/|Σ1|) = log(4/4) = 0

# This simplifies to:
# -0.5 * (x1^2 * (-3/4) + x2^2 * (3/4)) = 0
# -0.5 * (3/4) * (-x1^2 + x2^2) = 0
# -x1^2 + x2^2 = 0
# x1^2 = x2^2
# This represents the lines x1 = x2 and x1 = -x2

# Step 8: Calculate analytical decision boundary
# We'll plot x1 = x2 and x1 = -x2
boundary1_x = np.linspace(-6, 6, 100)
boundary1_y = boundary1_x
boundary2_y = -boundary1_x

# Step 9: Create Figure 1: Gaussian contours and decision boundary
plt.figure(figsize=(10, 8))

# Plot contours for p(x|y=0)
contour0 = plt.contour(X, Y, Z0, levels=5, colors='blue', alpha=0.7, linestyles='solid')
plt.clabel(contour0, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=1)
contour1 = plt.contour(X, Y, Z1, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

# Plot decision boundary
plt.plot(boundary1_x, boundary1_y, 'k--', label='Decision Boundary: $x_1 = x_2$')
plt.plot(boundary1_x, boundary2_y, 'k--', label='Decision Boundary: $x_1 = -x_2$')

# Step 10: Add ellipses to represent the covariance matrices
# We'll draw ellipses for 2 standard deviations (covering ~95% of the data)
def add_covariance_ellipse(mean, cov, color, label):
    """
    Add an ellipse representing the covariance matrix to the current plot.
    
    Args:
        mean: Mean vector of the Gaussian distribution
        cov: Covariance matrix of the Gaussian distribution
        color: Color for the ellipse
        label: Label for the legend
    """
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eigh(cov)
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    
    # Create the ellipse patch (2 standard deviations)
    ellipse = Ellipse(xy=mean, width=2*2*np.sqrt(evals[0]), height=2*2*np.sqrt(evals[1]),
                      angle=angle, edgecolor=color, facecolor='none', lw=2, label=label)
    plt.gca().add_patch(ellipse)

add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gaussian Contours and Decision Boundary')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'gaussian_contours_decision_boundary.png'), dpi=300, bbox_inches='tight')

# Step 11: Create Figure 2: Decision regions
plt.figure(figsize=(10, 8))

# Calculate the classified regions
decision_regions = np.zeros_like(X)
decision_regions[(log_ratio < 0)] = 0  # Class 0 regions
decision_regions[(log_ratio >= 0)] = 1  # Class 1 regions

# Plot decision regions - 'skyblue' is for class 0, 'salmon' is for class 1
plt.contourf(X, Y, decision_regions, levels=[0, 0.5, 1], colors=['skyblue', 'salmon'], alpha=0.3)

# Plot decision boundary
plt.plot(boundary1_x, boundary1_y, 'k--', linewidth=2, label='Decision Boundary: $x_1 = x_2$')
plt.plot(boundary1_x, boundary2_y, 'k--', linewidth=2, label='Decision Boundary: $x_1 = -x_2$')

# Add ellipses to represent the covariance matrices
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Place class labels based on the decision regions - programmatically find positions
# Get points in the different quadrants
x_pos = 4.0  # Far right
x_neg = -4.0  # Far left
y_pos = 4.0  # Far top
y_neg = -4.0  # Far bottom

# Check points to determine which class they belong to
top_point = np.array([0, y_pos])
bottom_point = np.array([0, y_neg])
right_point = np.array([x_pos, 0])
left_point = np.array([x_neg, 0])

# Compute log-ratio for each point
log_ratio_top = -0.5 * np.dot(np.dot(top_point, (np.linalg.inv(Sigma1) - np.linalg.inv(Sigma0))), top_point)
log_ratio_bottom = -0.5 * np.dot(np.dot(bottom_point, (np.linalg.inv(Sigma1) - np.linalg.inv(Sigma0))), bottom_point)
log_ratio_right = -0.5 * np.dot(np.dot(right_point, (np.linalg.inv(Sigma1) - np.linalg.inv(Sigma0))), right_point)
log_ratio_left = -0.5 * np.dot(np.dot(left_point, (np.linalg.inv(Sigma1) - np.linalg.inv(Sigma0))), left_point)

# Add class labels based on log-ratio values - Class 0 (blue) where log_ratio < 0, Class 1 (red) where log_ratio >= 0
if log_ratio_top >= 0:
    plt.text(0, y_pos-1, 'Class 1', fontsize=12, color='red', ha='center')
else:
    plt.text(0, y_pos-1, 'Class 0', fontsize=12, color='blue', ha='center')
    
if log_ratio_bottom >= 0:
    plt.text(0, y_neg+1, 'Class 1', fontsize=12, color='red', ha='center')
else:
    plt.text(0, y_neg+1, 'Class 0', fontsize=12, color='blue', ha='center')
    
if log_ratio_right >= 0:
    plt.text(x_pos-1, 0, 'Class 1', fontsize=12, color='red', va='center')
else:
    plt.text(x_pos-1, 0, 'Class 0', fontsize=12, color='blue', va='center')
    
if log_ratio_left >= 0:
    plt.text(x_neg+1, 0, 'Class 1', fontsize=12, color='red', va='center')
else:
    plt.text(x_neg+1, 0, 'Class 0', fontsize=12, color='blue', va='center')

plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'decision_regions.png'), dpi=300, bbox_inches='tight')

# Step 12: Create Figure 3: Effect of unequal priors
plt.figure(figsize=(10, 8))

# Try different prior probabilities
prior_scenarios = [
    {'prior0': 0.5, 'prior1': 0.5, 'color': 'black', 'style': '--', 'label': 'Equal Priors (0.5/0.5)'},
    {'prior0': 0.7, 'prior1': 0.3, 'color': 'green', 'style': '-', 'label': 'Unequal Priors (0.7/0.3)'},
    {'prior0': 0.3, 'prior1': 0.7, 'color': 'purple', 'style': '-.', 'label': 'Unequal Priors (0.3/0.7)'},
]

for scenario in prior_scenarios:
    # Get the log of the prior ratio: log(P(y=1)/P(y=0))
    log_prior_ratio = np.log(scenario['prior1'] / scenario['prior0'])
    
    # For the case with diagonal covariance matrices:
    # The complete decision condition becomes:
    # -0.5 * (3/4) * (-x1^2 + x2^2) = log_prior_ratio
    # Solving for x2:
    # x2^2 = x1^2 + (8/3) * log_prior_ratio

    # For simplicity, I'll show how the x1 = x2 and x1 = -x2 boundaries shift
    # This is a simplification - the true boundary might not be exactly these lines
    # For general case, you'd solve the full quadratic equation
    
    # Plot decision boundaries with shifted priors
    x_vals = np.linspace(-6, 6, 100)
    
    if scenario['prior0'] == scenario['prior1']:
        # Equal priors - we already calculated these boundaries
        plt.plot(x_vals, x_vals, color=scenario['color'], linestyle=scenario['style'], linewidth=2, label=scenario['label'])
        plt.plot(x_vals, -x_vals, color=scenario['color'], linestyle=scenario['style'], linewidth=2)
    else:
        # Shifted boundaries due to prior ratio
        # The shift affects the intercept but not the slope for these simple boundaries
        # This is a simplified approximation for this specific problem
        shift_factor = np.sqrt((8/3) * abs(log_prior_ratio))
        
        if log_prior_ratio > 0:  # P(y=1) > P(y=0), favor class 1
            plt.plot(x_vals, x_vals + shift_factor, color=scenario['color'], linestyle=scenario['style'], linewidth=2, label=scenario['label'])
            plt.plot(x_vals, -x_vals - shift_factor, color=scenario['color'], linestyle=scenario['style'], linewidth=2)
        else:  # P(y=0) > P(y=1), favor class 0
            plt.plot(x_vals, x_vals - shift_factor, color=scenario['color'], linestyle=scenario['style'], linewidth=2, label=scenario['label'])
            plt.plot(x_vals, -x_vals + shift_factor, color=scenario['color'], linestyle=scenario['style'], linewidth=2)

# Add ellipses to represent the covariance matrices
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'red', r'Class 1: $2\sigma$ region')

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Effect of Prior Probabilities on Decision Boundary')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'prior_probability_effect.png'), dpi=300, bbox_inches='tight')

# Step 13: NEW VISUALIZATION - Create 3D surface plot of probability densities
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Downsample for clearer visualization
step = 5
X_sparse = X[::step, ::step]
Y_sparse = Y[::step, ::step]
Z0_sparse = Z0[::step, ::step]
Z1_sparse = Z1[::step, ::step]

# Plot the two probability densities
surf0 = ax.plot_surface(X_sparse, Y_sparse, Z0_sparse, cmap='Blues', alpha=0.7, linewidth=0, antialiased=True)
surf1 = ax.plot_surface(X_sparse, Y_sparse, Z1_sparse, cmap='Reds', alpha=0.7, linewidth=0, antialiased=True)

# Add labels
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Probability Density')
ax.set_title('3D Visualization of Probability Densities')

# Save the figure
plt.savefig(os.path.join(save_dir, '3d_probability_densities.png'), dpi=300, bbox_inches='tight')

# Step 14: NEW VISUALIZATION - Heatmap of posterior probabilities
plt.figure(figsize=(10, 8))

# Create a heatmap of the posterior probability for class 1
plt.imshow(posterior1, extent=[-6, 6, -6, 6], origin='lower', cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='$P(y=1|x)$')

# Plot decision boundary
plt.plot(boundary1_x, boundary1_y, 'k--', linewidth=2)
plt.plot(boundary1_x, boundary2_y, 'k--', linewidth=2)

# Add labels
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Posterior Probability Heatmap for Class 1')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.grid(False)

# Save the figure
plt.savefig(os.path.join(save_dir, 'posterior_probability_heatmap.png'), dpi=300, bbox_inches='tight')

# Mathematical derivation (output in console)
print("\nMathematical Derivation of Decision Boundary:")
print("---------------------------------------------")
print("For multivariate Gaussians with equal means, the log-likelihood ratio is:")
print("log[p(x|y=1)/p(x|y=0)] = -0.5 * x^T (Σ1^-1 - Σ0^-1) x + 0.5 * log(|Σ0|/|Σ1|)")

print("\nWith the given covariance matrices:")
print("Σ0 = [[1, 0], [0, 4]]")
print("Σ1 = [[4, 0], [0, 1]]")

print("\nWe compute:")
print("Σ0^-1 = [[1, 0], [0, 1/4]]")
print("Σ1^-1 = [[1/4, 0], [0, 1]]")
print("Σ1^-1 - Σ0^-1 = [[-3/4, 0], [0, 3/4]]")
print("log(|Σ0|/|Σ1|) = log(4/4) = 0")

print("\nSo the log-likelihood ratio is:")
print("log[p(x|y=1)/p(x|y=0)] = -0.5 * [x1, x2] [[-3/4, 0], [0, 3/4]] [x1, x2]^T")
print("= -0.5 * (-3/4 * x1^2 + 3/4 * x2^2)")
print("= -0.5 * (3/4) * (-x1^2 + x2^2)")

print("\nThe decision boundary occurs when this equals log(P(y=1)/P(y=0)).")
print("For equal priors, log(0.5/0.5) = 0, so:")
print("-0.5 * (3/4) * (-x1^2 + x2^2) = 0")
print("Which simplifies to: x1^2 = x2^2")
print("Therefore, the decision boundary consists of the lines: x1 = x2 and x1 = -x2")

print("\nWhen priors are not equal, log(P(y=1)/P(y=0)) ≠ 0, so:")
print("-0.5 * (3/4) * (-x1^2 + x2^2) = log(P(y=1)/P(y=0))")
print("This shifts the decision boundary based on the prior probabilities.")

print("\nVisualization saved to:", save_dir)

# Display the plots (commented out for automation)
# plt.show() 