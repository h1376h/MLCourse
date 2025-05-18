import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Define the problem parameters
# Step 1: Define the mean vectors and covariance matrices for all classes
mu0 = np.array([0, 0])  # Mean for class 0
mu1 = np.array([4, 0])  # Mean for class 1
mu2 = np.array([2, 3])  # Mean for class 2

# Initial covariance matrices (all identity)
Sigma0 = np.array([[1, 0], [0, 1]])  # Covariance matrix for class 0
Sigma1 = np.array([[1, 0], [0, 1]])  # Covariance matrix for class 1
Sigma2 = np.array([[1, 0], [0, 1]])  # Covariance matrix for class 2

# Modified covariance matrix for class 2 (for part 4)
Sigma2_modified = np.array([[3, 0], [0, 3]])  # Modified covariance matrix for class 2

# Prior probabilities (equal)
prior0 = 1/3  # Prior probability for class 0
prior1 = 1/3  # Prior probability for class 1
prior2 = 1/3  # Prior probability for class 2

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
x = np.linspace(-2, 6, 300)
y = np.linspace(-2, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack((X, Y), axis=2)  # Stack all (x,y) pairs

# Step 4: Compute probability density functions for all classes
Z0 = np.zeros_like(X)
Z1 = np.zeros_like(X)
Z2 = np.zeros_like(X)
Z2_modified = np.zeros_like(X)  # For modified covariance

# Calculate PDF values at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z0[i, j] = gaussian_pdf(point, mu0, Sigma0)
        Z1[i, j] = gaussian_pdf(point, mu1, Sigma1)
        Z2[i, j] = gaussian_pdf(point, mu2, Sigma2)
        Z2_modified[i, j] = gaussian_pdf(point, mu2, Sigma2_modified)

# Step 5: Compute posterior probabilities
# p(y=k|x) = p(x|y=k)p(y=k) / [sum_j p(x|y=j)p(y=j)]

# For original case with equal covariances
evidence = prior0 * Z0 + prior1 * Z1 + prior2 * Z2
posterior0 = prior0 * Z0 / evidence
posterior1 = prior1 * Z1 / evidence
posterior2 = prior2 * Z2 / evidence

# For modified case with different covariance for class 2
evidence_modified = prior0 * Z0 + prior1 * Z1 + prior2 * Z2_modified
posterior0_modified = prior0 * Z0 / evidence_modified
posterior1_modified = prior1 * Z1 / evidence_modified
posterior2_modified = prior2 * Z2_modified / evidence_modified

# Step 6: Function to add covariance ellipses
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

# Step 7: Function to derive and plot decision boundaries
def plot_decision_boundaries(mu0, mu1, mu2, Sigma0, Sigma1, Sigma2, title, filename):
    """
    Derive and plot decision boundaries for the three-class classification problem.
    
    Args:
        mu0, mu1, mu2: Mean vectors for the three classes
        Sigma0, Sigma1, Sigma2: Covariance matrices for the three classes
        title: Plot title
        filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate decision boundaries
    # For equal covariance matrices, the decision boundary between two classes
    # is a straight line that is the perpendicular bisector of the line connecting the means
    # For a more general case with different covariances, we need to use the full formula
    
    # Get inverse covariance matrices
    Sigma0_inv = np.linalg.inv(Sigma0)
    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)
    
    # Get determinants
    det0 = np.linalg.det(Sigma0)
    det1 = np.linalg.det(Sigma1)
    det2 = np.linalg.det(Sigma2)
    
    # Function to calculate the discriminant function (log posterior)
    def discriminant(x, mu, Sigma_inv, det, prior):
        return -0.5 * np.dot(np.dot((x - mu).T, Sigma_inv), (x - mu)) - 0.5 * np.log(det) + np.log(prior)
    
    # Calculate discriminant values for each class at each point in the grid
    D0 = np.zeros_like(X)
    D1 = np.zeros_like(X)
    D2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            D0[i, j] = discriminant(point, mu0, Sigma0_inv, det0, prior0)
            D1[i, j] = discriminant(point, mu1, Sigma1_inv, det1, prior1)
            D2[i, j] = discriminant(point, mu2, Sigma2_inv, det2, prior2)
    
    # Determine the predicted class at each point
    predicted_class = np.zeros_like(X)
    predicted_class[D1 > D0] = 1
    predicted_class[(D2 > D0) & (D2 > D1)] = 2
    
    # Plot the decision regions
    plt.contourf(X, Y, predicted_class, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.3, 
                 colors=['lightblue', 'lightgreen', 'salmon'])
    
    # Plot the contours of the discriminant functions to show the decision boundaries
    # The decision boundary between classes i and j is where Di = Dj
    plt.contour(X, Y, D0 - D1, levels=[0], colors=['blue'], linestyles=['--'], linewidths=2)
    plt.contour(X, Y, D0 - D2, levels=[0], colors=['red'], linestyles=['--'], linewidths=2)
    plt.contour(X, Y, D1 - D2, levels=[0], colors=['green'], linestyles=['--'], linewidths=2)
    
    # Add covariance ellipses
    add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
    add_covariance_ellipse(mu1, Sigma1, 'green', r'Class 1: $2\sigma$ region')
    add_covariance_ellipse(mu2, Sigma2, 'red', r'Class 2: $2\sigma$ region')
    
    # Add class means as points
    plt.scatter(mu0[0], mu0[1], color='blue', s=100, marker='o', label='Class 0 Mean')
    plt.scatter(mu1[0], mu1[1], color='green', s=100, marker='o', label='Class 1 Mean')
    plt.scatter(mu2[0], mu2[1], color='red', s=100, marker='o', label='Class 2 Mean')
    
    # Add labels for clarity
    plt.text(mu0[0]+0.3, mu0[1]+0.3, r'$\mu_0$', fontsize=12)
    plt.text(mu1[0]+0.3, mu1[1]+0.3, r'$\mu_1$', fontsize=12)
    plt.text(mu2[0]+0.3, mu2[1]+0.3, r'$\mu_2$', fontsize=12)
    
    # Draw the coordinate axes
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and legend
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Set the x and y limits
    plt.xlim([-2, 6])
    plt.ylim([-2, 6])
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    
    custom_lines = [
        Line2D([0], [0], color='blue', linestyle='--', lw=2),
        Line2D([0], [0], color='red', linestyle='--', lw=2),
        Line2D([0], [0], color='green', linestyle='--', lw=2)
    ]
    
    plt.legend(custom_lines, ['Decision Boundary 0-1', 'Decision Boundary 0-2', 'Decision Boundary 1-2'],
               loc='lower right')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    
    return predicted_class

# Step 8: Plot Gaussian contours for all classes
plt.figure(figsize=(10, 8))

# Plot contours for p(x|y=0)
contour0 = plt.contour(X, Y, Z0, levels=5, colors='blue', alpha=0.7, linestyles='solid')
plt.clabel(contour0, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=1)
contour1 = plt.contour(X, Y, Z1, levels=5, colors='green', alpha=0.7, linestyles='solid')
plt.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

# Plot contours for p(x|y=2)
contour2 = plt.contour(X, Y, Z2, levels=5, colors='red', alpha=0.7, linestyles='solid')
plt.clabel(contour2, inline=True, fontsize=8, fmt='%.2f')

# Add covariance ellipses
add_covariance_ellipse(mu0, Sigma0, 'blue', r'Class 0: $2\sigma$ region')
add_covariance_ellipse(mu1, Sigma1, 'green', r'Class 1: $2\sigma$ region')
add_covariance_ellipse(mu2, Sigma2, 'red', r'Class 2: $2\sigma$ region')

# Add class means as points
plt.scatter(mu0[0], mu0[1], color='blue', s=100, marker='o', label='Class 0 Mean')
plt.scatter(mu1[0], mu1[1], color='green', s=100, marker='o', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], color='red', s=100, marker='o', label='Class 2 Mean')

# Add labels for clarity
plt.text(mu0[0]+0.3, mu0[1]+0.3, r'$\mu_0$', fontsize=12)
plt.text(mu1[0]+0.3, mu1[1]+0.3, r'$\mu_1$', fontsize=12)
plt.text(mu2[0]+0.3, mu2[1]+0.3, r'$\mu_2$', fontsize=12)

# Draw the coordinate axes
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add labels and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gaussian Distributions for Three Classes')
plt.grid(True, alpha=0.3)

# Set the x and y limits
plt.xlim([-2, 6])
plt.ylim([-2, 6])

# Save the figure
plt.savefig(os.path.join(save_dir, 'gaussian_distributions.png'), dpi=300, bbox_inches='tight')

# Step 9: Derive and plot decision boundaries for original case (identical covariances)
predicted_class = plot_decision_boundaries(
    mu0, mu1, mu2, Sigma0, Sigma1, Sigma2,
    'Decision Boundaries with Identical Covariance Matrices',
    'decision_boundaries_original.png'
)

# Step 10: Derive and plot decision boundaries for modified case (different covariance for class 2)
predicted_class_modified = plot_decision_boundaries(
    mu0, mu1, mu2, Sigma0, Sigma1, Sigma2_modified,
    'Decision Boundaries with Modified Covariance Matrix for Class 2',
    'decision_boundaries_modified.png'
)

# Step 11: Calculate and visualize posterior probabilities for original case
plt.figure(figsize=(15, 5))

# Plot posterior probability for class 0
plt.subplot(1, 3, 1)
plt.imshow(posterior0, extent=[-2, 6, -2, 6], origin='lower', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label='$P(y=0|x)$')
plt.title('Posterior Probability for Class 0')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot posterior probability for class 1
plt.subplot(1, 3, 2)
plt.imshow(posterior1, extent=[-2, 6, -2, 6], origin='lower', cmap='Greens', vmin=0, vmax=1)
plt.colorbar(label='$P(y=1|x)$')
plt.title('Posterior Probability for Class 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot posterior probability for class 2
plt.subplot(1, 3, 3)
plt.imshow(posterior2, extent=[-2, 6, -2, 6], origin='lower', cmap='Reds', vmin=0, vmax=1)
plt.colorbar(label='$P(y=2|x)$')
plt.title('Posterior Probability for Class 2')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_probabilities_original.png'), dpi=300, bbox_inches='tight')

# Step 12: Calculate and visualize posterior probabilities for modified case
plt.figure(figsize=(15, 5))

# Plot posterior probability for class 0
plt.subplot(1, 3, 1)
plt.imshow(posterior0_modified, extent=[-2, 6, -2, 6], origin='lower', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label='$P(y=0|x)$')
plt.title('Posterior Probability for Class 0')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot posterior probability for class 1
plt.subplot(1, 3, 2)
plt.imshow(posterior1_modified, extent=[-2, 6, -2, 6], origin='lower', cmap='Greens', vmin=0, vmax=1)
plt.colorbar(label='$P(y=1|x)$')
plt.title('Posterior Probability for Class 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Plot posterior probability for class 2
plt.subplot(1, 3, 3)
plt.imshow(posterior2_modified, extent=[-2, 6, -2, 6], origin='lower', cmap='Reds', vmin=0, vmax=1)
plt.colorbar(label='$P(y=2|x)$')
plt.title('Posterior Probability for Class 2')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'posterior_probabilities_modified.png'), dpi=300, bbox_inches='tight')

# Step 13: Mathematical derivation (output to console)
print("\nMathematical Derivation of Decision Boundaries:")
print("-" * 50)

print("\nFor classes with equal covariance matrices Σ = I:")
print("The discriminant function for class k is:")
print("  δₖ(x) = -0.5(x - μₖ)ᵀ(x - μₖ) + log(P(y=k))")
print("  δₖ(x) = -0.5(x - μₖ)ᵀ(x - μₖ) - 0.5log(|Σ|) + log(P(y=k))")
print("Since the covariance matrices are identical, and the priors are equal,")
print("we can simplify by dropping constant terms that are the same for all classes.")
print("The decision boundary between class i and j is where:")
print("  -0.5(x - μᵢ)ᵀ(x - μᵢ) = -0.5(x - μⱼ)ᵀ(x - μⱼ)")
print("Expanding and simplifying:")
print("  -0.5(xᵀx - 2μᵢᵀx + μᵢᵀμᵢ) = -0.5(xᵀx - 2μⱼᵀx + μⱼᵀμⱼ)")
print("  -xᵀx + 2μᵢᵀx - μᵢᵀμᵢ = -xᵀx + 2μⱼᵀx - μⱼᵀμⱼ")
print("  2μᵢᵀx - μᵢᵀμᵢ = 2μⱼᵀx - μⱼᵀμⱼ")
print("  2(μᵢ - μⱼ)ᵀx = μᵢᵀμᵢ - μⱼᵀμⱼ")
print("  (μᵢ - μⱼ)ᵀx = 0.5(μᵢᵀμᵢ - μⱼᵀμⱼ)")
print("This is the equation of a line perpendicular to (μᵢ - μⱼ) with a specific offset.")

print("\nFor the modified case where Σ₂ = [[3, 0], [0, 3]] = 3I:")
print("Since Σ₂ = 3I is still a scalar multiple of the identity matrix,")
print("the decision boundaries are still linear, but they shift.")
print("The discriminant function for class 2 becomes:")
print("  δ₂(x) = -0.5(x - μ₂)ᵀ(3I)⁻¹(x - μ₂) - 0.5log(|3I|) + log(P(y=2))")
print("  δ₂(x) = -0.5(x - μ₂)ᵀ(1/3)I(x - μ₂) - 0.5log(3²) + log(P(y=2))")
print("  δ₂(x) = -(1/6)(x - μ₂)ᵀ(x - μ₂) - 0.5log(9) + log(P(y=2))")
print("The increased covariance makes the quadratic term smaller, indicating")
print("that the density decreases more slowly as we move away from the mean.")
print("The decision boundary between class i and class 2 is where:")
print("  -0.5(x - μᵢ)ᵀ(x - μᵢ) = -(1/6)(x - μ₂)ᵀ(x - μ₂) - 0.5log(9)")
print("This is no longer a simple line; it becomes a quadratic curve.")

print("\nConclusion:")
print("1. With identical covariance matrices, decision boundaries are straight lines.")
print("2. With different covariance matrices, decision boundaries become quadratic curves.")
print("3. The shape of the decision regions reflects the structure of the covariance matrices.")

print(f"\nAll visualizations have been saved to: {save_dir}")

# Display the plots (commented for automation)
# plt.show() 