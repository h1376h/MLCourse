import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy.stats import multivariate_normal
import matplotlib as mpl

# Set a more modern style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

print("Question 9: Linear Discriminant Analysis (LDA) Assumptions")
print("=======================================================")

# Step 1: Explain the key assumptions of LDA
print("\nStep 1: Key assumptions of LDA")
print("----------------------------")

print("Linear Discriminant Analysis (LDA) makes several important assumptions:")
print("1. The classes have multivariate Gaussian (normal) distributions")
print("2. The classes share the same covariance matrix (homoscedasticity)")
print("3. The features are not perfectly correlated (no multi-collinearity)")
print("4. The sample size is larger than the number of features")
print("\nWhen these assumptions are met, LDA provides the optimal decision boundary")
print("in the sense of minimizing the Bayes error rate.")

# Step 2: Visualize the LDA assumptions
print("\nStep 2: Visualizing LDA assumptions")
print("--------------------------------")

# Set random seed for reproducibility
np.random.seed(42)

# Define the parameters for our classes in a more general way
# Instead of hardcoding values, we'll use variables
class_means = {
    1: np.array([2, 3]),  # Class 1 mean
    2: np.array([4, 1])   # Class 2 mean
}
cov_matrix = np.array([[2, 0], [0, 1]])  # Shared covariance matrix

# Generate data for classes
n_samples = 200
X = {}  # Dictionary to store data for each class
for class_label, mean in class_means.items():
    X[class_label] = np.random.multivariate_normal(mean, cov_matrix, n_samples)

# Plot data and equal-probability contours with cleaner aesthetics
plt.figure(figsize=(10, 8))
colors = {1: '#3498db', 2: '#e74c3c'}  # Blue for class 1, red for class 2
labels = {1: 'Class 1', 2: 'Class 2'}

# Plot the data points
for class_label, data in X.items():
    plt.scatter(data[:, 0], data[:, 1], c=colors[class_label], alpha=0.5, 
                edgecolors='none', label=labels[class_label])

# Plot class means
for class_label, mean in class_means.items():
    plt.scatter(mean[0], mean[1], c=colors[class_label], s=150, marker='X', 
                edgecolor='k', linewidth=2, label=f'{labels[class_label]} Mean')

# Function to create an ellipse representing equal-probability contours
def plot_ellipse(mean, cov, color, alpha=0.3, std_multiplier=2):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * std_multiplier * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                     facecolor=color, alpha=alpha, edgecolor='black')
    return ellipse

# Add equal-probability contours (2-sigma)
ax = plt.gca()
for class_label, mean in class_means.items():
    ax.add_patch(plot_ellipse(mean, cov_matrix, colors[class_label], alpha=0.2))

# Add axis labels and title with better formatting
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Assumption: Equal Covariance Matrices', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, frameon=True, framealpha=0.9)
plt.tight_layout()

# Add text explaining the assumption visually
plt.text(0.02, 0.98, "Both classes have the same shape and orientation\n(equal covariance matrices)",
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Save the plot
plt.savefig(os.path.join(save_dir, "lda_assumptions_equal_covariance.png"), dpi=300, bbox_inches='tight')

# Step 3: Calculate the LDA projection vector w - Detailed step-by-step calculation
print("\nStep 3: Calculating the LDA projection vector")
print("-----------------------------------------")

# Use the means from the problem statement
mu1 = np.array([2, 3])  # Class 1 mean from problem statement
mu2 = np.array([4, 1])  # Class 2 mean from problem statement
Sigma = np.array([[2, 0], [0, 1]])  # Shared covariance matrix

print("DETAILED STEP-BY-STEP CALCULATION OF LDA PROJECTION VECTOR:")
print("--------------------------------------------------")
print(f"Given information:")
print(f"  Class 1 mean (μ₁): {mu1} (from problem statement)")
print(f"  Class 2 mean (μ₂): {mu2} (from problem statement)")
print(f"  Shared covariance matrix (Σ):\n{Sigma}")

# Step 3.1: Calculate the inverse of the covariance matrix
print("\nStep 3.1: Calculate the inverse of the covariance matrix:")
try:
    # Check if matrix is invertible
    det = np.linalg.det(Sigma)
    print(f"  Determinant of Σ: {det:.4f}")
    
    if det == 0:
        print("  Error: Covariance matrix is singular (not invertible)")
    else:
        # Calculate inverse
        Sigma_inv = np.linalg.inv(Sigma)
        print(f"  Σ⁻¹ =\n{Sigma_inv}")
        
        # Show calculation details for 2x2 matrix
        if Sigma.shape == (2, 2):
            a, b = Sigma[0, 0], Sigma[0, 1]
            c, d = Sigma[1, 0], Sigma[1, 1]
            print(f"  For a 2×2 matrix [[a, b], [c, d]] = [[{a}, {b}], [{c}, {d}]]:")
            print(f"  Determinant = ad - bc = ({a})({d}) - ({b})({c}) = {det:.4f}")
            print(f"  Inverse = 1/det * [[d, -b], [-c, a]] = 1/{det:.4f} * [[{d}, {-b}], [{-c}, {a}]]")
            print(f"  Σ⁻¹ = [[{Sigma_inv[0,0]:.4f}, {Sigma_inv[0,1]:.4f}], [{Sigma_inv[1,0]:.4f}, {Sigma_inv[1,1]:.4f}]]")
except np.linalg.LinAlgError:
    print("  Error: Could not compute inverse - matrix is singular")

# Step 3.2: Calculate the difference between class means
print("\nStep 3.2: Calculate the difference between class means:")
mean_diff = mu1 - mu2
print(f"  μ₁ - μ₂ = {mu1} - {mu2} = {mean_diff}")

# Step 3.3: Calculate the LDA projection vector w = Σ^(-1) * (μ₁ - μ₂)
print("\nStep 3.3: Calculate the projection vector w = Σ⁻¹ · (μ₁ - μ₂):")
w = Sigma_inv.dot(mean_diff)
print(f"  w = Σ⁻¹ · (μ₁ - μ₂) = {Sigma_inv} · {mean_diff} = {w}")

# Show detailed calculation
print(f"  Detailed calculation:")
for i in range(len(w)):
    calculation = " + ".join([f"{Sigma_inv[i,j]:.4f} × ({mean_diff[j]})" for j in range(len(mean_diff))])
    print(f"  w[{i}] = {calculation} = {w[i]:.4f}")

# Step 3.4: Calculate the norm of w
w_norm = np.linalg.norm(w)
print(f"\nStep 3.4: Calculate the norm of w:")
print(f"  |w| = √({w[0]}² + {w[1]}²) = √({w[0]**2} + {w[1]**2}) = {w_norm:.4f}")

# Step 3.5: Normalize w to unit length (optional)
w_unit = w / w_norm
print(f"\nStep 3.5: Normalize w to unit length (optional):")
print(f"  w_unit = w/|w| = {w}/{w_norm:.4f} = {w_unit}")

# Step 4: Visualize the LDA projection with improved visuals
print("\nStep 4: Visualizing the LDA projection")
print("----------------------------------")

# Create a cleaner, more elegant figure to visualize the LDA projection
plt.figure(figsize=(10, 8))

# Generate data points for the visualization
n_points = 100
X1 = np.random.multivariate_normal(mu1, Sigma, n_points)
X2 = np.random.multivariate_normal(mu2, Sigma, n_points)

# Plot the data points with cleaner aesthetics
plt.scatter(X1[:, 0], X1[:, 1], c='#3498db', alpha=0.4, s=50, edgecolors='none', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='#e74c3c', alpha=0.4, s=50, edgecolors='none', label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='#3498db', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='#e74c3c', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 2 Mean')

# Add equal-probability contours
ax = plt.gca()
ax.add_patch(plot_ellipse(mu1, Sigma, '#3498db', alpha=0.2))
ax.add_patch(plot_ellipse(mu2, Sigma, '#e74c3c', alpha=0.2))

# Calculate the midpoint between the class means
midpoint = (mu1 + mu2) / 2

# Calculate the direction perpendicular to w (for the decision boundary)
perp_w = np.array([-w[1], w[0]])
perp_w = perp_w / np.linalg.norm(perp_w)

# Draw the projection vector w starting from the midpoint with better arrow
scale = 3  # Scale for vector visualization
plt.arrow(midpoint[0], midpoint[1], scale * w[0], scale * w[1], 
          head_width=0.2, head_length=0.3, fc='#2ecc71', ec='#2ecc71', 
          width=0.05, length_includes_head=True, zorder=10)

# Draw the decision boundary (perpendicular to w, passing through midpoint)
boundary_x = np.array([midpoint[0] - 5 * perp_w[0], midpoint[0] + 5 * perp_w[0]])
boundary_y = np.array([midpoint[1] - 5 * perp_w[1], midpoint[1] + 5 * perp_w[1]])
plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

# Draw projection lines from class means to the projection direction
for i, (mean, color) in enumerate(zip([mu1, mu2], ['#3498db', '#e74c3c'])):
    # Project the mean onto the w direction
    proj = midpoint + (np.dot(mean - midpoint, w) / np.dot(w, w)) * w
    plt.plot([mean[0], proj[0]], [mean[1], proj[1]], color=color, linestyle='--', alpha=0.7)
    plt.annotate(f'Projection of μ{i+1}', xy=(proj[0], proj[1]), 
                 xytext=(proj[0] + 0.3, proj[1] - 0.3), fontsize=11,
                 arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, alpha=0.7))

# Add annotation for w
plt.annotate('w = Σ⁻¹(μ₁-μ₂)', xy=(midpoint[0] + scale * w[0]/2, midpoint[1] + scale * w[1]/2), 
             xytext=(midpoint[0] + 1, midpoint[1] - 1), fontsize=12,
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7))

# Add axis labels and title
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Projection and Decision Boundary', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, frameon=True, framealpha=0.9)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# Step 5: Calculate the threshold for classification with detailed explanation
print("\nStep 5: Calculating the threshold for classification")
print("------------------------------------------------")

print("STEP-BY-STEP CALCULATION OF LDA CLASSIFICATION THRESHOLD:")
print("--------------------------------------------------------")

# Step 5.1: Project the class means onto the direction w
print("Step 5.1: Project the class means onto the direction w:")
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)

# Show detailed calculation
print(f"  w·μ₁ = {w} · {mu1} = ", end="")
calculation = " + ".join([f"({w[i]}) × ({mu1[i]})" for i in range(len(w))])
print(f"{calculation} = {proj_mu1:.4f}")

print(f"  w·μ₂ = {w} · {mu2} = ", end="")
calculation = " + ".join([f"({w[i]}) × ({mu2[i]})" for i in range(len(w))])
print(f"{calculation} = {proj_mu2:.4f}")

# Step 5.2: Calculate the threshold as the midpoint of projected means
print("\nStep 5.2: Calculate the threshold as the midpoint of projected means:")
threshold = (proj_mu1 + proj_mu2) / 2
print(f"  threshold = (w·μ₁ + w·μ₂)/2 = ({proj_mu1:.4f} + {proj_mu2:.4f})/2 = {threshold:.4f}")

# Step 5.3: Derive the general form of the threshold
print("\nStep 5.3: Derive the general form of the threshold (for equal prior probabilities):")
print("  For equal prior probabilities P(C₁) = P(C₂), the threshold is:")
print("  threshold = (w·μ₁ + w·μ₂)/2")
print("  This corresponds to the point where the posterior probabilities are equal:")
print("  P(C₁|x) = P(C₂|x) = 0.5")

# Step 5.4: Express the decision rule
print("\nStep 5.4: Express the decision rule:")
print(f"  If w·x > {threshold:.4f}, classify as Class 1")
print(f"  If w·x < {threshold:.4f}, classify as Class 2")
print(f"  This can be rewritten as: {w[0]:.4f}x₁ + {w[1]:.4f}x₂ {'>' if threshold >= 0 else '<'} {abs(threshold):.4f}")

# Step 6: Classify new data points with detailed workthrough
print("\nStep 6: Classifying new data points")
print("-------------------------------")

# Given new data points to classify
x1 = np.array([2, 1])
x2 = np.array([0, 3])

print("STEP-BY-STEP CLASSIFICATION OF NEW DATA POINTS:")
print("----------------------------------------------")

# Function to classify and show work
def classify_point(x, w, threshold, class_labels={1, 2}):
    # Project the point onto w
    proj = np.dot(w, x)
    
    # Show calculation details
    print(f"  Point x = {x}")
    print(f"  Projection onto w = w·x = {w} · {x} = ", end="")
    calculation = " + ".join([f"({w[i]}) × ({x[i]})" for i in range(len(w))])
    print(f"{calculation} = {proj:.4f}")
    
    print(f"  Threshold = {threshold:.4f}")
    print(f"  Since w·x = {proj:.4f} {'>' if proj > threshold else '<'} {threshold:.4f} (threshold)")
    
    # Determine the class
    if proj > threshold:
        class_label = min(class_labels)  # Typically 1
    else:
        class_label = max(class_labels)  # Typically 2
        
    print(f"  Classification: Class {class_label}")
    return class_label, proj

# Classify both points
print("Classification of point x₁:")
class_x1, proj_x1 = classify_point(x1, w, threshold)
print("\nClassification of point x₂:")
class_x2, proj_x2 = classify_point(x2, w, threshold)

# Step 7: Visualize the classification of new points with improved clarity
print("\nStep 7: Visualizing the classification of new points")
print("------------------------------------------------")

plt.figure(figsize=(10, 8))

# Plot the distribution of classes (fewer points for clarity)
np.random.seed(42)
X1_sample = np.random.multivariate_normal(mu1, Sigma, 50)
X2_sample = np.random.multivariate_normal(mu2, Sigma, 50)

plt.scatter(X1_sample[:, 0], X1_sample[:, 1], c='#3498db', alpha=0.3, label='Class 1')
plt.scatter(X2_sample[:, 0], X2_sample[:, 1], c='#e74c3c', alpha=0.3, label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='#3498db', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='#e74c3c', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 2 Mean')

# Plot the new points to classify with larger markers for visibility
point_colors = {1: '#2ecc71', 2: '#9b59b6'}  # Green for class 1, purple for class 2
point_markers = ['o', 's']  # Circle for x1, square for x2

for i, (point, proj, class_label, marker) in enumerate(zip([x1, x2], [proj_x1, proj_x2], 
                                                   [class_x1, class_x2], point_markers)):
    plt.scatter(point[0], point[1], c=point_colors[class_label], s=120, marker=marker, 
                edgecolor='k', linewidth=1.5, zorder=10, 
                label=f'New Point x{i+1} (Class {class_label})')

# Draw the decision boundary
plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

# Draw projection lines for the new points with better annotations
for i, (point, proj, name, marker, class_label) in enumerate(zip(
        [x1, x2], [proj_x1, proj_x2], ['x₁', 'x₂'], point_markers, [class_x1, class_x2])):
    
    # Find the projection point on the w direction
    # First find the projection onto the decision boundary (perpendicular to w)
    boundary_point = midpoint + (np.dot(point - midpoint, perp_w) * perp_w)
    
    # Draw projection line to the boundary
    plt.plot([point[0], boundary_point[0]], [point[1], boundary_point[1]], 
             'k:', linewidth=1, alpha=0.7)
    
    # Calculate and annotate the signed distance to boundary
    dist = np.linalg.norm(point - boundary_point)
    sign = 1 if (proj > threshold) else -1
    plt.annotate(f'Distance = {sign * dist:.2f}', 
                 xy=((point[0] + boundary_point[0])/2, (point[1] + boundary_point[1])/2), 
                 xytext=((point[0] + boundary_point[0])/2 + 0.3, 
                          (point[1] + boundary_point[1])/2 + 0.3), 
                 fontsize=10, color='#555555',
                 arrowprops=dict(facecolor='#555555', shrink=0.05, width=1, alpha=0.8))

# Add equal-probability contours with lower opacity for clarity
ax = plt.gca()
ax.add_patch(plot_ellipse(mu1, Sigma, '#3498db', alpha=0.1))
ax.add_patch(plot_ellipse(mu2, Sigma, '#e74c3c', alpha=0.1))

# Add the decision rule as an annotation
plt.text(0.02, 0.02, 
         f"Decision Rule:\nIf {w[0]:.2f}x₁ + {w[1]:.2f}x₂ > {threshold:.2f} → Class 1\n"
         f"If {w[0]:.2f}x₁ + {w[1]:.2f}x₂ < {threshold:.2f} → Class 2", 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Classification of New Points with LDA', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_classification.png"), dpi=300, bbox_inches='tight')

# Step 7.5: Visualize posterior probabilities
print("\nStep 7.5: Visualizing posterior probabilities")
print("----------------------------------------")

# Define a grid of points for visualization
x1_range = np.linspace(-1, 7, 100)
x2_range = np.linspace(-1, 5, 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
grid_points = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T

# Calculate posterior for class 1 using Bayes' rule with equal priors
def lda_posterior(x, mu1, mu2, Sigma, prior1=0.5, prior2=0.5):
    # Multivariate Gaussian PDFs for each class
    pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=Sigma)
    pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=Sigma)
    
    # Apply Bayes' rule
    posterior1 = (pdf1 * prior1) / (pdf1 * prior1 + pdf2 * prior2)
    return posterior1

# Calculate posterior probabilities for grid points
posteriors = np.array([lda_posterior(x, mu1, mu2, Sigma) for x in grid_points])
posterior_grid = posteriors.reshape(X1_grid.shape)

# Create visualization of the posterior probabilities
plt.figure(figsize=(9, 7))

# Use a better colormap for the posterior probability
cmap = plt.cm.RdBu_r
contour = plt.contourf(X1_grid, X2_grid, posterior_grid, levels=np.linspace(0, 1, 21), 
                       cmap=cmap, alpha=0.8)
cbar = plt.colorbar(contour, label='P(Class 1 | x)')

# Add contour line for decision boundary (P(Class 1 | x) = 0.5)
decision_contour = plt.contour(X1_grid, X2_grid, posterior_grid, levels=[0.5], 
                              colors='k', linewidths=2, linestyles='--')
plt.clabel(decision_contour, inline=True, fontsize=10, fmt='P = 0.5')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='white', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='white', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 2 Mean')

# Annotate the class means
plt.annotate('$\\mu_1$', xy=(mu1[0], mu1[1]), xytext=(mu1[0]+0.2, mu1[1]+0.2), 
             fontsize=14, color='k')
plt.annotate('$\\mu_2$', xy=(mu2[0], mu2[1]), xytext=(mu2[0]+0.2, mu2[1]+0.2), 
             fontsize=14, color='k')

# Add annotations for the decision boundary
plt.text(0.02, 0.95, 
        "Decision Boundary: P(Class 1 | x) = 0.5\n"
        f"Equation: {w[0]:.2f}x₁ + {w[1]:.2f}x₂ = {threshold:.2f}", 
        transform=plt.gca().transAxes, fontsize=12, va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Posterior Probability P(Class 1 | x)', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_posterior.png"), dpi=300, bbox_inches='tight')

# Step 8: Special case - LDA with identity covariance matrix
print("\nStep 8: Special case - LDA with Σ = I (Identity Matrix)")
print("----------------------------------------------------")

# Use the identity matrix for covariance
I_matrix = np.eye(2)  # 2x2 identity matrix
print(f"For the special case where Σ = I (identity matrix):")
print(f"  Σ = {I_matrix}")

# Calculate w for the identity covariance case
w_identity = np.dot(I_matrix, mu1 - mu2)
print(f"\nStep 8.1: Calculate w = Σ⁻¹(μ₁ - μ₂) = I(μ₁ - μ₂) = μ₁ - μ₂")
print(f"  w = {mu1} - {mu2} = {w_identity}")

# Calculate threshold for identity case
proj_mu1_identity = np.dot(w_identity, mu1)
proj_mu2_identity = np.dot(w_identity, mu2)
threshold_identity = (proj_mu1_identity + proj_mu2_identity) / 2

print(f"\nStep 8.2: Calculate the threshold:")
print(f"  w·μ₁ = {w_identity} · {mu1} = {proj_mu1_identity:.4f}")
print(f"  w·μ₂ = {w_identity} · {mu2} = {proj_mu2_identity:.4f}")
print(f"  threshold = (w·μ₁ + w·μ₂)/2 = ({proj_mu1_identity:.4f} + {proj_mu2_identity:.4f})/2 = {threshold_identity:.4f}")

print(f"\nStep 8.3: Express the decision boundary equation:")
print(f"  At the decision boundary, w·x = threshold")
print(f"  {w_identity[0]}x₁ + {w_identity[1]}x₂ = {threshold_identity:.4f}")

# Find the most convenient form
if w_identity[0] != 0:
    boundary_slope = -w_identity[1]/w_identity[0]
    boundary_intercept = threshold_identity/w_identity[0]
    print(f"  Solving for x₁: x₁ = {boundary_slope:.4f}x₂ + {boundary_intercept:.4f}")
else:
    boundary_constant = threshold_identity/w_identity[1]
    print(f"  Solving for x₂: x₂ = {boundary_constant:.4f}")

# Compute the general form in terms of means
mu1_squared_norm = np.dot(mu1, mu1)
mu2_squared_norm = np.dot(mu2, mu2)
boundary_constant_general = 0.5 * (mu1_squared_norm - mu2_squared_norm)
print(f"\nStep 8.4: General formula for the decision boundary with Σ = I:")
print(f"  (μ₁ - μ₂)ᵀx = 0.5(||μ₁||² - ||μ₂||²)")
print(f"  (μ₁ - μ₂)ᵀx = 0.5({mu1_squared_norm:.4f} - {mu2_squared_norm:.4f}) = {boundary_constant_general:.4f}")
print(f"  {w_identity[0]}x₁ + {w_identity[1]}x₂ = {boundary_constant_general:.4f}")

# Visualize LDA with identity covariance
plt.figure(figsize=(9, 7))

# Generate data with identity covariance
n_points = 80
X1_identity = np.random.multivariate_normal(mu1, I_matrix, n_points)
X2_identity = np.random.multivariate_normal(mu2, I_matrix, n_points)

# Plot the data points
plt.scatter(X1_identity[:, 0], X1_identity[:, 1], c=colors[1], alpha=0.4, 
            edgecolors='none', label='Class 1')
plt.scatter(X2_identity[:, 0], X2_identity[:, 1], c=colors[2], alpha=0.4, 
            edgecolors='none', label='Class 2')

# Plot class means
plt.scatter(mu1[0], mu1[1], c=colors[1], s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c=colors[2], s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 2 Mean')

# Add equal-probability contours (circles for identity covariance)
ax = plt.gca()
ax.add_patch(plot_ellipse(mu1, I_matrix, colors[1], alpha=0.15))
ax.add_patch(plot_ellipse(mu2, I_matrix, colors[2], alpha=0.15))

# Calculate the midpoint between the class means
midpoint_identity = (mu1 + mu2) / 2

# Draw projection vector (w = μ₁ - μ₂ for identity case)
scale = 0.5  # Scale for vector visualization
plt.arrow(midpoint_identity[0], midpoint_identity[1], 
          scale * w_identity[0], scale * w_identity[1], 
          head_width=0.2, head_length=0.3, fc='#2ecc71', ec='#2ecc71', 
          width=0.05, length_includes_head=True, zorder=10)

# Draw decision boundary (perpendicular to w, passing through midpoint)
perp_w_identity = np.array([-w_identity[1], w_identity[0]])
perp_w_identity = perp_w_identity / np.linalg.norm(perp_w_identity)

boundary_x_identity = np.array([midpoint_identity[0] - 5 * perp_w_identity[0], 
                               midpoint_identity[0] + 5 * perp_w_identity[0]])
boundary_y_identity = np.array([midpoint_identity[1] - 5 * perp_w_identity[1], 
                               midpoint_identity[1] + 5 * perp_w_identity[1]])
plt.plot(boundary_x_identity, boundary_y_identity, 'k--', linewidth=2, 
         label='Decision Boundary')

# Annotate the decision boundary equation
plt.text(0.02, 0.95, 
        f"When Σ = I (identity matrix):\n"
        f"w = μ₁ - μ₂ = [{w_identity[0]}, {w_identity[1]}]ᵀ\n"
        f"Decision boundary: {w_identity[0]:.2f}x₁ + {w_identity[1]:.2f}x₂ = {boundary_constant_general:.2f}", 
        transform=plt.gca().transAxes, fontsize=12, va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Add key insight: w direction is orthogonal to decision boundary
plt.arrow(midpoint_identity[0], midpoint_identity[1], 
          3 * perp_w_identity[0], 3 * perp_w_identity[1], 
          head_width=0.2, head_length=0.3, fc='gray', ec='gray', 
          width=0.05, alpha=0.5, length_includes_head=True, zorder=9)
plt.annotate('Boundary\ndirection', 
             xy=(midpoint_identity[0] + 1.5 * perp_w_identity[0], 
                 midpoint_identity[1] + 1.5 * perp_w_identity[1]),
             xytext=(midpoint_identity[0] + 2 * perp_w_identity[0], 
                    midpoint_identity[1] + 2 * perp_w_identity[1]),
             fontsize=10, color='gray')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA with Identity Covariance: w = μ₁ - μ₂', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_identity_covariance.png"), dpi=300, bbox_inches='tight')

# Step 9: New Visualization - Feature Importance in LDA
print("\nStep 9: NEW - Visualizing Feature Importance in LDA")
print("------------------------------------------------")

plt.figure(figsize=(9, 7))

# Calculate explained variance for each feature
# This is based on how much each feature contributes to w relative to its variability
w_abs = np.abs(w)  # Take absolute values
feature_importance = w_abs / np.sum(w_abs)  # Normalize to get relative importance

# Create a bar chart of feature importance
features = ['$x_1$', '$x_2$']
plt.bar(features, feature_importance, color=['#3498db', '#2ecc71'])
plt.ylabel('Relative Importance', fontsize=14)
plt.title('Feature Importance in LDA Decision Boundary', fontsize=16)

# Add annotation explaining the interpretation
plt.text(0.02, 0.95, 
        "Feature importance in LDA is proportional to\n"
        "the magnitude of weights in w = Σ⁻¹(μ₁ - μ₂)\n"
        f"w = [{w[0]:.2f}, {w[1]:.2f}]", 
        transform=plt.gca().transAxes, fontsize=12, va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Add the exact values as text on the bars
for i, v in enumerate(feature_importance):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12)

plt.ylim(0, max(feature_importance) + 0.1)  # Add some padding at the top
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_feature_importance.png"), dpi=300, bbox_inches='tight')

# Step 10: Compare LDA with Perceptron with improved visualization
print("\nStep 10: Comparing LDA with Perceptron")
print("----------------------------------")

# Create a cleaner figure to compare LDA with Perceptron
plt.figure(figsize=(10, 8))

# Plot the distribution of classes (fewer points for clarity)
plt.scatter(X1_sample[:, 0], X1_sample[:, 1], c='#3498db', alpha=0.5, label='Class 1')
plt.scatter(X2_sample[:, 0], X2_sample[:, 1], c='#e74c3c', alpha=0.5, label='Class 2')

# Plot the class means
plt.scatter(mu1[0], mu1[1], c='#3498db', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], c='#e74c3c', s=150, marker='X', edgecolor='k', 
            linewidth=2, label='Class 2 Mean')

# Draw the LDA decision boundary
plt.plot(boundary_x, boundary_y, '#2ecc71', linestyle='--', linewidth=2.5, label='LDA Boundary')

# Simulate a Perceptron decision boundary (using a simplified approach)
# For demonstration, we'll use a direction that doesn't account for covariance
perc_vector = mu1 - mu2  # Simplified perceptron direction
perc_midpoint = (mu1 + mu2) / 2
perc_perp = np.array([-perc_vector[1], perc_vector[0]])
perc_perp = perc_perp / np.linalg.norm(perc_perp)

perc_boundary_x = np.array([perc_midpoint[0] - 5 * perc_perp[0], perc_midpoint[0] + 5 * perc_perp[0]])
perc_boundary_y = np.array([perc_midpoint[1] - 5 * perc_perp[1], perc_midpoint[1] + 5 * perc_perp[1]])
plt.plot(perc_boundary_x, perc_boundary_y, '#9b59b6', linestyle='-', linewidth=2.5, 
        label='Perceptron Boundary (Simplified)')

# Add arrows showing the direction vectors for both methods
scale = 2
# LDA direction
plt.arrow(perc_midpoint[0], perc_midpoint[1], scale * w[0], scale * w[1],
         head_width=0.2, head_length=0.3, fc='#2ecc71', ec='#2ecc71', width=0.05, 
         length_includes_head=True, zorder=10, alpha=0.7)
# Perceptron direction 
plt.arrow(perc_midpoint[0], perc_midpoint[1], scale * perc_vector[0], scale * perc_vector[1],
         head_width=0.2, head_length=0.3, fc='#9b59b6', ec='#9b59b6', width=0.05, 
         length_includes_head=True, zorder=10, alpha=0.7)

# Add explanatory text box
plt.text(0.02, 0.98, 
        "LDA: Uses Σ⁻¹(μ₁-μ₂) as direction\n"
        "- Accounts for covariance structure\n"
        "- Optimal in Bayes sense when assumptions met\n\n"
        "Perceptron: Uses (μ₁-μ₂) direction\n"
        "- Ignores covariance structure\n"
        "- Finds any separating hyperplane", 
        transform=plt.gca().transAxes, fontsize=11, va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Comparison: LDA vs Perceptron Decision Boundaries', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_vs_perceptron.png"), dpi=300, bbox_inches='tight')

# Step 11: Summary of findings
print("\nStep 11: Summary of findings")
print("-------------------------")

print("1. Key assumptions of LDA:")
print("   - Classes follow multivariate Gaussian distributions")
print("   - Classes share the same covariance matrix")
print("   - The covariance matrix is invertible (no perfect multicollinearity)")
print("\n2. LDA projection direction:")
print(f"   w = Σ⁻¹(μ₁ - μ₂) = [{w[0]:.4f}, {w[1]:.4f}]^T")
print("\n3. Classification threshold (equal priors):")
print(f"   threshold = {threshold:.4f}")
print("\n4. Classification of new points:")
print(f"   x₁ = [{x1[0]}, {x1[1]}]^T is classified as Class {class_x1}")
print(f"   x₂ = [{x2[0]}, {x2[1]}]^T is classified as Class {class_x2}")
print("\n5. Difference from Perceptron:")
print("   - LDA takes a probabilistic approach based on class distributions")
print("   - LDA finds the optimal boundary in terms of statistical separability")
print("   - Perceptron simply tries to find any hyperplane that separates the classes")
print("   - LDA is more robust when assumptions are met, but more constrained by assumptions") 