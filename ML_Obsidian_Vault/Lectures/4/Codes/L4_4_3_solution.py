import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 3: Linear Discriminant Analysis")
print("=======================================")

# Step 1: Define the given parameters
print("\nStep 1: Define the parameters of the problem")
print("------------------------------------------")

# Given data
mu1 = np.array([1, 2])
mu2 = np.array([3, 0])
cov = np.array([[2, 0], [0, 1]])

print("Class 1 mean (μ1):", mu1)
print("Class 2 mean (μ2):", mu2)
print("Shared covariance matrix (Σ):")
print(cov)

# Step 2: Calculate the direction of LDA projection
print("\nStep 2: Calculate the direction of LDA projection")
print("------------------------------------------------")

# Calculate inverse of covariance matrix
cov_inv = np.linalg.inv(cov)
print("Inverse of covariance matrix (Σ^-1):")
print(cov_inv)

# Calculate the difference between means
mean_diff = mu1 - mu2
print("Difference between means (μ1 - μ2):", mean_diff)

# Calculate the LDA projection direction w = Σ^-1(μ1 - μ2)
w = np.dot(cov_inv, mean_diff)
print("LDA projection direction (w = Σ^-1(μ1 - μ2)):", w)

# Print detailed step-by-step calculation for the markdown file
print("\nDetailed step-by-step calculation of w:")
print("----------------------------------------")
print(f"Σ^-1 = [[ 1/2, 0   ],")
print(f"        [ 0,   1   ]]")
print(f"      = [[ {cov_inv[0,0]}, {cov_inv[0,1]} ],")
print(f"         [ {cov_inv[1,0]}, {cov_inv[1,1]} ]]")
print()
print(f"μ1 - μ2 = [ {mu1[0]} ] - [ {mu2[0]} ] = [ {mean_diff[0]} ]")
print(f"          [ {mu1[1]} ]   [ {mu2[1]} ]   [ {mean_diff[1]} ]")
print()
print(f"w = Σ^-1(μ1 - μ2)")
print(f"  = [[ {cov_inv[0,0]}, {cov_inv[0,1]} ],  × [ {mean_diff[0]} ]")
print(f"     [ {cov_inv[1,0]}, {cov_inv[1,1]} ]]    [ {mean_diff[1]} ]")
print()
print(f"  = [ {cov_inv[0,0]} × {mean_diff[0]} + {cov_inv[0,1]} × {mean_diff[1]} ]")
print(f"    [ {cov_inv[1,0]} × {mean_diff[0]} + {cov_inv[1,1]} × {mean_diff[1]} ]")
print()
print(f"  = [ {cov_inv[0,0] * mean_diff[0]} + {cov_inv[0,1] * mean_diff[1]} ]")
print(f"    [ {cov_inv[1,0] * mean_diff[0]} + {cov_inv[1,1] * mean_diff[1]} ]")
print()
print(f"  = [ {w[0]} ]")
print(f"    [ {w[1]} ]")

# Step 3: Calculate the threshold value for classification
print("\nStep 3: Calculate the threshold value for classification")
print("------------------------------------------------------")

# Project the means onto the direction w
proj_mu1 = np.dot(w, mu1)
proj_mu2 = np.dot(w, mu2)
print(f"Projection of μ1 onto w: {proj_mu1:.4f}")
print(f"Projection of μ2 onto w: {proj_mu2:.4f}")

# Calculate the threshold (midpoint of projected means)
threshold = (proj_mu1 + proj_mu2) / 2
print(f"Classification threshold: {threshold:.4f}")

# Print detailed calculations for projections
print("\nDetailed projection calculations:")
print("---------------------------------")
print(f"proj_μ1 = w^T × μ1")
print(f"       = [{w[0]}, {w[1]}] × [{mu1[0]}, {mu1[1]}]^T")
print(f"       = {w[0]} × {mu1[0]} + {w[1]} × {mu1[1]}")
print(f"       = {w[0] * mu1[0]} + {w[1] * mu1[1]}")
print(f"       = {proj_mu1:.4f}")
print()
print(f"proj_μ2 = w^T × μ2")
print(f"       = [{w[0]}, {w[1]}] × [{mu2[0]}, {mu2[1]}]^T")
print(f"       = {w[0]} × {mu2[0]} + {w[1]} × {mu2[1]}")
print(f"       = {w[0] * mu2[0]} + {w[1] * mu2[1]}")
print(f"       = {proj_mu2:.4f}")
print()
print(f"threshold = (proj_μ1 + proj_μ2) / 2")
print(f"         = ({proj_mu1:.4f} + {proj_mu2:.4f}) / 2")
print(f"         = {proj_mu1 + proj_mu2:.4f} / 2")
print(f"         = {threshold:.4f}")

# Step 4: Classify the first new data point
print("\nStep 4: Classify the first new data point")
print("--------------------------------------")

# New data point
x_new = np.array([2, 1])
print("First new data point (x):", x_new)

# Project the new point onto w
proj_x = np.dot(w, x_new)
print(f"Projection of first point onto w: {proj_x:.4f}")

# Classify the point
if proj_x > threshold:
    classification = "Class 1"
    print(f"Since {proj_x:.4f} > {threshold:.4f}, x is classified as Class 1")
else:
    classification = "Class 2"
    print(f"Since {proj_x:.4f} <= {threshold:.4f}, x is classified as Class 2")

# Detailed calculation for new point projection
print("\nDetailed calculation for first new point projection:")
print("--------------------------------------------------")
print(f"proj_x = w^T × x")
print(f"      = [{w[0]}, {w[1]}] × [{x_new[0]}, {x_new[1]}]^T")
print(f"      = {w[0]} × {x_new[0]} + {w[1]} × {x_new[1]}")
print(f"      = {w[0] * x_new[0]} + {w[1] * x_new[1]}")
print(f"      = {proj_x:.4f}")
print()
print(f"Decision rule: If proj_x > {threshold:.4f}, classify as Class 1, otherwise Class 2")
print(f"Since {proj_x:.4f} {'>' if proj_x > threshold else '<='} {threshold:.4f}, x is classified as {classification}")

# Step 5: Classify the second new data point
print("\nStep 5: Classify the second new data point")
print("---------------------------------------")

# Second new data point
x_new2 = np.array([0, 3])
print("Second new data point (x2):", x_new2)

# Project the second new point onto w
proj_x2 = np.dot(w, x_new2)
print(f"Projection of second point onto w: {proj_x2:.4f}")

# Classify the second point
if proj_x2 > threshold:
    classification2 = "Class 1"
    print(f"Since {proj_x2:.4f} > {threshold:.4f}, x2 is classified as Class 1")
else:
    classification2 = "Class 2"
    print(f"Since {proj_x2:.4f} <= {threshold:.4f}, x2 is classified as Class 2")

# Detailed calculation for second new point projection
print("\nDetailed calculation for second new point projection:")
print("---------------------------------------------------")
print(f"proj_x2 = w^T × x2")
print(f"       = [{w[0]}, {w[1]}] × [{x_new2[0]}, {x_new2[1]}]^T")
print(f"       = {w[0]} × {x_new2[0]} + {w[1]} × {x_new2[1]}")
print(f"       = {w[0] * x_new2[0]} + {w[1] * x_new2[1]}")
print(f"       = {proj_x2:.4f}")
print()
print(f"Decision rule: If proj_x2 > {threshold:.4f}, classify as Class 1, otherwise Class 2")
print(f"Since {proj_x2:.4f} {'>' if proj_x2 > threshold else '<='} {threshold:.4f}, x2 is classified as {classification2}")

# Step 6: Create simple visualizations
print("\nStep 6: Create simple visualizations")
print("----------------------------------")

# Generate sample data for visualization
def generate_samples(mean, cov, n_samples=100):
    return np.random.multivariate_normal(mean, cov, n_samples)

# Generate samples for both classes
np.random.seed(42)  # For reproducibility
samples_class1 = generate_samples(mu1, cov)
samples_class2 = generate_samples(mu2, cov)

# Calculate the boundary line
# For the line w[0]*x + w[1]*y + b = 0
# We can rewrite as y = -(w[0]*x + b)/w[1] where b is chosen to make the line pass through the midpoint
midpoint = (mu1 + mu2) / 2
b = -np.dot(w, midpoint)

# Generate points for the decision boundary line
x_line = np.linspace(-2, 6, 1000)
if w[1] != 0:  # Avoid division by zero
    y_line = -(w[0] * x_line + b) / w[1]
else:
    # If w[1] is zero, it's a vertical line at x = -b/w[0]
    x_line = np.array([-b / w[0], -b / w[0]])
    y_line = np.array([-3, 5])

# Print the equation of the decision boundary
print("\nDecision boundary equation:")
print("--------------------------")
print(f"w[0]*x + w[1]*y + b = 0")
print(f"{w[0]}*x + {w[1]}*y + {b} = 0")
if w[1] != 0:
    print(f"Solving for y: y = -({w[0]}*x + {b})/{w[1]}")
    print(f"y = {-w[0]/w[1]}*x + {-b/w[1]}")

# Visualization 1: Basic elements with both new points
plt.figure(figsize=(10, 8))
plt.scatter(mu1[0], mu1[1], color='blue', s=150, marker='X', label='Class 1 Mean (μ1)')
plt.scatter(mu2[0], mu2[1], color='orange', s=150, marker='X', label='Class 2 Mean (μ2)')
plt.scatter(x_new[0], x_new[1], color='red', s=150, marker='*', label=f'First new point (x1)')
plt.scatter(x_new2[0], x_new2[1], color='purple', s=150, marker='*', label=f'Second new point (x2)')
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

# Add the midpoint
plt.scatter(midpoint[0], midpoint[1], color='green', s=100, marker='o', label='Midpoint of means')

# Add direction vector w
plt.arrow(midpoint[0], midpoint[1], w[0], w[1], head_width=0.2, head_length=0.3, 
         fc='green', ec='green', label='LDA direction w')

# Improve plot styling
plt.grid(True, alpha=0.3)
plt.title('LDA Decision Boundary and Classification Points', fontsize=16)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlim(-2, 6)
plt.ylim(-3, 5)

plt.savefig(os.path.join(save_dir, "lda_basic_elements_with_two_points.png"), dpi=300, bbox_inches='tight')

# Visualization 2: Show the projection process for both new points
plt.figure(figsize=(10, 8))
plt.scatter(mu1[0], mu1[1], color='blue', s=150, marker='X', label='Class 1 Mean (μ1)')
plt.scatter(mu2[0], mu2[1], color='orange', s=150, marker='X', label='Class 2 Mean (μ2)')
plt.scatter(x_new[0], x_new[1], color='red', s=150, marker='*', label=f'First new point (x1)')
plt.scatter(x_new2[0], x_new2[1], color='purple', s=150, marker='*', label=f'Second new point (x2)')
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

# Add direction vector w
plt.arrow(midpoint[0], midpoint[1], w[0], w[1], head_width=0.2, head_length=0.3, 
         fc='green', ec='green', label='LDA direction w')

# Calculate endpoints of projection onto w
w_unit = w / np.linalg.norm(w)
proj_length1 = np.dot(mu1 - midpoint, w_unit)
proj_length2 = np.dot(mu2 - midpoint, w_unit)
proj_length_new = np.dot(x_new - midpoint, w_unit)
proj_length_new2 = np.dot(x_new2 - midpoint, w_unit)

midpoint_to_mu1 = midpoint + proj_length1 * w_unit
midpoint_to_mu2 = midpoint + proj_length2 * w_unit
midpoint_to_new = midpoint + proj_length_new * w_unit
midpoint_to_new2 = midpoint + proj_length_new2 * w_unit

# Add projection lines for means and first new point
plt.plot([mu1[0], midpoint_to_mu1[0]], [mu1[1], midpoint_to_mu1[1]], 'b--', alpha=0.7, linewidth=2)
plt.plot([mu2[0], midpoint_to_mu2[0]], [mu2[1], midpoint_to_mu2[1]], 'orange', linestyle='--', alpha=0.7, linewidth=2)
plt.plot([x_new[0], midpoint_to_new[0]], [x_new[1], midpoint_to_new[1]], 'r--', alpha=0.7, linewidth=2)
plt.plot([x_new2[0], midpoint_to_new2[0]], [x_new2[1], midpoint_to_new2[1]], 'purple', linestyle='--', alpha=0.7, linewidth=2)

# Add LDA projection line
proj_line_x = np.linspace(midpoint_to_mu2[0] - 1, midpoint_to_mu1[0] + 1, 100)
proj_direction = w_unit / np.linalg.norm(w_unit)
proj_line_y = np.zeros_like(proj_line_x)
for i, x_val in enumerate(proj_line_x):
    t = (x_val - midpoint[0]) / proj_direction[0] if proj_direction[0] != 0 else 0
    proj_line_y[i] = midpoint[1] + t * proj_direction[1]

plt.plot(proj_line_x, proj_line_y, 'g-', alpha=0.7, linewidth=2, label='LDA projection line')

# Add projected points on the projection line
plt.scatter(midpoint_to_mu1[0], midpoint_to_mu1[1], color='blue', s=100, marker='o')
plt.scatter(midpoint_to_mu2[0], midpoint_to_mu2[1], color='orange', s=100, marker='o')
plt.scatter(midpoint_to_new[0], midpoint_to_new[1], color='red', s=100, marker='o')
plt.scatter(midpoint_to_new2[0], midpoint_to_new2[1], color='purple', s=100, marker='o')

plt.grid(True, alpha=0.3)
plt.title('LDA Projection Process for Both New Points', fontsize=16)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlim(-2, 6)
plt.ylim(-3, 5)

plt.savefig(os.path.join(save_dir, "lda_projection_process_two_points.png"), dpi=300, bbox_inches='tight')

# Visualization 3: Improved 1D Projection (Simplified)
plt.figure(figsize=(14, 5))

# Set up a cleaner 1D axis with more vertical space
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)

# Add the projections with clearer markings
plt.scatter([proj_mu1], [0], color='blue', s=200, marker='o', label='Projected μ1')
plt.scatter([proj_mu2], [0], color='orange', s=200, marker='o', label='Projected μ2')
plt.scatter([proj_x], [0], color='red', s=200, marker='*', label='Projected x1')
plt.scatter([proj_x2], [0], color='purple', s=200, marker='*', label='Projected x2')
plt.scatter([threshold], [0], color='green', s=200, marker='|', label='Threshold')

# Place labels with better spacing and no overlaps
plt.annotate(f'μ1: {proj_mu1:.2f}', xy=(proj_mu1, 0), xytext=(proj_mu1, 0.15), 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.annotate(f'μ2: {proj_mu2:.2f}', xy=(proj_mu2, 0), xytext=(proj_mu2, 0.15), 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.annotate(f'x2: {proj_x2:.2f}', xy=(proj_x2, 0), xytext=(proj_x2, -0.15), 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.annotate(f'x1: {proj_x:.2f}', xy=(proj_x, 0), xytext=(proj_x, 0.15), 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.annotate(f'threshold: {threshold:.2f}', xy=(threshold, 0), xytext=(threshold, -0.15), 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Color the regions with clearer boundaries
plt.axvspan(threshold, 7, alpha=0.2, color='blue')
plt.axvspan(-4, threshold, alpha=0.2, color='orange')

# Add simple decision rule text with better positioning
plt.text(0, -0.3, 'If projection > 0, classify as Class 1, else Class 2', 
         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Clean up the plot
plt.title('1D Projection Visualization', fontsize=16)
plt.xlim(-4, 7)  # Wider range to accommodate all points
plt.ylim(-0.5, 0.5)  # More vertical space
plt.yticks([])
plt.grid(False)
plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.01, 1.15))

plt.savefig(os.path.join(save_dir, "lda_1d_projection_simple.png"), dpi=300, bbox_inches='tight')

# Add back the class distributions visualization
plt.figure(figsize=(10, 8))
plt.scatter(samples_class1[:, 0], samples_class1[:, 1], alpha=0.4, label='Class 1 samples', s=20)
plt.scatter(samples_class2[:, 0], samples_class2[:, 1], alpha=0.4, label='Class 2 samples', s=20)
plt.scatter(mu1[0], mu1[1], color='blue', s=150, marker='X', label='Class 1 Mean (μ1)')
plt.scatter(mu2[0], mu2[1], color='orange', s=150, marker='X', label='Class 2 Mean (μ2)')
plt.scatter(x_new[0], x_new[1], color='red', s=150, marker='*', label=f'First new point (x1)')
plt.scatter(x_new2[0], x_new2[1], color='purple', s=150, marker='*', label=f'Second new point (x2)')
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

# Add confidence ellipses for both classes
def plot_ellipse(ax, mean, cov, color, alpha=0.2):
    # Eigenvalues and eigenvectors of covariance matrix
    v, w = np.linalg.eigh(cov)
    # Order of eigenvectors
    order = v.argsort()[::-1]
    v, w = v[order], w[:, order]
    # Angle of the first eigenvector
    angle = np.arctan2(w[1, 0], w[0, 0])
    # 95% confidence interval
    ellipse = Ellipse(xy=mean, width=2*np.sqrt(5.991*v[0]), height=2*np.sqrt(5.991*v[1]),
                      angle=np.degrees(angle), facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(ellipse)

plot_ellipse(plt.gca(), mu1, cov, 'blue')
plot_ellipse(plt.gca(), mu2, cov, 'orange')

plt.grid(True, alpha=0.3)
plt.title('LDA Decision Boundary with Class Distributions', fontsize=16)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlim(-2, 6)
plt.ylim(-3, 5)

plt.savefig(os.path.join(save_dir, "lda_with_distributions.png"), dpi=300, bbox_inches='tight')

# Add back the 1D histogram projection
plt.figure(figsize=(14, 6))

# Project all sample points
proj_samples1 = np.dot(samples_class1, w)
proj_samples2 = np.dot(samples_class2, w)

# Calculate 1D histograms of projections
hist_bins = np.linspace(-6, 8, 50)

plt.hist(proj_samples1, bins=hist_bins, alpha=0.5, color='blue', label='Class 1 projections')
plt.hist(proj_samples2, bins=hist_bins, alpha=0.5, color='orange', label='Class 2 projections')

# Mark the projected means and threshold
plt.axvline(x=proj_mu1, color='blue', linestyle='-', linewidth=2, label='Projected μ1')
plt.axvline(x=proj_mu2, color='orange', linestyle='-', linewidth=2, label='Projected μ2')
plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
plt.axvline(x=proj_x, color='red', linestyle='-', linewidth=2, label='Projected x1')
plt.axvline(x=proj_x2, color='purple', linestyle='-', linewidth=2, label='Projected x2')

# Fill the regions to indicate classification areas
plt.axvspan(threshold, 8, alpha=0.2, color='blue')
plt.axvspan(-6, threshold, alpha=0.2, color='orange')

plt.title('LDA 1D Projection Histogram', fontsize=16)
plt.xlabel('Projection value along LDA direction', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add key values with better positioning
plt.annotate(f'μ1: {proj_mu1:.2f}', xy=(proj_mu1, 0), xytext=(proj_mu1+0.5, 5),
            arrowprops=dict(facecolor='blue', width=1.5, shrink=0.05), ha='left', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
plt.annotate(f'μ2: {proj_mu2:.2f}', xy=(proj_mu2, 0), xytext=(proj_mu2-0.5, 10),
            arrowprops=dict(facecolor='orange', width=1.5, shrink=0.05), ha='right', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
plt.annotate(f'x1: {proj_x:.2f}', xy=(proj_x, 0), xytext=(proj_x+0.3, 15),
            arrowprops=dict(facecolor='red', width=1.5, shrink=0.05), ha='left', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
plt.annotate(f'x2: {proj_x2:.2f}', xy=(proj_x2, 0), xytext=(proj_x2-0.5, 20),
            arrowprops=dict(facecolor='purple', width=1.5, shrink=0.05), ha='right', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
plt.annotate(f'threshold: {threshold:.2f}', xy=(threshold, 0), xytext=(threshold, 25),
            arrowprops=dict(facecolor='green', width=1.5, shrink=0.05), ha='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

plt.savefig(os.path.join(save_dir, "lda_1d_projection_histogram.png"), dpi=300, bbox_inches='tight')

# Add a visualization showing the decision boundary equation explicitly
plt.figure(figsize=(10, 8))
plt.scatter(mu1[0], mu1[1], color='blue', s=150, marker='X', label='Class 1 Mean (μ1)')
plt.scatter(mu2[0], mu2[1], color='orange', s=150, marker='X', label='Class 2 Mean (μ2)')
plt.scatter(x_new[0], x_new[1], color='red', s=150, marker='*', label=f'First new point (x1)')
plt.scatter(x_new2[0], x_new2[1], color='purple', s=150, marker='*', label=f'Second new point (x2)')
plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

# Add the midpoint
plt.scatter(midpoint[0], midpoint[1], color='green', s=100, marker='o', label='Midpoint of means')

# Add some points on each side of the boundary for illustration
test_points_class1 = np.array([[1, 1.5], [2, 2], [0, 2]])
test_points_class2 = np.array([[3, 1], [4, 0.5], [2, 0.5]])

plt.scatter(test_points_class1[:,0], test_points_class1[:,1], color='blue', alpha=0.5, s=80)
plt.scatter(test_points_class2[:,0], test_points_class2[:,1], color='orange', alpha=0.5, s=80)

# Add equation text to the plot
boundary_eq = f"Decision Boundary Equation:\n${w[0]}x_1 + {w[1]}x_2 + {b} = 0$"
alternative_eq = f"Equivalent Form:\n$x_2 = {-w[0]/w[1]:.1f}x_1 + {-b/w[1]:.1f}$"

plt.text(0.05, 0.95, boundary_eq, transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.05, 0.85, alternative_eq, transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

# Improve plot styling
plt.grid(True, alpha=0.3)
plt.title('LDA Decision Boundary and Equation', fontsize=16)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')
plt.xlim(-2, 6)
plt.ylim(-3, 5)

plt.savefig(os.path.join(save_dir, "lda_decision_boundary_equation.png"), dpi=300, bbox_inches='tight')

# Additional detailed step-by-step calculations for the markdown file
print("\nVery detailed calculations for the LDA direction (pen and paper style):")
print("------------------------------------------------------------------")
print("Step 1: Compute the inverse of the covariance matrix")
print("Σ = [ 2  0 ]")
print("    [ 0  1 ]")
print()
print("For a diagonal matrix, the inverse is simply:")
print("Σ^(-1) = [ 1/2  0   ]")
print("          [ 0    1   ]")
print("        = [ 0.5  0.0 ]")
print("          [ 0.0  1.0 ]")
print()
print("Step 2: Calculate the difference between the class means")
print("μ₁ - μ₂ = [ 1 ] - [ 3 ] = [ -2 ]")
print("          [ 2 ]   [ 0 ]   [  2 ]")
print()
print("Step 3: Multiply the inverse covariance matrix by the mean difference")
print("w = Σ^(-1)(μ₁ - μ₂)")
print("  = [ 0.5  0.0 ] × [ -2 ]")
print("    [ 0.0  1.0 ]   [  2 ]")
print()
print("Computing each element:")
print("w₁ = 0.5 × (-2) + 0.0 × 2 = -1.0")
print("w₂ = 0.0 × (-2) + 1.0 × 2 = 2.0")
print()
print("Therefore, w = [ -1.0 ]")
print("                [  2.0 ]")

print("\nDetailed projection calculations for the class means:")
print("---------------------------------------------------")
print("Step 1: Project μ₁ onto w")
print("proj_μ₁ = w^T × μ₁")
print("        = [ -1.0  2.0 ] × [ 1 ]")
print("                          [ 2 ]")
print("        = (-1.0 × 1) + (2.0 × 2)")
print("        = -1.0 + 4.0")
print("        = 3.0")
print()
print("Step 2: Project μ₂ onto w")
print("proj_μ₂ = w^T × μ₂")
print("        = [ -1.0  2.0 ] × [ 3 ]")
print("                          [ 0 ]")
print("        = (-1.0 × 3) + (2.0 × 0)")
print("        = -3.0 + 0.0")
print("        = -3.0")
print()
print("Step 3: Calculate the threshold")
print("threshold = (proj_μ₁ + proj_μ₂) / 2")
print("          = (3.0 + (-3.0)) / 2")
print("          = 0 / 2")
print("          = 0.0")

print("\nDetailed projection and classification of the first new point:")
print("-----------------------------------------------------------")
print("Step 1: Project x₁ = [2, 1] onto w")
print("proj_x₁ = w^T × x₁")
print("        = [ -1.0  2.0 ] × [ 2 ]")
print("                          [ 1 ]")
print("        = (-1.0 × 2) + (2.0 × 1)")
print("        = -2.0 + 2.0")
print("        = 0.0")
print()
print("Step 2: Apply the classification rule")
print("If proj_x > threshold, classify as Class 1")
print("If proj_x ≤ threshold, classify as Class 2")
print()
print("Since proj_x₁ = 0.0 and threshold = 0.0, we have proj_x₁ = threshold")
print("By convention, when the projection equals the threshold,")
print("the point is assigned to Class 2.")
print()
print("Therefore, x₁ = [2, 1] is classified as Class 2.")

print("\nDetailed projection and classification of the second new point:")
print("------------------------------------------------------------")
print("Step 1: Project x₂ = [0, 3] onto w")
print("proj_x₂ = w^T × x₂")
print("        = [ -1.0  2.0 ] × [ 0 ]")
print("                          [ 3 ]")
print("        = (-1.0 × 0) + (2.0 × 3)")
print("        = 0.0 + 6.0")
print("        = 6.0")
print()
print("Step 2: Apply the classification rule")
print("If proj_x > threshold, classify as Class 1")
print("If proj_x ≤ threshold, classify as Class 2")
print()
print("Since proj_x₂ = 6.0 and threshold = 0.0, we have proj_x₂ > threshold")
print("Therefore, x₂ = [0, 3] is classified as Class 1.")

print("\nDetailed derivation of the decision boundary equation:")
print("---------------------------------------------------")
print("Step 1: The decision boundary is defined by the set of points where w^T x + b = 0")
print("With w = [-1.0, 2.0] and b = -np.dot(w, midpoint) = -(-1.0 × 2.0 + 2.0 × 1.0) = -(0.0) = 0.0")
print()
print("Step 2: Substituting these values, the decision boundary equation is:")
print("-1.0 × x₁ + 2.0 × x₂ + 0.0 = 0")
print()
print("Step 3: Simplifying the equation:")
print("-1.0 × x₁ + 2.0 × x₂ = 0")
print("2.0 × x₂ = 1.0 × x₁")
print("x₂ = 0.5 × x₁")
print()
print("Therefore, the decision boundary is the line x₂ = 0.5 × x₁")
print("This line passes through the origin (0, 0) and has a slope of 0.5.")
print("It also passes through the midpoint of the class means (2, 1).")

# Step 7: Verify with sklearn implementation
print("\nStep 7: Verify with sklearn implementation")
print("----------------------------------------")

# Create a simple dataset
X = np.vstack([samples_class1, samples_class2])
y = np.array([0] * len(samples_class1) + [1] * len(samples_class2))

# Train LDA with sklearn
lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
lda.fit(X, y)

# Check direction
print("sklearn LDA coefficients (scaled):", lda.coef_[0])
print("Our calculated w (LDA direction):", w)

# Verify prediction for the new points
sklearn_prediction1 = lda.predict([x_new])[0]
sklearn_prediction2 = lda.predict([x_new2])[0]
print(f"sklearn prediction for first point: Class {sklearn_prediction1+1}")
print(f"Our prediction for first point: {classification}")
print(f"sklearn prediction for second point: Class {sklearn_prediction2+1}")
print(f"Our prediction for second point: {classification2}")

# Print the decision scores
sklearn_score1 = lda.decision_function([x_new])[0]
sklearn_score2 = lda.decision_function([x_new2])[0]
print(f"sklearn decision score for first point: {sklearn_score1:.4f}")
print(f"Our calculated projection relative to threshold for first point: {proj_x - threshold:.4f}")
print(f"sklearn decision score for second point: {sklearn_score2:.4f}")
print(f"Our calculated projection relative to threshold for second point: {proj_x2 - threshold:.4f}")

print("\nConclusion:")
print("-----------")
print(f"1. The LDA projection direction is w = {w}")
print(f"2. The threshold value for classification is {threshold:.4f}")
print(f"3. For the first new data point x1 = {x_new}, the projection is {proj_x:.4f}")
print(f"   Since {proj_x:.4f} {'>' if proj_x > threshold else '<='} {threshold:.4f}, LDA assigns it to {classification}")
print(f"4. For the second new data point x2 = {x_new2}, the projection is {proj_x2:.4f}")
print(f"   Since {proj_x2:.4f} {'>' if proj_x2 > threshold else '<='} {threshold:.4f}, LDA assigns it to {classification2}")
print("5. The LDA direction focuses on maximizing separation between the projected means") 