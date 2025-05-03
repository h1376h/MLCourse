import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# Fix font issues by using more basic LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "figure.dpi": 150,
})

# Set color palette for better visuals with higher contrast
CLASS1_COLOR = '#3498db'  # Blue
CLASS2_COLOR = '#e74c3c'  # Red
MEAN_COLOR1 = '#2980b9'   # Darker blue
MEAN_COLOR2 = '#c0392b'   # Darker red
NEW_POINT_COLOR = '#2ecc71'  # Green
BOUNDARY_COLOR = '#34495e'  # Dark blue/gray
PROJECTION_COLOR = '#9b59b6'  # Brighter purple for better visibility

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 23: LDA Classification")
print("==============================")

# Step 1: Calculate the mean vectors for each class
print("\nStep 1: Calculate the mean vectors for each class")
print("------------------------------------------------")

# Given data points - stored as separate points for detailed calculations
class1_points = [
    np.array([1, 0]),
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([0, -1])
]

class2_points = [
    np.array([0, 1]),
    np.array([0, -1]),
    np.array([2, 1]),
    np.array([2, -1])
]

# Store points in arrays for easier manipulation
class1 = np.array(class1_points)
class2 = np.array(class2_points)

# Calculate means explicitly
print("Class 1 points:")
for i, point in enumerate(class1_points):
    print(f"x_{i+1} = [{point[0]}, {point[1]}]")

mu1 = np.zeros(2)
for point in class1_points:
    mu1 += point
mu1 /= len(class1_points)

print("\nCalculating mean of Class 1:")
print(f"μ₁ = (1/4) * ([1, 0] + [-1, 0] + [0, 1] + [0, -1])")
print(f"μ₁ = (1/4) * ([0, 0])")
print(f"μ₁ = [{mu1[0]}, {mu1[1]}]")

print("\nClass 2 points:")
for i, point in enumerate(class2_points):
    print(f"x_{i+1} = [{point[0]}, {point[1]}]")

mu2 = np.zeros(2)
for point in class2_points:
    mu2 += point
mu2 /= len(class2_points)

print("\nCalculating mean of Class 2:")
print(f"μ₂ = (1/4) * ([0, 1] + [0, -1] + [2, 1] + [2, -1])")
print(f"μ₂ = (1/4) * ([4, 0])")
print(f"μ₂ = [{mu2[0]}, {mu2[1]}]")

print(f"\nFinal mean vectors:")
print(f"Mean vector for Class 1 (μ₁): [{mu1[0]}, {mu1[1]}]")
print(f"Mean vector for Class 2 (μ₂): [{mu2[0]}, {mu2[1]}]")

# Step 2: Calculate the shared covariance matrix
print("\nStep 2: Calculate the shared covariance matrix")
print("---------------------------------------------")

# Function to calculate covariance matrix for a class
def calculate_covariance(points, mean):
    n = len(points)
    cov = np.zeros((2, 2))
    
    print(f"\nCalculating covariance with mean = [{mean[0]}, {mean[1]}]")
    for i, point in enumerate(points):
        centered = point - mean
        outer_product = np.outer(centered, centered)
        print(f"Point {i+1}: [{point[0]}, {point[1]}]")
        print(f"  Centered: [{centered[0]}, {centered[1]}]")
        print(f"  Outer product:")
        print(f"  [{outer_product[0,0]:.4f}, {outer_product[0,1]:.4f}]")
        print(f"  [{outer_product[1,0]:.4f}, {outer_product[1,1]:.4f}]")
        cov += outer_product
    
    cov /= n
    return cov

# Calculate individual covariance matrices with detailed steps
print("\nCovariance matrix for Class 1:")
cov1 = calculate_covariance(class1_points, mu1)
print(f"\nFinal covariance matrix for Class 1:")
print(f"S₁ = [{cov1[0,0]:.4f}, {cov1[0,1]:.4f}]")
print(f"    [{cov1[1,0]:.4f}, {cov1[1,1]:.4f}]")

print("\nCovariance matrix for Class 2:")
cov2 = calculate_covariance(class2_points, mu2)
print(f"\nFinal covariance matrix for Class 2:")
print(f"S₂ = [{cov2[0,0]:.4f}, {cov2[0,1]:.4f}]")
print(f"    [{cov2[1,0]:.4f}, {cov2[1,1]:.4f}]")

# Calculate shared covariance (average of the two)
n1 = len(class1_points)
n2 = len(class2_points)
Sigma = ((n1 * cov1) + (n2 * cov2)) / (n1 + n2)

print(f"\nCalculating shared covariance matrix:")
print(f"Σ = (n₁·S₁ + n₂·S₂)/(n₁ + n₂)")
print(f"Σ = (4·S₁ + 4·S₂)/8")
print(f"Σ = (1/2)·S₁ + (1/2)·S₂")
print(f"Σ = (1/2)·[{cov1[0,0]:.4f}, {cov1[0,1]:.4f}] + (1/2)·[{cov2[0,0]:.4f}, {cov2[0,1]:.4f}]")
print(f"    (1/2)·[{cov1[1,0]:.4f}, {cov1[1,1]:.4f}] + (1/2)·[{cov2[1,0]:.4f}, {cov2[1,1]:.4f}]")
print(f"Σ = [{Sigma[0,0]:.4f}, {Sigma[0,1]:.4f}]")
print(f"    [{Sigma[1,0]:.4f}, {Sigma[1,1]:.4f}]")

# Step 3: Calculate the LDA projection direction
print("\nStep 3: Calculate the LDA projection direction")
print("--------------------------------------------")

# Calculate w = Σ^(-1)(μ₁ - μ₂)
Sigma_inv = np.linalg.inv(Sigma)
mean_diff = mu1 - mu2

print(f"Calculating mean difference (μ₁ - μ₂):")
print(f"μ₁ - μ₂ = [{mu1[0]}, {mu1[1]}] - [{mu2[0]}, {mu2[1]}]")
print(f"μ₁ - μ₂ = [{mean_diff[0]}, {mean_diff[1]}]")

print(f"\nCalculating inverse of shared covariance matrix (Σ⁻¹):")
print(f"Σ⁻¹ = inverse of [{Sigma[0,0]:.4f}, {Sigma[0,1]:.4f}]")
print(f"                 [{Sigma[1,0]:.4f}, {Sigma[1,1]:.4f}]")
print(f"Σ⁻¹ = [{Sigma_inv[0,0]:.4f}, {Sigma_inv[0,1]:.4f}]")
print(f"      [{Sigma_inv[1,0]:.4f}, {Sigma_inv[1,1]:.4f}]")

w = np.dot(Sigma_inv, mean_diff)

print(f"\nCalculating LDA projection direction w = Σ⁻¹(μ₁ - μ₂):")
print(f"w = [{Sigma_inv[0,0]:.4f}, {Sigma_inv[0,1]:.4f}] · [{mean_diff[0]}, {mean_diff[1]}]")
print(f"    [{Sigma_inv[1,0]:.4f}, {Sigma_inv[1,1]:.4f}]")
print(f"w = [{Sigma_inv[0,0]:.4f} · {mean_diff[0]} + {Sigma_inv[0,1]:.4f} · {mean_diff[1]}]")
print(f"    [{Sigma_inv[1,0]:.4f} · {mean_diff[0]} + {Sigma_inv[1,1]:.4f} · {mean_diff[1]}]")
print(f"w = [{w[0]:.4f}, {w[1]:.4f}]")

# Normalize w for visualization purposes
w_norm = w / np.linalg.norm(w)
print(f"\nNormalizing the projection direction:")
print(f"||w|| = √({w[0]:.4f}² + {w[1]:.4f}²) = {np.linalg.norm(w):.4f}")
print(f"w_norm = w/||w|| = [{w[0]:.4f}, {w[1]:.4f}]/{np.linalg.norm(w):.4f} = [{w_norm[0]:.4f}, {w_norm[1]:.4f}]")

# Step 4: Calculate the threshold value
print("\nStep 4: Calculate the threshold value")
print("-----------------------------------")

# Assuming equal prior probabilities
# The threshold is at the midpoint between the projected means
projected_mu1 = np.dot(w, mu1)
projected_mu2 = np.dot(w, mu2)

print(f"Calculating projected mean for Class 1:")
print(f"w·μ₁ = [{w[0]:.4f}, {w[1]:.4f}] · [{mu1[0]}, {mu1[1]}]")
print(f"w·μ₁ = {w[0]:.4f} · {mu1[0]} + {w[1]:.4f} · {mu1[1]} = {projected_mu1:.4f}")

print(f"\nCalculating projected mean for Class 2:")
print(f"w·μ₂ = [{w[0]:.4f}, {w[1]:.4f}] · [{mu2[0]}, {mu2[1]}]")
print(f"w·μ₂ = {w[0]:.4f} · {mu2[0]} + {w[1]:.4f} · {mu2[1]} = {projected_mu2:.4f}")

threshold = (projected_mu1 + projected_mu2) / 2

print(f"\nCalculating threshold (midpoint between projected means):")
print(f"threshold = (w·μ₁ + w·μ₂)/2 = ({projected_mu1:.4f} + {projected_mu2:.4f})/2 = {threshold:.4f}")

# Step 5: Classify a new point
print("\nStep 5: Classify a new point")
print("---------------------------")

x_new = np.array([1, 0])
projected_x_new = np.dot(w, x_new)

print(f"New point x_new = [{x_new[0]}, {x_new[1]}]")
print(f"\nCalculating projection of new point:")
print(f"w·x_new = [{w[0]:.4f}, {w[1]:.4f}] · [{x_new[0]}, {x_new[1]}]")
print(f"w·x_new = {w[0]:.4f} · {x_new[0]} + {w[1]:.4f} · {x_new[1]} = {projected_x_new:.4f}")

# Classify based on threshold
print(f"\nClassification decision:")
print(f"If w·x_new > threshold, assign to Class 1")
print(f"If w·x_new < threshold, assign to Class 2")
print(f"{projected_x_new:.4f} {'>' if projected_x_new > threshold else '<'} {threshold:.4f}")

class_assignment = 1 if projected_x_new > threshold else 2
print(f"Therefore, x_new is assigned to Class {class_assignment}")

# Print the decision boundary equation
print("\nDecision Boundary Equation:")
print(f"w^T x = θ")
print(f"{w[0]:.2f}x₁ + {w[1]:.2f}x₂ = {threshold:.2f}")

# Step 6: Visualize the data and decision boundary
print("\nStep 6: Visualize the data and decision boundary")
print("---------------------------------------------")

# IMPROVED VISUALIZATION 1: LDA Classification with Decision Boundary
plt.figure(figsize=(10, 8))

# Add a light grid
plt.grid(True, alpha=0.3, linestyle='--')

# Plot the data points with better aesthetics
plt.scatter(class1[:, 0], class1[:, 1], color=CLASS1_COLOR, s=120, marker='o', 
            label=r'Class 1', edgecolor='white', linewidth=1.5, alpha=0.8)
plt.scatter(class2[:, 0], class2[:, 1], color=CLASS2_COLOR, s=120, marker='x', 
            label=r'Class 2', linewidth=2, alpha=0.8)

# Plot the means with distinct markers - using simpler LaTeX notations
plt.scatter(mu1[0], mu1[1], color=MEAN_COLOR1, s=200, marker='*', 
            label=r'Mean of Class 1 ($\mu_1$)', edgecolor='white', linewidth=1.5, zorder=3)
plt.scatter(mu2[0], mu2[1], color=MEAN_COLOR2, s=200, marker='*', 
            label=r'Mean of Class 2 ($\mu_2$)', edgecolor='white', linewidth=1.5, zorder=3)

# Plot the new point
plt.scatter(x_new[0], x_new[1], color=NEW_POINT_COLOR, s=150, marker='D', 
            label=r'New Point ($x_{new}$)', edgecolor='white', linewidth=1.5, zorder=3)

# Add point labels with better positioning to avoid overlap
for i, point in enumerate(class1):
    # Adjust label positions to avoid overlap
    offset_x = 10
    offset_y = 10
    if point[0] == -1 and point[1] == 0:  # Second point in class 1
        offset_x = -25
    
    plt.annotate(f'$C_1^{{{i+1}}}$', (point[0], point[1]), 
                 xytext=(offset_x, offset_y), textcoords='offset points', 
                 fontsize=12, color=CLASS1_COLOR, fontweight='bold')

for i, point in enumerate(class2):
    # Adjust label positions to avoid overlap
    offset_x = -25
    offset_y = -15
    if point[0] == 0 and point[1] == 1:  # First point in class 2
        offset_x = 15
        offset_y = 15
    elif point[0] == 0 and point[1] == -1:  # Second point in class 2
        offset_x = 15
        offset_y = -25
    
    plt.annotate(f'$C_2^{{{i+1}}}$', (point[0], point[1]), 
                 xytext=(offset_x, offset_y), textcoords='offset points', 
                 fontsize=12, color=CLASS2_COLOR, fontweight='bold')

# Plot the LDA direction with a much larger, more visible arrow
scale = 6  # Increased scale for better visibility
arrow_props = dict(
    arrowstyle='-|>', 
    lw=4,  # Thicker line width
    shrinkA=0, 
    shrinkB=0, 
    fc=PROJECTION_COLOR, 
    ec=PROJECTION_COLOR, 
    alpha=1.0
)

# Add background for LDA direction for greater contrast
plt.plot([0, scale * w_norm[0]], [0, scale * w_norm[1]], 
         color='white', linewidth=6, alpha=0.8, zorder=2)

# Add the arrow on top
plt.annotate('', xy=(scale * w_norm[0], scale * w_norm[1]), xytext=(0, 0), 
             arrowprops=arrow_props, zorder=3)

# Add a clearer label for the LDA direction with better positioning
lda_label_x = scale * w_norm[0] * 0.5
lda_label_y = scale * w_norm[1] * 0.5
plt.annotate(r'LDA Direction ($w$)', 
             xy=(lda_label_x, lda_label_y), 
             xytext=(30, 15), textcoords='offset points', 
             fontsize=14, color=PROJECTION_COLOR, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=PROJECTION_COLOR))

# Add an explicit callout to the LDA direction arrow for extra emphasis
plt.annotate('', xy=(scale * w_norm[0] * 0.8, scale * w_norm[1] * 0.8), 
             xytext=(scale * w_norm[0] * 0.8 + 1.5, scale * w_norm[1] * 0.8 + 0.5), 
             arrowprops=dict(arrowstyle='->', lw=2, color=PROJECTION_COLOR), zorder=4)

# Calculate and plot the decision boundary with shaded regions
if w[1] != 0:  # Not a vertical line
    x_vals = np.array([-3, 3])
    y_vals = (threshold - w[0] * x_vals) / w[1]
    plt.plot(x_vals, y_vals, color=BOUNDARY_COLOR, linestyle='--', linewidth=2, 
             label=r'Decision Boundary', alpha=0.8)
    
    # Create fill regions
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = w[0]*xx + w[1]*yy - threshold
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=[CLASS2_COLOR, CLASS1_COLOR], alpha=0.1)
else:  # Vertical line
    plt.axvline(x=threshold/w[0], color=BOUNDARY_COLOR, linestyle='--', 
                linewidth=2, label=r'Decision Boundary', alpha=0.8)
    
    # Create fill regions for vertical line
    if w[0] > 0:
        plt.axvspan(threshold/w[0], 3, alpha=0.1, color=CLASS1_COLOR)
        plt.axvspan(-3, threshold/w[0], alpha=0.1, color=CLASS2_COLOR)
    else:
        plt.axvspan(-3, threshold/w[0], alpha=0.1, color=CLASS1_COLOR)
        plt.axvspan(threshold/w[0], 3, alpha=0.1, color=CLASS2_COLOR)

# Final plot settings
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.title(r'LDA Classification with Decision Boundary', fontsize=18, pad=20)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Create a custom legend with larger font and better positioning
plt.legend(fontsize=12, loc='upper right', framealpha=0.9, 
           edgecolor=BOUNDARY_COLOR, fancybox=True)

# Make the whole plot look better
plt.tight_layout()

# Save the figure with high quality
plt.savefig(os.path.join(save_dir, "lda_classification.png"), dpi=300, bbox_inches='tight')

# IMPROVED VISUALIZATION 2: 1D Projection with Classification Regions
plt.figure(figsize=(14, 4))

# Project all points onto a 1D line
proj_class1 = np.array([np.dot(w, point) for point in class1])
proj_class2 = np.array([np.dot(w, point) for point in class2])
proj_x_new = np.dot(w, x_new)

# Print the projected points
print("\nProjected points onto LDA direction:")
print("Class 1 projections:", [f"{p:.4f}" for p in proj_class1])
print("Class 2 projections:", [f"{p:.4f}" for p in proj_class2])
print(f"New point projection: {proj_x_new:.4f}")

# Define the 1D axis range
x_range = np.linspace(min(np.min(proj_class1), np.min(proj_class2)) - 0.5, 
                      max(np.max(proj_class1), np.max(proj_class2)) + 0.5, 1000)

# Mark the decision regions with better aesthetics
plt.axvspan(x_range[0], threshold, alpha=0.15, color=CLASS2_COLOR, label='Class 2 Region')
plt.axvspan(threshold, x_range[-1], alpha=0.15, color=CLASS1_COLOR, label='Class 1 Region')

# Add a subtle axis line
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

# Prepare sorted points and make sure there's no text overlap
# Sort points for better visualization
class1_proj_sorted = sorted(list(zip(proj_class1, range(len(proj_class1)))))
class2_proj_sorted = sorted(list(zip(proj_class2, range(len(proj_class2)))))

# Create distinct vertical positions to avoid overlap
# For class 1 (top half)
class1_offsets = []
for i in range(len(class1_proj_sorted)):
    if i > 0 and abs(class1_proj_sorted[i][0] - class1_proj_sorted[i-1][0]) < 0.5:
        # If this projection is close to the previous one, adjust offset
        class1_offsets.append(class1_offsets[-1] - 0.04)
    else:
        class1_offsets.append(0.12 - 0.02 * i)  # Start at 0.12 and decrease

# For class 2 (bottom half)
class2_offsets = []
for i in range(len(class2_proj_sorted)):
    if i > 0 and abs(class2_proj_sorted[i][0] - class2_proj_sorted[i-1][0]) < 0.5:
        # If this projection is close to the previous one, adjust offset
        class2_offsets.append(class2_offsets[-1] + 0.04)
    else:
        class2_offsets.append(-0.06 - 0.02 * i)  # Start at -0.06 and decrease

# Plot Class 1 points with optimized vertical offsets
for i, (proj, idx) in enumerate(class1_proj_sorted):
    offset_y = class1_offsets[i]
    label = f'$C_1^{{{idx+1}}}$'
    plt.scatter(proj, offset_y, color=CLASS1_COLOR, s=100, marker='o', 
                edgecolor='white', linewidth=1, zorder=3)
    plt.annotate(label, (proj, offset_y), 
                 xytext=(0, 10), textcoords='offset points', 
                 fontsize=12, ha='center', color=CLASS1_COLOR, 
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.plot([proj, proj], [0, offset_y], 'k--', alpha=0.3)

# Plot Class 2 points with optimized vertical offsets
for i, (proj, idx) in enumerate(class2_proj_sorted):
    offset_y = class2_offsets[i]
    label = f'$C_2^{{{idx+1}}}$'
    plt.scatter(proj, offset_y, color=CLASS2_COLOR, s=100, marker='x', 
                linewidth=2, zorder=3)
    plt.annotate(label, (proj, offset_y), 
                 xytext=(0, -18), textcoords='offset points', 
                 fontsize=12, ha='center', color=CLASS2_COLOR, 
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.plot([proj, proj], [0, offset_y], 'k--', alpha=0.3)

# Plot means with clearer labels
plt.scatter(projected_mu1, 0.20, color=MEAN_COLOR1, s=180, marker='*', zorder=4)
plt.annotate(r'$\mu_1$', (projected_mu1, 0.20), 
             xytext=(0, 12), textcoords='offset points', 
             fontsize=14, ha='center', color=MEAN_COLOR1, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.plot([projected_mu1, projected_mu1], [0, 0.20], 'k--', alpha=0.3)

plt.scatter(projected_mu2, -0.20, color=MEAN_COLOR2, s=180, marker='*', zorder=4)
plt.annotate(r'$\mu_2$', (projected_mu2, -0.20), 
             xytext=(0, -20), textcoords='offset points', 
             fontsize=14, ha='center', color=MEAN_COLOR2, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.plot([projected_mu2, projected_mu2], [0, -0.20], 'k--', alpha=0.3)

# Plot new point with a clear label
plt.scatter(proj_x_new, 0, color=NEW_POINT_COLOR, s=150, marker='D', 
            edgecolor='white', linewidth=1, zorder=5)
plt.annotate(r'$x_{new}$', (proj_x_new, 0), 
             xytext=(0, 20), textcoords='offset points', 
             fontsize=14, ha='center', color=NEW_POINT_COLOR, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Plot the threshold with a nicer label and line
plt.axvline(x=threshold, color=PROJECTION_COLOR, linestyle='--', linewidth=2.5)
plt.annotate(r'$\theta$', 
             xy=(threshold, 0.25),
             xytext=(0, -7), textcoords='offset points',
             fontsize=14, ha='center', color=PROJECTION_COLOR,
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Better axis labels and title
plt.xlim(x_range[0], x_range[-1])
plt.ylim(-0.3, 0.3)
plt.yticks([])  # Hide y-axis ticks since this is a 1D projection
plt.xlabel(r'Projection onto LDA Direction ($w^T x$)', fontsize=16)
plt.title(r'1D Projection of Data Points with Threshold', fontsize=18, pad=20)
plt.grid(True, alpha=0.3, axis='x')

# Add a custom legend
plt.legend(fontsize=12, loc='upper right', framealpha=0.9, 
           edgecolor=BOUNDARY_COLOR, fancybox=True)

plt.tight_layout()

# Save the improved 1D projection figure
plt.savefig(os.path.join(save_dir, "lda_1d_projection.png"), dpi=300, bbox_inches='tight')

# IMPROVED VISUALIZATION 3: Fisher's criterion
plt.figure(figsize=(10, 6))

# Calculate Fisher's criterion for different projection directions
angles = np.linspace(0, 180, 180)  # degrees
J_fisher = np.zeros_like(angles)

for i, angle_deg in enumerate(angles):
    # Convert degrees to radians
    angle_rad = angle_deg * np.pi / 180
    
    # Create projection direction
    w_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Project means
    mu1_proj = np.dot(w_dir, mu1)
    mu2_proj = np.dot(w_dir, mu2)
    
    # Project data and calculate variances
    class1_proj = np.array([np.dot(w_dir, p) for p in class1])
    class2_proj = np.array([np.dot(w_dir, p) for p in class2])
    
    s1_proj = np.var(class1_proj)
    s2_proj = np.var(class2_proj)
    
    # Calculate Fisher's criterion: J(w) = |μ₁ - μ₂|²/(s₁² + s₂²)
    between_class = (mu1_proj - mu2_proj)**2
    within_class = s1_proj + s2_proj
    
    # Avoid division by zero
    if within_class > 1e-10:
        J_fisher[i] = between_class / within_class
    else:
        J_fisher[i] = 0

# Print Fisher's criterion formula
print("\nFisher's Criterion Formula:")
print("J(w) = (μ₁ - μ₂)²/(s₁² + s₂²)")
print("Where:")
print("  (μ₁ - μ₂)² is the squared difference between projected means (between-class variance)")
print("  s₁² + s₂² is the sum of projected class variances (within-class variance)")

# Plot Fisher's criterion vs angle with better aesthetics
plt.plot(angles, J_fisher, color='#3498db', linewidth=2.5)
plt.fill_between(angles, J_fisher, alpha=0.2, color='#3498db')
plt.xlabel(r'Projection Direction Angle (degrees)', fontsize=14)
plt.ylabel(r"Fisher's Criterion $J(w)$", fontsize=14)
plt.title(r"Fisher's Criterion for Different Projection Directions", fontsize=18, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')

# Get LDA direction angle
lda_angle_rad = np.arctan2(w[1], w[0])
if lda_angle_rad < 0:
    lda_angle_rad += np.pi  # Convert to [0, π] range
lda_angle_deg = lda_angle_rad * 180 / np.pi

# Find the maximum Fisher's criterion
max_idx = np.argmax(J_fisher)
max_angle = angles[max_idx]

# Print the angle information
print(f"\nLDA Direction Angle: {lda_angle_deg:.1f}°")
print(f"Maximum Fisher's Criterion Angle: {max_angle:.1f}°")
print(f"Maximum Fisher's Criterion Value: {J_fisher[max_idx]:.4f}")

# Highlight the LDA direction and maximum
plt.axvline(x=lda_angle_deg, color='#e74c3c', linestyle='--', linewidth=2)
plt.axvline(x=max_angle, color='#2ecc71', linestyle='--', linewidth=2)

# Calculate the nearest valid index for the lda_angle_deg
nearest_idx = min(int(lda_angle_deg), len(angles)-1)
if nearest_idx < 0:
    nearest_idx = 0

# Simpler annotations without text boxes
plt.annotate('LDA Direction', 
             xy=(lda_angle_deg, J_fisher[nearest_idx]),
             xytext=(30, 30), textcoords='offset points',
             fontsize=12, color='#e74c3c',
             arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3,rad=.2',
                            color='#e74c3c'),
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.annotate('Maximum',
             xy=(max_angle, J_fisher[max_idx]),
             xytext=(-50, 50), textcoords='offset points',
             fontsize=12, color='#2ecc71',
             arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3,rad=-.2',
                            color='#2ecc71'),
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.legend([
    'Fisher\'s Criterion',
    'LDA Direction',
    'Maximum'
], loc='upper right', fontsize=12)

plt.tight_layout()

# Save the Fisher's criterion plot
plt.savefig(os.path.join(save_dir, "fishers_criterion.png"), dpi=300, bbox_inches='tight')

print("\nConclusion:")
print("-----------")
print(f"1. The mean vectors are μ₁ = [{mu1[0]}, {mu1[1]}] and μ₂ = [{mu2[0]}, {mu2[1]}]")
print(f"2. The shared covariance matrix is:")
print(f"   Σ = [{Sigma[0,0]:.4f}, {Sigma[0,1]:.4f}]")
print(f"       [{Sigma[1,0]:.4f}, {Sigma[1,1]:.4f}]")
print(f"3. The LDA projection direction is w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"4. The threshold value for classification is {threshold:.4f}")
print(f"5. The new point x_new = [{x_new[0]}, {x_new[1]}] is classified as Class {class_assignment}")
print(f"   - Projected value: {projected_x_new:.4f}")
print(f"   - Threshold: {threshold:.4f}")
print("6. All visualizations have been saved to the Images/L4_4_Quiz_23 directory")
print("   - lda_classification.png: Shows the decision boundary and class regions")
print("   - lda_1d_projection.png: Shows the 1D projection of all points")
print("   - fishers_criterion.png: Shows Fisher's criterion for different projection angles") 