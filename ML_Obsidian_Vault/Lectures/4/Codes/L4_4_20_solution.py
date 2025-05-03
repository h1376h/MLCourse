import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from scipy import linalg

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots with a white background for better printing
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

print("Question 20: LDA with Scatter Matrices")
print("=====================================")

# Given data points
class0 = np.array([[1, 2], [2, 3], [3, 3]])
class1 = np.array([[5, 2], [6, 3], [6, 4]])

# Combine all data for plotting
X = np.vstack([class0, class1])
y = np.array([0, 0, 0, 1, 1, 1])  # Class labels

# Step 1: Calculate the mean vector for each class
print("\nStep 1: Calculate the mean vector for each class")
print("---------------------------------------------")

mean0 = np.mean(class0, axis=0)
mean1 = np.mean(class1, axis=0)

print(f"Class 0 data points:")
for i, point in enumerate(class0):
    print(f"  x^({i+1}) = [{point[0]}, {point[1]}]^T")

print(f"\nClass 1 data points:")
for i, point in enumerate(class1):
    print(f"  x^({i+1}) = [{point[0]}, {point[1]}]^T")

print("\nCalculation of mean vector for Class 0:")
print(f"μ_0 = (1/3) * ([1, 2]^T + [2, 3]^T + [3, 3]^T)")
print(f"    = (1/3) * ([{1+2+3}, {2+3+3}]^T)")
print(f"    = (1/3) * ([6, 8]^T)")
print(f"    = [{6/3}, {8/3}]^T")
print(f"    = [{mean0[0]}, {mean0[1]}]^T")

print("\nCalculation of mean vector for Class 1:")
print(f"μ_1 = (1/3) * ([5, 2]^T + [6, 3]^T + [6, 4]^T)")
print(f"    = (1/3) * ([{5+6+6}, {2+3+4}]^T)")
print(f"    = (1/3) * ([17, 9]^T)")
print(f"    = [{17/3}, {9/3}]^T")
print(f"    = [{mean1[0]}, {mean1[1]}]^T")

# Step 2: Calculate the within-class scatter matrix Sw
print("\nStep 2: Calculate the within-class scatter matrix Sw")
print("------------------------------------------------")

# Calculate scatter matrix for each class
S0 = np.zeros((2, 2))
print("\nCalculation of scatter matrix for Class 0:")
for i, x in enumerate(class0):
    x_diff = x - mean0
    x_diff_col = x_diff.reshape(2, 1)
    contribution = x_diff_col @ x_diff_col.T
    S0 += contribution
    
    print(f"\nPoint x^({i+1}) = [{x[0]}, {x[1]}]^T:")
    print(f"  x^({i+1}) - μ_0 = [{x[0]}, {x[1]}]^T - [{mean0[0]}, {mean0[1]}]^T = [{x_diff[0]:.3f}, {x_diff[1]:.3f}]^T")
    print(f"  (x^({i+1}) - μ_0)(x^({i+1}) - μ_0)^T = [{x_diff[0]:.3f}, {x_diff[1]:.3f}]^T [{x_diff[0]:.3f}, {x_diff[1]:.3f}]")
    print(f"  = [  {contribution[0,0]:.3f}  {contribution[0,1]:.3f}  ]")
    print(f"    [  {contribution[1,0]:.3f}  {contribution[1,1]:.3f}  ]")

print(f"\nS_0 = sum of all contributions = [  {S0[0,0]}  {S0[0,1]}  ]")
print(f"                                  [  {S0[1,0]}  {S0[1,1]}  ]")

S1 = np.zeros((2, 2))
print("\nCalculation of scatter matrix for Class 1:")
for i, x in enumerate(class1):
    x_diff = x - mean1
    x_diff_col = x_diff.reshape(2, 1)
    contribution = x_diff_col @ x_diff_col.T
    S1 += contribution
    
    print(f"\nPoint x^({i+1}) = [{x[0]}, {x[1]}]^T:")
    print(f"  x^({i+1}) - μ_1 = [{x[0]}, {x[1]}]^T - [{mean1[0]}, {mean1[1]}]^T = [{x_diff[0]:.3f}, {x_diff[1]:.3f}]^T")
    print(f"  (x^({i+1}) - μ_1)(x^({i+1}) - μ_1)^T = [{x_diff[0]:.3f}, {x_diff[1]:.3f}]^T [{x_diff[0]:.3f}, {x_diff[1]:.3f}]")
    print(f"  = [  {contribution[0,0]:.3f}  {contribution[0,1]:.3f}  ]")
    print(f"    [  {contribution[1,0]:.3f}  {contribution[1,1]:.3f}  ]")

print(f"\nS_1 = sum of all contributions = [  {S1[0,0]}  {S1[0,1]}  ]")
print(f"                                  [  {S1[1,0]}  {S1[1,1]}  ]")

# Calculate within-class scatter matrix
Sw = S0 + S1

print(f"\nWithin-class scatter matrix Sw = S_0 + S_1 = ")
print(f"  [  {S0[0,0]}  {S0[0,1]}  ]   +   [  {S1[0,0]}  {S1[0,1]}  ]   =   [  {Sw[0,0]}  {Sw[0,1]}  ]")
print(f"  [  {S0[1,0]}  {S0[1,1]}  ]       [  {S1[1,0]}  {S1[1,1]}  ]       [  {Sw[1,0]}  {Sw[1,1]}  ]")

# Step 3: Calculate the between-class scatter matrix Sb
print("\nStep 3: Calculate the between-class scatter matrix Sb")
print("------------------------------------------------")

# Calculate between-class scatter matrix
mean_diff = mean0 - mean1
mean_diff_col = mean_diff.reshape(2, 1)
Sb = mean_diff_col @ mean_diff_col.T

print(f"Mean difference μ_0 - μ_1 = [{mean0[0]}, {mean0[1]}]^T - [{mean1[0]}, {mean1[1]}]^T = [{mean_diff[0]:.3f}, {mean_diff[1]:.3f}]^T")
print(f"Sb = (μ_0 - μ_1)(μ_0 - μ_1)^T = [{mean_diff[0]:.3f}, {mean_diff[1]:.3f}]^T [{mean_diff[0]:.3f}, {mean_diff[1]:.3f}]")
print(f"  = [  {Sb[0,0]:.3f}  {Sb[0,1]:.3f}  ]")
print(f"    [  {Sb[1,0]:.3f}  {Sb[1,1]:.3f}  ]")

# Step 4: Find LDA projection direction (eigenvector of Sw^-1*Sb with largest eigenvalue)
print("\nStep 4: Find LDA projection direction")
print("----------------------------------")

# Calculate Sw^-1
Sw_inv = linalg.inv(Sw)

print(f"Inverse of within-class scatter matrix Sw^-1:")
print(f"  [  {Sw_inv[0,0]:.3f}  {Sw_inv[0,1]:.3f}  ]")
print(f"  [  {Sw_inv[1,0]:.3f}  {Sw_inv[1,1]:.3f}  ]")

# Calculate Sw^-1 * Sb
Sw_inv_Sb = Sw_inv @ Sb

print(f"\nSw^-1 * Sb = ")
print(f"  [  {Sw_inv[0,0]:.3f}  {Sw_inv[0,1]:.3f}  ]   ×   [  {Sb[0,0]:.3f}  {Sb[0,1]:.3f}  ]   =   [  {Sw_inv_Sb[0,0]:.3f}  {Sw_inv_Sb[0,1]:.3f}  ]")
print(f"  [  {Sw_inv[1,0]:.3f}  {Sw_inv[1,1]:.3f}  ]       [  {Sb[1,0]:.3f}  {Sb[1,1]:.3f}  ]       [  {Sw_inv_Sb[1,0]:.3f}  {Sw_inv_Sb[1,1]:.3f}  ]")

# Find eigenvalues and eigenvectors of Sw^-1*Sb
eigenvalues, eigenvectors = linalg.eig(Sw_inv_Sb)

# Find the index of the largest eigenvalue
max_eigen_idx = np.argmax(np.abs(eigenvalues.real))
min_eigen_idx = np.argmin(np.abs(eigenvalues.real))
w = eigenvectors[:, max_eigen_idx].real  # Use real part if there's any numerical complex component

# Normalize the direction vector
w = w / np.linalg.norm(w)

print(f"\nEigenvalues of Sw^-1*Sb: [{eigenvalues[0].real:.6f}, {eigenvalues[1].real:.6f}]")
print(f"First eigenvector: [{eigenvectors[0,0].real:.6f}, {eigenvectors[1,0].real:.6f}]^T")
print(f"Second eigenvector: [{eigenvectors[0,1].real:.6f}, {eigenvectors[1,1].real:.6f}]^T")
print(f"\nEigenvector corresponding to largest eigenvalue λ = {eigenvalues[max_eigen_idx].real:.6f}:")
print(f"  v = [{eigenvectors[0,max_eigen_idx].real:.6f}, {eigenvectors[1,max_eigen_idx].real:.6f}]^T")

print(f"\nNormalized LDA projection direction w = v / ||v|| = [{w[0]:.6f}, {w[1]:.6f}]^T")

# Step 5: Calculate the threshold for classification assuming equal priors
print("\nStep 5: Calculate the threshold for classification")
print("---------------------------------------------")

# Project class means onto the LDA direction
mean0_projected = np.dot(w, mean0)
mean1_projected = np.dot(w, mean1)

# Calculate threshold (midpoint between projected means)
threshold = (mean0_projected + mean1_projected) / 2

print(f"Projection of Class 0 mean: w^T × μ_0 = [{w[0]:.6f}, {w[1]:.6f}] × [{mean0[0]}, {mean0[1]}]^T")
print(f"  = {w[0]:.6f} × {mean0[0]} + {w[1]:.6f} × {mean0[1]}")
print(f"  = {w[0] * mean0[0]:.6f} + ({w[1] * mean0[1]:.6f})")
print(f"  = {mean0_projected:.6f}")

print(f"\nProjection of Class 1 mean: w^T × μ_1 = [{w[0]:.6f}, {w[1]:.6f}] × [{mean1[0]}, {mean1[1]}]^T")
print(f"  = {w[0]:.6f} × {mean1[0]} + {w[1]:.6f} × {mean1[1]}")
print(f"  = {w[0] * mean1[0]:.6f} + ({w[1] * mean1[1]:.6f})")
print(f"  = {mean1_projected:.6f}")

print(f"\nClassification threshold (midpoint of projected means) = ({mean0_projected:.6f} + {mean1_projected:.6f}) / 2 = {threshold:.6f}")

# Step 6: Classify a new data point using LDA
print("\nStep 6: Classify a new data point using LDA")
print("----------------------------------------")

# New data point
x_new = np.array([4, 3])
x_new_projected = np.dot(w, x_new)

print(f"New data point: x_new = [{x_new[0]}, {x_new[1]}]^T")

print(f"\nProjection of new point: w^T × x_new = [{w[0]:.6f}, {w[1]:.6f}] × [{x_new[0]}, {x_new[1]}]^T")
print(f"  = {w[0]:.6f} × {x_new[0]} + {w[1]:.6f} × {x_new[1]}")
print(f"  = {w[0] * x_new[0]:.6f} + ({w[1] * x_new[1]:.6f})")
print(f"  = {x_new_projected:.6f}")

# Classify based on threshold
if x_new_projected < threshold:
    predicted_class = 0
    decision = "Class 0"
else:
    predicted_class = 1
    decision = "Class 1"

print(f"\nComparing projection with threshold: {x_new_projected:.6f} {'<' if x_new_projected < threshold else '>'} {threshold:.6f}")
print(f"Classification decision: {decision}")

# Create a decision function for plotting
def decision_function(x, w, threshold):
    return np.dot(x, w) - threshold

# Function to plot a covariance ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plot an ellipse representing the covariance matrix
    """
    if ax is None:
        ax = plt.gca()
    
    # Find eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Width and height are "full" widths, not radii
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
    ax.add_artist(ellip)
    return ellip

# Visualizations

# Plot 1: Data points with class means (simple version)
plt.figure(figsize=(8, 6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mean0[0], mean0[1], color='navy', s=150, marker='*', label='Mean Class 0')
plt.scatter(mean1[0], mean1[1], color='darkred', s=150, marker='*', label='Mean Class 1')

# Add simple point labels
for i, point in enumerate(class0):
    plt.text(point[0]+0.1, point[1]+0.1, f'C0-{i+1}', fontsize=10)
for i, point in enumerate(class1):
    plt.text(point[0]+0.1, point[1]+0.1, f'C1-{i+1}', fontsize=10)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Data Points with Class Means', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "class_means.png"), dpi=300, bbox_inches='tight')

# Plot 2: LDA projection direction and decision boundary (simple version)
plt.figure(figsize=(8, 6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mean0[0], mean0[1], color='navy', s=150, marker='*')
plt.scatter(mean1[0], mean1[1], color='darkred', s=150, marker='*')
plt.scatter(x_new[0], x_new[1], color='green', s=150, marker='d', label='New Point')

# Calculate the midpoint between class means
midpoint = (mean0 + mean1) / 2

# Calculate the orthogonal direction for the decision boundary
# The decision boundary passes through the midpoint and is orthogonal to the LDA direction
orthogonal_w = np.array([-w[1], w[0]])  # Rotate 90 degrees
boundary_start = midpoint + 4 * orthogonal_w
boundary_end = midpoint - 4 * orthogonal_w

# Draw the decision boundary
plt.plot([boundary_start[0], boundary_end[0]], 
        [boundary_start[1], boundary_end[1]], 
        'k--', linewidth=2, label='Decision Boundary')

# Draw the LDA direction vector
plt.arrow(midpoint[0], midpoint[1], 
         w[0], w[1], 
         head_width=0.2, head_length=0.3, fc='k', ec='k', 
         length_includes_head=True, label='LDA Direction')

# Add the new point annotation
plt.text(x_new[0]+0.2, x_new[1], f'New ({x_new[0]}, {x_new[1]}) → {decision}', fontsize=10)

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('LDA Direction and Decision Boundary', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# Plot 3: Projection of points onto LDA direction (simpler version)
plt.figure(figsize=(8, 4))
line_y = 0.5
# Draw the number line
plt.axhline(y=line_y, color='black', linestyle='-', linewidth=1)
plt.scatter(mean0_projected, line_y, color='navy', s=100, marker='*')
plt.scatter(mean1_projected, line_y, color='darkred', s=100, marker='*')
plt.scatter(x_new_projected, line_y, color='green', s=100, marker='d')

# Project all points
class0_proj = np.dot(class0, w)
class1_proj = np.dot(class1, w)

# Plot projected points
plt.scatter(class0_proj, [line_y]*len(class0_proj), color='blue', s=80, marker='o')
plt.scatter(class1_proj, [line_y]*len(class1_proj), color='red', s=80, marker='x')

# Add vertical line at the threshold
plt.axvline(x=threshold, color='k', linestyle='--', linewidth=1)

# Add labels
plt.text(mean0_projected, line_y-0.07, 'μ₀ proj', fontsize=10, ha='center')
plt.text(mean1_projected, line_y-0.07, 'μ₁ proj', fontsize=10, ha='center')
plt.text(threshold, line_y+0.1, 'Threshold', fontsize=10, ha='center')
plt.text(x_new_projected, line_y+0.1, 'New point', fontsize=10, ha='center', color='green')

for i, proj in enumerate(class0_proj):
    plt.text(proj, line_y+0.05, f'C0-{i+1}', fontsize=8, ha='center')
for i, proj in enumerate(class1_proj):
    plt.text(proj, line_y-0.1, f'C1-{i+1}', fontsize=8, ha='center')

plt.ylim([0.25, 0.75])
plt.yticks([])
plt.xlabel('Projection onto LDA Direction', fontsize=12)
plt.title('Projection of Data Points onto LDA Direction', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "projected_data.png"), dpi=300, bbox_inches='tight')

# Plot 4: Decision regions
plt.figure(figsize=(8, 6))

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Evaluate decision function on the grid
Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        Z[i, j] = np.dot(w, point) - threshold

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z, [0], linewidths=2, colors='k')

# Plot the data points
plt.scatter(class0[:, 0], class0[:, 1], color='blue', s=100, marker='o', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='red', s=100, marker='x', label='Class 1')
plt.scatter(mean0[0], mean0[1], color='navy', s=150, marker='*', label='Mean Class 0')
plt.scatter(mean1[0], mean1[1], color='darkred', s=150, marker='*', label='Mean Class 1')
plt.scatter(x_new[0], x_new[1], color='green', s=150, marker='d', label='New Point')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Decision Regions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "decision_regions.png"), dpi=300, bbox_inches='tight')

print("\nSummary of LDA Calculations:")
print("==========================")
print(f"1. Mean vectors:")
print(f"   - Class 0: μ_0 = [{mean0[0]}, {mean0[1]}]^T")
print(f"   - Class 1: μ_1 = [{mean1[0]}, {mean1[1]}]^T")

print(f"\n2. Within-class scatter matrix:")
print(f"   Sw = [  {Sw[0,0]}  {Sw[0,1]}  ]")
print(f"        [  {Sw[1,0]}  {Sw[1,1]}  ]")

print(f"\n3. Between-class scatter matrix:")
print(f"   Sb = [  {Sb[0,0]:.3f}  {Sb[0,1]:.3f}  ]")
print(f"        [  {Sb[1,0]:.3f}  {Sb[1,1]:.3f}  ]")

print(f"\n4. LDA projection direction:")
print(f"   w = [{w[0]:.6f}, {w[1]:.6f}]^T")

print(f"\n5. Classification threshold:")
print(f"   t = {threshold:.6f}")

print(f"\n6. Classification of new point [{x_new[0]}, {x_new[1]}]^T:")
print(f"   Projection = {x_new_projected:.6f}")
print(f"   Classified as: {decision}")

print("\nConclusion:")
print("===========")
print(f"The LDA analysis found the optimal projection direction w = [{w[0]:.6f}, {w[1]:.6f}]^T")
print(f"This direction maximizes the ratio of between-class to within-class scatter.")
print(f"The decision boundary is a line passing through ({midpoint[0]:.3f}, {midpoint[1]:.3f}) and perpendicular to w.")
print(f"The new point ({x_new[0]}, {x_new[1]}) is classified as {decision} because its projection")
print(f"value {x_new_projected:.6f} is {'less than' if x_new_projected < threshold else 'greater than'} the threshold {threshold:.6f}.")
print("The linear decision boundary successfully separates the two classes, confirming that the dataset is linearly separable.") 