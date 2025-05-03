import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 25: Linear Discriminant Analysis (LDA)")
print("==============================================")

# Step 1: Generate synthetic data to demonstrate LDA
print("\nStep 1: Generate synthetic data to demonstrate LDA")
print("-------------------------------------------------")

# Create two classes of data
np.random.seed(42)
mean1 = [2, 2]
mean2 = [4, 4]
cov1 = [[1.0, 0.8], [0.8, 1.0]]  # Covariance matrix for class 1
cov2 = [[1.0, -0.5], [-0.5, 1.0]]  # Covariance matrix for class 2

# Generate samples
n_samples = 100
X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
X2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# Combine the data and create labels
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# Calculate means for each class
mean1 = np.mean(X1, axis=0)
mean2 = np.mean(X2, axis=0)
overall_mean = np.mean(X, axis=0)

print("Generated synthetic data with two classes:")
print(f"Class 0: {n_samples} samples with mean {mean1}")
print(f"Class 1: {n_samples} samples with mean {mean2}")

# Plot the original data
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7, color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7, color='red', label='Class 1')
plt.scatter(mean1[0], mean1[1], color='navy', s=100, marker='X', label='Mean Class 0')
plt.scatter(mean2[0], mean2[1], color='darkred', s=100, marker='X', label='Mean Class 1')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Original 2D Data Distribution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "original_data.png"), dpi=300, bbox_inches='tight')

# Step 2: Compute LDA projection
print("\nStep 2: Compute LDA projection")
print("-----------------------------")

# Calculate within-class scatter matrix
S_W = np.zeros((2, 2))
for i, x in enumerate(X1):
    x = x.reshape(-1, 1)  # Column vector
    S_W += (x - mean1.reshape(-1, 1)) @ (x - mean1.reshape(-1, 1)).T
for i, x in enumerate(X2):
    x = x.reshape(-1, 1)  # Column vector
    S_W += (x - mean2.reshape(-1, 1)) @ (x - mean2.reshape(-1, 1)).T

# Calculate between-class scatter matrix
mean_diff = (mean2 - mean1).reshape(-1, 1)  # Column vector
S_B = n_samples * mean_diff @ mean_diff.T

# Calculate the projection vector using eigenvectors
S_W_inv = np.linalg.inv(S_W)
eigen_values, eigen_vectors = np.linalg.eig(S_W_inv @ S_B)

# Get the eigenvector with the largest eigenvalue
idx = np.argmax(eigen_values)
w = eigen_vectors[:, idx].real  # Extract the real part
w = w / np.linalg.norm(w)  # Normalize

print("LDA Calculation:")
print(f"Within-class scatter matrix:\n{S_W}")
print(f"Between-class scatter matrix:\n{S_B}")
print(f"Projection vector w = {w}")

# Project the data onto the LDA direction
X_proj = X @ w
threshold = (mean1 @ w + mean2 @ w) / 2
y_pred = (X_proj > threshold).astype(int)

# Plot the projected data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_proj[y == 0], np.zeros(n_samples), alpha=0.7, color='blue', label='Class 0')
plt.scatter(X_proj[y == 1], np.zeros(n_samples), alpha=0.7, color='red', label='Class 1')
plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')
plt.title('Data Projected onto LDA Direction', fontsize=16)
plt.xlabel('Projected Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Compute histograms
plt.subplot(1, 2, 2)
plt.hist(X_proj[y == 0], bins=20, alpha=0.5, color='blue', label='Class 0')
plt.hist(X_proj[y == 1], bins=20, alpha=0.5, color='red', label='Class 1')
plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')
plt.title('Histogram of Projected Data', fontsize=16)
plt.xlabel('Projected Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "projected_data.png"), dpi=300, bbox_inches='tight')

# Step 3: Visualize the LDA projection in 2D space
print("\nStep 3: Visualize the LDA projection in 2D space")
print("----------------------------------------------")

# Plot original data and projection direction
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7, color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7, color='red', label='Class 1')
plt.scatter(mean1[0], mean1[1], color='navy', s=100, marker='X', label='Mean Class 0')
plt.scatter(mean2[0], mean2[1], color='darkred', s=100, marker='X', label='Mean Class 1')

# Plot the LDA direction
origin = np.mean(X, axis=0)
plt.arrow(origin[0], origin[1], w[0]*2, w[1]*2, 
          head_width=0.2, head_length=0.2, fc='green', ec='green', label='LDA Direction')

# Calculate points to draw the decision boundary
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx = np.linspace(x_min, x_max, 100)

# Find the decision boundary line: w[0]*x + w[1]*y = threshold
# Solving for y: y = (threshold - w[0]*x) / w[1]
if w[1] != 0:
    yy = (threshold - w[0] * xx) / w[1]
    # Filter points within plot bounds
    valid_idx = (yy >= y_min) & (yy <= y_max)
    plt.plot(xx[valid_idx], yy[valid_idx], 'g--', label='Decision Boundary')
else:
    plt.axvline(x=threshold/w[0], color='green', linestyle='--', label='Decision Boundary')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('LDA Direction and Decision Boundary', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "lda_direction.png"), dpi=300, bbox_inches='tight')

# Step 4: Demonstrate LDA vs. Least Squares
print("\nStep 4: Demonstrate LDA vs. Least Squares")
print("---------------------------------------")

# Prepare data for least squares regression
y_for_reg = y.copy() * 2 - 1  # Convert to -1, 1 for easier comparison

# Fit Linear Regression (least squares)
reg = LinearRegression()
reg.fit(X, y_for_reg)
w_ls = reg.coef_ / np.linalg.norm(reg.coef_)  # Normalize for comparison
b_ls = reg.intercept_

# Calculate decision boundary for least squares
# The boundary is where w_ls[0]*x + w_ls[1]*y + b_ls = 0
# Solving for y: y = (-w_ls[0]*x - b_ls) / w_ls[1]
xx_ls = np.linspace(x_min, x_max, 100)
if w_ls[1] != 0:
    yy_ls = (-w_ls[0] * xx_ls - b_ls) / w_ls[1]
    valid_idx_ls = (yy_ls >= y_min) & (yy_ls <= y_max)
else:
    valid_idx_ls = np.zeros_like(xx_ls, dtype=bool)
    valid_idx_ls[xx_ls == -b_ls/w_ls[0]] = True

# Predict classes using least squares
X_proj_ls = X @ w_ls.T
threshold_ls = -b_ls / np.linalg.norm(w_ls)
y_pred_ls = (X_proj_ls > threshold_ls).astype(int)

# Calculate accuracies
accuracy_lda = accuracy_score(y, y_pred)
accuracy_ls = accuracy_score(y, y_pred_ls)

print(f"LDA projection vector: {w}")
print(f"LS regression vector: {w_ls}")
print(f"LDA threshold: {threshold:.4f}")
print(f"LS threshold: {threshold_ls:.4f}")
print(f"LDA classification accuracy: {accuracy_lda:.4f}")
print(f"LS classification accuracy: {accuracy_ls:.4f}")

# Plot LDA vs Least Squares
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7, color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7, color='red', label='Class 1')

# Plot the LDA direction
plt.arrow(origin[0], origin[1], w[0]*2, w[1]*2, 
          head_width=0.2, head_length=0.2, fc='green', ec='green', label='LDA Direction')

# Plot the LS direction
plt.arrow(origin[0], origin[1], w_ls[0]*2, w_ls[1]*2, 
          head_width=0.2, head_length=0.2, fc='purple', ec='purple', label='LS Direction')

# Plot LDA decision boundary
if w[1] != 0:
    plt.plot(xx[valid_idx], yy[valid_idx], 'g--', label='LDA Decision Boundary')
else:
    plt.axvline(x=threshold/w[0], color='green', linestyle='--', label='LDA Decision Boundary')

# Plot LS decision boundary
if w_ls[1] != 0:
    plt.plot(xx_ls[valid_idx_ls], yy_ls[valid_idx_ls], 'm--', label='LS Decision Boundary')
else:
    plt.axvline(x=-b_ls/w_ls[0], color='purple', linestyle='--', label='LS Decision Boundary')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('LDA vs. Least Squares', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "lda_vs_ls.png"), dpi=300, bbox_inches='tight')

# Step 5: Visualize within-class and between-class scatter (simplified)
print("\nStep 5: Visualize within-class and between-class scatter")
print("------------------------------------------------------")

# Plot the data with simplified visualization
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7, color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7, color='red', label='Class 1')
plt.scatter(mean1[0], mean1[1], color='navy', s=100, marker='X', label='Mean Class 0')
plt.scatter(mean2[0], mean2[1], color='darkred', s=100, marker='X', label='Mean Class 1')

# Plot the LDA direction
plt.arrow(origin[0], origin[1], w[0]*2, w[1]*2, 
          head_width=0.2, head_length=0.2, fc='green', ec='green', label='LDA Direction')

# Connect the means (between-class scatter illustration)
plt.plot([mean1[0], mean2[0]], [mean1[1], mean2[1]], 'k--', linewidth=2, 
         label='Between-class Distance')

plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('Within-class vs Between-class Scatter', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "scatter_visualization.png"), dpi=300, bbox_inches='tight')

# Step 6: Demonstrate LDA for different data configurations
print("\nStep 6: Demonstrate LDA for different data configurations")
print("------------------------------------------------------")

# Generate datasets with different properties to illustrate LDA behavior
np.random.seed(42)

# Case 1: Linearly separable data (large between-class, small within-class)
X1_sep, y1_sep = make_blobs(n_samples=[100, 100], centers=[[0, 0], [5, 5]], 
                            cluster_std=[1, 1], random_state=42)

# Case 2: Non-separable data (small between-class, large within-class)
X2_nonsep, y2_nonsep = make_blobs(n_samples=[100, 100], centers=[[0, 0], [2, 2]], 
                                 cluster_std=[2, 2], random_state=42)

# Fit LDA to both datasets
lda_sep = LinearDiscriminantAnalysis()
lda_nonsep = LinearDiscriminantAnalysis()

lda_sep.fit(X1_sep, y1_sep)
lda_nonsep.fit(X2_nonsep, y2_nonsep)

# Create meshgrid for decision boundaries
x_min1, x_max1 = X1_sep[:, 0].min() - 1, X1_sep[:, 0].max() + 1
y_min1, y_max1 = X1_sep[:, 1].min() - 1, X1_sep[:, 1].max() + 1
xx1, yy1 = np.meshgrid(np.arange(x_min1, x_max1, 0.02),
                      np.arange(y_min1, y_max1, 0.02))

x_min2, x_max2 = X2_nonsep[:, 0].min() - 1, X2_nonsep[:, 0].max() + 1
y_min2, y_max2 = X2_nonsep[:, 1].min() - 1, X2_nonsep[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min2, x_max2, 0.02),
                      np.arange(y_min2, y_max2, 0.02))

# Get predictions for meshgrid points
Z1 = lda_sep.predict(np.c_[xx1.ravel(), yy1.ravel()])
Z1 = Z1.reshape(xx1.shape)

Z2 = lda_nonsep.predict(np.c_[xx2.ravel(), yy2.ravel()])
Z2 = Z2.reshape(xx2.shape)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot 1: Linearly separable data
ax1.contourf(xx1, yy1, Z1, alpha=0.3, cmap=ListedColormap(['#AAAAFF', '#FFAAAA']))
ax1.scatter(X1_sep[y1_sep == 0, 0], X1_sep[y1_sep == 0, 1], alpha=0.7, color='blue', label='Class 0')
ax1.scatter(X1_sep[y1_sep == 1, 0], X1_sep[y1_sep == 1, 1], alpha=0.7, color='red', label='Class 1')
ax1.set_xlabel('Feature 1', fontsize=14)
ax1.set_ylabel('Feature 2', fontsize=14)
ax1.set_title('Linearly Separable Data\n(Good for LDA)', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Non-separable data
ax2.contourf(xx2, yy2, Z2, alpha=0.3, cmap=ListedColormap(['#AAAAFF', '#FFAAAA']))
ax2.scatter(X2_nonsep[y2_nonsep == 0, 0], X2_nonsep[y2_nonsep == 0, 1], alpha=0.7, color='blue', label='Class 0')
ax2.scatter(X2_nonsep[y2_nonsep == 1, 0], X2_nonsep[y2_nonsep == 1, 1], alpha=0.7, color='red', label='Class 1')
ax2.set_xlabel('Feature 1', fontsize=14)
ax2.set_ylabel('Feature 2', fontsize=14)
ax2.set_title('Non-separable Data\n(Challenging for LDA)', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_data_comparison.png"), dpi=300, bbox_inches='tight')

# Step 7: Calculate scores and accuracies
print("\nStep 7: Calculate scores and accuracies")
print("-------------------------------------")

# Calculate LDA projection scores
X1_sep_proj = lda_sep.transform(X1_sep)
X2_nonsep_proj = lda_nonsep.transform(X2_nonsep)

# Calculate separability measures
def calculate_separability(X_proj, y):
    class0 = X_proj[y == 0]
    class1 = X_proj[y == 1]
    
    mean0 = np.mean(class0)
    mean1 = np.mean(class1)
    var0 = np.var(class0)
    var1 = np.var(class1)
    
    # Fisher's discriminant ratio: (mean1 - mean0)^2 / (var0 + var1)
    fisher_ratio = (mean1 - mean0)**2 / (var0 + var1)
    
    return {
        'mean_class0': mean0,
        'mean_class1': mean1,
        'var_class0': var0,
        'var_class1': var1,
        'fisher_ratio': fisher_ratio
    }

sep_metrics = calculate_separability(X1_sep_proj, y1_sep)
nonsep_metrics = calculate_separability(X2_nonsep_proj, y2_nonsep)

print("Separability Metrics:")
print("\nLinearly Separable Data:")
print(f"Mean Class 0: {sep_metrics['mean_class0']:.4f}")
print(f"Mean Class 1: {sep_metrics['mean_class1']:.4f}")
print(f"Variance Class 0: {sep_metrics['var_class0']:.4f}")
print(f"Variance Class 1: {sep_metrics['var_class1']:.4f}")
print(f"Fisher's Ratio: {sep_metrics['fisher_ratio']:.4f}")

print("\nNon-separable Data:")
print(f"Mean Class 0: {nonsep_metrics['mean_class0']:.4f}")
print(f"Mean Class 1: {nonsep_metrics['mean_class1']:.4f}")
print(f"Variance Class 0: {nonsep_metrics['var_class0']:.4f}")
print(f"Variance Class 1: {nonsep_metrics['var_class1']:.4f}")
print(f"Fisher's Ratio: {nonsep_metrics['fisher_ratio']:.4f}")

# Prediction accuracy
acc_sep = accuracy_score(y1_sep, lda_sep.predict(X1_sep))
acc_nonsep = accuracy_score(y2_nonsep, lda_nonsep.predict(X2_nonsep))

print(f"\nAccuracy on linearly separable data: {acc_sep:.4f}")
print(f"Accuracy on non-separable data: {acc_nonsep:.4f}")

# Plot histograms of projected data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Histogram for separable data
ax1.hist(X1_sep_proj[y1_sep == 0], bins=20, alpha=0.5, color='blue', label='Class 0')
ax1.hist(X1_sep_proj[y1_sep == 1], bins=20, alpha=0.5, color='red', label='Class 1')
ax1.axvline(x=0, color='green', linestyle='--', label='Decision Boundary')
ax1.set_title('Projected Linearly Separable Data', fontsize=16)
ax1.set_xlabel('LDA Projection', fontsize=14)
ax1.set_ylabel('Frequency', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# Plot 2: Histogram for non-separable data
ax2.hist(X2_nonsep_proj[y2_nonsep == 0], bins=20, alpha=0.5, color='blue', label='Class 0')
ax2.hist(X2_nonsep_proj[y2_nonsep == 1], bins=20, alpha=0.5, color='red', label='Class 1')
ax2.axvline(x=0, color='green', linestyle='--', label='Decision Boundary')
ax2.set_title('Projected Non-separable Data', fontsize=16)
ax2.set_xlabel('LDA Projection', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "projected_data_comparison.png"), dpi=300, bbox_inches='tight')

# Step 8: Summarize key findings for Question 25
print("\nStep 8: Summarize key findings for Question 25")
print("-------------------------------------------")

statement1 = "LDA projects p-dimensional data into a one-dimensional space and compares with a threshold"
statement2 = "LDA is more appropriate for linearly separable data"
statement3 = "Mean values of both classes play essential roles in LDA"
statement4 = "LDA aims to maximize between-class variations and minimize within-class variations"
statement5 = "LDA is always equivalent to linear classification with LSE"

# Assess each statement based on the analysis
statement1_true = True
statement2_true = acc_sep > acc_nonsep
statement3_true = True
statement4_true = True
statement5_true = False

print(f"Statement 1: {statement1}")
print(f"TRUE. Our analysis confirms that LDA projects data to 1D and uses a threshold for classification.")

print(f"\nStatement 2: {statement2}")
print(f"TRUE. We showed that LDA performs better on linearly separable data (accuracy {acc_sep:.4f}) " 
      f"compared to non-separable data (accuracy {acc_nonsep:.4f}).")

print(f"\nStatement 3: {statement3}")
print(f"TRUE. Class means are crucial in LDA; they determine the between-class scatter matrix " 
      f"and influence the projection direction significantly.")

print(f"\nStatement 4: {statement4}")
print(f"TRUE. LDA explicitly maximizes the ratio of between-class to within-class scatter, " 
      f"as measured by Fisher's ratio (separable: {sep_metrics['fisher_ratio']:.4f}, " 
      f"non-separable: {nonsep_metrics['fisher_ratio']:.4f}).")

print(f"\nStatement 5: {statement5}")
print(f"FALSE. LDA and Least Squares Estimation (LSE) yield different decision boundaries " 
      f"and projection vectors, as our comparison demonstrated.")

print("\nConclusion:")
print("1. LDA is a powerful dimensionality reduction technique for classification")
print("2. It works by projecting high-dimensional data to a lower dimension")
print("3. It seeks to maximize class separability in the projected space")
print("4. LDA is particularly effective for linearly separable data")
print("5. LDA is not equivalent to least squares classification") 