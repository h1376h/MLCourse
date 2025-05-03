import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 12: LDA vs. Logistic Regression")
print("======================================")

# Part 1: Finding the direction that maximizes class separation in LDA
print("\nPart 1: Finding the direction for maximal class separation in LDA")
print("--------------------------------------------------------------")

# Given matrices
S_B = np.array([[4, 2], [2, 1]])
S_W = np.array([[2, 0], [0, 2]])

print("Between-class scatter matrix S_B:")
print(S_B)
print("\nWithin-class scatter matrix S_W:")
print(S_W)

# Calculate S_W^-1 * S_B
S_W_inv = np.linalg.inv(S_W)
S_W_inv_S_B = np.dot(S_W_inv, S_B)

print("\nS_W^-1 * S_B:")
print(S_W_inv_S_B)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S_W_inv_S_B)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues:")
for i, ev in enumerate(eigenvalues):
    print(f"λ{i+1} = {ev.real:.4f}")

print("\nEigenvectors (columns):")
print(eigenvectors)

# The direction that maximizes class separation is the eigenvector with the largest eigenvalue
max_direction = eigenvectors[:, 0]
print("\nDirection that maximizes class separation:")
print(max_direction)

# Visualize the direction and scatter matrices
plt.figure(figsize=(10, 8))

# Plot the between-class scatter matrix as an ellipse
def plot_ellipse(matrix, color, label, alpha=0.5):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)
    
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=(0, 0),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=alpha,
        edgecolor='black',
        label=label
    )
    plt.gca().add_patch(ellipse)

plot_ellipse(S_B, 'blue', 'Between-class scatter S_B')
plot_ellipse(S_W, 'red', 'Within-class scatter S_W')

# Plot the direction vector
plt.arrow(0, 0, max_direction[0], max_direction[1], head_width=0.1, head_length=0.2, 
          fc='green', ec='green', width=0.05, label='LDA Direction')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('LDA Direction and Scatter Matrices')
plt.legend()

plt.savefig(os.path.join(save_dir, "lda_direction.png"), dpi=300, bbox_inches='tight')

# Part 2: Decision Boundary for Equal Prior Probabilities
print("\nPart 2: LDA Decision Boundary with Equal Prior Probabilities")
print("----------------------------------------------------------")

# Generate synthetic data with two well-separated classes
np.random.seed(42)
n_samples = 200
mean1 = np.array([1, 2])
mean2 = np.array([4, 0])
cov = np.identity(2)  # same covariance matrix for both classes

class1_samples = np.random.multivariate_normal(mean1, cov, n_samples // 2)
class2_samples = np.random.multivariate_normal(mean2, cov, n_samples // 2)

X = np.vstack([class1_samples, class2_samples])
y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Visualize the data and decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], color='blue', alpha=0.6, label='Class 1')
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], color='red', alpha=0.6, label='Class 2')

# Plot means
plt.scatter(mean1[0], mean1[1], color='blue', s=200, marker='*', label='Mean Class 1')
plt.scatter(mean2[0], mean2[1], color='red', s=200, marker='*', label='Mean Class 2')

# Show the midpoint
midpoint = (mean1 + mean2) / 2
plt.scatter(midpoint[0], midpoint[1], color='purple', s=200, marker='o', label='Midpoint')

# Calculate direction perpendicular to the line connecting means (for the decision boundary)
direction = mean2 - mean1
perp_direction = np.array([-direction[1], direction[0]])  # Perpendicular vector

# Get points for the decision boundary line
boundary_x = np.linspace(-1, 6, 100)
slope = perp_direction[1] / perp_direction[0]
boundary_y = slope * (boundary_x - midpoint[0]) + midpoint[1]

plt.plot(boundary_x, boundary_y, 'k--', linewidth=2, label='Decision Boundary')

# Connect means with a line
plt.plot([mean1[0], mean2[0]], [mean1[1], mean2[1]], 'g-', linewidth=2, label='Means Connection')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('LDA Decision Boundary with Equal Priors')
plt.legend()
plt.grid(True)
plt.xlim(-1, 6)
plt.ylim(-2, 4)

plt.savefig(os.path.join(save_dir, "lda_decision_boundary.png"), dpi=300, bbox_inches='tight')

print("For binary classification with equal prior probabilities, the LDA decision boundary is:")
print("1. Perpendicular to the line connecting the two class means")
print("2. Intersects the line connecting the means at the midpoint between them")
print(f"In this example, the midpoint is at {midpoint}")

# Part 3: LDA vs Logistic Regression with Outliers
print("\nPart 3: LDA vs Logistic Regression with Outliers")
print("----------------------------------------------")

# Generate data with outliers
np.random.seed(42)
X_clean, y_clean = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=42, n_clusters_per_class=1)

# Add outliers to one class
outlier_indices = np.random.choice(np.where(y_clean == 1)[0], 10, replace=False)
X_outliers = X_clean.copy()
X_outliers[outlier_indices, :] += np.random.normal(0, 10, (10, 2))  # Add large noise to create outliers

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_outliers, y_clean, test_size=0.3, random_state=42)

# Train LDA and Logistic Regression
lda = LinearDiscriminantAnalysis()
logreg = LogisticRegression(max_iter=1000)

lda.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# Create a meshgrid for visualization
x_min, x_max = X_outliers[:, 0].min() - 1, X_outliers[:, 0].max() + 1
y_min, y_max = X_outliers[:, 1].min() - 1, X_outliers[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get predictions
Z_lda = lda.predict_proba(grid_points)[:, 1].reshape(xx.shape)
Z_logreg = logreg.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Create a figure comparing LDA and Logistic Regression
plt.figure(figsize=(18, 6))

# Plot LDA
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_lda, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_clean, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.scatter(X_outliers[outlier_indices, 0], X_outliers[outlier_indices, 1], 
           s=100, facecolors='none', edgecolors='green', linewidth=2, label='Outliers')
plt.contour(xx, yy, Z_lda, levels=[0.5], colors='k', linewidths=2)
plt.title('Linear Discriminant Analysis (LDA)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot Logistic Regression
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_logreg, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_clean, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.scatter(X_outliers[outlier_indices, 0], X_outliers[outlier_indices, 1], 
           s=100, facecolors='none', edgecolors='green', linewidth=2, label='Outliers')
plt.contour(xx, yy, Z_logreg, levels=[0.5], colors='k', linewidths=2)
plt.title('Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_vs_logreg_outliers.png"), dpi=300, bbox_inches='tight')

print("Comparison with outliers:")
print("- LDA is more sensitive to outliers because it models the class-conditional distributions")
print("- Logistic Regression is typically more robust to outliers as it directly models the decision boundary")

# Part 4: When to prefer Logistic Regression over LDA
print("\nPart 4: When to Prefer Logistic Regression over LDA")
print("------------------------------------------------")

# Generate non-Gaussian distributed data
np.random.seed(42)
# Create uniformly distributed data (non-Gaussian)
n_samples = 200
class1_samples_uniform = np.random.uniform(-3, 0, (n_samples // 2, 2))
class2_samples_uniform = np.random.uniform(0, 3, (n_samples // 2, 2))

X_uniform = np.vstack([class1_samples_uniform, class2_samples_uniform])
y_uniform = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_uniform, y_uniform, test_size=0.3, random_state=42)

# Train models
lda_uniform = LinearDiscriminantAnalysis()
logreg_uniform = LogisticRegression(max_iter=1000)

lda_uniform.fit(X_train, y_train)
logreg_uniform.fit(X_train, y_train)

# Evaluate models
lda_train_score = lda_uniform.score(X_train, y_train)
lda_test_score = lda_uniform.score(X_test, y_test)
logreg_train_score = logreg_uniform.score(X_train, y_train)
logreg_test_score = logreg_uniform.score(X_test, y_test)

print("Performance on non-Gaussian data:")
print(f"LDA - Train accuracy: {lda_train_score:.4f}, Test accuracy: {lda_test_score:.4f}")
print(f"Logistic Regression - Train accuracy: {logreg_train_score:.4f}, Test accuracy: {logreg_test_score:.4f}")

# Create a meshgrid for visualization
x_min, x_max = X_uniform[:, 0].min() - 1, X_uniform[:, 0].max() + 1
y_min, y_max = X_uniform[:, 1].min() - 1, X_uniform[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get predictions
Z_lda = lda_uniform.predict_proba(grid_points)[:, 1].reshape(xx.shape)
Z_logreg = logreg_uniform.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Create a figure comparing LDA and Logistic Regression on non-Gaussian data
plt.figure(figsize=(18, 6))

# Plot LDA
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_lda, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_uniform[:, 0], X_uniform[:, 1], c=y_uniform, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z_lda, levels=[0.5], colors='k', linewidths=2)
plt.title(f'LDA on Non-Gaussian Data\nTest Acc: {lda_test_score:.4f}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Replace inset histograms with a simple text annotation about the distribution
plt.annotate("Non-Gaussian Distribution\n(Uniform)", xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Plot Logistic Regression
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_logreg, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_uniform[:, 0], X_uniform[:, 1], c=y_uniform, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z_logreg, levels=[0.5], colors='k', linewidths=2)
plt.title(f'Logistic Regression on Non-Gaussian Data\nTest Acc: {logreg_test_score:.4f}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Replace inset histograms with a simple text annotation about the distribution
plt.annotate("Non-Gaussian Distribution\n(Uniform)", xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_vs_logreg_non_gaussian.png"), dpi=300, bbox_inches='tight')

print("\nScenarios where Logistic Regression is preferred over LDA:")
print("1. When the data distributions are not Gaussian (LDA assumes Gaussian distributions)")
print("2. When classes have very different covariance matrices (LDA assumes equal covariances)")
print("3. When the dataset contains outliers (LDA is more sensitive to outliers)")
print("4. When direct probability estimation is more important than generative modeling")
print("5. When the training set is large (LDA may overfit with fewer parameters)")

# Summary
print("\nOverall Conclusions:")
print("1. LDA maximizes the ratio of between-class variance to within-class variance")
print("2. With equal priors, the LDA decision boundary is perpendicular to the line")
print("   connecting class means and passes through their midpoint")
print("3. LDA is more sensitive to outliers than Logistic Regression")
print("4. Logistic Regression performs better when data distributions are non-Gaussian")
print("5. LDA is a generative model that models class-conditional densities")
print("   while Logistic Regression is a discriminative model that directly models posterior probabilities") 