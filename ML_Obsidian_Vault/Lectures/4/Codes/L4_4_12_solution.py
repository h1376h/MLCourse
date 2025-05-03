import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Calculate S_W^-1 manually to show the steps
print("\nCalculating S_W^-1 (inverse of within-class scatter matrix):")
# For a 2x2 matrix [[a, b], [c, d]], the inverse is 1/(ad-bc) * [[d, -b], [-c, a]]
a, b = S_W[0, 0], S_W[0, 1]
c, d = S_W[1, 0], S_W[1, 1]
det_S_W = a*d - b*c
print(f"Determinant of S_W = {a}*{d} - {b}*{c} = {det_S_W}")

S_W_inv_manual = np.array([[d, -b], [-c, a]]) / det_S_W
print("S_W^-1 (calculated manually):")
print(S_W_inv_manual)

# Double-check with NumPy's inverse function
S_W_inv = np.linalg.inv(S_W)
print("\nS_W^-1 (using numpy.linalg.inv):")
print(S_W_inv)

# Calculate S_W^-1 * S_B manually
print("\nCalculating S_W^-1 * S_B manually:")
S_W_inv_S_B_manual = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            S_W_inv_S_B_manual[i, j] += S_W_inv[i, k] * S_B[k, j]
            print(f"S_W_inv_S_B[{i},{j}] += S_W_inv[{i},{k}] * S_B[{k},{j}] = {S_W_inv[i, k]} * {S_B[k, j]} = {S_W_inv[i, k] * S_B[k, j]}")

print("\nS_W^-1 * S_B (calculated manually):")
print(S_W_inv_S_B_manual)

# Also calculate using NumPy's dot product for verification
S_W_inv_S_B = np.dot(S_W_inv, S_B)
print("\nS_W^-1 * S_B (using numpy.dot):")
print(S_W_inv_S_B)

# Calculate eigenvalues and eigenvectors manually for a 2x2 matrix
print("\nCalculating eigenvalues and eigenvectors manually:")
A = S_W_inv_S_B
a, b = A[0, 0], A[0, 1]
c, d = A[1, 0], A[1, 1]

# The characteristic equation is: λ^2 - (a+d)λ + (ad-bc) = 0
trace = a + d
determinant = a*d - b*c
print(f"Characteristic equation: λ^2 - {trace}λ + {determinant} = 0")

# Use the quadratic formula to find eigenvalues
# λ = (trace ± sqrt(trace^2 - 4*determinant))/2
discriminant = trace**2 - 4*determinant
print(f"Discriminant = {trace}^2 - 4*{determinant} = {discriminant}")

if discriminant >= 0:
    lambda1 = (trace + np.sqrt(discriminant))/2
    lambda2 = (trace - np.sqrt(discriminant))/2
    print(f"λ1 = ({trace} + sqrt({discriminant}))/2 = {lambda1}")
    print(f"λ2 = ({trace} - sqrt({discriminant}))/2 = {lambda2}")
else:
    print("Complex eigenvalues")

# Find eigenvectors for each eigenvalue
# For λ1: (A - λ1*I)v = 0
print(f"\nFinding eigenvector for λ1 = {lambda1}:")
A_minus_lambda1I = A - lambda1*np.eye(2)
print(f"A - λ1*I = \n{A_minus_lambda1I}")

# For a 2x2 matrix, we can find the eigenvector by solving one of the rows
# and normalizing the result
if abs(A_minus_lambda1I[0, 0]) > abs(A_minus_lambda1I[1, 0]):
    # Use first row
    v1 = np.array([A_minus_lambda1I[0, 1], -A_minus_lambda1I[0, 0]])
    print(f"Using first row: v1 = [{A_minus_lambda1I[0, 1]}, {-A_minus_lambda1I[0, 0]}]")
else:
    # Use second row
    v1 = np.array([A_minus_lambda1I[1, 1], -A_minus_lambda1I[1, 0]])
    print(f"Using second row: v1 = [{A_minus_lambda1I[1, 1]}, {-A_minus_lambda1I[1, 0]}]")

# Normalize v1
v1_norm = np.linalg.norm(v1)
v1_normalized = v1 / v1_norm
print(f"Normalizing: v1 / ||v1|| = {v1} / {v1_norm} = {v1_normalized}")

# Similarly for λ2
print(f"\nFinding eigenvector for λ2 = {lambda2}:")
A_minus_lambda2I = A - lambda2*np.eye(2)
print(f"A - λ2*I = \n{A_minus_lambda2I}")

if abs(A_minus_lambda2I[0, 0]) > abs(A_minus_lambda2I[1, 0]):
    # Use first row
    v2 = np.array([A_minus_lambda2I[0, 1], -A_minus_lambda2I[0, 0]])
    print(f"Using first row: v2 = [{A_minus_lambda2I[0, 1]}, {-A_minus_lambda2I[0, 0]}]")
else:
    # Use second row
    v2 = np.array([A_minus_lambda2I[1, 1], -A_minus_lambda2I[1, 0]])
    print(f"Using second row: v2 = [{A_minus_lambda2I[1, 1]}, {-A_minus_lambda2I[1, 0]}]")

# Normalize v2
v2_norm = np.linalg.norm(v2)
v2_normalized = v2 / v2_norm
print(f"Normalizing: v2 / ||v2|| = {v2} / {v2_norm} = {v2_normalized}")

# Verify with NumPy's eigenvalue function
eigenvalues, eigenvectors = np.linalg.eig(S_W_inv_S_B)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues (using numpy.linalg.eig):")
for i, ev in enumerate(eigenvalues):
    print(f"λ{i+1} = {ev.real:.4f}")

print("\nEigenvectors (columns, using numpy.linalg.eig):")
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

# Define the class means 
mean1 = np.array([1, 2])
mean2 = np.array([4, 0])

print(f"Class 1 mean: {mean1}")
print(f"Class 2 mean: {mean2}")

# Equal covariance matrix for both classes (identity matrix)
cov = np.identity(2)
print(f"Shared covariance matrix:\n{cov}")

# Generate samples from multivariate normal distributions
class1_samples = np.random.multivariate_normal(mean1, cov, n_samples // 2)
class2_samples = np.random.multivariate_normal(mean2, cov, n_samples // 2)

X = np.vstack([class1_samples, class2_samples])
y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Calculate midpoint between class means
midpoint = (mean1 + mean2) / 2
print(f"Midpoint between class means: {midpoint}")

# Calculate direction from mean1 to mean2
direction = mean2 - mean1
print(f"Direction from mean1 to mean2: {direction}")

# Calculate perpendicular direction for the decision boundary
# A vector perpendicular to [a, b] is [-b, a]
perp_direction = np.array([-direction[1], direction[0]])
print(f"Direction perpendicular to (mean2 - mean1): {perp_direction}")

# Normalize perpendicular direction
perp_direction_norm = np.linalg.norm(perp_direction)
perp_direction_normalized = perp_direction / perp_direction_norm
print(f"Normalized perpendicular direction: {perp_direction_normalized}")

# For equal prior probabilities, the LDA decision boundary is a line
# passing through the midpoint and perpendicular to the line connecting the means
# The equation is: perp_direction_normalized ⋅ (x - midpoint) = 0

# For visualization, express the boundary in the form: ax + by + c = 0
a, b = perp_direction_normalized
c = -(a * midpoint[0] + b * midpoint[1])
print(f"Decision boundary equation: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")

# Get points for the decision boundary line for plotting
boundary_x = np.linspace(-1, 6, 100)
# From ax + by + c = 0, we get y = (-ax - c) / b
boundary_y = (-a * boundary_x - c) / b

# Visualize the data and decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], color='blue', alpha=0.6, label='Class 1')
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], color='red', alpha=0.6, label='Class 2')

# Plot means
plt.scatter(mean1[0], mean1[1], color='blue', s=200, marker='*', label='Mean Class 1')
plt.scatter(mean2[0], mean2[1], color='red', s=200, marker='*', label='Mean Class 2')

# Show the midpoint
plt.scatter(midpoint[0], midpoint[1], color='purple', s=200, marker='o', label='Midpoint')

# Plot the decision boundary
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

print("\nFor binary classification with equal prior probabilities, the LDA decision boundary:")
print("1. Is perpendicular to the line connecting the two class means")
print("2. Passes through the midpoint between the class means")
print(f"3. Has the equation: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")

# Part 3: LDA vs Logistic Regression with Outliers
print("\nPart 3: LDA vs Logistic Regression with Outliers")
print("----------------------------------------------")

# Generate clean data for binary classification
np.random.seed(42)
X_clean, y_clean = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=42, n_clusters_per_class=1)

print(f"Generated {len(X_clean)} samples with 2 features for binary classification")
print(f"Class 0: {np.sum(y_clean == 0)} samples")
print(f"Class 1: {np.sum(y_clean == 1)} samples")

# Add outliers to one class
n_outliers = 10
print(f"Adding {n_outliers} outliers to class 1")

outlier_indices = np.random.choice(np.where(y_clean == 1)[0], n_outliers, replace=False)
X_outliers = X_clean.copy()

# Add large noise to create outliers
outlier_noise_mean = 0
outlier_noise_std = 10
print(f"Adding noise with mean={outlier_noise_mean}, std={outlier_noise_std} to create outliers")

outlier_noise = np.random.normal(outlier_noise_mean, outlier_noise_std, (n_outliers, 2))
X_outliers[outlier_indices, :] += outlier_noise

print("Outlier modifications:")
for i, idx in enumerate(outlier_indices):
    print(f"Outlier {i+1}: Original point {X_clean[idx]} → Modified point {X_outliers[idx]}")

# Split data for training and testing
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_outliers, y_clean, test_size=test_size, random_state=42)
print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

# Train LDA and Logistic Regression
lda = LinearDiscriminantAnalysis()
logreg = LogisticRegression(max_iter=1000)

lda.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# Evaluate performance
lda_train_acc = accuracy_score(y_train, lda.predict(X_train))
lda_test_acc = accuracy_score(y_test, lda.predict(X_test))
logreg_train_acc = accuracy_score(y_train, logreg.predict(X_train))
logreg_test_acc = accuracy_score(y_test, logreg.predict(X_test))

print("\nModel Performance:")
print(f"LDA: Training accuracy = {lda_train_acc:.4f}, Test accuracy = {lda_test_acc:.4f}")
print(f"Logistic Regression: Training accuracy = {logreg_train_acc:.4f}, Test accuracy = {logreg_test_acc:.4f}")

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

print("\nComparison with outliers:")
print("- LDA is more sensitive to outliers because it estimates distribution parameters (means and covariance)")
print("- Logistic Regression typically shows more robustness to outliers as it directly models the decision boundary")
if lda_test_acc < logreg_test_acc:
    print(f"- In this example, Logistic Regression outperforms LDA on the test set ({logreg_test_acc:.4f} vs {lda_test_acc:.4f})")
elif lda_test_acc > logreg_test_acc:
    print(f"- In this example, LDA outperforms Logistic Regression on the test set ({lda_test_acc:.4f} vs {logreg_test_acc:.4f})")
else:
    print(f"- In this example, both models perform equally on the test set (accuracy = {lda_test_acc:.4f})")

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

print(f"Generated non-Gaussian (uniform) data with {n_samples} samples")
print(f"Class 0: {np.sum(y_uniform == 0)} samples with uniform distribution in [-3,0] × [-3,0]")
print(f"Class 1: {np.sum(y_uniform == 1)} samples with uniform distribution in [0,3] × [0,3]")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_uniform, y_uniform, test_size=0.3, random_state=42)
print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

# Train models
lda_uniform = LinearDiscriminantAnalysis()
logreg_uniform = LogisticRegression(max_iter=1000)

lda_uniform.fit(X_train, y_train)
logreg_uniform.fit(X_train, y_train)

# Evaluate performance
lda_uniform_train_acc = accuracy_score(y_train, lda_uniform.predict(X_train))
lda_uniform_test_acc = accuracy_score(y_test, lda_uniform.predict(X_test))
logreg_uniform_train_acc = accuracy_score(y_train, logreg_uniform.predict(X_train))
logreg_uniform_test_acc = accuracy_score(y_test, logreg_uniform.predict(X_test))

print("\nModel Performance on Non-Gaussian Data:")
print(f"LDA: Training accuracy = {lda_uniform_train_acc:.4f}, Test accuracy = {lda_uniform_test_acc:.4f}")
print(f"Logistic Regression: Training accuracy = {logreg_uniform_train_acc:.4f}, Test accuracy = {logreg_uniform_test_acc:.4f}")

# Visualize models on non-Gaussian data
x_min, x_max = X_uniform[:, 0].min() - 1, X_uniform[:, 0].max() + 1
y_min, y_max = X_uniform[:, 1].min() - 1, X_uniform[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get predictions
Z_lda_uniform = lda_uniform.predict_proba(grid_points)[:, 1].reshape(xx.shape)
Z_logreg_uniform = logreg_uniform.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Plot the results
plt.figure(figsize=(18, 6))

# Plot LDA
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_lda_uniform, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_uniform[:, 0], X_uniform[:, 1], c=y_uniform, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z_lda_uniform, levels=[0.5], colors='k', linewidths=2)
plt.title('LDA on Non-Gaussian Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot Logistic Regression
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_logreg_uniform, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_uniform[:, 0], X_uniform[:, 1], c=y_uniform, edgecolors='k', cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z_logreg_uniform, levels=[0.5], colors='k', linewidths=2)
plt.title('Logistic Regression on Non-Gaussian Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lda_vs_logreg_non_gaussian.png"), dpi=300, bbox_inches='tight')

print("\nWhen to prefer Logistic Regression over LDA:")
print("1. Non-Gaussian data: LDA assumes Gaussian class-conditional densities, while Logistic Regression makes no such assumption")
print("2. Different covariance structures: LDA typically assumes classes share the same covariance matrix")
print("3. Presence of outliers: As shown earlier, Logistic Regression is often more robust to outliers")
print("4. Direct probability estimation: When accurate probability calibration is more important than understanding the data generation process")
print("5. Sufficient training data: Logistic Regression may need more data but can model more complex decision boundaries")

if logreg_uniform_test_acc > lda_uniform_test_acc:
    print(f"\nIn this non-Gaussian data example, Logistic Regression indeed outperforms LDA: {logreg_uniform_test_acc:.4f} vs {lda_uniform_test_acc:.4f}")
else:
    print(f"\nIn this particular non-Gaussian data example, LDA still performs well despite violated assumptions: {lda_uniform_test_acc:.4f} vs {logreg_uniform_test_acc:.4f}")

# Summary
print("\nSummary of LDA vs. Logistic Regression:")
print("-------------------------------------")
print("1. LDA is a generative model that estimates class-conditional densities p(x|C_k) and applies Bayes' rule")
print("2. Logistic Regression is a discriminative model that directly estimates P(C_k|x)")
print("3. The LDA decision boundary for equal priors is perpendicular to the line connecting class means and passes through their midpoint")
print("4. LDA is more sensitive to outliers than Logistic Regression due to its parameter estimation approach")
print("5. Logistic Regression is preferred when data is non-Gaussian, has different covariance structures, or contains outliers") 