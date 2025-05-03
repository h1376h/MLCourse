import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_11")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 11: Combining Concepts")
print("==============================")

# Part 1: Bias-Variance Tradeoff Visualization
print("\nPart 1: Bias-Variance Tradeoff with Linear vs. Non-linear Boundaries")
print("------------------------------------------------------------------")

# Generate data with a non-linear decision boundary (circle)
np.random.seed(42)
n_samples = 300

# Generate points in a circle
radius = 5
center_x, center_y = 0, 0
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = np.sqrt(np.random.uniform(0, radius**2, n_samples))
x1 = center_x + r * np.cos(theta)
x2 = center_y + r * np.sin(theta)

# Add some noise to make it more realistic
noise = np.random.normal(0, 0.5, size=(n_samples, 2))
X = np.column_stack([x1, x2]) + noise

# Create labels: points inside a circle of radius 3 are class 1, others are class 0
y = (np.sqrt(X[:, 0]**2 + X[:, 1]**2) < 3).astype(int)

# Split the data for training and visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a meshgrid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Train different models with varying complexity
models = [
    ('Linear (High Bias)', LogisticRegression(C=1.0, max_iter=1000)),
    ('Quadratic (Medium Complexity)', Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('logreg', LogisticRegression(C=1.0, max_iter=1000))
    ])),
    ('Higher Order (Potential High Variance)', Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('logreg', LogisticRegression(C=1.0, max_iter=1000))
    ]))
]

# Plot the data and decision boundaries
plt.figure(figsize=(18, 6))

for i, (name, model) in enumerate(models):
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Make predictions on the grid for visualization
    if name == 'Linear (High Bias)':
        Z = model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
    else:
        Z = model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.RdBu_r, edgecolors='k')
    
    plt.title(f"{name}\nTrain Acc: {train_accuracy:.3f}, Test Acc: {test_accuracy:.3f}")
    plt.xlabel('Feature $x_1$')
    plt.ylabel('Feature $x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "bias_variance_tradeoff.png"), dpi=300, bbox_inches='tight')

print("Generated visualization comparing linear (high bias) vs non-linear models (potential high variance)")
print("The linear model has high bias as it cannot capture the circular decision boundary.")
print("The higher-order polynomial model may have high variance, potentially overfitting the data.")

# Part 2: Regularization Effects on Overlapping Classes
print("\nPart 2: Regularization Effects with Overlapping Classes")
print("--------------------------------------------------")

# Generate overlapping data
np.random.seed(42)
n_samples = 300

# Create two overlapping Gaussian distributions
mean1 = [0, 0]
mean2 = [2, 2]
cov = [[2, 0.5], [0.5, 2]]  # Covariance matrix with some correlation

# Generate samples from each distribution
X1 = np.random.multivariate_normal(mean1, cov, n_samples // 2)
X2 = np.random.multivariate_normal(mean2, cov, n_samples // 2)
X_overlap = np.vstack([X1, X2])
y_overlap = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_overlap, y_overlap, test_size=0.3, random_state=42)

# Create a meshgrid for visualization
x_min, x_max = X_overlap[:, 0].min() - 1, X_overlap[:, 0].max() + 1
y_min, y_max = X_overlap[:, 1].min() - 1, X_overlap[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Train models with different regularization strengths
C_values = [0.01, 1.0, 100.0]  # From strong to weak regularization
models_reg = [
    (f'Strong Reg (C={C_values[0]})', LogisticRegression(C=C_values[0], max_iter=1000)),
    (f'Medium Reg (C={C_values[1]})', LogisticRegression(C=C_values[1], max_iter=1000)),
    (f'Weak Reg (C={C_values[2]})', LogisticRegression(C=C_values[2], max_iter=1000))
]

# Plot the data and decision boundaries
plt.figure(figsize=(18, 6))

for i, (name, model) in enumerate(models_reg):
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Make predictions on the grid for visualization
    Z = model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
    plt.scatter(X_overlap[:, 0], X_overlap[:, 1], c=y_overlap, s=20, cmap=plt.cm.RdBu_r, edgecolors='k')
    
    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
    
    plt.title(f"{name}\nTrain Acc: {train_accuracy:.3f}, Test Acc: {test_accuracy:.3f}")
    plt.xlabel('Feature $x_1$')
    plt.ylabel('Feature $x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "regularization_effects.png"), dpi=300, bbox_inches='tight')

print("Generated visualization showing effects of regularization on overlapping classes")
print("Strong regularization creates smoother decision boundaries, which may generalize better.")
print("Weak regularization allows more complex boundaries that may overfit to the training data.")

# Part 3: Circular Decision Boundary with Feature Transformation
print("\nPart 3: Circular Decision Boundary with Feature Transformation")
print("--------------------------------------------------------")

# Generate data with a perfect circular decision boundary
np.random.seed(43)
n_samples = 300

# Generate points uniformly in a square
x1 = np.random.uniform(-5, 5, n_samples)
x2 = np.random.uniform(-5, 5, n_samples)
X_circle = np.column_stack([x1, x2])

# Create labels: points inside a circle of radius 3 are class 1, others are class 0
y_circle = (np.sqrt(X_circle[:, 0]**2 + X_circle[:, 1]**2) < 3).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_circle, y_circle, test_size=0.3, random_state=42)

# Create a meshgrid for visualization
x_min, x_max = X_circle[:, 0].min() - 1, X_circle[:, 0].max() + 1
y_min, y_max = X_circle[:, 1].min() - 1, X_circle[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Create models
linear_model = LogisticRegression(C=1.0, max_iter=1000)
quadratic_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('logreg', LogisticRegression(C=1.0, max_iter=1000))
])

# Train models
linear_model.fit(X_train, y_train)
quadratic_model.fit(X_train, y_train)

# Calculate training and test accuracy
linear_train_acc = accuracy_score(y_train, linear_model.predict(X_train))
linear_test_acc = accuracy_score(y_test, linear_model.predict(X_test))
quad_train_acc = accuracy_score(y_train, quadratic_model.predict(X_train))
quad_test_acc = accuracy_score(y_test, quadratic_model.predict(X_test))

# Make predictions for visualization
Z_linear = linear_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
Z_quad = quadratic_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

# Create 2D feature transformation visualization
plt.figure(figsize=(16, 7))

# Plot linear model
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_linear, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, s=20, cmap=plt.cm.RdBu_r, edgecolors='k')
plt.contour(xx, yy, Z_linear, levels=[0.5], colors='k', linewidths=2)
plt.title(f"Linear Classifier\nTrain Acc: {linear_train_acc:.3f}, Test Acc: {linear_test_acc:.3f}")
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')

# Plot quadratic model
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_quad, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
plt.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle, s=20, cmap=plt.cm.RdBu_r, edgecolors='k')
plt.contour(xx, yy, Z_quad, levels=[0.5], colors='k', linewidths=2)
plt.title(f"Quadratic Feature Transform\nTrain Acc: {quad_train_acc:.3f}, Test Acc: {quad_test_acc:.3f}")
plt.xlabel('Feature $x_1$')
plt.ylabel('Feature $x_2$')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "circular_boundary.png"), dpi=300, bbox_inches='tight')

# Show the feature transformation details
poly = PolynomialFeatures(degree=2)
X_sample = np.array([[1, 2]])
X_transformed = poly.fit_transform(X_sample)
feature_names = poly.get_feature_names_out(['x1', 'x2'])

print("Generated visualization comparing linear classifier vs quadratic feature transform on circular data")
print(f"Linear model accuracy: Train {linear_train_acc:.3f}, Test {linear_test_acc:.3f}")
print(f"Quadratic model accuracy: Train {quad_train_acc:.3f}, Test {quad_test_acc:.3f}")
print("\nQuadratic feature transformation details:")
print("Original features [x1, x2] = [1, 2]")
print("Transformed features:", feature_names)
print("Transformed values:", X_transformed[0])
print("\nA quadratic feature transform can perfectly capture a circular decision boundary")
print("because it includes quadratic terms (x1², x2², x1·x2) that can model the equation of a circle: x1² + x2² = r²")

# Print overall conclusions
print("\nOverall Conclusions:")
print("1. Bias-Variance tradeoff: Linear models have high bias (unable to capture non-linear boundaries),")
print("   while complex models may have high variance (potential overfitting).")
print("2. With overlapping classes, regularization helps create more generalizable decision boundaries.")
print("3. For data with circular decision boundaries, quadratic feature transforms are more appropriate")
print("   as they can model the equation of a circle: x1² + x2² = r²") 