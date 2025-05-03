import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 4: Feature Transformations for Non-Linear Separability")
print("=============================================================")

# Step 1: Generate non-linearly separable datasets
print("\nStep 1: Generate non-linearly separable datasets")
print("---------------------------------------------")

# Generate two types of non-linearly separable datasets
n_samples = 200

# 1. Concentric circles
X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
# 2. Two interleaving half-moons
X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# Function to plot datasets
def plot_dataset(X, y, title, filename):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', s=40, marker='o', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', s=40, marker='x', label='Class 1')
    plt.title(title, fontsize=16)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Plot the datasets
plot_dataset(X_circles, y_circles, "Non-Linearly Separable Dataset: Concentric Circles", "circles_dataset.png")
plot_dataset(X_moons, y_moons, "Non-Linearly Separable Dataset: Half Moons", "moons_dataset.png")

print("Created two non-linearly separable datasets:")
print("1. Concentric circles: Inner circle of one class surrounded by outer circle of another class")
print("2. Half moons: Two interleaving crescent-shaped classes")
print("These datasets cannot be separated by a linear decision boundary in their original 2D space.")

# Step 2: Demonstrate linear classifiers failing on non-linear data
print("\nStep 2: Show why linear classifiers fail on this data")
print("------------------------------------------------")

# Function to plot decision boundaries
def plot_decision_boundary(X, y, model, title, filename, feature_transform=None):
    plt.figure(figsize=(10, 8))
    
    # Set the step size for the mesh grid
    h = 0.02  # smaller step size for finer boundaries
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for all grid points
    if feature_transform is not None:
        # Apply the feature transformation first
        Z = model.predict(feature_transform(np.c_[xx.ravel(), yy.ravel()]))
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    cm = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.3)
    plt.contour(xx, yy, Z, colors='black', linewidths=1)
    
    # Plot the original points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', s=40, marker='o', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', s=40, marker='x', label='Class 1')
    
    plt.title(title, fontsize=16)
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Train a linear classifier on circles data
linear_model_circles = LogisticRegression(C=1.0)
linear_model_circles.fit(X_circles, y_circles)
circles_linear_accuracy = linear_model_circles.score(X_circles, y_circles)

# Train a linear classifier on moons data
linear_model_moons = LogisticRegression(C=1.0)
linear_model_moons.fit(X_moons, y_moons)
moons_linear_accuracy = linear_model_moons.score(X_moons, y_moons)

# Plot the results
plot_decision_boundary(X_circles, y_circles, linear_model_circles, 
                      f"Linear Classifier on Circles (Accuracy: {circles_linear_accuracy:.2f})",
                      "circles_linear.png")

plot_decision_boundary(X_moons, y_moons, linear_model_moons, 
                      f"Linear Classifier on Moons (Accuracy: {moons_linear_accuracy:.2f})",
                      "moons_linear.png")

print(f"Linear classifier accuracy on circles dataset: {circles_linear_accuracy:.2f}")
print(f"Linear classifier accuracy on moons dataset: {moons_linear_accuracy:.2f}")
print("As expected, the linear classifiers perform poorly because these datasets are not linearly separable.")
print("The decision boundaries are straight lines that cannot properly separate the classes.")

# Step 3: Polynomial feature transformation
print("\nStep 3: Apply polynomial feature transformation")
print("-------------------------------------------")

# Define the quadratic feature transformation
def quadratic_transform(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    return poly.fit_transform(X)

# Create a pipeline for polynomial transformation + linear classifier for circles
quad_model_circles = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('log_reg', LogisticRegression(C=1.0))
])
quad_model_circles.fit(X_circles, y_circles)
circles_quad_accuracy = quad_model_circles.score(X_circles, y_circles)

# Create a pipeline for polynomial transformation + linear classifier for moons
quad_model_moons = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('log_reg', LogisticRegression(C=1.0))
])
quad_model_moons.fit(X_moons, y_moons)
moons_quad_accuracy = quad_model_moons.score(X_moons, y_moons)

# Function to plot decision boundaries with transformed features
def plot_transformed_boundary(X, y, model, title, filename):
    # Use the plot_decision_boundary function but pass the transformation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train the model on transformed features
    model.fit(X_poly, y)
    
    # Plot the decision boundary
    plot_decision_boundary(X, y, model, title, filename, 
                          feature_transform=lambda X: poly.transform(X))

# Plot transformed decision boundaries
plot_transformed_boundary(X_circles, y_circles, LogisticRegression(C=1.0),
                         f"Quadratic Features on Circles (Accuracy: {circles_quad_accuracy:.2f})",
                         "circles_quadratic.png")

plot_transformed_boundary(X_moons, y_moons, LogisticRegression(C=1.0),
                         f"Quadratic Features on Moons (Accuracy: {moons_quad_accuracy:.2f})",
                         "moons_quadratic.png")

# Show the quadratic feature transformation for a few points
print("\nDemonstration of quadratic feature transformation:")
original_points = np.array([[1.0, 2.0], [0.5, -1.0]])
transformed_points = quadratic_transform(original_points)

print("Original features (x₁, x₂):")
for point in original_points:
    print(f"({point[0]}, {point[1]})")

print("\nTransformed features [x₁, x₂, x₁², x₂², x₁x₂]:")
for i, point in enumerate(transformed_points):
    og = original_points[i]
    print(f"[{og[0]}, {og[1]}, {og[0]**2}, {og[1]**2}, {og[0]*og[1]}] = {point}")

print(f"\nQuadratic transformation accuracy on circles dataset: {circles_quad_accuracy:.2f}")
print(f"Quadratic transformation accuracy on moons dataset: {moons_quad_accuracy:.2f}")
print("By transforming the features to a higher-dimensional space, we've made the data linearly separable.")
print("The quadratic transformation maps: φ(x₁, x₂) = [x₁, x₂, x₁², x₂², x₁x₂]")

# Step 4: Kernel methods
print("\nStep 4: Demonstrate the kernel trick")
print("--------------------------------")

# Create SVM models with different kernels
kernels = ['linear', 'poly', 'rbf']
circle_accuracies = {}
moon_accuracies = {}

for kernel in kernels:
    # SVM for circles dataset
    svm_circles = SVC(kernel=kernel, degree=3 if kernel == 'poly' else 2, gamma='scale')
    svm_circles.fit(X_circles, y_circles)
    circle_accuracies[kernel] = svm_circles.score(X_circles, y_circles)
    
    # SVM for moons dataset
    svm_moons = SVC(kernel=kernel, degree=3 if kernel == 'poly' else 2, gamma='scale')
    svm_moons.fit(X_moons, y_moons)
    moon_accuracies[kernel] = svm_moons.score(X_moons, y_moons)
    
    # Plot the decision boundaries
    plot_decision_boundary(X_circles, y_circles, svm_circles, 
                          f"SVM with {kernel.upper()} Kernel on Circles (Accuracy: {circle_accuracies[kernel]:.2f})",
                          f"circles_{kernel}_kernel.png")
    
    plot_decision_boundary(X_moons, y_moons, svm_moons, 
                          f"SVM with {kernel.upper()} Kernel on Moons (Accuracy: {moon_accuracies[kernel]:.2f})",
                          f"moons_{kernel}_kernel.png")

print("SVM Kernel accuracy comparison:")
print("\nCircles dataset:")
for kernel, acc in circle_accuracies.items():
    print(f"  {kernel.upper()} kernel: {acc:.2f}")

print("\nMoons dataset:")
for kernel, acc in moon_accuracies.items():
    print(f"  {kernel.upper()} kernel: {acc:.2f}")

print("\nThe kernel trick allows SVMs to implicitly operate in a higher-dimensional space")
print("without explicitly computing the feature transformation, which is more computationally efficient.")

# Step 5: Advantages and disadvantages of feature transformations
print("\nStep 5: Advantages and disadvantages of feature transformations")
print("---------------------------------------------------------")

# Create a sample with dimensionality issues
n_features = 2
X_original = np.random.randn(10, n_features)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_transformed = poly.fit_transform(X_original)

print(f"Original data dimensions: {X_original.shape}")
print(f"Transformed data dimensions with quadratic features: {X_transformed.shape}")
print(f"Feature names after transformation: {poly.get_feature_names_out(['x1', 'x2'])}")

# Feature explosion calculation
feature_counts = []
degrees = range(1, 6)
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_temp = poly.fit_transform(np.zeros((1, n_features)))
    feature_counts.append(X_temp.shape[1])

# Plot feature growth with polynomial degree
plt.figure(figsize=(10, 6))
plt.plot(degrees, feature_counts, marker='o', linestyle='-', linewidth=2)
plt.title('Growth of Feature Count with Polynomial Degree', fontsize=16)
plt.xlabel('Polynomial Degree', fontsize=14)
plt.ylabel('Number of Features', fontsize=14)
plt.xticks(degrees)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "feature_growth.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\nAdvantages of feature transformations:")
print("1. Can make non-linearly separable data linearly separable")
print("2. Can work with standard linear models which are well-understood and easy to interpret")
print("3. Can capture complex relationships in the data that linear models would miss")
print("4. No need to modify existing linear algorithms, just transform the input")

print("\nDisadvantages of feature transformations:")
print("1. Can lead to feature explosion: # of features grows polynomial or exponentially with degree")
print("2. Higher computational cost due to increased dimensionality")
print("3. Risk of overfitting, especially with higher-degree transformations")
print("4. May require feature selection or regularization to manage complexity")
print("5. Can be difficult to interpret the importance of transformed features")

# Example of feature explosion
print(f"\nFeature explosion example: 2 original features → {X_transformed.shape[1]} features with degree 2")
print(f"With degree 5, this would grow to {feature_counts[4]} features")

print("\nConclusion:")
print("-----------")
print("1. Non-linearly separable data cannot be separated by a linear boundary in the original feature space")
print("2. Two ways to make such data linearly separable:")
print("   a. Feature transformation (e.g., polynomial features, radial basis functions)")
print("   b. Kernel methods (kernel trick)")
print("3. Quadratic feature transform φ(x₁, x₂) = [x₁, x₂, x₁², x₂², x₁x₂] helps by mapping data to a higher-dimensional space where it becomes linearly separable")
print("4. The kernel trick allows efficient computation in the transformed space without explicitly computing the transformation")
print("5. Feature transformations offer improved separability but may lead to computational challenges and overfitting") 