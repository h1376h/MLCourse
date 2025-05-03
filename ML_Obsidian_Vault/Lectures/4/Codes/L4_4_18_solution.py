import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Helper function to show step-by-step matrix calculations
def print_matrix(name, matrix):
    """Print a named matrix in a nicely formatted way."""
    print(f"{name}:")
    for row in matrix:
        print("[", end=" ")
        for val in row:
            print(f"{val:8.4f}", end=" ")
        print("]")
    print()

# Helper function to calculate matrix multiplication step by step
def matrix_multiply(A, B, name_A="A", name_B="B", name_result="A*B"):
    """Calculate and print step-by-step matrix multiplication."""
    print(f"Calculating {name_result} = {name_A} * {name_B}:")
    
    # Check if B is a vector (1D array)
    if B.ndim == 1:
        result = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            row_sum = 0
            for j in range(A.shape[1]):
                product = A[i, j] * B[j]
                row_sum += product
                print(f"  {name_A}[{i},{j}] * {name_B}[{j}] = {A[i,j]:.4f} * {B[j]:.4f} = {product:.4f}")
            result[i] = row_sum
            print(f"  {name_result}[{i}] = {row_sum:.4f}")
    else:
        # Matrix-matrix multiplication
        result = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                cell_sum = 0
                for k in range(A.shape[1]):
                    product = A[i, k] * B[k, j]
                    cell_sum += product
                    print(f"  {name_A}[{i},{k}] * {name_B}[{k},{j}] = {A[i,k]:.4f} * {B[k,j]:.4f} = {product:.4f}")
                result[i, j] = cell_sum
                print(f"  {name_result}[{i},{j}] = {cell_sum:.4f}")
                
    return result

# Helper function to calculate dot product step by step
def dot_product(a, b, name_a="a", name_b="b"):
    """Calculate and print step-by-step dot product of two vectors."""
    print(f"Calculating dot product {name_a} · {name_b}:")
    result = 0
    for i in range(len(a)):
        term = a[i] * b[i]
        result += term
        print(f"  {name_a}[{i}] * {name_b}[{i}] = {a[i]:.4f} * {b[i]:.4f} = {term:.4f}")
    print(f"  {name_a} · {name_b} = {result:.4f}")
    return result

# Helper function to calculate covariance matrix step by step
def calculate_covariance(X, mean, name="Covariance"):
    """Calculate and print step-by-step covariance matrix."""
    n = X.shape[0]
    print(f"Calculating {name} matrix with {n} data points and {X.shape[1]} features:")
    
    # Center the data by subtracting the mean
    X_centered = X - mean
    print("Centered data (X - μ):")
    for i, row in enumerate(X_centered):
        print(f"  Data point {i+1}: [{row[0]:.4f}, {row[1]:.4f}]")
    
    # Calculate (X - μ)ᵀ(X - μ)
    cov = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            sum_term = 0
            for k in range(n):
                term = X_centered[k, i] * X_centered[k, j]
                sum_term += term
            cov[i, j] = sum_term / (n - 1)
            print(f"  {name}[{i},{j}] = (1/{n-1}) * {sum_term:.4f} = {cov[i,j]:.4f}")
    
    return cov

# Helper function for matrix inversion step by step (2x2 case)
def invert_2x2_matrix(A, name="A"):
    """Calculate and print step-by-step inversion of a 2x2 matrix."""
    print(f"Calculating inverse of 2x2 matrix {name}:")
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    det = a*d - b*c
    print(f"  Determinant = {a:.4f}*{d:.4f} - {b:.4f}*{c:.4f} = {det:.4f}")
    
    if abs(det) < 1e-10:
        print("  Error: Matrix is singular, cannot compute inverse")
        return None
    
    inv = np.zeros((2, 2))
    inv[0, 0] = d/det
    inv[0, 1] = -b/det
    inv[1, 0] = -c/det
    inv[1, 1] = a/det
    
    print(f"  {name}⁻¹[0,0] = {d:.4f}/{det:.4f} = {inv[0,0]:.6f}")
    print(f"  {name}⁻¹[0,1] = -{b:.4f}/{det:.4f} = {inv[0,1]:.6f}")
    print(f"  {name}⁻¹[1,0] = -{c:.4f}/{det:.4f} = {inv[1,0]:.6f}")
    print(f"  {name}⁻¹[1,1] = {a:.4f}/{det:.4f} = {inv[1,1]:.6f}")
    
    return inv

# Helper function to calculate confidence ellipses for visualization
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

print("Question 18: LDA for Medical Diagnosis")
print("======================================")

# Given data
# Tumor Size (mm), Age (years), y (Malignant: 1 = malignant, 0 = benign)
data = np.array([
    [15, 20, 0],
    [65, 30, 0],
    [30, 50, 1],
    [90, 20, 1],
    [44, 35, 0],
    [20, 70, 1],
    [50, 40, 1],
    [36, 25, 0]
])

# Extract features and labels
X = data[:, :2]
y = data[:, 2]

# Separate the data by class
X_malignant = X[y == 1]
X_benign = X[y == 0]

print("\nStep 1: Calculate the mean vectors for each class")
print("------------------------------------------------")

# Show the raw data for each class
print("Malignant class data points:")
for i, point in enumerate(X_malignant):
    print(f"  Data point {i+1}: [{point[0]}, {point[1]}]")

print("\nBenign class data points:")
for i, point in enumerate(X_benign):
    print(f"  Data point {i+1}: [{point[0]}, {point[1]}]")

# Calculate mean for malignant class (class 1) step by step
print("\nCalculating mean vector for malignant class (y=1):")
n_malignant = X_malignant.shape[0]
sum_malignant = np.zeros(2)
for i, point in enumerate(X_malignant):
    sum_malignant += point
    print(f"  Adding point {i+1}: {point} → Running sum: {sum_malignant}")

mean_malignant = sum_malignant / n_malignant
print(f"  Mean = Sum / {n_malignant} = {sum_malignant} / {n_malignant} = [{mean_malignant[0]:.2f}, {mean_malignant[1]:.2f}]")

# Calculate mean for benign class (class 0) step by step
print("\nCalculating mean vector for benign class (y=0):")
n_benign = X_benign.shape[0]
sum_benign = np.zeros(2)
for i, point in enumerate(X_benign):
    sum_benign += point
    print(f"  Adding point {i+1}: {point} → Running sum: {sum_benign}")

mean_benign = sum_benign / n_benign
print(f"  Mean = Sum / {n_benign} = {sum_benign} / {n_benign} = [{mean_benign[0]:.2f}, {mean_benign[1]:.2f}]")

print("\nStep 2: Calculate the shared covariance matrix")
print("--------------------------------------------")

# Calculate covariance matrix for malignant class step by step
cov_malignant = calculate_covariance(X_malignant, mean_malignant, "Malignant covariance")
print("\nMalignant class covariance matrix:")
print_matrix("Σ₁", cov_malignant)

# Calculate covariance matrix for benign class step by step
cov_benign = calculate_covariance(X_benign, mean_benign, "Benign covariance")
print("\nBenign class covariance matrix:")
print_matrix("Σ₀", cov_benign)

# Calculate shared (pooled) covariance matrix step by step
print("\nCalculating shared (pooled) covariance matrix:")
n_total = n_malignant + n_benign

# Show the weighted calculation
print(f"  Σ = ((n₁-1)Σ₁ + (n₀-1)Σ₀) / (n₁ + n₀ - 2)")
print(f"  Σ = (({n_malignant}-1)Σ₁ + ({n_benign}-1)Σ₀) / ({n_malignant} + {n_benign} - 2)")
print(f"  Σ = ({n_malignant-1}Σ₁ + {n_benign-1}Σ₀) / {n_total-2}")

shared_cov = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        term1 = (n_malignant - 1) * cov_malignant[i, j]
        term2 = (n_benign - 1) * cov_benign[i, j]
        sum_term = term1 + term2
        shared_cov[i, j] = sum_term / (n_total - 2)
        print(f"  Σ[{i},{j}] = ({n_malignant-1} * {cov_malignant[i,j]:.4f} + {n_benign-1} * {cov_benign[i,j]:.4f}) / {n_total-2} = {shared_cov[i,j]:.4f}")

print("\nShared covariance matrix:")
print_matrix("Σ", shared_cov)

print("\nStep 3: Determine the LDA projection direction")
print("-------------------------------------------")

# Calculate the difference between class means step by step
print("Calculating difference between class means:")
mean_diff = np.zeros(2)
for i in range(2):
    mean_diff[i] = mean_malignant[i] - mean_benign[i]
    print(f"  μ₁[{i}] - μ₀[{i}] = {mean_malignant[i]:.4f} - {mean_benign[i]:.4f} = {mean_diff[i]:.4f}")

print(f"  Mean difference vector (μ₁ - μ₀): [{mean_diff[0]:.2f}, {mean_diff[1]:.2f}]")

# Calculate the inverse of the shared covariance matrix step by step
shared_cov_inv = invert_2x2_matrix(shared_cov, "Σ")
print("\nInverse of shared covariance matrix:")
print_matrix("Σ⁻¹", shared_cov_inv)

# Calculate the LDA projection direction step by step
print("\nCalculating LDA projection direction w = Σ⁻¹(μ₁ - μ₀):")
w = matrix_multiply(shared_cov_inv, mean_diff, "Σ⁻¹", "μ₁ - μ₀", "w")
print(f"LDA projection direction w = Σ⁻¹(μ₁ - μ₀): [{w[0]:.6f}, {w[1]:.6f}]")

# Normalize the projection direction to unit length for visualization
w_norm_factor = np.linalg.norm(w)
print(f"\nNormalizing w to unit length:")
print(f"  ||w|| = sqrt({w[0]:.6f}² + {w[1]:.6f}²) = sqrt({w[0]**2:.6f} + {w[1]**2:.6f}) = {w_norm_factor:.6f}")
w_norm = w / w_norm_factor
print(f"  w_normalized = w / ||w|| = [{w[0]:.6f}, {w[1]:.6f}] / {w_norm_factor:.6f} = [{w_norm[0]:.6f}, {w_norm[1]:.6f}]")

print("\nStep 4: Calculate the threshold value for classification")
print("---------------------------------------------------")

# Calculate projections of the class means onto w step by step
print("Calculating projections of class means onto w:")
proj_malignant = dot_product(mean_malignant, w, "μ₁", "w")
proj_benign = dot_product(mean_benign, w, "μ₀", "w")

# Calculate the threshold (assuming equal prior probabilities)
print("\nCalculating the threshold (assuming equal prior probabilities):")
threshold = (proj_malignant + proj_benign) / 2
print(f"  threshold = (μ₁·w + μ₀·w) / 2 = ({proj_malignant:.6f} + {proj_benign:.6f}) / 2 = {threshold:.6f}")

print("\nStep 5: Classify a new patient")
print("---------------------------")

# New patient data: tumor size = 40mm, age = 45 years
new_patient = np.array([40, 45])
print(f"New patient data - Tumor size: {new_patient[0]}mm, Age: {new_patient[1]} years")

# Project the new patient's data onto w step by step
print("\nCalculating projection of new patient data onto w:")
new_patient_proj = dot_product(new_patient, w, "x_new", "w")

# Classify the new patient
print("\nClassifying the new patient:")
print(f"  Comparing projection {new_patient_proj:.6f} with threshold {threshold:.6f}")
if new_patient_proj > threshold:
    prediction = "Malignant (y=1)"
    print(f"  {new_patient_proj:.6f} > {threshold:.6f}, so prediction is Malignant (y=1)")
else:
    prediction = "Benign (y=0)"
    print(f"  {new_patient_proj:.6f} < {threshold:.6f}, so prediction is Benign (y=0)")

# Distance from the threshold
distance_from_threshold = abs(new_patient_proj - threshold)
print(f"  Distance from the threshold: |{new_patient_proj:.6f} - {threshold:.6f}| = {distance_from_threshold:.6f}")

# Verify our results using scikit-learn
print("\nVerification using scikit-learn's LDA implementation:")
lda = LinearDiscriminantAnalysis(store_covariance=True)
lda.fit(X, y)
sklearn_prediction = lda.predict([new_patient])[0]
sklearn_pred_proba = lda.predict_proba([new_patient])[0]

print(f"scikit-learn LDA prediction: {'Malignant (y=1)' if sklearn_prediction == 1 else 'Benign (y=0)'}")
print(f"Prediction probabilities - Benign: {sklearn_pred_proba[0]:.4f}, Malignant: {sklearn_pred_proba[1]:.4f}")

# Visualizations
print("\nCreating visualizations:")

# 1. Scatter plot of data with class means and covariance ellipses
plt.figure(figsize=(10, 8))
plt.scatter(X_benign[:, 0], X_benign[:, 1], color='blue', marker='o', s=100, label='Benign (y=0)')
plt.scatter(X_malignant[:, 0], X_malignant[:, 1], color='red', marker='x', s=100, label='Malignant (y=1)')
plt.scatter(mean_benign[0], mean_benign[1], color='blue', marker='*', s=300, edgecolor='k', label='Mean Benign')
plt.scatter(mean_malignant[0], mean_malignant[1], color='red', marker='*', s=300, edgecolor='k', label='Mean Malignant')
plt.scatter(new_patient[0], new_patient[1], color='green', marker='D', s=200, edgecolor='k', label='New Patient')

# Add confidence ellipses for each class
confidence_ellipse(X_benign[:, 0], X_benign[:, 1], plt.gca(), n_std=2.0, 
                   edgecolor='blue', linestyle='--', label='Benign 95% CI')
confidence_ellipse(X_malignant[:, 0], X_malignant[:, 1], plt.gca(), n_std=2.0, 
                   edgecolor='red', linestyle='--', label='Malignant 95% CI')

# Add labels for each data point
for i in range(len(X)):
    label = 'M' if y[i] == 1 else 'B'
    plt.annotate(f"{label}({X[i][0]}, {X[i][1]})", (X[i][0], X[i][1]), 
                 xytext=(7, 0), textcoords='offset points', fontsize=10)

plt.annotate(f"New({new_patient[0]}, {new_patient[1]})", (new_patient[0], new_patient[1]),
             xytext=(7, 0), textcoords='offset points', fontsize=10)

plt.xlabel('Tumor Size (mm)', fontsize=14)
plt.ylabel('Age (years)', fontsize=14)
plt.title('Tumor Data with Class Means and 95% Confidence Ellipses', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "tumor_data_plot.png"), dpi=300, bbox_inches='tight')

# 2. Decision boundary visualization - Improved
plt.figure(figsize=(12, 10))

# Create a meshgrid to visualize the decision boundary
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Get predictions for all points in the meshgrid
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary with improved colors
plt.figure(figsize=(12, 10), facecolor='white')
contour = plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu_r)

# Add a white background to make the boundary more visible
plt.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max], color='white', zorder=-1)

# Plot the original data points
plt.scatter(X_benign[:, 0], X_benign[:, 1], color='blue', marker='o', s=120, label='Benign (y=0)', edgecolor='black', zorder=5)
plt.scatter(X_malignant[:, 0], X_malignant[:, 1], color='red', marker='x', s=120, linewidth=2, label='Malignant (y=1)', zorder=5)
plt.scatter(mean_benign[0], mean_benign[1], color='blue', marker='*', s=350, edgecolor='black', linewidth=1.5, label='Mean Benign', zorder=6)
plt.scatter(mean_malignant[0], mean_malignant[1], color='red', marker='*', s=350, edgecolor='black', linewidth=1.5, label='Mean Malignant', zorder=6)
plt.scatter(new_patient[0], new_patient[1], color='green', marker='D', s=200, edgecolor='black', linewidth=1.5, label='New Patient', zorder=7)

# Add confidence ellipses for each class
confidence_ellipse(X_benign[:, 0], X_benign[:, 1], plt.gca(), n_std=2.0, 
                   edgecolor='blue', linestyle='--', linewidth=2)
confidence_ellipse(X_malignant[:, 0], X_malignant[:, 1], plt.gca(), n_std=2.0, 
                   edgecolor='red', linestyle='--', linewidth=2)

# Calculate and plot the LDA direction as a vector from the centroid
centroid = (mean_malignant + mean_benign) / 2
plt.arrow(centroid[0], centroid[1], w_norm[0]*30, w_norm[1]*30, 
          head_width=3, head_length=5, fc='black', ec='black', linewidth=2, zorder=8, label='LDA Direction')

# Draw the decision boundary explicitly as a line
t = np.array([-40, 40])  # Parameter for line equation
boundary_points = np.vstack([centroid[0] - t * w_norm[1], centroid[1] + t * w_norm[0]]).T
plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', lw=3, label='Decision Boundary', zorder=8)

# Add the equation of the decision boundary in a better position with clearer text
boundary_eq = f"LDA Equation: {w[0]:.4f}×Tumor Size + {w[1]:.4f}×Age"
plt.annotate(boundary_eq, xy=(0.02, 0.95), xycoords='axes fraction', 
             fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

# Add labels for each data point
font_props = {'fontsize': 11, 'weight': 'bold', 'backgroundcolor': 'white', 'zorder': 10}
for i in range(len(X)):
    label = f"M{i+1}" if y[i] == 1 else f"B{i+1}"
    plt.annotate(label, (X[i][0], X[i][1]), 
                 xytext=(8, 0), textcoords='offset points', 
                 **font_props)

plt.annotate("New", (new_patient[0], new_patient[1]),
             xytext=(8, 0), textcoords='offset points', 
             color='green', **font_props)

# Improved formatting
plt.grid(True, alpha=0.3, linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=12)

# Improved labels and title
plt.xlabel('Tumor Size (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Age (years)', fontsize=16, fontweight='bold')
plt.title('LDA Decision Boundary for Tumor Classification', fontsize=18, fontweight='bold', pad=15)

# Create a more readable and organized legend
leg = plt.legend(fontsize=14, loc='upper left', framealpha=0.9, edgecolor='black')
leg.get_frame().set_boxstyle('round,pad=0.5')

# Set plot limits with some padding
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.savefig(os.path.join(save_dir, "lda_decision_boundary.png"), dpi=300, bbox_inches='tight', facecolor='white')

# 3. Improved projection visualization with better readability
plt.figure(figsize=(16, 9), facecolor='white')

# Calculate projections of all data points onto w
projections = np.dot(X, w)
projections_benign = np.dot(X_benign, w)
projections_malignant = np.dot(X_malignant, w)
new_projection = np.dot(new_patient, w)

# Create a cleaner background for the number line
min_proj, max_proj = min(projections.min(), new_projection) - 5, max(projections.max(), new_projection) + 5
plt.fill_between([min_proj, threshold], [-1.5, -1.5], [1.5, 1.5], color='#E6F0FF', alpha=0.7, zorder=1)  # Light blue for benign
plt.fill_between([threshold, max_proj], [-1.5, -1.5], [1.5, 1.5], color='#FFE6E6', alpha=0.7, zorder=1)  # Light red for malignant

# Create a number line for the projections
plt.axhline(y=0, color='k', linestyle='-', alpha=0.7, linewidth=2, zorder=2)
plt.scatter(projections_benign, np.zeros_like(projections_benign), color='blue', s=120, marker='o', label='Benign Projections', edgecolor='black', zorder=3)
plt.scatter(projections_malignant, np.zeros_like(projections_malignant), color='red', s=120, marker='x', linewidth=2, label='Malignant Projections', zorder=3)
plt.scatter(new_projection, 0, color='green', s=180, marker='D', label='New Patient Projection', edgecolor='black', zorder=4)
plt.axvline(x=threshold, color='purple', linestyle='--', linewidth=2.5, label='Decision Threshold', zorder=3)

# Add clear labels for benign projections with improved spacing
benign_positions = [-30, -60, -90, -120]  # More space between labels
for i, proj in enumerate(projections_benign):
    plt.annotate(f"B{i+1}: {proj:.2f}", (proj, 0), 
                 xytext=(0, benign_positions[i % len(benign_positions)]), 
                 textcoords='offset points', 
                 ha='center', fontsize=12, color='blue', weight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="blue", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5),
                 zorder=5)

# Add clear labels for malignant projections with improved spacing
malignant_positions = [30, 60, 90, 120]  # More space between labels
for i, proj in enumerate(projections_malignant):
    plt.annotate(f"M{i+1}: {proj:.2f}", (proj, 0), 
                 xytext=(0, malignant_positions[i % len(malignant_positions)]), 
                 textcoords='offset points', 
                 ha='center', fontsize=12, color='red', weight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5),
                 zorder=5)

# Add label for the new patient with improved visibility
plt.annotate(f"New Patient: {new_projection:.2f}", (new_projection, 0),
             xytext=(0, -150), textcoords='offset points',
             ha='center', fontsize=14, weight='bold', color='green',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green", alpha=0.9),
             arrowprops=dict(arrowstyle="->", color="green", linewidth=2),
             zorder=5)

# Add label for the threshold with improved visibility
plt.annotate(f"Threshold: {threshold:.2f}", (threshold, 0),
             xytext=(0, 150), textcoords='offset points',
             ha='center', fontsize=14, weight='bold', color='purple',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="purple", alpha=0.9),
             arrowprops=dict(arrowstyle="->", color="purple", linewidth=2),
             zorder=5)

# Add clear region labels
plt.text(min_proj + (threshold - min_proj)/2, 0.75, "Benign Region", 
         ha='center', fontsize=16, color='blue', weight='bold', 
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7),
         zorder=4)
plt.text(threshold + (max_proj - threshold)/2, 0.75, "Malignant Region", 
         ha='center', fontsize=16, color='red', weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
         zorder=4)

# Add a title showing the LDA projection formula with improved clarity
plt.text(0.5, 0.95, 
         f"LDA Projection: {w[0]:.4f} × Tumor Size + {w[1]:.4f} × Age", 
         transform=plt.gca().transAxes, 
         ha='center', fontsize=18, weight='bold',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.9),
         zorder=6)

# Add distance markers on the scale
tick_positions = np.arange(np.floor(min_proj), np.ceil(max_proj) + 1, 1.0)
plt.xticks(tick_positions, fontsize=12)
plt.grid(True, alpha=0.3, axis='x', linestyle='--')

# Improved formatting
plt.xlabel('Projection Value', fontsize=16, fontweight='bold')
plt.title('Projections of Data Points onto LDA Direction', fontsize=18, fontweight='bold', pad=20)
plt.xlim(min_proj, max_proj)
plt.ylim(-1.5, 1.5)
plt.yticks([])

# Create a more readable and organized legend
leg = plt.legend(fontsize=14, loc='upper left', framealpha=0.9, edgecolor='black')
leg.get_frame().set_boxstyle('round,pad=0.5')

plt.savefig(os.path.join(save_dir, "lda_projections.png"), dpi=300, bbox_inches='tight', facecolor='white')

# 4. Add a new visualization: Fisher's Criterion maximization
plt.figure(figsize=(10, 8))

# Create a function to calculate Fisher's criterion for different projection directions
def fishers_criterion(direction, mean1, mean2, cov1, cov2):
    # Normalize direction to unit length
    direction = direction / np.linalg.norm(direction)
    # Project means onto direction
    mean_diff = np.abs(np.dot(mean1 - mean2, direction))
    # Calculate projected variances
    var1 = np.dot(np.dot(direction, cov1), direction)
    var2 = np.dot(np.dot(direction, cov2), direction)
    # Fisher's criterion: (between-class variance) / (within-class variance)
    return mean_diff**2 / (var1 + var2)

# Create a grid of possible projection directions (in polar coordinates)
thetas = np.linspace(0, np.pi, 180)
criterion_values = []

for theta in thetas:
    # Convert angle to unit vector
    direction = np.array([np.cos(theta), np.sin(theta)])
    # Calculate Fisher's criterion for this direction
    criterion = fishers_criterion(direction, mean_malignant, mean_benign, cov_malignant, cov_benign)
    criterion_values.append(criterion)

# Find the optimal direction
optimal_idx = np.argmax(criterion_values)
optimal_theta = thetas[optimal_idx]
optimal_direction = np.array([np.cos(optimal_theta), np.sin(optimal_theta)])

# Plot the criterion values as a function of projection angle
plt.plot(thetas * 180 / np.pi, criterion_values, 'b-', lw=2)
plt.axvline(x=optimal_theta * 180 / np.pi, color='r', linestyle='--', 
            label=f'Optimal θ = {optimal_theta * 180 / np.pi:.2f}°')

# Mark the LDA direction on the plot
lda_theta = np.arctan2(w_norm[1], w_norm[0])
if lda_theta < 0:
    lda_theta += np.pi
plt.axvline(x=lda_theta * 180 / np.pi, color='g', linestyle='--', 
            label=f'LDA θ = {lda_theta * 180 / np.pi:.2f}°')

# Add annotations and formatting
plt.grid(True, alpha=0.3)
plt.title("Fisher's Criterion vs. Projection Direction", fontsize=16)
plt.xlabel('Projection Angle (degrees)', fontsize=14)
plt.ylabel("Fisher's Criterion (J)", fontsize=14)
plt.legend(fontsize=12)
plt.savefig(os.path.join(save_dir, "fishers_criterion.png"), dpi=300, bbox_inches='tight')

# 5. Add a 3D visualization of the projected data
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D representation with the original features and the LDA projection
# The z-axis represents the projection value
X_proj = np.dot(X, w)

# Plot benign points
ax.scatter(X_benign[:, 0], X_benign[:, 1], np.dot(X_benign, w), 
           color='blue', marker='o', s=100, label='Benign')

# Plot malignant points
ax.scatter(X_malignant[:, 0], X_malignant[:, 1], np.dot(X_malignant, w), 
           color='red', marker='x', s=100, label='Malignant')

# Plot new patient point
ax.scatter(new_patient[0], new_patient[1], np.dot(new_patient, w), 
           color='green', marker='D', s=200, label='New Patient')

# Add the decision boundary plane
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
zz = np.ones_like(xx) * threshold
ax.plot_surface(xx, yy, zz, alpha=0.3, color='purple', label='Decision Boundary')

# Add labels
ax.set_xlabel('Tumor Size (mm)', fontsize=14)
ax.set_ylabel('Age (years)', fontsize=14)
ax.set_zlabel('LDA Projection', fontsize=14)
ax.set_title('3D Visualization of LDA Projection', fontsize=16)
ax.legend(fontsize=12)

# Adjust viewing angle
ax.view_init(elev=30, azim=45)
plt.savefig(os.path.join(save_dir, "lda_3d_visualization.png"), dpi=300, bbox_inches='tight')

print("\nSummary:")
print("---------")
print(f"1. Mean vector for malignant class (y=1): [{mean_malignant[0]:.2f}, {mean_malignant[1]:.2f}]")
print(f"2. Mean vector for benign class (y=0): [{mean_benign[0]:.2f}, {mean_benign[1]:.2f}]")
print(f"3. Shared covariance matrix:\n{shared_cov}")
print(f"4. LDA projection direction w = Σ⁻¹(μ₁ - μ₀): [{w[0]:.6f}, {w[1]:.6f}]")
print(f"5. Classification threshold (equal priors): {threshold:.6f}")
print(f"6. Prediction for new patient (tumor size = 40mm, age = 45 years): {prediction}") 