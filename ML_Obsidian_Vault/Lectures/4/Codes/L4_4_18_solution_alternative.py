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

# Helper function to calculate scatter matrices step by step
def calculate_scatter_matrix(X, mean, name="Scatter"):
    """Calculate and print step-by-step scatter matrix for a class."""
    X_centered = X - mean
    print(f"\nCalculating {name} matrix step by step:")
    scatter_matrix = np.zeros((2, 2))
    
    for i, x_i in enumerate(X_centered):
        outer_product = np.outer(x_i, x_i)
        print(f"  Data point {i+1}: [{X[i][0]}, {X[i][1]}] - Mean: [{mean[0]:.2f}, {mean[1]:.2f}] = [{x_i[0]:.2f}, {x_i[1]:.2f}]")
        print(f"  Outer product: [")
        print(f"    [{x_i[0]:.2f} * {x_i[0]:.2f}, {x_i[0]:.2f} * {x_i[1]:.2f}]")
        print(f"    [{x_i[1]:.2f} * {x_i[0]:.2f}, {x_i[1]:.2f} * {x_i[1]:.2f}]")
        print(f"  ] = [")
        print(f"    [{outer_product[0, 0]:.2f}, {outer_product[0, 1]:.2f}]")
        print(f"    [{outer_product[1, 0]:.2f}, {outer_product[1, 1]:.2f}]")
        print(f"  ]")
        
        scatter_matrix += outer_product
        print(f"  Running sum of scatter matrix after point {i+1}:")
        print(f"  [")
        print(f"    [{scatter_matrix[0, 0]:.2f}, {scatter_matrix[0, 1]:.2f}]")
        print(f"    [{scatter_matrix[1, 0]:.2f}, {scatter_matrix[1, 1]:.2f}]")
        print(f"  ]\n")
    
    return scatter_matrix

# Helper function to print matrices clearly
def print_matrix(name, matrix):
    """Print a named matrix in a nicely formatted way."""
    print(f"{name}:")
    print("[")
    for row in matrix:
        print("  [", end=" ")
        for val in row:
            print(f"{val:8.2f}", end=" ")
        print("]")
    print("]")

# Helper function to invert a 2x2 matrix step by step
def invert_2x2_matrix(A, name="A"):
    """Calculate and print step-by-step inversion of a 2x2 matrix."""
    print(f"\nCalculating inverse of 2x2 matrix {name}:")
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    det = a*d - b*c
    print(f"  Determinant = {a:.2f}*{d:.2f} - {b:.2f}*{c:.2f} = {det:.2f}")
    
    if abs(det) < 1e-10:
        print("  Error: Matrix is singular, cannot compute inverse")
        return None
    
    inv = np.zeros((2, 2))
    inv[0, 0] = d/det
    inv[0, 1] = -b/det
    inv[1, 0] = -c/det
    inv[1, 1] = a/det
    
    print(f"  {name}⁻¹[0,0] = {d:.2f}/{det:.2f} = {inv[0,0]:.6f}")
    print(f"  {name}⁻¹[0,1] = -{b:.2f}/{det:.2f} = {inv[0,1]:.6f}")
    print(f"  {name}⁻¹[1,0] = -{c:.2f}/{det:.2f} = {inv[1,0]:.6f}")
    print(f"  {name}⁻¹[1,1] = {a:.2f}/{det:.2f} = {inv[1,1]:.6f}")
    
    print_matrix(f"{name}⁻¹", inv)
    return inv

# Helper function to calculate dot products step by step
def dot_product(a, b, name_a="a", name_b="b"):
    """Calculate and print step-by-step dot product of two vectors."""
    print(f"\nCalculating dot product {name_a} · {name_b}:")
    result = 0
    for i in range(len(a)):
        term = a[i] * b[i]
        result += term
        print(f"  {name_a}[{i}] * {name_b}[{i}] = {a[i]:.2f} * {b[i]:.2f} = {term:.2f}")
    print(f"  {name_a} · {name_b} = {result:.2f}")
    return result

# Helper function for matrix-vector multiplication
def matrix_vector_multiply(A, b, name_A="A", name_b="b", name_result="result"):
    """Calculate and print step-by-step matrix-vector multiplication."""
    print(f"\nCalculating {name_result} = {name_A} * {name_b}:")
    result = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        row_sum = 0
        for j in range(A.shape[1]):
            product = A[i, j] * b[j]
            row_sum += product
            print(f"  {name_A}[{i},{j}] * {name_b}[{j}] = {A[i,j]:.6f} * {b[j]:.2f} = {product:.6f}")
        result[i] = row_sum
        print(f"  {name_result}[{i}] = {row_sum:.6f}")
    return result

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
print("Malignant class data points (y=1):")
for i, point in enumerate(X_malignant):
    print(f"  Data point {i+1}: [{point[0]}, {point[1]}]")

print("\nBenign class data points (y=0):")
for i, point in enumerate(X_benign):
    print(f"  Data point {i+1}: [{point[0]}, {point[1]}]")

# Calculate mean for malignant class (class 1)
print("\nCalculating mean vector for malignant class (y=1):")
n_malignant = X_malignant.shape[0]
sum_malignant = np.zeros(2)
for i, point in enumerate(X_malignant):
    sum_malignant += point
    print(f"  Adding point {i+1}: {point} → Running sum: {sum_malignant}")

m1 = sum_malignant / n_malignant
print(f"  m1 = Sum / {n_malignant} = {sum_malignant} / {n_malignant} = [{m1[0]:.2f}, {m1[1]:.2f}]")

# Calculate mean for benign class (class 0)
print("\nCalculating mean vector for benign class (y=0):")
n_benign = X_benign.shape[0]
sum_benign = np.zeros(2)
for i, point in enumerate(X_benign):
    sum_benign += point
    print(f"  Adding point {i+1}: {point} → Running sum: {sum_benign}")

m2 = sum_benign / n_benign
print(f"  m2 = Sum / {n_benign} = {sum_benign} / {n_benign} = [{m2[0]:.2f}, {m2[1]:.2f}]")

print("\nStep 2: Calculate the within-class scatter matrix Sw")
print("------------------------------------------------")

# Calculate scatter matrix for malignant class
S1 = calculate_scatter_matrix(X_malignant, m1, "Malignant scatter")
print_matrix("Scatter matrix for malignant class (S1)", S1)

# Calculate scatter matrix for benign class
S2 = calculate_scatter_matrix(X_benign, m2, "Benign scatter")
print_matrix("Scatter matrix for benign class (S2)", S2)

# Calculate within-class scatter matrix Sw
print("\nCalculating within-class scatter matrix Sw = S1 + S2:")
Sw = S1 + S2
print_matrix("Within-class scatter matrix (Sw)", Sw)

print("\nStep 3: Determine the LDA projection direction")
print("-------------------------------------------")

# Calculate the mean difference vector (m2 - m1)
print("\nCalculating difference between means (m2 - m1):")
mean_diff = m2 - m1
print(f"  m2 - m1 = [{m2[0]:.2f}, {m2[1]:.2f}] - [{m1[0]:.2f}, {m1[1]:.2f}] = [{mean_diff[0]:.2f}, {mean_diff[1]:.2f}]")

# Calculate the inverse of Sw
Sw_inv = invert_2x2_matrix(Sw, "Sw")

# Calculate the LDA projection direction theta ∝ Sw^(-1)(m2 - m1)
theta_unnormalized = matrix_vector_multiply(Sw_inv, mean_diff, "Sw⁻¹", "m2-m1", "θ_unnormalized")

# Since we're projecting in the opposite direction, we need to flip the sign
print("\nSince we got negative values and we're only interested in the direction, we'll flip the signs:")
theta_unnormalized = -theta_unnormalized
print(f"  θ_unnormalized = [{theta_unnormalized[0]:.6f}, {theta_unnormalized[1]:.6f}]")

# Normalize theta to have unit length
print("\nNormalizing θ to have unit length:")
theta_norm = np.linalg.norm(theta_unnormalized)
print(f"  ||θ|| = sqrt({theta_unnormalized[0]:.6f}² + {theta_unnormalized[1]:.6f}²) = sqrt({theta_unnormalized[0]**2:.6f} + {theta_unnormalized[1]**2:.6f}) = {theta_norm:.6f}")

theta = theta_unnormalized / theta_norm
print(f"  θ = θ_unnormalized / ||θ|| = [{theta_unnormalized[0]:.6f}, {theta_unnormalized[1]:.6f}] / {theta_norm:.6f} = [{theta[0]:.2f}, {theta[1]:.2f}]")

print("\nStep 4: Calculate the threshold value for classification")
print("---------------------------------------------------")

# Calculate projections of the class means onto theta
print("Calculating projections of class means onto θ:")
proj_m1 = dot_product(m1, theta, "m1", "θ")
proj_m2 = dot_product(m2, theta, "m2", "θ")

# Calculate the threshold (assuming equal prior probabilities)
print("\nCalculating the threshold (assuming equal prior probabilities):")
mean_sum = m1 + m2
print(f"  m1 + m2 = [{m1[0]:.2f}, {m1[1]:.2f}] + [{m2[0]:.2f}, {m2[1]:.2f}] = [{mean_sum[0]:.2f}, {mean_sum[1]:.2f}]")

threshold = dot_product(theta, mean_sum/2, "θ", "(m1+m2)/2")
print(f"  Threshold c = θᵀ * (m1 + m2)/2 = {threshold:.2f}")

print("\nStep 5: Classify a new patient")
print("---------------------------")

# New patient data: tumor size = 30mm, age = 50 years
new_patient = np.array([30, 50])
print(f"New patient data - Tumor size: {new_patient[0]}mm, Age: {new_patient[1]} years")

# Project the new patient's data onto theta
print("\nCalculating projection of new patient data onto θ:")
new_patient_proj = dot_product(new_patient, theta, "x_new", "θ")

# Classify the new patient
print("\nClassifying the new patient:")
print(f"  Comparing projection {new_patient_proj:.2f} with threshold {threshold:.2f}")
if new_patient_proj > threshold:
    prediction = "Malignant (y=1)"
    print(f"  {new_patient_proj:.2f} > {threshold:.2f}, so prediction is Malignant (y=1)")
else:
    prediction = "Benign (y=0)"
    print(f"  {new_patient_proj:.2f} < {threshold:.2f}, so prediction is Benign (y=0)")

# Distance from the threshold
distance_from_threshold = abs(new_patient_proj - threshold)
print(f"  Distance from the threshold: |{new_patient_proj:.2f} - {threshold:.2f}| = {distance_from_threshold:.2f}")

# Visualizations
print("\nCreating visualizations:")

# 1. Scatter plot of data with class means
plt.figure(figsize=(10, 8))
plt.scatter(X_benign[:, 0], X_benign[:, 1], color='blue', marker='o', s=100, label='Benign (y=0)')
plt.scatter(X_malignant[:, 0], X_malignant[:, 1], color='red', marker='x', s=100, label='Malignant (y=1)')
plt.scatter(m2[0], m2[1], color='blue', marker='*', s=300, edgecolor='k', label='Mean Benign (m2)')
plt.scatter(m1[0], m1[1], color='red', marker='*', s=300, edgecolor='k', label='Mean Malignant (m1)')
plt.scatter(new_patient[0], new_patient[1], color='green', marker='D', s=200, edgecolor='k', label='New Patient')

# Add labels for each data point
for i in range(len(X)):
    label = 'M' if y[i] == 1 else 'B'
    plt.annotate(f"{label}({X[i][0]}, {X[i][1]})", (X[i][0], X[i][1]), 
                 xytext=(7, 0), textcoords='offset points', fontsize=10)

plt.annotate(f"New({new_patient[0]}, {new_patient[1]})", (new_patient[0], new_patient[1]),
             xytext=(7, 0), textcoords='offset points', fontsize=10)

plt.xlabel('Tumor Size (mm)', fontsize=14)
plt.ylabel('Age (years)', fontsize=14)
plt.title('Tumor Data with Class Means', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "tumor_data_plot.png"), dpi=300, bbox_inches='tight')

# 2. Decision boundary visualization
plt.figure(figsize=(12, 10))

# Create a meshgrid to visualize the decision boundary
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Calculate decision boundary based on theta and threshold
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], theta) > threshold
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(12, 10), facecolor='white')
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu_r)

# Plot the original data points
plt.scatter(X_benign[:, 0], X_benign[:, 1], color='blue', marker='o', s=120, label='Benign (y=0)', edgecolor='black', zorder=5)
plt.scatter(X_malignant[:, 0], X_malignant[:, 1], color='red', marker='x', s=120, linewidth=2, label='Malignant (y=1)', zorder=5)
plt.scatter(m2[0], m2[1], color='blue', marker='*', s=350, edgecolor='black', linewidth=1.5, label='Mean Benign (m2)', zorder=6)
plt.scatter(m1[0], m1[1], color='red', marker='*', s=350, edgecolor='black', linewidth=1.5, label='Mean Malignant (m1)', zorder=6)
plt.scatter(new_patient[0], new_patient[1], color='green', marker='D', s=200, edgecolor='black', linewidth=1.5, label='New Patient', zorder=7)

# Calculate and plot the LDA direction as a vector from the centroid
centroid = (m1 + m2) / 2
plt.arrow(centroid[0], centroid[1], theta[0]*15, theta[1]*15, 
          head_width=3, head_length=5, fc='black', ec='black', linewidth=2, zorder=8, label='LDA Direction')

# Draw the decision boundary explicitly as a line
t = np.array([-40, 40])  # Parameter for line equation
boundary_points = np.vstack([centroid[0] - t * theta[1], centroid[1] + t * theta[0]]).T
plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k--', lw=3, label='Decision Boundary', zorder=8)

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

# 3. Projection visualization 
plt.figure(figsize=(16, 9), facecolor='white')

# Calculate projections of all data points onto theta
projections_benign = np.dot(X_benign, theta)
projections_malignant = np.dot(X_malignant, theta)

# Create a number line for the projections
plt.axhline(y=0, color='k', linestyle='-', alpha=0.7, linewidth=2)
plt.scatter(projections_benign, np.zeros_like(projections_benign), color='blue', s=120, marker='o', label='Benign Projections', edgecolor='black')
plt.scatter(projections_malignant, np.zeros_like(projections_malignant), color='red', s=120, marker='x', linewidth=2, label='Malignant Projections')
plt.scatter(new_patient_proj, 0, color='green', s=180, marker='D', label='New Patient Projection', edgecolor='black')
plt.axvline(x=threshold, color='purple', linestyle='--', linewidth=2.5, label='Decision Threshold')

# Add a title showing the LDA projection formula
plt.text(0.5, 0.95, 
         f"LDA Projection: {theta[0]:.2f} × Tumor Size + {theta[1]:.2f} × Age", 
         transform=plt.gca().transAxes, 
         ha='center', fontsize=18, weight='bold',
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.9))

# Formatting
plt.xlabel('Projection Value', fontsize=16, fontweight='bold')
plt.title('Projections of Data Points onto LDA Direction', fontsize=18, fontweight='bold', pad=20)
plt.legend(fontsize=14, loc='upper left')
plt.grid(True, alpha=0.3, axis='x')

plt.savefig(os.path.join(save_dir, "lda_projections.png"), dpi=300, bbox_inches='tight', facecolor='white')

print("\nSummary:")
print("---------")
print(f"1. Mean vector for malignant class (m1): [{m1[0]:.2f}, {m1[1]:.2f}]")
print(f"2. Mean vector for benign class (m2): [{m2[0]:.2f}, {m2[1]:.2f}]")
print(f"3. Within-class scatter matrix Sw:")
print(f"   [{Sw[0,0]:.2f}, {Sw[0,1]:.2f}]")
print(f"   [{Sw[1,0]:.2f}, {Sw[1,1]:.2f}]")
print(f"4. LDA projection direction θ: [{theta[0]:.2f}, {theta[1]:.2f}]")
print(f"5. Classification threshold (equal priors): {threshold:.2f}")
print(f"6. Prediction for new patient (tumor size = 30mm, age = 50 years): {prediction}")