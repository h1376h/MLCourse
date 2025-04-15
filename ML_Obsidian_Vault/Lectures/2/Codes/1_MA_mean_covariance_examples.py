import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D

print("\n=== MEAN VECTOR AND COVARIANCE MATRIX EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Analysis")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Car Features Analysis
print("Example 1: Car Features Analysis")

# Dataset with 3 variables: horsepower (X1), weight (X2), and MPG (X3)
data = np.array([
    [130, 1.9, 27],
    [165, 2.2, 24],
    [200, 2.5, 20],
    [110, 1.8, 32],
    [220, 2.8, 18]
])

print("Dataset with 3 variables: horsepower (X₁), weight in tons (X₂), and miles per gallon (X₃)")
print("\n| Car | Horsepower (X₁) | Weight (X₂) | MPG (X₃) |")
print("|-----|----------------|------------|---------|")
for i, row in enumerate(data):
    print(f"| {i+1}   | {row[0]:<14} | {row[1]:<10} | {row[2]:<7} |")

# Step 1: Calculate the Mean Vector
print("\nStep 1: Calculate the Mean Vector")

mean_vector = np.mean(data, axis=0)

for i, var_name in enumerate(["X₁ (Horsepower)", "X₂ (Weight)", "X₃ (MPG)"]):
    values = data[:, i]
    print(f"\nμ_{var_name} = ({' + '.join(map(str, values))}) / {len(values)} = {sum(values)} / {len(values)} = {mean_vector[i]}")

print("\nTherefore, the mean vector μ = [", end="")
print(", ".join([f"{val}" for val in mean_vector]), end="")
print("]")

# Step 2: Calculate Deviations from the Mean
print("\nStep 2: Calculate Deviations from the Mean")

deviations = data - mean_vector

print("\n| Car | X₁ - μ₁ | X₂ - μ₂ | X₃ - μ₃ |")
print("|-----|---------|---------|---------|")
for i, row in enumerate(deviations):
    print(f"| {i+1}   | {row[0]:<7.1f} | {row[1]:<7.2f} | {row[2]:<7.1f} |")

# Step 3: Compute the Covariance Matrix Elements
print("\nStep 3: Compute the Covariance Matrix Elements")

# Number of samples
n = data.shape[0]
# Number of variables
p = data.shape[1]

# Initialize covariance matrix
cov_matrix = np.zeros((p, p))

# Calculate each element of the covariance matrix
var_names = ["X₁", "X₂", "X₃"]

print("\nCalculating variances (diagonal elements):")
for i in range(p):
    # Calculate variance (diagonal elements)
    variance = np.sum(deviations[:, i]**2) / (n - 1)
    cov_matrix[i, i] = variance
    
    # Print the calculation steps
    squared_devs = deviations[:, i]**2
    print(f"\nσ_{var_names[i]}{var_names[i]} = Var({var_names[i]}) = ", end="")
    print(f"({' + '.join([f'({dev:.2f})²' for dev in deviations[:, i]])}) / {n-1} = ", end="")
    print(f"({' + '.join([f'{dev:.2f}' for dev in squared_devs])}) / {n-1} = ", end="")
    print(f"{np.sum(squared_devs):.2f} / {n-1} = {variance:.4f}")

print("\nCalculating covariances (off-diagonal elements):")
for i in range(p):
    for j in range(i+1, p):
        # Calculate covariance (off-diagonal elements)
        covariance = np.sum(deviations[:, i] * deviations[:, j]) / (n - 1)
        cov_matrix[i, j] = covariance
        cov_matrix[j, i] = covariance  # Covariance matrix is symmetric
        
        # Print the calculation steps
        products = deviations[:, i] * deviations[:, j]
        print(f"\nσ_{var_names[i]}{var_names[j]} = Cov({var_names[i]}, {var_names[j]}) = ", end="")
        print(f"({' + '.join([f'({dev_i:.2f})({dev_j:.2f})' for dev_i, dev_j in zip(deviations[:, i], deviations[:, j])])}) / {n-1} = ", end="")
        print(f"({' + '.join([f'{prod:.2f}' for prod in products])}) / {n-1} = ", end="")
        print(f"{np.sum(products):.2f} / {n-1} = {covariance:.4f}")

# Step 4: Assemble the Covariance Matrix
print("\nStep 4: Assemble the Covariance Matrix")
print("\nThe complete covariance matrix is:")
print(f"\n{cov_matrix}")

print("\nIn a more readable format:")
print("\n┌" + "─" * 35 + "┐")
for i in range(p):
    print("│ ", end="")
    for j in range(p):
        print(f"{cov_matrix[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 35 + "┘")

# Interpretation
print("\nInterpretation:")
print("- The positive covariance (19.0) between horsepower and weight indicates that as horsepower increases, weight tends to increase")
print("- The negative covariance (-253.75) between horsepower and MPG indicates that as horsepower increases, fuel efficiency tends to decrease")
print("- The negative covariance (-2.235) between weight and MPG indicates that as weight increases, fuel efficiency tends to decrease")
print("- These findings align with mechanical engineering principles: more powerful engines tend to be heavier and consume more fuel")

# Visualization: Scatter plot matrix
plt.figure(figsize=(14, 10))
labels = ["Horsepower", "Weight (tons)", "MPG"]

# Create a scatter plot matrix
for i in range(p):
    for j in range(p):
        plt.subplot(p, p, i*p + j + 1)
        
        if i == j:  # Histogram on the diagonal
            plt.hist(data[:, j], bins=5, color='skyblue', alpha=0.7)
            plt.axvline(mean_vector[j], color='red', linestyle='--', linewidth=2)
            plt.text(mean_vector[j], plt.ylim()[1]*0.8, f'μ = {mean_vector[j]:.1f}', color='red', 
                     ha='center', bbox=dict(facecolor='white', alpha=0.5))
        else:  # Scatter plot for off-diagonal
            plt.scatter(data[:, j], data[:, i], c='blue', alpha=0.7, s=100)
            plt.axvline(mean_vector[j], color='red', linestyle='--', alpha=0.3)
            plt.axhline(mean_vector[i], color='red', linestyle='--', alpha=0.3)
            
            # Add correlation text
            corr = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
            plt.text(np.mean(plt.xlim()), np.mean(plt.ylim()), 
                     f'Cov = {cov_matrix[i, j]:.1f}\nCorr = {corr:.2f}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        if i == p-1:  # Last row
            plt.xlabel(labels[j])
        if j == 0:    # First column
            plt.ylabel(labels[i])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'car_features_scatterplot_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normalize the data for better visualization
x_normalized = (data[:, 0] - mean_vector[0]) / np.sqrt(cov_matrix[0, 0])
y_normalized = (data[:, 1] - mean_vector[1]) / np.sqrt(cov_matrix[1, 1])
z_normalized = (data[:, 2] - mean_vector[2]) / np.sqrt(cov_matrix[2, 2])

ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=100, alpha=0.7)
ax.scatter([mean_vector[0]], [mean_vector[1]], [mean_vector[2]], c='red', s=150, alpha=1, marker='*')

# Add car labels
for i in range(n):
    ax.text(data[i, 0], data[i, 1], data[i, 2], f'Car {i+1}', fontsize=8)

ax.set_xlabel('Horsepower')
ax.set_ylabel('Weight (tons)')
ax.set_zlabel('MPG')
ax.set_title('3D Scatter Plot of Car Features')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'car_features_3d_scatter.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Heatmap of Correlation Matrix
plt.figure(figsize=(8, 6))
correlation_matrix = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        correlation_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

# Create a heatmap of the correlation matrix using matplotlib instead of seaborn
plt.figure(figsize=(8, 6))
im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation')

# Add text annotations
for i in range(p):
    for j in range(p):
        plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                 ha='center', va='center', color='black' if abs(correlation_matrix[i, j]) < 0.7 else 'white')

plt.xticks(range(p), labels)
plt.yticks(range(p), labels)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'car_features_correlation_heatmap.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Student Exam Scores Analysis
print("\n\nExample 2: Student Exam Scores Analysis")

# Dataset with 4 variables: Math, Physics, Chemistry, and Biology scores
exam_data = np.array([
    [85, 78, 82, 90],
    [90, 85, 88, 92],
    [70, 65, 72, 68],
    [80, 75, 80, 82],
    [95, 90, 92, 88],
    [60, 55, 58, 62]
])

print("Dataset with exam scores for 6 students across 4 subjects")
print("\n| Student | Math | Physics | Chemistry | Biology |")
print("|---------|------|---------|-----------|---------|")
for i, row in enumerate(exam_data):
    print(f"| {i+1}       | {row[0]:<4} | {row[1]:<7} | {row[2]:<9} | {row[3]:<7} |")

# Step 1: Calculate the Mean Vector
print("\nStep 1: Calculate the Mean Vector")

exam_mean_vector = np.mean(exam_data, axis=0)

for i, subject in enumerate(["Math", "Physics", "Chemistry", "Biology"]):
    values = exam_data[:, i]
    print(f"\nμ_{subject} = ({' + '.join(map(str, values))}) / {len(values)} = {sum(values)} / {len(values)} = {exam_mean_vector[i]}")

print("\nTherefore, the mean vector μ = [", end="")
print(", ".join([f"{val:.2f}" for val in exam_mean_vector]), end="")
print("]")

# Step 2: Calculate Sample Covariance Matrix
print("\nStep 2: Calculate Sample Covariance Matrix")

n_students = exam_data.shape[0]
n_subjects = exam_data.shape[1]

# Calculate deviations from the mean
exam_deviations = exam_data - exam_mean_vector

# Calculate the covariance matrix
exam_cov_matrix = np.zeros((n_subjects, n_subjects))
for i in range(n_subjects):
    for j in range(n_subjects):
        exam_cov_matrix[i, j] = np.sum(exam_deviations[:, i] * exam_deviations[:, j]) / (n_students - 1)

print("\nThe sample covariance matrix is:")
print("\n┌" + "─" * 47 + "┐")
for i in range(n_subjects):
    print("│ ", end="")
    for j in range(n_subjects):
        print(f"{exam_cov_matrix[i, j]:10.2f} ", end="")
    print("│")
print("└" + "─" * 47 + "┘")

# Calculate the correlation matrix
exam_corr_matrix = np.zeros((n_subjects, n_subjects))
for i in range(n_subjects):
    for j in range(n_subjects):
        exam_corr_matrix[i, j] = exam_cov_matrix[i, j] / np.sqrt(exam_cov_matrix[i, i] * exam_cov_matrix[j, j])

print("\nStep 3: Calculate the Correlation Matrix")
print("\nThe correlation matrix is:")
print("\n┌" + "─" * 47 + "┐")
for i in range(n_subjects):
    print("│ ", end="")
    for j in range(n_subjects):
        print(f"{exam_corr_matrix[i, j]:10.4f} ", end="")
    print("│")
print("└" + "─" * 47 + "┘")

# Interpretation
print("\nInterpretation:")
print("- The variances (diagonal elements) show the spread of scores in each subject")
print("- The high positive correlations between subjects indicate students who do well in one subject tend to do well in others")
print("- Math and Physics have a particularly high correlation, as expected due to the overlap in required skills")
print("- A student's performance can generally be predicted across subjects based on their performance in a single subject")

# Visualization: Heatmap of Correlation Matrix for Exam Scores
plt.figure(figsize=(8, 6))
subject_labels = ["Math", "Physics", "Chemistry", "Biology"]

# Create a heatmap of the correlation matrix using matplotlib instead of seaborn
plt.figure(figsize=(8, 6))
im = plt.imshow(exam_corr_matrix, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im, label='Correlation')

# Add text annotations
for i in range(n_subjects):
    for j in range(n_subjects):
        plt.text(j, i, f'{exam_corr_matrix[i, j]:.2f}', 
                ha='center', va='center', color='black' if exam_corr_matrix[i, j] < 0.7 else 'white')

plt.xticks(range(n_subjects), subject_labels)
plt.yticks(range(n_subjects), subject_labels)
plt.title('Correlation Matrix of Exam Scores')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exam_scores_correlation_heatmap.png'), dpi=100, bbox_inches='tight')
plt.close()

# Visualization: Scatter plot matrix for exam scores
plt.figure(figsize=(14, 10))

# Create a scatter plot matrix
for i in range(n_subjects):
    for j in range(n_subjects):
        plt.subplot(n_subjects, n_subjects, i*n_subjects + j + 1)
        
        if i == j:  # Histogram on the diagonal
            plt.hist(exam_data[:, j], bins=5, color='lightgreen', alpha=0.7)
            plt.axvline(exam_mean_vector[j], color='red', linestyle='--', linewidth=2)
            plt.text(exam_mean_vector[j], plt.ylim()[1]*0.8, f'μ = {exam_mean_vector[j]:.1f}', color='red', 
                     ha='center', bbox=dict(facecolor='white', alpha=0.5))
        else:  # Scatter plot for off-diagonal
            plt.scatter(exam_data[:, j], exam_data[:, i], c='green', alpha=0.7, s=80)
            plt.axvline(exam_mean_vector[j], color='red', linestyle='--', alpha=0.3)
            plt.axhline(exam_mean_vector[i], color='red', linestyle='--', alpha=0.3)
            
            # Add correlation text
            plt.text(np.mean(plt.xlim()), np.mean(plt.ylim()), 
                     f'r = {exam_corr_matrix[i, j]:.2f}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        if i == n_subjects-1:  # Last row
            plt.xlabel(subject_labels[j])
        if j == 0:    # First column
            plt.ylabel(subject_labels[i])

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'exam_scores_scatterplot_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll mean vector and covariance matrix example images created successfully.") 