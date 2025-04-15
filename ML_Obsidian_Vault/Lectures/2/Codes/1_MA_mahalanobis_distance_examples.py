import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

print("\n=== MAHALANOBIS DISTANCE EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Analysis")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Function to calculate Mahalanobis distance (detailed)
def calculate_mahalanobis_distance(x, mean, cov_inv, show_steps=True):
    """Calculate Mahalanobis distance with detailed steps."""
    # Calculate deviation from the mean
    deviation = x - mean
    
    if show_steps:
        print(f"Step 1: Calculate deviation from mean (x - μ):")
        print(f"x = {x}")
        print(f"μ = {mean}")
        print(f"x - μ = {deviation}")
        
        print("\nStep 2: Calculate (x - μ)ᵀ Σ⁻¹ (x - μ):")
        print(f"Σ⁻¹ = \n{cov_inv}")
        
        # Show intermediate calculation: Σ⁻¹(x - μ)
        intermediate = np.dot(cov_inv, deviation)
        print(f"\nΣ⁻¹(x - μ) = {intermediate}")
        
        # Show final calculation
        squared_dist = np.dot(deviation, intermediate)
        print(f"\n(x - μ)ᵀΣ⁻¹(x - μ) = {squared_dist}")
        
        # Calculate square root
        distance = np.sqrt(squared_dist)
        print(f"\nStep 3: Calculate square root:")
        print(f"Mahalanobis distance = √({squared_dist}) = {distance}")
    else:
        # Just calculate without showing steps
        squared_dist = np.dot(deviation, np.dot(cov_inv, deviation))
        distance = np.sqrt(squared_dist)
    
    return distance

# Function to draw confidence ellipses
def draw_confidence_ellipse(ax, mean, cov, color='blue', alpha=0.3, n_std=2.0, label=None):
    """Draw a covariance confidence ellipse."""
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color, alpha=alpha, label=label)
    
    # Scale in data coordinates
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = mean
    
    transform = transforms.Affine2D() \
        .rotate_deg(45 if pearson > 0 else -45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

# Example 1: Outlier Detection in Credit Card Transactions
print("Example 1: Outlier Detection in Credit Card Transactions")

# Dataset with transaction amount (X₁) and transaction frequency (X₂)
normal_data = np.array([
    [120, 5],
    [150, 4],
    [100, 3],
    [130, 6],
    [140, 5],
    [110, 4],
    [160, 7],
    [125, 5],
    [115, 4],
    [145, 6]
])

# Potential fraudulent transactions to test
test_transactions = np.array([
    [135, 5],  # Normal transaction
    [800, 12], # Clearly fraudulent
    [200, 20], # Unusual frequency
    [50, 1]    # Small and infrequent (not fraudulent but unusual)
])

print("Dataset of 10 normal credit card transactions with 2 variables:")
print("\n| Transaction | Amount (X₁) | Frequency per week (X₂) |")
print("|-------------|--------------|--------------------------|")
for i, row in enumerate(normal_data):
    print(f"| {i+1:<11} | ${row[0]:<11.0f} | {row[1]:<24.0f} |")

print("\nPotential transactions to analyze:")
print("\n| Transaction | Amount (X₁) | Frequency per week (X₂) |")
print("|-------------|--------------|--------------------------|")
for i, row in enumerate(test_transactions):
    print(f"| Test {i+1:<7} | ${row[0]:<11.0f} | {row[1]:<24.0f} |")

# Step 1: Calculate the Mean Vector and Covariance Matrix
print("\nStep 1: Calculate the Mean Vector and Covariance Matrix of normal transactions")

# Mean vector
mean_vector = np.mean(normal_data, axis=0)
print(f"\nMean Vector: μ = [{mean_vector[0]:.2f}, {mean_vector[1]:.2f}]")

# Covariance matrix
cov_matrix = np.cov(normal_data, rowvar=False)
print(f"\nCovariance Matrix: Σ = \n{cov_matrix}")

# Step 2: Calculate the Inverse of the Covariance Matrix
print("\nStep 2: Calculate the Inverse of the Covariance Matrix")
cov_inv = np.linalg.inv(cov_matrix)
print(f"\nInverse Covariance Matrix: Σ⁻¹ = \n{cov_inv}")

# Step 3: Calculate Mahalanobis Distance for each test transaction
print("\nStep 3: Calculate Mahalanobis Distance for each test transaction")

mahalanobis_distances = []
for i, transaction in enumerate(test_transactions):
    print(f"\n--- Test Transaction {i+1}: ${transaction[0]:.0f}, {transaction[1]} times per week ---")
    dist = calculate_mahalanobis_distance(transaction, mean_vector, cov_inv)
    mahalanobis_distances.append(dist)

# Step 4: Determine outliers using chi-squared distribution
print("\nStep 4: Determine outliers using chi-squared distribution")

# With 2 degrees of freedom (2 variables)
# Critical values for different significance levels
alpha_levels = [0.05, 0.01, 0.001]
critical_values = [stats.chi2.ppf(1 - alpha, df=2) for alpha in alpha_levels]

print("\nCritical values from chi-squared distribution with 2 degrees of freedom:")
for alpha, cv in zip(alpha_levels, critical_values):
    print(f"Significance level α = {alpha}: Critical value = {cv:.3f}")

print("\nOutlier detection results:")
print("\n| Transaction | Mahalanobis Distance | Outlier at α=0.05? | Outlier at α=0.01? | Outlier at α=0.001? |")
print("|-------------|---------------------|-----------------|-----------------|------------------|")
for i, dist in enumerate(mahalanobis_distances):
    outlier_05 = "Yes" if dist > critical_values[0] else "No"
    outlier_01 = "Yes" if dist > critical_values[1] else "No"
    outlier_001 = "Yes" if dist > critical_values[2] else "No"
    print(f"| Test {i+1:<7} | {dist:<21.3f} | {outlier_05:<17} | {outlier_01:<17} | {outlier_001:<18} |")

# Interpretation
print("\nInterpretation:")
interpretation_results = []
for i, dist in enumerate(mahalanobis_distances):
    if i == 0:
        interpretation_results.append(f"- Test Transaction 1 (${test_transactions[i,0]:.0f}, {test_transactions[i,1]} times): With Mahalanobis distance of {dist:.3f}, "
                                      f"this transaction is similar to normal patterns and is not flagged as an outlier.")
    elif i == 1:
        interpretation_results.append(f"- Test Transaction 2 (${test_transactions[i,0]:.0f}, {test_transactions[i,1]} times): With Mahalanobis distance of {dist:.3f}, "
                                      f"this transaction is highly unusual in both amount and frequency, strongly indicating fraud.")
    elif i == 2:
        interpretation_results.append(f"- Test Transaction 3 (${test_transactions[i,0]:.0f}, {test_transactions[i,1]} times): With Mahalanobis distance of {dist:.3f}, "
                                      f"this transaction has a normal amount but unusually high frequency, suggesting possible account sharing or automated payments.")
    elif i == 3:
        interpretation_results.append(f"- Test Transaction 4 (${test_transactions[i,0]:.0f}, {test_transactions[i,1]} time): With Mahalanobis distance of {dist:.3f}, "
                                      f"this transaction is unusual for being small and infrequent, but not necessarily fraudulent - could be a seldom-used card.")

for interp in interpretation_results:
    print(interp)

# Visualization: Scatter plot with Mahalanobis distance contours
plt.figure(figsize=(10, 8))
plt.scatter(normal_data[:, 0], normal_data[:, 1], c='blue', marker='o', alpha=0.7, label='Normal Transactions')

# Plot test transactions with markers indicating detection level
for i, (tx, dist) in enumerate(zip(test_transactions, mahalanobis_distances)):
    if dist <= critical_values[0]:  # Not an outlier
        plt.scatter(tx[0], tx[1], c='green', marker='o', s=100, edgecolors='black', linewidths=2, label='_nolegend_')
        plt.text(tx[0]+5, tx[1], f'Test {i+1}', fontsize=10)
    elif dist <= critical_values[1]:  # Outlier at 0.05
        plt.scatter(tx[0], tx[1], c='yellow', marker='s', s=100, edgecolors='black', linewidths=2, label='_nolegend_')
        plt.text(tx[0]+5, tx[1], f'Test {i+1}', fontsize=10)
    elif dist <= critical_values[2]:  # Outlier at 0.01
        plt.scatter(tx[0], tx[1], c='orange', marker='s', s=100, edgecolors='black', linewidths=2, label='_nolegend_')
        plt.text(tx[0]+5, tx[1], f'Test {i+1}', fontsize=10)
    else:  # Outlier at 0.001
        plt.scatter(tx[0], tx[1], c='red', marker='s', s=100, edgecolors='black', linewidths=2, label='_nolegend_')
        plt.text(tx[0]+5, tx[1], f'Test {i+1}', fontsize=10)

# Add mean vector
plt.scatter(mean_vector[0], mean_vector[1], c='black', marker='X', s=150, label='Mean')

# Add confidence ellipses
draw_confidence_ellipse(plt.gca(), mean_vector, cov_matrix, n_std=np.sqrt(critical_values[0]), 
                       color='green', alpha=0.1, label=f'95% Region (α=0.05)')
draw_confidence_ellipse(plt.gca(), mean_vector, cov_matrix, n_std=np.sqrt(critical_values[1]), 
                       color='yellow', alpha=0.1, label=f'99% Region (α=0.01)')
draw_confidence_ellipse(plt.gca(), mean_vector, cov_matrix, n_std=np.sqrt(critical_values[2]), 
                       color='red', alpha=0.1, label=f'99.9% Region (α=0.001)')

plt.xlabel('Transaction Amount ($)')
plt.ylabel('Transaction Frequency (per week)')
plt.title('Credit Card Transactions with Mahalanobis Distance Contours')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'credit_card_mahalanobis.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Multivariate Outliers in Manufacturing Quality Control
print("\n\nExample 2: Multivariate Outliers in Manufacturing Quality Control")

# Dataset with 3 variables: length (mm), width (mm), and weight (g) of manufactured parts
normal_parts = np.array([
    [50.1, 25.2, 120.5],
    [49.8, 24.9, 119.8],
    [50.2, 25.0, 120.2],
    [50.0, 25.1, 120.0],
    [49.9, 25.0, 119.9],
    [50.1, 24.8, 120.1],
    [49.7, 24.9, 119.5],
    [50.3, 25.1, 120.6],
    [50.0, 25.0, 120.0],
    [49.9, 24.8, 119.7],
    [50.2, 25.2, 120.4],
    [49.8, 25.0, 119.9]
])

# Test parts for quality control
test_parts = np.array([
    [50.0, 25.0, 120.1],  # Normal part
    [48.5, 24.0, 116.0],  # Undersized part
    [52.0, 26.0, 125.0],  # Oversized part
    [50.0, 25.0, 110.0],  # Right dimensions but too light
    [50.0, 20.0, 120.0]   # Abnormal width but normal length and weight
])

print("Dataset of 12 normal manufactured parts with 3 variables:")
print("\n| Part | Length (mm) | Width (mm) | Weight (g) |")
print("|------|-------------|------------|------------|")
for i, row in enumerate(normal_parts):
    print(f"| {i+1:<4} | {row[0]:<11.1f} | {row[1]:<10.1f} | {row[2]:<10.1f} |")

print("\nTest parts to analyze:")
print("\n| Part | Length (mm) | Width (mm) | Weight (g) |")
print("|------|-------------|------------|------------|")
for i, row in enumerate(test_parts):
    print(f"| Test {i+1} | {row[0]:<11.1f} | {row[1]:<10.1f} | {row[2]:<10.1f} |")

# Step 1: Calculate the Mean Vector and Covariance Matrix
print("\nStep 1: Calculate the Mean Vector and Covariance Matrix of normal parts")

# Mean vector
parts_mean = np.mean(normal_parts, axis=0)
print(f"\nMean Vector: μ = [{parts_mean[0]:.2f}, {parts_mean[1]:.2f}, {parts_mean[2]:.2f}]")

# Covariance matrix
parts_cov = np.cov(normal_parts, rowvar=False)
print(f"\nCovariance Matrix: Σ = \n{parts_cov}")

# Step 2: Calculate the Inverse of the Covariance Matrix
print("\nStep 2: Calculate the Inverse of the Covariance Matrix")
parts_cov_inv = np.linalg.inv(parts_cov)
print(f"\nInverse Covariance Matrix: Σ⁻¹ = \n{parts_cov_inv}")

# Step 3: Calculate Mahalanobis Distance for each test part
print("\nStep 3: Calculate Mahalanobis Distance for each test part")

print("\nDetailed calculation for Test Part 1 only:")
parts_distances = [calculate_mahalanobis_distance(test_parts[0], parts_mean, parts_cov_inv)]

print("\nCalculating for remaining test parts (summary):")
for i in range(1, len(test_parts)):
    dist = calculate_mahalanobis_distance(test_parts[i], parts_mean, parts_cov_inv, False)
    parts_distances.append(dist)
    print(f"Test Part {i+1}: Mahalanobis distance = {dist:.3f}")

# Step 4: Determine outliers using chi-squared distribution
print("\nStep 4: Determine outliers using chi-squared distribution")

# With 3 degrees of freedom (3 variables)
# Critical values for different significance levels
parts_critical_values = [stats.chi2.ppf(1 - alpha, df=3) for alpha in alpha_levels]

print("\nCritical values from chi-squared distribution with 3 degrees of freedom:")
for alpha, cv in zip(alpha_levels, parts_critical_values):
    print(f"Significance level α = {alpha}: Critical value = {cv:.3f}")

print("\nOutlier detection results:")
print("\n| Part | Mahalanobis Distance | Outlier at α=0.05? | Outlier at α=0.01? | Outlier at α=0.001? |")
print("|------|---------------------|-----------------|-----------------|------------------|")
for i, dist in enumerate(parts_distances):
    outlier_05 = "Yes" if dist > parts_critical_values[0] else "No"
    outlier_01 = "Yes" if dist > parts_critical_values[1] else "No"
    outlier_001 = "Yes" if dist > parts_critical_values[2] else "No"
    print(f"| Test {i+1} | {dist:<21.3f} | {outlier_05:<17} | {outlier_01:<17} | {outlier_001:<18} |")

# Interpretation
print("\nInterpretation:")
parts_interpretation = []
for i, dist in enumerate(parts_distances):
    if i == 0:
        parts_interpretation.append(f"- Test Part 1 ({test_parts[i][0]:.1f}mm × {test_parts[i][1]:.1f}mm, {test_parts[i][2]:.1f}g): With Mahalanobis distance of {dist:.3f}, "
                                     f"this part is well within manufacturing specifications and passes quality control.")
    elif i == 1:
        parts_interpretation.append(f"- Test Part 2 ({test_parts[i][0]:.1f}mm × {test_parts[i][1]:.1f}mm, {test_parts[i][2]:.1f}g): With Mahalanobis distance of {dist:.3f}, "
                                     f"this undersized part is clearly defective and should be rejected.")
    elif i == 2:
        parts_interpretation.append(f"- Test Part 3 ({test_parts[i][0]:.1f}mm × {test_parts[i][1]:.1f}mm, {test_parts[i][2]:.1f}g): With Mahalanobis distance of {dist:.3f}, "
                                     f"this oversized part is outside acceptable limits and should be rejected.")
    elif i == 3:
        parts_interpretation.append(f"- Test Part 4 ({test_parts[i][0]:.1f}mm × {test_parts[i][1]:.1f}mm, {test_parts[i][2]:.1f}g): With Mahalanobis distance of {dist:.3f}, "
                                     f"this part has correct dimensions but is too light, suggesting material composition issues.")
    elif i == 4:
        parts_interpretation.append(f"- Test Part 5 ({test_parts[i][0]:.1f}mm × {test_parts[i][1]:.1f}mm, {test_parts[i][2]:.1f}g): With Mahalanobis distance of {dist:.3f}, "
                                     f"this part has abnormal width which indicates a manufacturing defect in one dimension.")

for interp in parts_interpretation:
    print(interp)

# Visualization: 3D scatter plot with Mahalanobis distances
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot normal parts
ax.scatter(normal_parts[:, 0], normal_parts[:, 1], normal_parts[:, 2], 
           c='blue', marker='o', alpha=0.7, label='Normal Parts')

# Plot test parts with colors indicating outlier status
for i, (part, dist) in enumerate(zip(test_parts, parts_distances)):
    if dist <= parts_critical_values[0]:  # Not an outlier
        color = 'green'
    elif dist <= parts_critical_values[1]:  # Outlier at 0.05
        color = 'yellow'
    elif dist <= parts_critical_values[2]:  # Outlier at 0.01
        color = 'orange'
    else:  # Outlier at 0.001
        color = 'red'
    
    ax.scatter(part[0], part[1], part[2], c=color, marker='s', s=100, edgecolors='black', linewidths=2)
    ax.text(part[0], part[1], part[2], f'  Test {i+1}', fontsize=9)

# Add mean point
ax.scatter(parts_mean[0], parts_mean[1], parts_mean[2], c='black', marker='X', s=150, label='Mean')

# Add wireframe ellipsoid for 95% confidence region
# This is a simple approximation - a proper ellipsoid would be more complex
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = parts_mean[0] + np.sqrt(parts_critical_values[0] * parts_cov[0, 0]) * np.outer(np.cos(u), np.sin(v))
y = parts_mean[1] + np.sqrt(parts_critical_values[0] * parts_cov[1, 1]) * np.outer(np.sin(u), np.sin(v))
z = parts_mean[2] + np.sqrt(parts_critical_values[0] * parts_cov[2, 2]) * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(x, y, z, color='green', alpha=0.1)

ax.set_xlabel('Length (mm)')
ax.set_ylabel('Width (mm)')
ax.set_zlabel('Weight (g)')
ax.set_title('Quality Control: Parts Analysis using Mahalanobis Distance')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'manufacturing_mahalanobis_3d.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create 2D projections for better visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Length vs Width
axs[0].scatter(normal_parts[:, 0], normal_parts[:, 1], c='blue', marker='o', alpha=0.7, label='Normal Parts')
for i, (part, dist) in enumerate(zip(test_parts, parts_distances)):
    if dist <= parts_critical_values[0]:  # Not an outlier
        color = 'green'
    elif dist <= parts_critical_values[1]:  # Outlier at 0.05
        color = 'yellow'
    elif dist <= parts_critical_values[2]:  # Outlier at 0.01
        color = 'orange'
    else:  # Outlier at 0.001
        color = 'red'
    
    axs[0].scatter(part[0], part[1], c=color, marker='s', s=100, edgecolors='black', linewidths=2)
    axs[0].text(part[0], part[1], f'  Test {i+1}', fontsize=9)

axs[0].scatter(parts_mean[0], parts_mean[1], c='black', marker='X', s=150, label='Mean')
axs[0].set_xlabel('Length (mm)')
axs[0].set_ylabel('Width (mm)')
axs[0].set_title('Length vs Width')
axs[0].grid(True, alpha=0.3)

# Length vs Weight
axs[1].scatter(normal_parts[:, 0], normal_parts[:, 2], c='blue', marker='o', alpha=0.7, label='Normal Parts')
for i, (part, dist) in enumerate(zip(test_parts, parts_distances)):
    if dist <= parts_critical_values[0]:  # Not an outlier
        color = 'green'
    elif dist <= parts_critical_values[1]:  # Outlier at 0.05
        color = 'yellow'
    elif dist <= parts_critical_values[2]:  # Outlier at 0.01
        color = 'orange'
    else:  # Outlier at 0.001
        color = 'red'
    
    axs[1].scatter(part[0], part[2], c=color, marker='s', s=100, edgecolors='black', linewidths=2)
    axs[1].text(part[0], part[2], f'  Test {i+1}', fontsize=9)

axs[1].scatter(parts_mean[0], parts_mean[2], c='black', marker='X', s=150, label='Mean')
axs[1].set_xlabel('Length (mm)')
axs[1].set_ylabel('Weight (g)')
axs[1].set_title('Length vs Weight')
axs[1].grid(True, alpha=0.3)

# Width vs Weight
axs[2].scatter(normal_parts[:, 1], normal_parts[:, 2], c='blue', marker='o', alpha=0.7, label='Normal Parts')
for i, (part, dist) in enumerate(zip(test_parts, parts_distances)):
    if dist <= parts_critical_values[0]:  # Not an outlier
        color = 'green'
    elif dist <= parts_critical_values[1]:  # Outlier at 0.05
        color = 'yellow'
    elif dist <= parts_critical_values[2]:  # Outlier at 0.01
        color = 'orange'
    else:  # Outlier at 0.001
        color = 'red'
    
    axs[2].scatter(part[1], part[2], c=color, marker='s', s=100, edgecolors='black', linewidths=2)
    axs[2].text(part[1], part[2], f'  Test {i+1}', fontsize=9)

axs[2].scatter(parts_mean[1], parts_mean[2], c='black', marker='X', s=150, label='Mean')
axs[2].set_xlabel('Width (mm)')
axs[2].set_ylabel('Weight (g)')
axs[2].set_title('Width vs Weight')
axs[2].grid(True, alpha=0.3)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'manufacturing_mahalanobis_2d.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll Mahalanobis distance example images created successfully.") 