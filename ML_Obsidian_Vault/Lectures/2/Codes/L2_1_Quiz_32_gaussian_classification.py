import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    file_path = os.path.join(save_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    plt.close(fig)

# Problem parameters
print("\n=== Problem Setup ===")
mu1 = np.array([0, 0])  # Mean vector for class 1
mu2 = np.array([0.5, 0.5])  # Mean vector for class 2
print("Mean vectors:")
print(f"μ₁ = {mu1}")
print(f"μ₂ = {mu2}")

# Common covariance matrix
Sigma = np.array([[0.8, 0.01], [0.01, 0.2]])
print("\nCovariance matrix Σ:")
print(Sigma)

# Equal prior probabilities
P_C1 = P_C2 = 0.5
print(f"\nPrior probabilities:")
print(f"P(C₁) = {P_C1}")
print(f"P(C₂) = {P_C2}")

# Test point to classify
x_test = np.array([0.1, 0.5])
print(f"\nTest point x = {x_test}")

print("\n=== Detailed Calculations ===")
print("\nStep 1: Calculate inverse covariance matrix")
print("For a 2x2 matrix, inverse is calculated as:")
print("Σ⁻¹ = (1/det(Σ)) * [[ Σ₂₂, -Σ₁₂], [-Σ₂₁, Σ₁₁]]")

# Manual calculation of determinant
a, b = Sigma[0, 0], Sigma[0, 1]
c, d = Sigma[1, 0], Sigma[1, 1]
det_manual = a*d - b*c
print(f"\nManual determinant calculation:")
print(f"det(Σ) = Σ₁₁Σ₂₂ - Σ₁₂Σ₂₁")
print(f"       = ({a} × {d}) - ({b} × {c})")
print(f"       = {det_manual}")

# Manual calculation of inverse
inv_manual = np.array([[d, -b], [-c, a]]) / det_manual
print("\nManual inverse calculation:")
print("Σ⁻¹ = (1/det(Σ)) × [[Σ₂₂, -Σ₁₂], [-Σ₂₁, Σ₁₁]]")
print(f"    = (1/{det_manual}) × [[{d}, {-b}], [{-c}, {a}]]")
print("    =")
print(inv_manual)

# Verify with numpy
Sigma_inv = np.linalg.inv(Sigma)
print("\nVerification with numpy.linalg.inv:")
print(Sigma_inv)

print("\nStep 2: Calculate determinant and log-determinant")
print("Manual determinant calculation shown above:")
print(f"|Σ| = {det_manual}")

log_det = np.log(det_manual)
print(f"\nLog-determinant calculation:")
print(f"ln|Σ| = ln({det_manual})")
print(f"      = {log_det}")

print("\nStep 3: Calculate discriminant for Class 1")
# For Class 1
diff1 = x_test - mu1
print("a. Difference vector (x - μ₁):")
print(f"x - μ₁ = [{x_test[0]} - {mu1[0]}, {x_test[1]} - {mu1[1]}]")
print(f"       = {diff1}")

print("\nb. Calculate (x - μ₁)ᵀ Σ⁻¹:")
temp1 = diff1.T @ Sigma_inv
print("Detailed matrix multiplication:")
print(f"[{diff1[0]} {diff1[1]}] × [[{Sigma_inv[0,0]:.8f} {Sigma_inv[0,1]:.8f}],")
print(f"                           [{Sigma_inv[1,0]:.8f} {Sigma_inv[1,1]:.8f}]]")
print(f"= [{temp1[0]:.8f} {temp1[1]:.8f}]")

print("\nc. Calculate quadratic term (x - μ₁)ᵀ Σ⁻¹ (x - μ₁):")
quad_term1 = temp1 @ diff1
print(f"[{temp1[0]:.8f} {temp1[1]:.8f}] × [{diff1[0]}]")
print(f"                                   [{diff1[1]}]")
print(f"= {quad_term1:.8f}")

# Calculate full discriminant function for Class 1
g1 = -0.5 * quad_term1 - 0.5 * log_det + np.log(P_C1)
print("\nd. Calculate discriminant function g₁(x):")
print("g₁(x) = -1/2(x - μ₁)ᵀ Σ⁻¹ (x - μ₁) - 1/2 ln|Σ| + ln P(C₁)")
print(f"      = -1/2({quad_term1:.8f}) - 1/2({log_det:.8f}) + ln({P_C1})")
print(f"      = {-0.5 * quad_term1:.8f} + {-0.5 * log_det:.8f} + {np.log(P_C1):.8f}")
print(f"      = {g1:.8f}")

print("\nStep 4: Calculate discriminant for Class 2")
# For Class 2
diff2 = x_test - mu2
print("a. Difference vector (x - μ₂):")
print(f"x - μ₂ = [{x_test[0]} - {mu2[0]}, {x_test[1]} - {mu2[1]}]")
print(f"       = {diff2}")

print("\nb. Calculate (x - μ₂)ᵀ Σ⁻¹:")
temp2 = diff2.T @ Sigma_inv
print("Detailed matrix multiplication:")
print(f"[{diff2[0]} {diff2[1]}] × [[{Sigma_inv[0,0]:.8f} {Sigma_inv[0,1]:.8f}],")
print(f"                           [{Sigma_inv[1,0]:.8f} {Sigma_inv[1,1]:.8f}]]")
print(f"= [{temp2[0]:.8f} {temp2[1]:.8f}]")

print("\nc. Calculate quadratic term (x - μ₂)ᵀ Σ⁻¹ (x - μ₂):")
quad_term2 = temp2 @ diff2
print(f"[{temp2[0]:.8f} {temp2[1]:.8f}] × [{diff2[0]}]")
print(f"                                   [{diff2[1]}]")
print(f"= {quad_term2:.8f}")

# Calculate full discriminant function for Class 2
g2 = -0.5 * quad_term2 - 0.5 * log_det + np.log(P_C2)
print("\nd. Calculate discriminant function g₂(x):")
print("g₂(x) = -1/2(x - μ₂)ᵀ Σ⁻¹ (x - μ₂) - 1/2 ln|Σ| + ln P(C₂)")
print(f"      = -1/2({quad_term2:.8f}) - 1/2({log_det:.8f}) + ln({P_C2})")
print(f"      = {-0.5 * quad_term2:.8f} + {-0.5 * log_det:.8f} + {np.log(P_C2):.8f}")
print(f"      = {g2:.8f}")

print("\n=== Classification Results ===")
print(f"Discriminant difference g₂(x) - g₁(x) = {g2 - g1:.8f}")

# Calculate posterior probabilities using softmax for numerical stability
print("\nCalculating posterior probabilities using softmax:")
max_g = max(g1, g2)
print(f"1. Find maximum discriminant value: max_g = {max_g:.8f}")

exp_g1 = np.exp(g1 - max_g)
exp_g2 = np.exp(g2 - max_g)
print(f"2. Calculate exp(g₁ - max_g) = {exp_g1:.8f}")
print(f"   Calculate exp(g₂ - max_g) = {exp_g2:.8f}")

sum_exp = exp_g1 + exp_g2
print(f"3. Sum of exponentials = {sum_exp:.8f}")

p1 = exp_g1/sum_exp
p2 = exp_g2/sum_exp
print("\nPosterior Probabilities:")
print(f"P(C₁|x) = {p1:.8f} ({p1:.2%})")
print(f"P(C₂|x) = {p2:.8f} ({p2:.2%})")

if g1 > g2:
    print("\nClassification Decision: Class 1")
    print(f"Confidence: {p1:.2%}")
else:
    print("\nClassification Decision: Class 2")
    print(f"Confidence: {p2:.2%}")

# Visualization functions
def discriminant(x, mu):
    """Calculate discriminant function value for a given point and mean."""
    diff = x - mu
    return -0.5 * (diff @ Sigma_inv @ diff.T)

def gaussian_pdf(x, mu):
    """Calculate Gaussian PDF value for a given point and mean."""
    diff = x - mu
    return np.exp(-0.5 * (diff @ Sigma_inv @ diff.T)) / (2 * np.pi * np.sqrt(det_manual))

# Create grid for plotting
x1, x2 = np.meshgrid(np.linspace(-1, 1.5, 100), np.linspace(-1, 1.5, 100))
grid_points = np.column_stack((x1.ravel(), x2.ravel()))

# Calculate discriminant values over grid
g1_grid = np.array([discriminant(x, mu1) for x in grid_points]).reshape(x1.shape)
g2_grid = np.array([discriminant(x, mu2) for x in grid_points]).reshape(x1.shape)

print("\n=== Generating Visualizations ===")

# 1. Decision Boundary Plot
print("\n1. Creating Decision Boundary Plot...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.contour(x1, x2, g1_grid - g2_grid, levels=[0], colors='k', linewidths=2)
ax.scatter(mu1[0], mu1[1], color='blue', marker='X', s=200, label='Class 1 Mean')
ax.scatter(mu2[0], mu2[1], color='red', marker='X', s=200, label='Class 2 Mean')
ax.scatter(x_test[0], x_test[1], color='green', marker='*', s=200, label='Test Point')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Decision Boundary')
ax.legend()
ax.grid(True)
save_figure(fig, "decision_boundary.png")

# 2. Class-conditional PDFs Contour Plot
print("\n2. Creating Class-conditional PDFs Plot...")
pdf1_grid = np.array([gaussian_pdf(x, mu1) for x in grid_points]).reshape(x1.shape)
pdf2_grid = np.array([gaussian_pdf(x, mu2) for x in grid_points]).reshape(x1.shape)

fig, ax = plt.subplots(figsize=(10, 8))
levels = np.linspace(0, max(pdf1_grid.max(), pdf2_grid.max()), 20)
ax.contour(x1, x2, pdf1_grid, levels=levels, colors='blue', alpha=0.5, label='Class 1 PDF')
ax.contour(x1, x2, pdf2_grid, levels=levels, colors='red', alpha=0.5, label='Class 2 PDF')
ax.contour(x1, x2, g1_grid - g2_grid, levels=[0], colors='k', linewidths=2, label='Decision Boundary')
ax.scatter(mu1[0], mu1[1], color='blue', marker='X', s=200, label='Class 1 Mean')
ax.scatter(mu2[0], mu2[1], color='red', marker='X', s=200, label='Class 2 Mean')
ax.scatter(x_test[0], x_test[1], color='green', marker='*', s=200, label='Test Point')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Class-conditional PDFs and Decision Boundary')
ax.legend()
ax.grid(True)
save_figure(fig, "class_conditional_pdfs.png")

# 3. 3D Surface Plot of Discriminant Functions
print("\n3. Creating 3D Discriminant Functions Plot...")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(x1, x2, g1_grid, cmap='Blues', alpha=0.5, label='g₁(x)')
surf2 = ax.plot_surface(x1, x2, g2_grid, cmap='Reds', alpha=0.5, label='g₂(x)')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('Discriminant Value')
ax.set_title('3D View of Discriminant Functions')

# Add test point
ax.scatter(x_test[0], x_test[1], g1_grid[50, 50], color='green', marker='*', s=200, label='Test Point')

# Create proxy artists for the legend
import matplotlib.patches as mpatches
proxy1 = mpatches.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.5)
proxy2 = mpatches.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5)
ax.legend([proxy1, proxy2, ax.scatter([], [], color='green', marker='*', s=200)],
         ['g₁(x)', 'g₂(x)', 'Test Point'])

save_figure(fig, "discriminant_functions_3d.png")
print("\nAll visualizations have been generated and saved.") 