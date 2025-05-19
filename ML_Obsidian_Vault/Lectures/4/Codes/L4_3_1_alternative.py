import numpy as np

# Define the problem parameters
mu = np.array([0, 0])  # Equal means for both classes
Sigma0 = np.array([[1, 0], [0, 4]])  # Covariance matrix for class 0
Sigma1 = np.array([[4, 0], [0, 1]])  # Covariance matrix for class 1

# Calculate determinants
det_Sigma0 = np.linalg.det(Sigma0)
det_Sigma1 = np.linalg.det(Sigma1)
print(f"Determinant of Sigma0: {det_Sigma0}")
print(f"Determinant of Sigma1: {det_Sigma1}")

# Calculate inverse matrices (precision matrices)
inv_Sigma0 = np.linalg.inv(Sigma0)
inv_Sigma1 = np.linalg.inv(Sigma1)

print(f"\nPrecision matrix for class 0:\n{inv_Sigma0}")
print(f"\nPrecision matrix for class 1:\n{inv_Sigma1}")

# Calculate the difference of precision matrices
diff_precision = inv_Sigma1 - inv_Sigma0
print(f"\nDifference of precision matrices (Sigma1^-1 - Sigma0^-1):\n{diff_precision}")

# For equal priors, the decision boundary is where:
# -0.5 * x^T (Sigma1^-1 - Sigma0^-1) x + 0.5 * log(det(Sigma0)/det(Sigma1)) = 0

# With our matrices, this simplifies to:
# -0.5 * x^T [[-3/4, 0], [0, 3/4]] x + 0.5 * log(4/4) = 0
# -0.5 * (-3/4*x1^2 + 3/4*x2^2) = 0
# 3/8*x1^2 - 3/8*x2^2 = 0
# x1^2 = x2^2

print("\nDecision boundary equation:")
print("x1^2 = x2^2")
print("This gives us two lines: x1 = x2 and x1 = -x2")

# Decision regions:
print("\nDecision regions:")
print("Class 0: |x2| > |x1| (vertical regions)")
print("Class 1: |x1| > |x2| (horizontal regions)")

# Effect of unequal priors:
print("\nEffect of unequal priors:")
print("For unequal priors P(y=0)=p0 and P(y=1)=p1, the decision boundary becomes:")
print("3/8*x1^2 - 3/8*x2^2 = log(p1/p0)")
print("This creates hyperbolas instead of straight lines when p0 ≠ p1")

# Key insights:
print("\nKey insights:")
print("1. Equal means don't imply identical distributions")
print("2. The decision boundary shape depends on the difference of precision matrices")
print("3. For equal determinants, the boundary is determined solely by the quadratic form")
print("4. For equal priors and determinants, the decision regions are symmetric around the origin")
print("5. Prior probabilities shift the boundary, favoring the class with higher prior") 