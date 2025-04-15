import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== UNIFORM DISTRIBUTION EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images/Uniform_Distribution relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Uniform_Distribution")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Basic Properties and Visualization
print("Example 1: Basic Properties and Visualization")
print("Step 1: Define the uniform distribution parameters")
a, b = 0, 1
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

print("\nStep 2: Calculate the PDF")
print(f"  PDF formula: f(x) = 1/(b-a) for x ∈ [a,b]")
print(f"  For our case: f(x) = 1/({b}-{a}) = {1/(b-a)}")

print("\nStep 3: Calculate the CDF")
print(f"  CDF formula: F(x) = (x-a)/(b-a) for x ∈ [a,b]")
print(f"  For our case: F(x) = (x-{a})/({b}-{a}) = x")

x = np.linspace(a-0.5, b+0.5, 1000)
pdf = np.where((x >= a) & (x <= b), 1/(b-a), 0)
cdf = np.where(x < a, 0, np.where(x > b, 1, (x-a)/(b-a)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, alpha=0.2)
plt.title('PDF of Uniform(0,1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', linewidth=2)
plt.title('CDF of Uniform(0,1)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'basic.png'), dpi=100)
plt.close()

print("\nStep 4: Verify Properties")
print(f"  Area under PDF curve: {np.trapz(pdf, x):.2f}")
print(f"  PDF is constant at {1/(b-a)} over [{a},{b}]")
print(f"  CDF increases linearly from 0 to 1 over [{a},{b}]")

# Example 2: Probability Calculations
print("\n\nExample 2: Probability Calculations")
print("Step 1: Define the uniform distribution parameters")
a, b = 2, 5
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

print("\nStep 2: Calculate P(X ≤ 3)")
x1 = 3
print(f"  P(X ≤ {x1}) = F({x1}) = ({x1}-{a})/({b}-{a})")
p1 = (x1 - a) / (b - a)
print(f"  P(X ≤ {x1}) = {p1:.4f}")

print("\nStep 3: Calculate P(2.5 ≤ X ≤ 4)")
x2_lower, x2 = 2.5, 4
print(f"  P({x2_lower} ≤ X ≤ {x2}) = F({x2}) - F({x2_lower})")
print(f"  = ({x2}-{a})/({b}-{a}) - ({x2_lower}-{a})/({b}-{a})")
p2 = (x2 - x2_lower) / (b - a)
print(f"  = {p2:.4f}")

print("\nStep 4: Calculate P(X > 4.5)")
x3 = 4.5
print(f"  P(X > {x3}) = 1 - F({x3})")
print(f"  = 1 - ({x3}-{a})/({b}-{a})")
p3 = 1 - (x3 - a) / (b - a)
print(f"  = {p3:.4f}")

# Plot probabilities
x = np.linspace(a-0.5, b+0.5, 1000)
pdf = np.where((x >= a) & (x <= b), 1/(b-a), 0)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, where=(x <= x1), alpha=0.3, color='green', label=f'P(X ≤ {x1})')
plt.fill_between(x, pdf, where=(x >= x2_lower) & (x <= x2), alpha=0.3, color='red', label=f'P({x2_lower} ≤ X ≤ {x2})')
plt.fill_between(x, pdf, where=(x > x3), alpha=0.3, color='purple', label=f'P(X > {x3})')
plt.title('Probability Calculations for Uniform(2,5)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'probabilities.png'), dpi=100)
plt.close()

# Example 3: Mean and Variance
print("\n\nExample 3: Mean and Variance")
print("Step 1: Define the uniform distribution parameters")
a, b = 3, 7
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

print("\nStep 2: Calculate the Mean")
print(f"  Mean formula: μ = (a + b)/2")
print(f"  For our case: μ = ({a} + {b})/2")
mean = (a + b) / 2
print(f"  Mean = {mean}")

print("\nStep 3: Calculate the Variance")
print(f"  Variance formula: σ² = (b-a)²/12")
print(f"  For our case: σ² = ({b}-{a})²/12")
variance = (b - a)**2 / 12
print(f"  Variance = {variance:.4f}")

print("\nStep 4: Calculate Standard Deviation")
print(f"  Standard deviation formula: σ = √σ²")
print(f"  For our case: σ = √{variance:.4f}")
std_dev = np.sqrt(variance)
print(f"  Standard deviation = {std_dev:.4f}")

# Plot distribution with mean and std dev
x = np.linspace(a-1, b+1, 1000)
pdf = np.where((x >= a) & (x <= b), 1/(b-a), 0)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean = {mean}')
plt.axvline(x=mean-std_dev, color='g', linestyle=':', label=f'μ ± σ')
plt.axvline(x=mean+std_dev, color='g', linestyle=':')
plt.title('Uniform Distribution with Mean and Standard Deviation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'moments.png'), dpi=100)
plt.close()

# Example 4: Simple Range Probability
print("\n\nExample 4: Simple Range Probability")
print("Step 1: Define the uniform distribution parameters")
a, b = 0, 10
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

print("\nStep 2: Define the range")
lower, upper = 2, 7
print(f"  Lower range bound = {lower}")
print(f"  Upper range bound = {upper}")

print("\nStep 3: Calculate the Probability")
print(f"  P({lower} ≤ X ≤ {upper}) = ({upper}-{lower})/({b}-{a})")
probability = (upper - lower) / (b - a)
print(f"  Probability = {probability:.2f}")

# Plot the probability
x = np.linspace(a-1, b+1, 1000)
pdf = np.where((x >= a) & (x <= b), 1/(b-a), 0)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, where=(x >= lower) & (x <= upper), alpha=0.3, color='green')
plt.title(f'P({lower} ≤ X ≤ {upper}) = {probability:.2f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'simple_range.png'), dpi=100)
plt.close()

# Example 5: Expected Value of a Function
print("\n\nExample 5: Expected Value of a Function")
print("Step 1: Define the uniform distribution parameters")
a, b = 0, 4
print(f"  Lower bound (a) = {a}")
print(f"  Upper bound (b) = {b}")

print("\nStep 2: Set up the Integral")
print(f"  E[X²] = ∫x²f(x)dx from {a} to {b}")
print(f"  f(x) = 1/({b}-{a}) = 1/{b-a}")
print(f"  E[X²] = ∫x²(1/{b-a})dx from {a} to {b}")

print("\nStep 3: Solve the Integral")
print(f"  = (1/{b-a}) * ∫x²dx from {a} to {b}")
print(f"  = (1/{b-a}) * [x³/3] from {a} to {b}")

print("\nStep 4: Evaluate the Integral")
expected_value = (b**3 - a**3) / (3 * (b - a))
print(f"  = (1/{b-a}) * ({b}³/3 - {a}³/3)")
print(f"  = {expected_value:.2f}")

# Plot the function and its expected value
x = np.linspace(a, b, 1000)
y = x**2
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
plt.axhline(y=expected_value, color='r', linestyle='--', label=f'E[X²] = {expected_value:.2f}')
plt.title(f'Expected Value of X² for Uniform({a},{b})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'expected_value.png'), dpi=100)
plt.close()

# Example 6: Transformation to Other Distributions
print("\n\nExample 6: Transformation to Other Distributions")
print("Step 1: Generate uniform random variables")
n_samples = 10000
u = np.random.uniform(0, 1, n_samples)
print(f"  Generated {n_samples} samples from Uniform(0,1)")

print("\nStep 2: Transform to Exponential Distribution")
lambda_exp = 2
print(f"  Using λ = {lambda_exp}")
print(f"  Transformation formula: X = -ln(1-U)/{lambda_exp}")
x_exp = -np.log(1-u)/lambda_exp

print("\nStep 3: Transform to Normal Distribution (Box-Muller)")
print("  Generate two independent uniform variables")
u1 = np.random.uniform(0, 1, n_samples)
u2 = np.random.uniform(0, 1, n_samples)
print("  Apply Box-Muller transform:")
print("  Z₁ = √(-2ln(U₁))cos(2πU₂)")
print("  Z₂ = √(-2ln(U₁))sin(2πU₂)")
z1 = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
z2 = np.sqrt(-2*np.log(u1)) * np.sin(2*np.pi*u2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(u, bins=50, density=True, alpha=0.7)
plt.title('Uniform(0,1) Samples')
plt.xlabel('x')
plt.ylabel('Density')

plt.subplot(1, 3, 2)
plt.hist(x_exp, bins=50, density=True, alpha=0.7)
x_exp_range = np.linspace(0, 5, 100)
plt.plot(x_exp_range, lambda_exp * np.exp(-lambda_exp * x_exp_range), 'r-')
plt.title('Exponential(λ=2) Samples')
plt.xlabel('x')
plt.ylabel('Density')

plt.subplot(1, 3, 3)
plt.hist(z1, bins=50, density=True, alpha=0.7)
x_norm = np.linspace(-4, 4, 100)
plt.plot(x_norm, stats.norm.pdf(x_norm), 'r-')
plt.title('Normal(0,1) Samples')
plt.xlabel('x')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'transformations.png'), dpi=100)
plt.close()

def f(x):
    return x**2

print("\nAll uniform distribution example images created successfully.") 