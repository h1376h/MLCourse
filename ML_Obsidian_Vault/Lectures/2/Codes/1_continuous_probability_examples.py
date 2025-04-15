import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

print("\n=== CONTINUOUS PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Continuous Random Variable with Given PDF
print("Example 1: Continuous Random Variable with Given PDF")
print("A machine learning algorithm produces prediction errors with the following probability density function:")
print("f(x) = 0.2 for -2 ≤ x ≤ 3, and 0 otherwise")
print("What is the probability that a randomly selected prediction has an error between -1 and 2?")

# Define the PDF function
def pdf_example1(x):
    if -2 <= x <= 3:
        return 0.2
    else:
        return 0

# Verify the PDF integrates to 1
x_range = np.linspace(-3, 4, 1000)
pdf_values = [pdf_example1(x) for x in x_range]
area = np.trapz(pdf_values, x_range)
print(f"\nVerifying the PDF integrates to 1: {area:.4f}")

# Calculate the probability
a, b = -1, 2
x_prob = np.linspace(a, b, 1000)
pdf_prob = [pdf_example1(x) for x in x_prob]
probability = np.trapz(pdf_prob, x_prob)
print(f"\nStep-by-step calculation:")
print(f"1. The PDF is constant (0.2) in the interval [-2, 3]")
print(f"2. The probability is the area under the PDF between -1 and 2")
print(f"3. For a constant PDF, the area is the height × width")
print(f"4. Height = 0.2, Width = 2 - (-1) = 3")
print(f"5. Probability = 0.2 × 3 = {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(x_range, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_prob, pdf_prob, alpha=0.3, color='blue')
plt.axvline(x=a, color='r', linestyle='--')
plt.axvline(x=b, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.xlabel('Error Value (x)', fontsize=12)
plt.ylabel('Probability Density f(x)', fontsize=12)
plt.title('Continuous Random Variable PDF', fontsize=14)
plt.ylim(0, 0.25)

# Add probability annotation
plt.annotate(f'P(-1 ≤ X ≤ 2) = {probability:.4f}', 
            xy=(0.5, 0.15),
            xytext=(0.5, 0.15),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'continuous_example1.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Continuous Random Variable with Linear PDF
print("\n\nExample 2: Continuous Random Variable with Linear PDF")
print("The lifetime of a certain electronic component has the following probability density function:")
print("f(x) = 0.04x for 0 ≤ x ≤ 5, and 0 otherwise")
print("What is the probability that a randomly selected component will last between 2 and 4 years?")

# Define the PDF function
def pdf_example2(x):
    if 0 <= x <= 5:
        return 0.04 * x
    else:
        return 0

# Verify the PDF integrates to 1
x_range = np.linspace(-1, 6, 1000)
pdf_values = [pdf_example2(x) for x in x_range]
area = np.trapz(pdf_values, x_range)
print(f"\nVerifying the PDF integrates to 1: {area:.4f}")

# Calculate the probability
a, b = 2, 4
x_prob = np.linspace(a, b, 1000)
pdf_prob = [pdf_example2(x) for x in x_prob]
probability = np.trapz(pdf_prob, x_prob)
print(f"\nStep-by-step calculation:")
print(f"1. The PDF is linear (0.04x) in the interval [0, 5]")
print(f"2. The probability is the area under the PDF between 2 and 4")
print(f"3. For a linear PDF, the area is the integral of 0.04x from 2 to 4")
print(f"4. ∫(0.04x)dx = 0.02x² + C")
print(f"5. Area = 0.02(4²) - 0.02(2²) = 0.32 - 0.08 = {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(x_range, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_prob, pdf_prob, alpha=0.3, color='blue')
plt.axvline(x=a, color='r', linestyle='--')
plt.axvline(x=b, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.xlabel('Lifetime (years)', fontsize=12)
plt.ylabel('Probability Density f(x)', fontsize=12)
plt.title('Component Lifetime PDF', fontsize=14)
plt.ylim(0, 0.25)

# Add probability annotation
plt.annotate(f'P(2 ≤ X ≤ 4) = {probability:.4f}', 
            xy=(3, 0.15),
            xytext=(3, 0.15),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'continuous_example2.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Continuous Random Variable with Quadratic PDF
print("\n\nExample 3: Continuous Random Variable with Quadratic PDF")
print("A machine learning algorithm randomly initializes weights with the following probability density function:")
print("f(x) = 0.75(1-x²) for -1 ≤ x ≤ 1, and 0 otherwise")
print("What is the probability that a randomly selected weight is between -0.5 and 0.7?")

# Define the PDF function
def pdf_example3(x):
    if -1 <= x <= 1:
        return 0.75 * (1 - x**2)
    else:
        return 0

# Verify the PDF integrates to 1
x_range = np.linspace(-1.5, 1.5, 1000)
pdf_values = [pdf_example3(x) for x in x_range]
area = np.trapz(pdf_values, x_range)
print(f"\nVerifying the PDF integrates to 1: {area:.4f}")

# Calculate the probability
a, b = -0.5, 0.7
x_prob = np.linspace(a, b, 1000)
pdf_prob = [pdf_example3(x) for x in x_prob]
probability = np.trapz(pdf_prob, x_prob)
print(f"\nStep-by-step calculation:")
print(f"1. The PDF is quadratic (0.75(1-x²)) in the interval [-1, 1]")
print(f"2. The probability is the area under the PDF between -0.5 and 0.7")
print(f"3. For a quadratic PDF, the area is the integral of 0.75(1-x²) from -0.5 to 0.7")
print(f"4. ∫(0.75(1-x²))dx = 0.75(x - x³/3) + C")
print(f"5. Area = 0.75(0.7 - 0.7³/3) - 0.75(-0.5 - (-0.5)³/3)")
print(f"6. Area = 0.75(0.7 - 0.1143) - 0.75(-0.5 - (-0.0417))")
print(f"7. Area = 0.75(0.5857) - 0.75(-0.4583)")
print(f"8. Area = 0.4393 + 0.3437 = {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(x_range, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_prob, pdf_prob, alpha=0.3, color='blue')
plt.axvline(x=a, color='r', linestyle='--')
plt.axvline(x=b, color='r', linestyle='--')
plt.grid(True, alpha=0.3)
plt.xlabel('Weight Value (x)', fontsize=12)
plt.ylabel('Probability Density f(x)', fontsize=12)
plt.title('Weight Initialization PDF', fontsize=14)
plt.ylim(0, 0.8)

# Add probability annotation
plt.annotate(f'P(-0.5 ≤ X ≤ 0.7) = {probability:.4f}', 
            xy=(0.1, 0.6),
            xytext=(0.1, 0.6),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'continuous_example3.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Continuous Random Variable with Piecewise PDF
print("\n\nExample 4: Continuous Random Variable with Piecewise PDF")
print("A data scientist is analyzing the time it takes to train a model. The training time has the following probability density function:")
print("f(x) = 0.5 for 0 ≤ x ≤ 1")
print("f(x) = 0.25 for 1 < x ≤ 3")
print("f(x) = 0 otherwise")
print("What is the probability that a randomly selected training takes between 0.5 and 2 hours?")

# Define the PDF function
def pdf_example4(x):
    if 0 <= x <= 1:
        return 0.5
    elif 1 < x <= 3:
        return 0.25
    else:
        return 0

# Verify the PDF integrates to 1
x_range = np.linspace(-1, 4, 1000)
pdf_values = [pdf_example4(x) for x in x_range]
area = np.trapz(pdf_values, x_range)
print(f"\nVerifying the PDF integrates to 1: {area:.4f}")

# Calculate the probability
a, b = 0.5, 2
x_prob = np.linspace(a, b, 1000)
pdf_prob = [pdf_example4(x) for x in x_prob]
probability = np.trapz(pdf_prob, x_prob)
print(f"\nStep-by-step calculation:")
print(f"1. The PDF is piecewise constant in the intervals [0, 1] and (1, 3]")
print(f"2. The probability is the area under the PDF between 0.5 and 2")
print(f"3. We need to split the calculation at x = 1:")
print(f"   - Area from 0.5 to 1: 0.5 × (1 - 0.5) = 0.25")
print(f"   - Area from 1 to 2: 0.25 × (2 - 1) = 0.25")
print(f"4. Total probability = 0.25 + 0.25 = {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(x_range, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_prob, pdf_prob, alpha=0.3, color='blue')
plt.axvline(x=a, color='r', linestyle='--')
plt.axvline(x=b, color='r', linestyle='--')
plt.axvline(x=1, color='g', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Training Time (hours)', fontsize=12)
plt.ylabel('Probability Density f(x)', fontsize=12)
plt.title('Model Training Time PDF', fontsize=14)
plt.ylim(0, 0.6)

# Add probability annotation
plt.annotate(f'P(0.5 ≤ X ≤ 2) = {probability:.4f}', 
            xy=(1.25, 0.4),
            xytext=(1.25, 0.4),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'continuous_example4.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll continuous probability example images created successfully.") 