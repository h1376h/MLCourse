import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 3: For any continuous random variable X, P(X = a) = 0 for any specific value a.")

# Image 1: Continuous PDF and zooming in on a point
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Plot normal PDF
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
ax1.plot(x, y, 'b-', linewidth=2)

# Pick a specific point
a = 1.0

# Mark the specific point
ax1.axvline(x=a, color='red', linestyle='--', linewidth=1.5)
ax1.plot(a, stats.norm.pdf(a), 'ro', markersize=6)

# Add labels
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Probability Density f(x)', fontsize=12)
ax1.set_title('Standard Normal Distribution PDF', fontsize=14)

# Save the figure
continuous_img_path1 = os.path.join(save_dir, "statement3_normal_pdf.png")
plt.savefig(continuous_img_path1, dpi=300, bbox_inches='tight')
plt.close()

# Image 2: Shrinking intervals visualization
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

# Plot PDF on the left
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
ax2.plot(x, y, 'b-', linewidth=2)

# Pick a specific point and intervals
a = 1.0
widths = [1.0, 0.5, 0.2, 0.1]
colors = ['purple', 'green', 'orange', 'red']

# Show different interval widths
for i, width in enumerate(widths):
    # Calculate probability for the interval
    prob = stats.norm.cdf(a + width/2) - stats.norm.cdf(a - width/2)
    
    # Fill the area
    interval = np.linspace(a - width/2, a + width/2, 100)
    ax2.fill_between(interval, 0, stats.norm.pdf(interval), color=colors[i], alpha=0.4)

# Mark the specific point
ax2.axvline(x=a, color='red', linestyle='--', linewidth=1.5)
ax2.plot(a, stats.norm.pdf(a), 'ro', markersize=6)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Probability Density f(x)', fontsize=12)
ax2.set_title('Intervals Around x = 1', fontsize=14)

# Plot shrinking probabilities on the right
interval_widths = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])
probs = []

for width in interval_widths:
    prob = stats.norm.cdf(a + width/2) - stats.norm.cdf(a - width/2)
    probs.append(prob)

# Create log-log plot of probabilities
ax3.loglog(interval_widths, probs, 'bo-', markersize=6)
ax3.set_xlabel('Interval Width', fontsize=12)
ax3.set_ylabel('Probability P(a-w/2 < X < a+w/2)', fontsize=12)
ax3.set_title('Probability Approaches Zero as Interval Shrinks', fontsize=14)
ax3.grid(True, which="both", ls="-")

plt.tight_layout()
# Save the figure
continuous_img_path2 = os.path.join(save_dir, "statement3_shrinking_intervals.png")
plt.savefig(continuous_img_path2, dpi=300, bbox_inches='tight')
plt.close()

# Image 3: Comparison of discrete vs continuous probability
fig3, axes = plt.subplots(2, 1, figsize=(10, 8))

# Discrete distribution (binomial)
n, p = 10, 0.3
x_discrete = np.arange(0, n+1)
pmf = stats.binom.pmf(x_discrete, n, p)

axes[0].bar(x_discrete, pmf, width=0.8, alpha=0.7, color='blue')
axes[0].set_xlim(-0.5, n+0.5)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('Probability Mass P(X = x)', fontsize=12)
axes[0].set_title('Discrete Distribution (Binomial)', fontsize=14)

# Highlight a specific point
specific_point = 3
axes[0].bar([specific_point], [pmf[specific_point]], width=0.8, alpha=0.9, color='red')
axes[0].text(specific_point, pmf[specific_point] + 0.01, f'P(X = {specific_point}) = {pmf[specific_point]:.4f}', 
             ha='center', fontsize=10)

# Continuous distribution (normal)
x_continuous = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x_continuous)
axes[1].plot(x_continuous, pdf, 'g-', linewidth=2)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('Probability Density f(x)', fontsize=12)
axes[1].set_title('Continuous Distribution (Normal)', fontsize=14)

# Highlight a specific point in continuous 
specific_point = 1.0
axes[1].plot([specific_point], [stats.norm.pdf(specific_point)], 'ro', markersize=6)
axes[1].text(specific_point, stats.norm.pdf(specific_point) + 0.02, 
             f'P(X = {specific_point}) = 0, f({specific_point}) = {stats.norm.pdf(specific_point):.4f}', 
             ha='center', fontsize=10)

plt.tight_layout()
# Save the figure
continuous_img_path3 = os.path.join(save_dir, "statement3_discrete_vs_continuous.png")
plt.savefig(continuous_img_path3, dpi=300, bbox_inches='tight')
plt.close()

# Explanation for continuous probabilities - Print to terminal
print("#### Mathematical Analysis of Continuous Probability")
print("For a continuous random variable X with probability density function f(x),")
print("probabilities are calculated as integrals over intervals:")
print("P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx")
print("")
print(f"#### Numerical Demonstration with a Standard Normal at x = {a}:")
widths = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
print("As we narrow the interval width around point a = 1:")
for width in widths:
    prob = stats.norm.cdf(a + width/2) - stats.norm.cdf(a - width/2)
    print(f"  P({a-width/2:.6f} < X < {a+width/2:.6f}) = {prob:.10f}")
print("")
print("#### Key Properties of Continuous Distributions:")
print("1. Probability is measured over intervals, not at individual points")
print("2. As the interval width approaches zero, the probability approaches zero")
print("3. P(X = a) corresponds to the integral over a single point, which has measure zero")
print("4. The probability density f(a) may be positive, even though P(X = a) = 0")
print("5. This differs from discrete distributions where P(X = a) can be positive")
print("")
print("#### Contrast with Discrete Distributions:")
print(f"Binomial example with n=10, p=0.3:")
print(f"  P(X = 3) = {pmf[3]:.6f} > 0")
print(f"Normal distribution at the same point:")
print(f"  P(X = 3) = 0, but f(3) = {stats.norm.pdf(3):.6f}")
print("")
print("#### Visual Verification:")
print(f"1. Standard normal distribution: {continuous_img_path1}")
print(f"2. Shrinking intervals around x=1: {continuous_img_path2}")
print(f"3. Comparison of discrete vs continuous: {continuous_img_path3}")
print("")
print("#### Conclusion:")
print("For continuous random variables, the probability of any specific value is zero,")
print("even though the probability density at that point may be positive.")
print("")
print("Therefore, Statement 3 is TRUE.") 