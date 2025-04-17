import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, binom
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Intro to the problem
print_step_header(0, "QUESTION 22: PROBABILITY FILL-IN-THE-BLANK")
print("In this question, we need to fill in the blanks with appropriate mathematical expressions or terms.")
print("Let's solve each part in detail with visualizations.")

# Part 1: Probability density function of a standard normal distribution
print_step_header(1, "THE PDF OF A STANDARD NORMAL DISTRIBUTION")

print("Question 1: The probability density function of a standard normal distribution is given by ________.")
print("\nSolution:")
print("The probability density function (PDF) of a standard normal distribution is given by:")
print("f(x) = (1/√(2π)) * e^(-(x²)/2)")
print("This is the standard normal distribution with mean μ = 0 and standard deviation σ = 1.")

# Create visualization for standard normal PDF
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, 0, 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, pdf, 'b-', linewidth=2)
ax.fill_between(x, 0, pdf, alpha=0.2, color='blue')

# Add the equation to the plot
equation = r"$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$"
ax.text(0.05, 0.9, equation, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Add vertical line at x=0 (mean)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Mean (μ = 0)')

# Annotate standard deviation
ax.arrow(0, 0.2, 1, 0, head_width=0.02, head_length=0.1, fc='r', ec='r')
ax.text(0.5, 0.22, 'σ = 1', color='r', ha='center', va='bottom')
ax.arrow(0, 0.2, -1, 0, head_width=0.02, head_length=0.1, fc='r', ec='r')
ax.text(-0.5, 0.22, 'σ = 1', color='r', ha='center', va='bottom')

# Indicate areas under the curve
ax.annotate('34.1%', xy=(-0.5, 0.1), xytext=(-0.5, 0.1), ha='center')
ax.annotate('34.1%', xy=(0.5, 0.1), xytext=(0.5, 0.1), ha='center')
ax.annotate('13.6%', xy=(-1.5, 0.05), xytext=(-1.5, 0.05), ha='center')
ax.annotate('13.6%', xy=(1.5, 0.05), xytext=(1.5, 0.05), ha='center')
ax.annotate('2.1%', xy=(-2.5, 0.02), xytext=(-2.5, 0.02), ha='center')
ax.annotate('2.1%', xy=(2.5, 0.02), xytext=(2.5, 0.02), ha='center')

ax.set_title('Standard Normal Distribution PDF', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability Density, f(x)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Save the figure
plt.tight_layout()
file_path = os.path.join(save_dir, "standard_normal_pdf.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Part 2: Variance of the sum of independent random variables
print_step_header(2, "VARIANCE OF THE SUM OF INDEPENDENT RANDOM VARIABLES")

print("Question 2: For two independent random variables X and Y, the variance of their sum X + Y equals ________.")
print("\nSolution:")
print("For two independent random variables X and Y, the variance of their sum X + Y equals:")
print("Var(X + Y) = Var(X) + Var(Y)")
print("This is a fundamental property when dealing with the sum of independent random variables.")

# Create visualization for variance of sum
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Symbolic representation
ax1.text(0.5, 0.5, r"$Var(X + Y) = Var(X) + Var(Y)$", 
         fontsize=16, ha='center', va='center')
ax1.text(0.5, 0.3, r"if $X$ and $Y$ are independent", 
         fontsize=12, ha='center', va='center', color='blue')
ax1.axis('off')

# Visual representation with normal distributions
np.random.seed(42)
X = np.random.normal(0, 1, 1000)  # mean=0, std=1
Y = np.random.normal(2, 1.5, 1000)  # mean=2, std=1.5
Z = X + Y  # Sum

# Plot histograms
ax2.hist(X, bins=30, alpha=0.5, label=r'$X \sim N(0,1)$')
ax2.hist(Y, bins=30, alpha=0.5, label=r'$Y \sim N(2,2.25)$')
ax2.set_title('Distributions of X and Y')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.legend()

ax3.hist(Z, bins=30, alpha=0.7, label=r'$Z = X + Y$')
ax3.set_title('Distribution of X + Y')
ax3.set_xlabel('Value')
ax3.legend()

# Annotations
var_x = np.var(X, ddof=0)
var_y = np.var(Y, ddof=0)
var_z = np.var(Z, ddof=0)
ax3.text(0.05, 0.9, f"Var(X) = {var_x:.2f}", transform=ax3.transAxes)
ax3.text(0.05, 0.85, f"Var(Y) = {var_y:.2f}", transform=ax3.transAxes)
ax3.text(0.05, 0.8, f"Var(X+Y) = {var_z:.2f}", transform=ax3.transAxes)
ax3.text(0.05, 0.75, f"Var(X) + Var(Y) = {var_x + var_y:.2f}", transform=ax3.transAxes)

plt.tight_layout()
file_path = os.path.join(save_dir, "variance_of_sum.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Part 3: Expected value and variance of a binomial distribution
print_step_header(3, "EXPECTED VALUE AND VARIANCE OF A BINOMIAL DISTRIBUTION")

print("Question 3: If X follows a binomial distribution with parameters n and p, then the expected value of X is ________ and its variance is ________.")
print("\nSolution:")
print("If X follows a binomial distribution with parameters n and p, then:")
print("Expected value of X: E[X] = n·p")
print("Variance of X: Var(X) = n·p·(1-p)")
print("These formulas are derived from the properties of the binomial distribution.")

# Create visualization for binomial distribution
n_values = [10, 20]
p_values = [0.3, 0.5, 0.7]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot PMFs for different n and p values
for i, n in enumerate(n_values):
    for j, p in enumerate(p_values):
        # Calculate the PMF
        x = np.arange(0, n+1)
        pmf = binom.pmf(x, n, p)
        
        # Expected value and variance
        expected = n * p
        variance = n * p * (1 - p)
        
        # Plot the PMF
        axes[i, j].bar(x, pmf, alpha=0.7)
        axes[i, j].axvline(x=expected, color='r', linestyle='--', 
                          label=f'E[X] = {expected:.1f}')
        
        # Add a span to visualize variance
        axes[i, j].axvspan(expected - np.sqrt(variance), 
                          expected + np.sqrt(variance), 
                          alpha=0.2, color='green',
                          label=f'Var(X) = {variance:.2f}')
        
        axes[i, j].set_title(f'Binomial(n={n}, p={p})')
        axes[i, j].set_xlabel('x')
        axes[i, j].set_ylabel('P(X=x)')
        axes[i, j].legend()

# Add formulas in the third column, third row
formula_ax = fig.add_subplot(2, 3, 6)
formula_ax.axis('off')
formula_ax.text(0.5, 0.7, r"$E[X] = n \cdot p$", fontsize=18, ha='center')
formula_ax.text(0.5, 0.5, r"$Var(X) = n \cdot p \cdot (1-p)$", fontsize=18, ha='center')
formula_ax.text(0.5, 0.3, "For X ~ Binomial(n, p)", fontsize=14, ha='center')

plt.tight_layout()
file_path = os.path.join(save_dir, "binomial_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Part 4: Expressing conditional probability using Bayes' theorem
print_step_header(4, "BAYES' THEOREM")

print("Question 4: The conditional probability P(A|B) can be expressed in terms of P(B|A) using ________ theorem.")
print("\nSolution:")
print("The conditional probability P(A|B) can be expressed in terms of P(B|A) using Bayes' theorem:")
print("P(A|B) = [P(B|A) × P(A)] / P(B)")
print("Bayes' theorem allows us to reverse conditional probabilities and is fundamental in Bayesian statistics.")

# Create visualization for Bayes' theorem
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)

# First subplot: Venn diagram representation
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_aspect('equal')

# Draw circles for events A and B
circle_a = plt.Circle((0.3, 0.5), 0.3, fill=False, edgecolor='blue', linewidth=2, label='A')
circle_b = plt.Circle((0.6, 0.5), 0.3, fill=False, edgecolor='red', linewidth=2, label='B')
ax1.add_patch(circle_a)
ax1.add_patch(circle_b)

# Shade the intersection
intersection_x = np.linspace(0.3, 0.6, 100)
intersection_y_upper = 0.5 + np.sqrt(0.3**2 - (intersection_x - 0.3)**2)
intersection_y_lower = 0.5 - np.sqrt(0.3**2 - (intersection_x - 0.3)**2)
ax1.fill_between(intersection_x, intersection_y_lower, intersection_y_upper, 
                color='purple', alpha=0.3, label='A ∩ B')

ax1.text(0.2, 0.5, 'A', fontsize=14, color='blue', ha='center', va='center')
ax1.text(0.7, 0.5, 'B', fontsize=14, color='red', ha='center', va='center')
ax1.text(0.45, 0.5, 'A ∩ B', fontsize=12, color='purple', ha='center', va='center')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('Events and Their Intersection')
ax1.axis('off')

# Second subplot: Bayes' theorem equation
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.7, r"Bayes' Theorem:", fontsize=16, ha='center', va='center')
ax2.text(0.5, 0.5, r"$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$", fontsize=20, ha='center', va='center')
ax2.text(0.5, 0.3, r"$P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)$", fontsize=16, ha='center', va='center')
ax2.axis('off')

# Third subplot: Example with medical testing
ax3 = fig.add_subplot(gs[1, :])

# Create a table of probabilities for a medical test example
disease_prevalence = 0.01  # Prior probability of disease P(D)
sensitivity = 0.95  # P(T|D)
specificity = 0.90  # P(T^c|D^c)

# Calculate P(D|T) using Bayes' theorem
false_positive_rate = 1 - specificity
p_positive = sensitivity * disease_prevalence + false_positive_rate * (1 - disease_prevalence)
p_disease_given_positive = (sensitivity * disease_prevalence) / p_positive

# Create a visual representation
ax3.text(0.5, 0.9, "Example: Medical Testing", fontsize=14, ha='center', va='center')
ax3.text(0.5, 0.8, "Disease (D) prevalence: 1%", fontsize=12, ha='center', va='center')
ax3.text(0.5, 0.7, "Test sensitivity P(T|D): 95%", fontsize=12, ha='center', va='center')
ax3.text(0.5, 0.6, "Test specificity P(T^c|D^c): 90%", fontsize=12, ha='center', va='center')
ax3.text(0.5, 0.5, "Using Bayes' Theorem:", fontsize=12, ha='center', va='center', weight='bold')
ax3.text(0.5, 0.4, r"$P(D|T) = \frac{P(T|D) \cdot P(D)}{P(T)}$", fontsize=14, ha='center', va='center')
ax3.text(0.5, 0.3, f"P(D|T) = {p_disease_given_positive:.4f} ≈ {p_disease_given_positive*100:.1f}%", 
        fontsize=14, ha='center', va='center', color='red')
ax3.text(0.5, 0.2, "Despite the high test accuracy, the probability of having the disease\ngiven a positive test is only about 9% due to the low prevalence.", 
        fontsize=12, ha='center', va='center')
ax3.axis('off')

plt.tight_layout()
file_path = os.path.join(save_dir, "bayes_theorem.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Part 5: Formula for covariance
print_step_header(5, "COVARIANCE FORMULA")

print("Question 5: The covariance between random variables X and Y can be calculated using the formula ________.")
print("\nSolution:")
print("The covariance between random variables X and Y can be calculated using the formula:")
print("Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]")
print("Covariance measures the joint variability of two random variables.")

# Create visualization for covariance
np.random.seed(42)
n = 100

# Create datasets with different correlations
means = [5, 7]
covs = [
    [[1, 0.8], [0.8, 1]],  # Positive correlation
    [[1, -0.8], [-0.8, 1]],  # Negative correlation
    [[1, 0], [0, 1]]  # No correlation
]

titles = ['Positive Covariance', 'Negative Covariance', 'Zero Covariance']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (cov_matrix, title) in enumerate(zip(covs, titles)):
    # Generate bivariate data
    data = np.random.multivariate_normal(means, cov_matrix, n)
    x = data[:, 0]
    y = data[:, 1]
    
    # Calculate means
    mean_x, mean_y = np.mean(x), np.mean(y)
    
    # Calculate covariance
    cov_xy = np.cov(x, y)[0, 1]
    
    # Plot the data
    axes[i].scatter(x, y, alpha=0.7)
    axes[i].axhline(y=mean_y, color='r', linestyle='--', alpha=0.3)
    axes[i].axvline(x=mean_x, color='r', linestyle='--', alpha=0.3)
    
    # Add means and covariance value
    axes[i].text(0.05, 0.95, f'E[X] = {mean_x:.2f}', transform=axes[i].transAxes)
    axes[i].text(0.05, 0.9, f'E[Y] = {mean_y:.2f}', transform=axes[i].transAxes)
    axes[i].text(0.05, 0.85, f'Cov(X,Y) = {cov_xy:.2f}', transform=axes[i].transAxes)
    
    axes[i].set_title(title)
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    
    # Add the covariance formula
    if i == 1:  # Add formula to middle plot
        axes[i].text(0.5, 0.1, r"$Cov(X,Y) = E[(X-E[X])(Y-E[Y])]$", 
                    transform=axes[i].transAxes, fontsize=12, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))
        axes[i].text(0.5, 0.03, r"$= E[XY] - E[X]E[Y]$", 
                    transform=axes[i].transAxes, fontsize=12, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "covariance_formula.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Summary of all answers
print_step_header(6, "SUMMARY OF ANSWERS")

print("Question 1: The probability density function of a standard normal distribution is given by:")
print("Answer: (1/√(2π)) * e^(-(x²)/2)")
print("\nQuestion 2: For two independent random variables X and Y, the variance of their sum X + Y equals:")
print("Answer: Var(X) + Var(Y)")
print("\nQuestion 3: If X follows a binomial distribution with parameters n and p, then the expected value of X is:")
print("Answer: n·p")
print("And its variance is:")
print("Answer: n·p·(1-p)")
print("\nQuestion 4: The conditional probability P(A|B) can be expressed in terms of P(B|A) using:")
print("Answer: Bayes'")
print("\nQuestion 5: The covariance between random variables X and Y can be calculated using the formula:")
print("Answer: E[(X - E[X])(Y - E[Y])] or E[XY] - E[X]E[Y]")

# Create a summary figure with all formulas
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

# Add title
ax.text(0.5, 0.95, "Question 22: Fill-in-the-Blank Summary", fontsize=20, ha='center', weight='bold')

formulas = [
    (0.9, "1. PDF of Standard Normal Distribution:", 
     r"$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$"),
    
    (0.75, "2. Variance of Sum of Independent Random Variables:", 
     r"$Var(X + Y) = Var(X) + Var(Y)$"),
    
    (0.6, "3a. Expected Value of Binomial Distribution:", 
     r"$E[X] = n \cdot p$"),
    
    (0.5, "3b. Variance of Binomial Distribution:", 
     r"$Var(X) = n \cdot p \cdot (1-p)$"),
    
    (0.35, "4. Bayes' Theorem:", 
     r"$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$"),
    
    (0.2, "5. Covariance Formula:", 
     r"$Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]$")
]

for y_pos, title, formula in formulas:
    ax.text(0.1, y_pos, title, fontsize=14)
    ax.text(0.15, y_pos-0.05, formula, fontsize=16)

plt.tight_layout()
file_path = os.path.join(save_dir, "fill_in_the_blank_summary.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nAll visualizations have been generated successfully!") 