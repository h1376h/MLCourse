import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os
from scipy import stats
from matplotlib.patches import FancyArrowPatch

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
})

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# Create directory for saving images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    plt.close(fig)

# Problem data
categories = ['A', 'B', 'C']
counts = [50, 30, 20]
total_examples = sum(counts)
probabilities = [count / total_examples for count in counts]

# Encoding schemes
scheme1_onehot = {
    'A': [1, 0, 0],
    'B': [0, 1, 0],
    'C': [0, 0, 1]
}

scheme2_binary = {
    'A': [0, 0],
    'B': [0, 1],
    'C': [1, 0]
}

# ==============================
# STEP 1: Derive the MLE for the Categorical Distribution
# ==============================
print_section_header("STEP 1: Derive the Maximum Likelihood Estimator")

print("Step 1.1: Define the probability distribution model")
print("We're modeling a categorical distribution with three categories (A, B, C)")
print("with respective probabilities θₐ, θᵦ, θc, where θₐ + θᵦ + θc = 1")

print("\nStep 1.2: Set up the likelihood function")
print("The likelihood function for a categorical distribution with multinomial counts is:")
print("L(θₐ, θᵦ, θc | data) = (n choose nₐ,nᵦ,n𝒸) × θₐ^nₐ × θᵦ^nᵦ × θc^n𝒸")

# Calculate multinomial coefficient
from math import factorial
def multinomial_coef(n, ks):
    """Calculate the multinomial coefficient for n objects into k groups"""
    if sum(ks) != n:
        raise ValueError("Sum of counts must equal total")
    numerator = factorial(n)
    denominator = np.prod([factorial(k) for k in ks])
    return numerator / denominator

multi_coef = multinomial_coef(total_examples, counts)

print(f"\nWhere:")
print(f"- n = {total_examples} (total examples)")
print(f"- nₐ = {counts[0]} (count of category A)")
print(f"- nᵦ = {counts[1]} (count of category B)")
print(f"- n𝒸 = {counts[2]} (count of category C)")
print(f"- (n choose nₐ,nᵦ,n𝒸) = {multi_coef:.3e}")

# Likelihood function
def likelihood(theta_a, theta_b, theta_c):
    """Calculate the likelihood for given probabilities"""
    return multi_coef * (theta_a ** counts[0]) * (theta_b ** counts[1]) * (theta_c ** counts[2])

# Test with our expected MLE values
mle_probs = probabilities
expected_likelihood = likelihood(mle_probs[0], mle_probs[1], mle_probs[2])

print(f"\nSubstituting our values:")
print(f"L(θₐ, θᵦ, θc | data) = {multi_coef:.3e} × θₐ^{counts[0]} × θᵦ^{counts[1]} × θc^{counts[2]}")

print("\nStep 1.3: Convert to log-likelihood for easier calculation")
print("log L(θₐ, θᵦ, θc | data) = log(multinomial coef) + nₐlog(θₐ) + nᵦlog(θᵦ) + n𝒸log(θc)")

# Log-likelihood function
def log_likelihood(theta_a, theta_b, theta_c):
    """Calculate the log-likelihood for given probabilities"""
    return np.log(multi_coef) + counts[0] * np.log(theta_a) + counts[1] * np.log(theta_b) + counts[2] * np.log(theta_c)

# Test with our expected MLE values
expected_log_likelihood = log_likelihood(mle_probs[0], mle_probs[1], mle_probs[2])

print(f"\nSubstituting our values:")
print(f"log L(θₐ, θᵦ, θc | data) = log({multi_coef:.3e}) + {counts[0]}×log(θₐ) + {counts[1]}×log(θᵦ) + {counts[2]}×log(θc)")
print(f"                         = {np.log(multi_coef):.4f} + {counts[0]}×log(θₐ) + {counts[1]}×log(θᵦ) + {counts[2]}×log(θc)")

print("\nStep 1.4: Maximize the log-likelihood using Lagrange multipliers")
print("We need to maximize log-likelihood subject to the constraint θₐ + θᵦ + θc = 1")
print("Using Lagrange multipliers with L(θₐ, θᵦ, θc, λ) = log-likelihood - λ(θₐ + θᵦ + θc - 1)")

print("\nTaking derivatives and setting them equal to zero:")
print("∂L/∂θₐ = nₐ/θₐ - λ = 0")
print("∂L/∂θᵦ = nᵦ/θᵦ - λ = 0")
print("∂L/∂θc = nc/θc - λ = 0")
print("∂L/∂λ = θₐ + θᵦ + θc - 1 = 0")

print("\nFrom the first three equations:")
print(f"θₐ = nₐ/λ = {counts[0]}/λ")
print(f"θᵦ = nᵦ/λ = {counts[1]}/λ")
print(f"θc = nc/λ = {counts[2]}/λ")

print("\nSubstituting into the constraint:")
print(f"θₐ + θᵦ + θc = {counts[0]}/λ + {counts[1]}/λ + {counts[2]}/λ = {sum(counts)}/λ = 1")
print(f"Solving for λ: λ = {sum(counts)}")

print(f"\nTherefore:")
print(f"θₐ = {counts[0]}/{sum(counts)} = {counts[0]/sum(counts):.2f}")
print(f"θᵦ = {counts[1]}/{sum(counts)} = {counts[1]/sum(counts):.2f}")
print(f"θc = {counts[2]}/{sum(counts)} = {counts[2]/sum(counts):.2f}")

print("\nStep 1.5: Verify our MLE solution")
print("For a categorical distribution, the MLE for each category probability is")
print("simply the proportion of observations in that category: θ̂ᵢ = nᵢ/n")

print(f"\nUsing this formula directly:")
print(f"θ̂ₐ = {counts[0]}/{sum(counts)} = {counts[0]/sum(counts):.2f}")
print(f"θ̂ᵦ = {counts[1]}/{sum(counts)} = {counts[1]/sum(counts):.2f}")
print(f"θ̂c = {counts[2]}/{sum(counts)} = {counts[2]/sum(counts):.2f}")

print(f"\nTherefore, the maximum likelihood estimate of the category distribution is:")
print(f"- P(A) = {mle_probs[0]}")
print(f"- P(B) = {mle_probs[1]}")
print(f"- P(C) = {mle_probs[2]}")

# Add detailed MLE derivation
print("\n=====================================================================")
print("DETAILED MLE DERIVATION (PEN AND PAPER STYLE)")
print("=====================================================================")
print("Given:")
print(f"- Data categories: {categories}")
# After the likelihood function definition, add detailed pen-and-paper style calculations

# After defining the multinomial coefficient
print("\nDetailed step-by-step MLE derivation (pen-and-paper style):")
print("\nStep 1: Understand the likelihood function structure")
print("The likelihood function represents the probability of observing our data given the parameters:")
print(f"L(θₐ, θᵦ, θc | data) = P(data | θₐ, θᵦ, θc)")
print(f"For a multinomial distribution with {counts[0]} A's, {counts[1]} B's, and {counts[2]} C's:")
print(f"L(θₐ, θᵦ, θc | data) = (multinomial coefficient) × P(A)^{counts[0]} × P(B)^{counts[1]} × P(C)^{counts[2]}")
print(f"L(θₐ, θᵦ, θc | data) = {multi_coef:.3e} × θₐ^{counts[0]} × θᵦ^{counts[1]} × θc^{counts[2]}")

print("\nStep 2: Convert to log-likelihood for easier maximization")
print("Taking the natural logarithm of both sides (which is monotonic, so maximizing log-likelihood is equivalent to maximizing likelihood):")
print(f"log L(θₐ, θᵦ, θc | data) = log({multi_coef:.3e}) + {counts[0]}log(θₐ) + {counts[1]}log(θᵦ) + {counts[2]}log(θc)")
print(f"log L(θₐ, θᵦ, θc | data) = {np.log(multi_coef):.4f} + {counts[0]}log(θₐ) + {counts[1]}log(θᵦ) + {counts[2]}log(θc)")
print("Note: The first term is a constant with respect to the parameters, so it doesn't affect the location of the maximum.")

print("\nStep 3: Incorporate the constraint using Lagrange multipliers")
print("We need to maximize log-likelihood subject to the constraint θₐ + θᵦ + θc = 1")
print("Set up the Lagrangian function:")
print(f"ℒ(θₐ, θᵦ, θc, λ) = {counts[0]}log(θₐ) + {counts[1]}log(θᵦ) + {counts[2]}log(θc) - λ(θₐ + θᵦ + θc - 1)")
print("where λ is the Lagrange multiplier.")

print("\nStep 4: Calculate partial derivatives and set them to zero")
print("For θₐ:")
print(f"∂ℒ/∂θₐ = {counts[0]}/θₐ - λ = 0")
print(f"Solving for θₐ: {counts[0]}/θₐ = λ")
print(f"θₐ = {counts[0]}/λ")

print("For θᵦ:")
print(f"∂ℒ/∂θᵦ = {counts[1]}/θᵦ - λ = 0")
print(f"Solving for θᵦ: {counts[1]}/θᵦ = λ")
print(f"θᵦ = {counts[1]}/λ")

print("For θc:")
print(f"∂ℒ/∂θc = {counts[2]}/θc - λ = 0")
print(f"Solving for θc: {counts[2]}/θc = λ")
print(f"θc = {counts[2]}/λ")

print("For λ (the constraint):")
print(f"∂ℒ/∂λ = -(θₐ + θᵦ + θc - 1) = 0")
print(f"θₐ + θᵦ + θc = 1")

print("\nStep 5: Solve the system of equations")
print("Substitute the expressions for θₐ, θᵦ, and θc into the constraint:")
print(f"θₐ + θᵦ + θc = {counts[0]}/λ + {counts[1]}/λ + {counts[2]}/λ = 1")
print(f"({counts[0]} + {counts[1]} + {counts[2]})/λ = 1")
print(f"{sum(counts)}/λ = 1")
print(f"λ = {sum(counts)}")

print("\nStep 6: Calculate the MLE values")
print(f"θₐ = {counts[0]}/λ = {counts[0]}/{sum(counts)} = {counts[0]/sum(counts)}")
print(f"θᵦ = {counts[1]}/λ = {counts[1]}/{sum(counts)} = {counts[1]/sum(counts)}")
print(f"θc = {counts[2]}/λ = {counts[2]}/{sum(counts)} = {counts[2]/sum(counts)}")

print("\nStep 7: Verify that the solution is indeed a maximum")
print("For a multinomial likelihood, the negative second derivatives of the log-likelihood are:")
print(f"∂²(log L)/∂θₐ² = -{counts[0]}/θₐ² < 0 (since both counts and probabilities are positive)")
print(f"∂²(log L)/∂θᵦ² = -{counts[1]}/θᵦ² < 0")
print(f"∂²(log L)/∂θc² = -{counts[2]}/θc² < 0")
print("Since these second derivatives are negative, the critical point is indeed a maximum.")

print("\nStep 8: Intuitive explanation of the MLE result")
print("The MLE for a categorical distribution turns out to be simply the proportion of each category in the sample.")
print(f"For category A: {counts[0]}/{total_examples} = {counts[0]/total_examples}")
print(f"For category B: {counts[1]}/{total_examples} = {counts[1]/total_examples}")
print(f"For category C: {counts[2]}/{total_examples} = {counts[2]/total_examples}")
print("This makes intuitive sense: the best estimate of the probability of a category is the frequency with which it occurred in the data.")

# Visualization for MLE derivation
fig_mle = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig_mle)

# Top left: Observed data histogram
ax1 = fig_mle.add_subplot(gs[0, 0])
bars = ax1.bar(categories, counts, color='skyblue', edgecolor='black')
ax1.set_title('Observed Data')
ax1.set_xlabel('Category')
ax1.set_ylabel('Count')
ax1.set_ylim(0, max(counts) * 1.2)

# Add count labels on top of bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'n = {count}', ha='center', va='bottom')

# Top right: Likelihood function visualization
ax2 = fig_mle.add_subplot(gs[0, 1])

# Create a contour plot of likelihood function
# We'll fix theta_c = 1 - theta_a - theta_b and plot in 2D
theta_a_range = np.linspace(0.01, 0.99, 100)
theta_b_range = np.linspace(0.01, 0.99, 100)
theta_a_grid, theta_b_grid = np.meshgrid(theta_a_range, theta_b_range)

# Calculate theta_c and mask invalid values
theta_c_grid = 1 - theta_a_grid - theta_b_grid
mask = (theta_c_grid > 0)  # Only keep points where all probabilities are positive

log_likelihood_grid = np.zeros_like(theta_a_grid)
for i in range(len(theta_a_range)):
    for j in range(len(theta_b_range)):
        if mask[j, i]:  # Use mask to filter valid points
            log_likelihood_grid[j, i] = log_likelihood(theta_a_grid[j, i], 
                                                    theta_b_grid[j, i], 
                                                    theta_c_grid[j, i])
        else:
            log_likelihood_grid[j, i] = np.nan

# Plot contour
contour = ax2.contourf(theta_a_grid, theta_b_grid, log_likelihood_grid, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2)
ax2.set_title('Log-Likelihood Function')
ax2.set_xlabel('θₐ (probability of A)')
ax2.set_ylabel('θᵦ (probability of B)')

# Mark the MLE point
ax2.scatter(mle_probs[0], mle_probs[1], color='red', s=100, marker='*', label='MLE')
ax2.annotate(f'MLE: ({mle_probs[0]}, {mle_probs[1]})',
           (mle_probs[0], mle_probs[1]), xytext=(10, -20),
           textcoords='offset points', color='red',
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

# Add constraint line θₐ + θᵦ + θc = 1 which in 2D is θₐ + θᵦ = 1 - θc which simplifies to θₐ + θᵦ = 1
constraint_x = np.linspace(0, 1, 100)
constraint_y = 1 - constraint_x
ax2.plot(constraint_x, constraint_y, 'r--', linewidth=2, label='Constraint: θₐ + θᵦ + θc = 1')
ax2.legend(loc='upper right')

# Print the explanation for the Lagrangian rather than adding it to the plot
print("\nLagrangian method for MLE of categorical distribution:")
print("1. Lagrangian: L = 50log(θₐ) + 30log(θᵦ) + 20log(θc) - λ(θₐ + θᵦ + θc - 1)")
print("\n2. Partial derivatives:")
print("   ∂L/∂θₐ = 50/θₐ - λ = 0")
print("   ∂L/∂θᵦ = 30/θᵦ - λ = 0")
print("   ∂L/∂θc = 20/θc - λ = 0") 
print("   ∂L/∂λ = θₐ + θᵦ + θc - 1 = 0")
print("\n3. From first three equations:")
print("   θₐ = 50/λ, θᵦ = 30/λ, θc = 20/λ")
print("   Substituting into constraint: 50/λ + 30/λ + 20/λ = 1")
print("   100/λ = 1, therefore λ = 100")
print("   θₐ = 50/100 = 0.5, θᵦ = 30/100 = 0.3, θc = 20/100 = 0.2")

# Add title to the visualization
fig_mle.suptitle('Maximum Likelihood Estimation for Categorical Distribution', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
save_figure(fig_mle, "step1_MLE_derivation.png")

# ==============================
# STEP 2: Calculate the Entropy of the MLE Distribution
# ==============================
print_section_header("STEP 2: Calculate the Entropy of the MLE Distribution")

print("Step 2.1: Apply the entropy formula")
print("H(X) = -∑ P(xᵢ)log₂(P(xᵢ))")

print("\nFor our MLE distribution:")
print(f"H(X) = -[P(A)log₂(P(A)) + P(B)log₂(P(B)) + P(C)log₂(P(C))]")
print(f"H(X) = -[{mle_probs[0]}log₂({mle_probs[0]}) + {mle_probs[1]}log₂({mle_probs[1]}) + {mle_probs[2]}log₂({mle_probs[2]})]")

print("\nDetailed step-by-step entropy calculation (pen-and-paper style):")
print("Entropy quantifies the average 'surprise' or uncertainty in a probability distribution.")
print("For a discrete distribution, entropy is calculated as: H(X) = -∑ P(x)log₂(P(x))")

print("\nStep 1: Identify the probability distribution")
print("Our MLE distribution (derived in previous step):")
for i, (cat, prob) in enumerate(zip(categories, mle_probs)):
    print(f"P({cat}) = {prob}")

print("\nStep 2: Calculate logarithm (base 2) for each probability")
log_values = []
for i, (cat, prob) in enumerate(zip(categories, mle_probs)):
    log_val = np.log2(prob)
    log_values.append(log_val)
    print(f"log₂(P({cat})) = log₂({prob})")
    
    # Show calculation for log base 2
    if cat == 'A':
        print(f"log₂(0.5) = log₂(1/2) = -log₂(2) = -1")
    elif cat == 'B':
        print(f"log₂(0.3) ≈ log₂(3/10)")
        print(f"Using log properties: log₂(3/10) = log₂(3) - log₂(10)")
        print(f"log₂(3) ≈ 1.585 and log₂(10) ≈ 3.322")
        print(f"So log₂(0.3) ≈ 1.585 - 3.322 ≈ -1.737")
    elif cat == 'C':
        print(f"log₂(0.2) = log₂(1/5) = -log₂(5)")
        print(f"log₂(5) ≈ 2.322")
        print(f"So log₂(0.2) ≈ -2.322")
    
    print(f"log₂(P({cat})) = {log_val:.4f}")

print("\nStep 3: Multiply each probability by its log value and negate")
for i, (cat, prob, log_val) in enumerate(zip(categories, mle_probs, log_values)):
    ent_term = -prob * log_val
    print(f"For category {cat}:")
    print(f"-P({cat})log₂(P({cat})) = -({prob}) × ({log_val:.4f})")
    print(f"                        = {ent_term:.4f} bits")

print("\nStep 4: Sum all terms to get total entropy")
total_entropy_manual = 0
entropy_terms = []
entropy_terms_str = []
for i, (cat, prob, log_val) in enumerate(zip(categories, mle_probs, log_values)):
    ent_term = -prob * log_val
    total_entropy_manual += ent_term
    entropy_terms.append(ent_term)
    entropy_terms_str.append(f"{ent_term:.4f}")

print(f"H(X) = {' + '.join(entropy_terms_str)}")
print(f"H(X) = {total_entropy_manual:.4f} bits")

print("\nStep 5: Interpret the result")
print(f"An entropy of {total_entropy_manual:.4f} bits means:")
print(f"1. This is the theoretical minimum number of bits needed to encode a symbol from this distribution")
print(f"2. The distribution has some uncertainty but isn't maximally uncertain (which would be log₂(3) ≈ 1.585 bits)")
print(f"3. Any lossless encoding of this distribution will require at least {total_entropy_manual:.4f} bits per symbol on average")
print(f"4. For fixed-length encoding, we would need at least ceil(log₂(3)) = 2 bits per symbol")

# Continue with the original entropy calculation code

# Visualization for entropy calculation
fig_entropy = plt.figure(figsize=(10, 6))

# Bar chart showing distribution and entropy contribution
bars = plt.bar(categories, mle_probs, color=sns.color_palette("pastel", 3), 
              alpha=0.8, width=0.6)

plt.title('MLE Distribution and Entropy Contribution')
plt.xlabel('Category')
plt.ylabel('Probability (MLE Estimate)')
plt.ylim(0, max(mle_probs) * 1.3)

# Add labels on each bar
for i, (bar, p, ent_term) in enumerate(zip(bars, mle_probs, entropy_terms_str)):
    height = bar.get_height()
    plt.text(i, height + 0.02, f'P={p:.2f}\n{ent_term} bits', 
            ha='center', va='bottom', fontweight='bold')

# Print formula steps rather than putting them in the figure
print("\nDetailed entropy calculation formula:")
print(f"H(X) = -[{mle_probs[0]}log₂({mle_probs[0]}) + {mle_probs[1]}log₂({mle_probs[1]}) + {mle_probs[2]}log₂({mle_probs[2]})]")
print(f"     = -{mle_probs[0]}×({np.log2(mle_probs[0]):.4f}) - {mle_probs[1]}×({np.log2(mle_probs[1]):.4f}) - {mle_probs[2]}×({np.log2(mle_probs[2]):.4f})")
print(f"     = {entropy_terms[0]:.4f} + {entropy_terms[1]:.4f} + {entropy_terms[2]:.4f}")
print(f"     = {total_entropy_manual:.4f} bits")

# Add total entropy annotation with a horizontal line
plt.axhline(y=0.1, color='red', linestyle='--', linewidth=2)
plt.text(len(categories)/2 - 0.5, 0.05, f'Total Entropy: {total_entropy_manual:.4f} bits', 
        ha='center', color='red', fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.8, pad=3))

plt.tight_layout()
save_figure(fig_entropy, "step2_entropy_calculation.png")

# ==============================
# STEP 3: Calculate Bits Required for Each Encoding Scheme
# ==============================
print_section_header("STEP 3: Calculate Bits Required for Each Encoding Scheme")

print("Step 3.1: Storage requirements for Scheme 1 (One-hot Encoding)")
print("\nOne-hot encoding scheme:")
for category, encoding in scheme1_onehot.items():
    print(f"- Category {category}: {encoding}")

bits_per_example_onehot = len(next(iter(scheme1_onehot.values())))
total_bits_onehot = bits_per_example_onehot * total_examples

print(f"\nFor one-hot encoding with {len(categories)} categories:")
print(f"- Bits per example = {bits_per_example_onehot}")
print(f"- For {total_examples} examples: Total bits = {bits_per_example_onehot} × {total_examples} = {total_bits_onehot} bits")

print("\nStep 3.2: Storage requirements for Scheme 2 (Binary Encoding)")
print("\nBinary encoding scheme:")
for category, encoding in scheme2_binary.items():
    print(f"- Category {category}: {encoding}")

bits_per_example_binary = len(next(iter(scheme2_binary.values())))
total_bits_binary = bits_per_example_binary * total_examples

print(f"\nFor binary encoding with {len(categories)} categories:")
print(f"- Bits per example = {bits_per_example_binary}")
print(f"- For {total_examples} examples: Total bits = {bits_per_example_binary} × {total_examples} = {total_bits_binary} bits")

# Add after the binary encoding requirements, before the visualization
print("\nDetailed analysis of encoding schemes (pen-and-paper style):")

print("\nStep 1: Analyze one-hot encoding")
print("One-hot encoding represents each category with a vector where exactly one position is 1 and all others are 0")
for cat, encoding in scheme1_onehot.items():
    print(f"Category {cat}: {encoding}")

print("\nTheoretical justification:")
print("For k distinct categories, we need k bits per example, one for each possible category")
print(f"With {len(categories)} categories, we need {bits_per_example_onehot} bits per example")

print("\nStep 2: Analyze binary encoding")
print("Binary encoding uses a more compact representation with a minimal number of bits")
for cat, encoding in scheme2_binary.items():
    print(f"Category {cat}: {encoding}")

print("\nTheoretical justification:")
print(f"For k distinct categories, we need ceil(log₂(k)) bits")
print(f"log₂({len(categories)}) = {np.log2(len(categories)):.4f}")
print(f"ceil(log₂({len(categories)})) = {np.ceil(np.log2(len(categories)))}")
print(f"Therefore, we need {bits_per_example_binary} bits per example")

print("\nStep 3: Calculate storage requirements for each encoding")
print("For one-hot encoding:")
category_bits_onehot = []
for cat, count in zip(categories, counts):
    bits = bits_per_example_onehot * count
    category_bits_onehot.append(bits)
    print(f"Category {cat}: {count} examples × {bits_per_example_onehot} bits = {bits} bits")
print(f"Total one-hot: {sum(category_bits_onehot)} bits")

print("\nFor binary encoding:")
category_bits_binary = []
for cat, count in zip(categories, counts):
    bits = bits_per_example_binary * count
    category_bits_binary.append(bits)
    print(f"Category {cat}: {count} examples × {bits_per_example_binary} bits = {bits} bits")
print(f"Total binary: {sum(category_bits_binary)} bits")

print("\nStep 4: Compare with theoretical limits")
entropy_bits = total_entropy_manual * total_examples
print(f"Theoretical minimum (entropy): {total_entropy_manual:.4f} bits/example × {total_examples} examples = {entropy_bits:.2f} bits")

print("\nOverhead analysis:")
onehot_overhead = bits_per_example_onehot - total_entropy_manual
onehot_overhead_percent = (onehot_overhead / total_entropy_manual) * 100
binary_overhead = bits_per_example_binary - total_entropy_manual
binary_overhead_percent = (binary_overhead / total_entropy_manual) * 100

print(f"One-hot overhead: {bits_per_example_onehot} - {total_entropy_manual:.4f} = {onehot_overhead:.4f} bits/example ({onehot_overhead_percent:.2f}%)")
print(f"Binary overhead: {bits_per_example_binary} - {total_entropy_manual:.4f} = {binary_overhead:.4f} bits/example ({binary_overhead_percent:.2f}%)")

print("\nStep 5: Analyze losslessness")
print("For an encoding to be lossless, each category must map to a unique code")

for cat, encoding in scheme1_onehot.items():
    print(f"One-hot: Category {cat} → {encoding}")
print("One-hot is lossless: each category has a unique representation")

for cat, encoding in scheme2_binary.items():
    print(f"Binary: Category {cat} → {encoding}")
print("Binary is lossless: each category has a unique representation")

print("\nStep 6: Calculate theoretical limits for variable-length codes")
print("Optimal variable-length codes (like Huffman coding) can approach the entropy limit")
print("For our distribution:")

expected_lengths = {}
for cat, prob in zip(categories, mle_probs):
    # Rough estimate of optimal code length based on information theory: -log₂(p)
    optimal_length = -np.log2(prob)
    expected_lengths[cat] = optimal_length
    print(f"Category {cat} (P = {prob}): Optimal length ≈ {optimal_length:.4f} bits")

avg_length = sum(p * expected_lengths[cat] for cat, p in zip(categories, mle_probs))
print(f"Expected average length: {avg_length:.4f} bits/example (equals entropy: {total_entropy_manual:.4f})")

print("\nIn practice, Huffman coding would give:")
# Simple Huffman-like code (not optimal but illustrative)
if mle_probs[0] >= 0.5:  # Most frequent gets shortest code
    huffman_codes = {'A': '0', 'B': '10', 'C': '11'}
else:
    huffman_codes = {'A': '00', 'B': '01', 'C': '1'}

huffman_lengths = {cat: len(code) for cat, code in huffman_codes.items()}
avg_huffman = sum(p * huffman_lengths[cat] for cat, p in zip(categories, mle_probs))

for cat, code in huffman_codes.items():
    prob = mle_probs[categories.index(cat)]
    print(f"Category {cat} (P = {prob}): Code '{code}', Length = {len(code)} bits")
print(f"Expected average length with Huffman: {avg_huffman:.4f} bits/example")
print(f"Overhead vs. entropy: {avg_huffman - total_entropy_manual:.4f} bits/example ({(avg_huffman - total_entropy_manual) / total_entropy_manual * 100:.2f}%)")

# Continue with the visualization code

# ==============================
# STEP 4: Compare the Efficiency of Both Encoding Schemes
# ==============================
print_section_header("STEP 4: Compare the Efficiency of Both Encoding Schemes")

print("Step 4.1: Calculate absolute bit savings")
reduction_bits = total_bits_onehot - total_bits_binary
print(f"Absolute bit savings = Bits_One-hot - Bits_Binary")
print(f"Absolute bit savings = {total_bits_onehot} - {total_bits_binary} = {reduction_bits} bits")

print("\nStep 4.2: Calculate percentage reduction")
reduction_percentage = (reduction_bits / total_bits_onehot) * 100
print(f"Percentage reduction = (Bits_One-hot - Bits_Binary) / Bits_One-hot × 100%")
print(f"Percentage reduction = ({total_bits_onehot} - {total_bits_binary}) / {total_bits_onehot} × 100%")
print(f"Percentage reduction = {reduction_bits} / {total_bits_onehot} × 100% = {reduction_percentage:.2f}%")

print(f"\nThis means binary encoding reduces storage requirements by {reduction_percentage:.2f}% compared to one-hot encoding.")

# ==============================
# STEP 5: Relate MLE to Cross-Entropy Minimization
# ==============================
print_section_header("STEP 5: Relate MLE to Cross-Entropy Minimization")

print("Step 5.1: Express the likelihood in terms of cross-entropy")
print("\nThe log-likelihood for a categorical distribution can be written as:")
print("log L(θ | data) = ∑ nᵢ log θᵢ")
print(f"For our dataset: log L(θ | data) = {counts[0]} log θₐ + {counts[1]} log θᵦ + {counts[2]} log θc")

# Define the empirical distribution q based on observed data
empirical_dist = {cat: count/total_examples for cat, count in zip(categories, counts)}
print("\nLet's define the empirical distribution q based on our observed data:")
for cat, prob in empirical_dist.items():
    print(f"q({cat}) = {counts[categories.index(cat)]}/{total_examples} = {prob}")

print("\nWe can rewrite the log-likelihood as:")
print("log L(θ | data) = n × ∑ q(i) log θᵢ")
print(f"log L(θ | data) = {total_examples} × [{empirical_dist['A']} log θₐ + {empirical_dist['B']} log θᵦ + {empirical_dist['C']} log θc]")

print("\nThe cross-entropy between distributions q and θ is defined as:")
print("H(q, θ) = -∑ q(i) log θᵢ")

print("\nTherefore:")
print("log L(θ | data) = -n × H(q, θ)")
print(f"log L(θ | data) = -{total_examples} × H(q, θ)")

print("\nStep 5.2: Explain the relationship")
print("\nWhen we perform MLE for a categorical distribution, we are finding the parameter values θ")
print("that minimize the cross-entropy between the empirical distribution of the observed data")
print("and our model distribution. This makes intuitive sense because:")
print("1. Cross-entropy measures the average number of bits needed to encode data from a true")
print("   distribution q using an estimated distribution θ")
print("2. Minimizing cross-entropy means finding the model that most efficiently encodes the observed data")
print("3. The most efficient encoding comes from the model that best matches the true data-generating process")

# Calculate cross-entropy for any theta distribution compared to empirical
def cross_entropy(q, theta):
    """Calculate cross-entropy between distributions q and theta"""
    return -sum(q[i] * np.log2(theta[i]) for i in range(len(q)))

# Create visualization for cross-entropy minimization
fig_cross_entropy = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 1, figure=fig_cross_entropy, height_ratios=[2, 1])

# Top plot: Cross-entropy landscape
ax1 = fig_cross_entropy.add_subplot(gs[0])

# We'll fix theta_c = 1 - theta_a - theta_b and plot in 2D as we did for likelihood
theta_a_range = np.linspace(0.01, 0.99, 100)
theta_b_range = np.linspace(0.01, 0.99, 100)
theta_a_grid, theta_b_grid = np.meshgrid(theta_a_range, theta_b_range)

# Calculate theta_c and mask invalid values
theta_c_grid = 1 - theta_a_grid - theta_b_grid
mask = (theta_c_grid > 0)  # Only keep points where all probabilities are positive

cross_entropy_grid = np.zeros_like(theta_a_grid)
for i in range(len(theta_a_range)):
    for j in range(len(theta_b_range)):
        if mask[j, i]:  # Use mask to filter valid points
            q_dist = [empirical_dist[cat] for cat in categories]
            theta_dist = [theta_a_grid[j, i], theta_b_grid[j, i], theta_c_grid[j, i]]
            cross_entropy_grid[j, i] = cross_entropy(q_dist, theta_dist)
        else:
            cross_entropy_grid[j, i] = np.nan

# Plot contour
contour = ax1.contourf(theta_a_grid, theta_b_grid, cross_entropy_grid, levels=20, cmap='plasma')
plt.colorbar(contour, ax=ax1, label='Cross-Entropy H(q, θ)')
ax1.set_title('Cross-Entropy Landscape')
ax1.set_xlabel('θₐ (probability of A)')
ax1.set_ylabel('θᵦ (probability of B)')

# Mark the MLE point (which minimizes cross-entropy)
ax1.scatter(mle_probs[0], mle_probs[1], color='lime', s=100, marker='*')
ax1.annotate(f'MLE: ({mle_probs[0]}, {mle_probs[1]})',
           (mle_probs[0], mle_probs[1]), xytext=(10, -20),
           textcoords='offset points', color='white',
           bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="lime", alpha=0.8))

# Add constraint line
constraint_x = np.linspace(0, 1, 100)
constraint_y = 1 - constraint_x
ax1.plot(constraint_x, constraint_y, 'w--', linewidth=2, label='Constraint: θₐ + θᵦ + θc = 1')
ax1.legend(loc='upper right')

# Print the explanation of relationship instead of putting it in the figure
print("\nDetailed derivation of MLE and cross-entropy relationship (pen-and-paper style):")

print("\nStep 1: Start with the log-likelihood function")
print(f"log L(θ | data) = {counts[0]}log(θₐ) + {counts[1]}log(θᵦ) + {counts[2]}log(θc)")

print("\nStep 2: Express counts in terms of empirical probabilities")
print(f"Let q(i) be the empirical probability of category i:")
for cat, prob in empirical_dist.items():
    print(f"q({cat}) = {counts[categories.index(cat)]}/{total_examples} = {prob}")

print("\nStep 3: Rewrite log-likelihood using empirical probabilities")
print(f"For each count nᵢ, we can write: nᵢ = n × q(i)")
print(f"Where n = {total_examples} is the total number of examples")
print(f"So n₁ = {total_examples} × q(A) = {total_examples} × {empirical_dist['A']} = {counts[0]}")
print(f"Similarly for n₂ and n₃")

print("\nStep 4: Substitute into log-likelihood")
print(f"log L(θ | data) = {total_examples} × q(A) × log(θₐ) + {total_examples} × q(B) × log(θᵦ) + {total_examples} × q(C) × log(θc)")
print(f"log L(θ | data) = {total_examples} × [q(A)log(θₐ) + q(B)log(θᵦ) + q(C)log(θc)]")
print(f"log L(θ | data) = {total_examples} × [Σᵢ q(i)log(θᵢ)]")

print("\nStep 5: Identify the cross-entropy term")
print("The cross-entropy between distributions q and θ is defined as:")
print("H(q, θ) = -Σᵢ q(i)log(θᵢ)")

print("\nStep 6: Express log-likelihood in terms of cross-entropy")
print("From steps 4 and 5:")
print("log L(θ | data) = n × [Σᵢ q(i)log(θᵢ)]")
print("                = n × [-(-Σᵢ q(i)log(θᵢ))]")
print("                = n × [-H(q, θ)]")
print("                = -n × H(q, θ)")

print("\nStep 7: Determine what maximizing log-likelihood means")
print("To maximize log-likelihood: maximize -n × H(q, θ)")
print("Since n is positive, this is equivalent to minimizing H(q, θ)")
print("Therefore, maximizing log-likelihood is equivalent to minimizing cross-entropy between")
print("the empirical distribution q and the model distribution θ")

print("\nStep 8: Verify with our specific distributions")
print("Our empirical distribution q:")
q_dist = [empirical_dist[cat] for cat in categories]
print(f"q = [{q_dist[0]}, {q_dist[1]}, {q_dist[2]}]")

print("\nCross-entropy if model θ exactly matches empirical distribution:")
theta_match = q_dist
ce_match = cross_entropy(q_dist, theta_match)
ll_match = -total_examples * ce_match
print(f"If θ = q = [{theta_match[0]}, {theta_match[1]}, {theta_match[2]}]:")
print(f"H(q, θ) = {ce_match:.4f} bits")
print(f"log L(θ | data) = -{total_examples} × {ce_match:.4f} = {ll_match:.4f}")

print("\nCross-entropy with some other distribution:")
theta_diff = [0.4, 0.4, 0.2]
ce_diff = cross_entropy(q_dist, theta_diff)
ll_diff = -total_examples * ce_diff
print(f"If θ = [{theta_diff[0]}, {theta_diff[1]}, {theta_diff[2]}]:")
print(f"H(q, θ) = {ce_diff:.4f} bits")
print(f"log L(θ | data) = -{total_examples} × {ce_diff:.4f} = {ll_diff:.4f}")

print(f"\nComparing log-likelihoods: {ll_match:.4f} vs {ll_diff:.4f}")
print(f"As expected, log-likelihood is higher (less negative) when θ matches q")

print("\nStep 9: Connect to information theory")
print("Cross-entropy H(q, θ) represents the average number of bits needed to encode")
print("data from distribution q using a code optimized for distribution θ")
print("The minimum cross-entropy occurs when θ = q, in which case H(q, θ) = H(q)")
print("Therefore, MLE finds the distribution θ that would be most efficient for encoding")
print("the observed data, in an information-theoretic sense")

# Continue with the original code for cross-entropy visualization

# ==============================
# STEP 6: Properties of MLE in the Context of Categorical Data
# ==============================
print_section_header("STEP 6: Properties of MLE for Categorical Data")

print("Step 6.1: Consistency")
print("\nConsistency means that as the sample size increases, the MLE converges in probability to the true parameter value.")
print("\nFor our categorical distribution:")
print("- With a small sample, the proportions might not reflect the true probabilities")
print("- As we increase the sample size, the MLE will get closer to the true distribution")
print("- In the limit as n→∞, the MLE will equal the true probabilities with probability 1")

# Let's demonstrate consistency with a simulation
print("\nDemonstration of consistency:")
# Assume a hypothetical true distribution
true_probs = [0.45, 0.35, 0.2]  # A bit different from our sample
print(f"Let's assume the true distribution is: P(A)={true_probs[0]}, P(B)={true_probs[1]}, P(C)={true_probs[2]}")

# Generate samples of different sizes from this distribution
np.random.seed(42)  # For reproducibility
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
samples = []
estimates = []

for n in sample_sizes:
    # Generate sample
    sample = np.random.choice(3, size=n, p=true_probs)
    samples.append(sample)
    
    # Calculate MLE
    counts = [np.sum(sample == i) for i in range(3)]
    mle = [count/n for count in counts]
    estimates.append(mle)
    
    print(f"n = {n}: MLE = [{mle[0]:.4f}, {mle[1]:.4f}, {mle[2]:.4f}]")

print("\nDetailed mathematical analysis of MLE properties (pen-and-paper style):")

print("\nProperty 1: Consistency - Mathematical Formulation")
print("A sequence of estimators θ̂ₙ is consistent if it converges in probability to the true parameter θ as n→∞:")
print("P(|θ̂ₙ - θ| > ε) → 0 as n → ∞, for any ε > 0")
print("For a categorical distribution with k categories, the MLE is:")
print("θ̂ᵢ = xᵢ/n, where xᵢ is the count of category i in a sample of size n")

print("\nProof sketch of consistency for categorical MLE:")
print("1. By the Law of Large Numbers, the sample proportion xᵢ/n converges in probability to the true probability θᵢ")
print("2. Since θ̂ᵢ = xᵢ/n, the MLE θ̂ᵢ also converges in probability to θᵢ")
print("3. This holds for all categories i = 1, 2, ..., k, so the entire parameter vector is consistent")

print("\nNumerical demonstration of consistency:")
print("True distribution:", [f"{p:.4f}" for p in true_probs])
for i, n in enumerate(sample_sizes):
    print(f"MLE with n = {n}: [{estimates[i][0]:.4f}, {estimates[i][1]:.4f}, {estimates[i][2]:.4f}]")
    errors = [abs(est - true) for est, true in zip(estimates[i], true_probs)]
    avg_error = sum(errors) / len(errors)
    print(f"Average absolute error: {avg_error:.4f}")

print("\nProperty 2: Asymptotic Normality - Mathematical Formulation")
print("For large sample sizes, the distribution of the MLE can be approximated by:")
print("√n(θ̂ - θ) → N(0, I(θ)⁻¹) as n → ∞")
print("where I(θ) is the Fisher Information Matrix")

print("\nFor a categorical distribution, the Fisher Information Matrix is diagonal with elements:")
print("I(θ)ᵢᵢ = n/θᵢ for i=1,2,...,k-1 (considering k-1 parameters due to the constraint Σᵢθᵢ = 1)")
print("The asymptotic variance of θ̂ᵢ is θᵢ(1-θᵢ)/n")

print("\nDerivation of asymptotic variance for categorical MLE:")
print("1. The log-likelihood for a multinomial/categorical distribution is:")
print("   l(θ) = constant + Σᵢ xᵢlog(θᵢ)")
print("2. The score function (first derivative) for each θᵢ is:")
print("   ∂l/∂θᵢ = xᵢ/θᵢ")
print("3. The Fisher information (negative expected second derivative) is:")
print("   I(θ)ᵢᵢ = E[-∂²l/∂θᵢ²] = E[xᵢ/θᵢ²] = n×θᵢ/θᵢ² = n/θᵢ")
print("4. Accounting for the constraint Σᵢθᵢ = 1, the asymptotic variance becomes:")
print("   Var(θ̂ᵢ) = θᵢ(1-θᵢ)/n")

print("\nNumerical demonstration of asymptotic normality:")
print("For category A with true probability", true_probs[0])
for n in [100, 1000, 10000]:
    se = np.sqrt(true_probs[0] * (1 - true_probs[0]) / n)
    print(f"n = {n}: Standard error = {se:.6f}")
    print(f"95% confidence interval: {true_probs[0] - 1.96*se:.6f} to {true_probs[0] + 1.96*se:.6f}")

# Calculate asymptotic covariance matrix for all parameters
n_large = 10000
cov_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if i == j:
            cov_matrix[i, j] = true_probs[i] * (1 - true_probs[i]) / n_large
        else:
            cov_matrix[i, j] = -true_probs[i] * true_probs[j] / n_large

print("\nFull asymptotic covariance matrix at n =", n_large)
for i in range(3):
    print(f"[{cov_matrix[i][0]:.8f}, {cov_matrix[i][1]:.8f}, {cov_matrix[i][2]:.8f}]")

print("\nProperty 3: Efficiency - Mathematical Formulation")
print("An estimator is efficient if it achieves the Cramér-Rao lower bound:")
print("Var(θ̂) ≥ 1/I(θ)")
print("where I(θ) is the Fisher Information")

print("\nFor a categorical distribution:")
print("1. The Cramér-Rao lower bound for Var(θ̂ᵢ) is θᵢ(1-θᵢ)/n")
print("2. The MLE θ̂ᵢ = xᵢ/n has Var(θ̂ᵢ) = θᵢ(1-θᵢ)/n")
print("3. Since the variance equals the lower bound, the MLE is efficient")

print("\nEfficiency comparison:")
print("Consider two estimators for the probability of category A:")
print("1. MLE: θ̂ₐ = xₐ/n")
print("2. Alternative estimator: θ̃ₐ = (xₐ+1)/(n+3) (a shrinkage estimator)")

n_test = 1000
var_mle = true_probs[0] * (1 - true_probs[0]) / n_test
bias_mle = 0  # MLE is unbiased
mse_mle = var_mle + bias_mle**2

# Calculate MSE for alternative estimator
bias_alt = (true_probs[0]*n_test + 1)/(n_test + 3) - true_probs[0]
var_alt = (true_probs[0]*(1-true_probs[0])*n_test)/(n_test+3)**2
mse_alt = var_alt + bias_alt**2

print(f"\nFor n = {n_test}:")
print(f"MLE estimator: Variance = {var_mle:.8f}, Bias = {bias_mle:.8f}, MSE = {mse_mle:.8f}")
print(f"Alternative estimator: Variance = {var_alt:.8f}, Bias = {bias_alt:.8f}, MSE = {mse_alt:.8f}")
print(f"Efficiency ratio: {mse_mle/mse_alt:.8f}")
if mse_mle < mse_alt:
    print("MLE is more efficient than the alternative estimator")
else:
    print("Alternative estimator is more efficient for this sample size due to bias-variance tradeoff")
    print("However, MLE becomes asymptotically efficient as n → ∞")

# Continue with the original MLE properties visualization

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Summary of question 29 on Maximum Likelihood Estimation for categorical data with information theory analysis:")
print(f"1. MLE for categorical distribution: P(A)={mle_probs[0]}, P(B)={mle_probs[1]}, P(C)={mle_probs[2]}")
print(f"2. Entropy of MLE distribution: {total_entropy_manual:.4f} bits per example")
print(f"3. Encoding efficiency: One-hot ({total_bits_onehot} bits) vs Binary ({total_bits_binary} bits)")
print(f"4. Binary encoding is {reduction_percentage:.2f}% more efficient than one-hot")
print("5. MLE is equivalent to minimizing cross-entropy between empirical and model distributions")
print("6. MLE properties: consistency and asymptotic normality ensure reliable estimation with sufficient data")

print("\nKey insights:")
print("- MLE provides a principled way to estimate probability distributions from data")
print("- The relationship with cross-entropy reveals the information-theoretic interpretation of MLE")
print("- Binary encoding offers significant efficiency advantages over one-hot encoding") 