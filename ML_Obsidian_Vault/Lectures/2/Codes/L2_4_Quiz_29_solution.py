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
print("with respective probabilities θₐ, θᵦ, θ𝒸, where θₐ + θᵦ + θ𝒸 = 1")

print("\nStep 1.2: Set up the likelihood function")
print("The likelihood function for a categorical distribution with multinomial counts is:")
print("L(θₐ, θᵦ, θ𝒸 | data) = (n choose nₐ,nᵦ,n𝒸) × θₐ^nₐ × θᵦ^nᵦ × θ𝒸^n𝒸")

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
print(f"L(θₐ, θᵦ, θ𝒸 | data) = {multi_coef:.3e} × θₐ^{counts[0]} × θᵦ^{counts[1]} × θ𝒸^{counts[2]}")

print("\nStep 1.3: Convert to log-likelihood for easier calculation")
print("log L(θₐ, θᵦ, θ𝒸 | data) = log(multinomial coef) + nₐlog(θₐ) + nᵦlog(θᵦ) + n𝒸log(θ𝒸)")

# Log-likelihood function
def log_likelihood(theta_a, theta_b, theta_c):
    """Calculate the log-likelihood for given probabilities"""
    return np.log(multi_coef) + counts[0] * np.log(theta_a) + counts[1] * np.log(theta_b) + counts[2] * np.log(theta_c)

# Test with our expected MLE values
expected_log_likelihood = log_likelihood(mle_probs[0], mle_probs[1], mle_probs[2])

print(f"\nSubstituting our values:")
print(f"log L(θₐ, θᵦ, θ𝒸 | data) = log({multi_coef:.3e}) + {counts[0]}×log(θₐ) + {counts[1]}×log(θᵦ) + {counts[2]}×log(θ𝒸)")
print(f"                         = {np.log(multi_coef):.4f} + {counts[0]}×log(θₐ) + {counts[1]}×log(θᵦ) + {counts[2]}×log(θ𝒸)")

print("\nStep 1.4: Maximize the log-likelihood using Lagrange multipliers")
print("We need to maximize log-likelihood subject to the constraint θₐ + θᵦ + θ𝒸 = 1")
print("Using Lagrange multipliers with L(θₐ, θᵦ, θ𝒸, λ) = log-likelihood - λ(θₐ + θᵦ + θ𝒸 - 1)")

print("\nTaking derivatives and setting them equal to zero:")
print("∂L/∂θₐ = nₐ/θₐ - λ = 0")
print("∂L/∂θᵦ = nᵦ/θᵦ - λ = 0")
print("∂L/∂θ𝒸 = n𝒸/θ𝒸 - λ = 0")
print("∂L/∂λ = θₐ + θᵦ + θ𝒸 - 1 = 0")

print("\nFrom the first three equations:")
print(f"θₐ = nₐ/λ = {counts[0]}/λ")
print(f"θᵦ = nᵦ/λ = {counts[1]}/λ")
print(f"θ𝒸 = n𝒸/λ = {counts[2]}/λ")

print("\nSubstituting into the constraint:")
print(f"θₐ + θᵦ + θ𝒸 = {counts[0]}/λ + {counts[1]}/λ + {counts[2]}/λ = {sum(counts)}/λ = 1")
print(f"Solving for λ: λ = {sum(counts)}")

print(f"\nTherefore:")
print(f"θₐ = {counts[0]}/{sum(counts)} = {counts[0]/sum(counts):.2f}")
print(f"θᵦ = {counts[1]}/{sum(counts)} = {counts[1]/sum(counts):.2f}")
print(f"θ𝒸 = {counts[2]}/{sum(counts)} = {counts[2]/sum(counts):.2f}")

print("\nStep 1.5: Verify our MLE solution")
print("For a categorical distribution, the MLE for each category probability is")
print("simply the proportion of observations in that category: θ̂ᵢ = nᵢ/n")

print(f"\nUsing this formula directly:")
print(f"θ̂ₐ = {counts[0]}/{sum(counts)} = {counts[0]/sum(counts):.2f}")
print(f"θ̂ᵦ = {counts[1]}/{sum(counts)} = {counts[1]/sum(counts):.2f}")
print(f"θ̂𝒸 = {counts[2]}/{sum(counts)} = {counts[2]/sum(counts):.2f}")

print(f"\nTherefore, the maximum likelihood estimate of the category distribution is:")
print(f"- P(A) = {mle_probs[0]}")
print(f"- P(B) = {mle_probs[1]}")
print(f"- P(C) = {mle_probs[2]}")

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

# Add constraint line θₐ + θᵦ + θ𝒸 = 1 which in 2D is θₐ + θᵦ = 1 - θ𝒸 which simplifies to θₐ + θᵦ = 1
constraint_x = np.linspace(0, 1, 100)
constraint_y = 1 - constraint_x
ax2.plot(constraint_x, constraint_y, 'r--', linewidth=2, label='Constraint: θₐ + θᵦ + θ𝒸 = 1')
ax2.legend(loc='upper right')

# Print the explanation for the Lagrangian rather than adding it to the plot
print("\nLagrangian method for MLE of categorical distribution:")
print("1. Lagrangian: L = 50log(θₐ) + 30log(θᵦ) + 20log(θ𝒸) - λ(θₐ + θᵦ + θ𝒸 - 1)")
print("\n2. Partial derivatives:")
print("   ∂L/∂θₐ = 50/θₐ - λ = 0")
print("   ∂L/∂θᵦ = 30/θᵦ - λ = 0")
print("   ∂L/∂θ𝒸 = 20/θ𝒸 - λ = 0") 
print("   ∂L/∂λ = θₐ + θᵦ + θ𝒸 - 1 = 0")
print("\n3. From first three equations:")
print("   θₐ = 50/λ, θᵦ = 30/λ, θ𝒸 = 20/λ")
print("   Substituting into constraint: 50/λ + 30/λ + 20/λ = 1")
print("   100/λ = 1, therefore λ = 100")
print("   θₐ = 50/100 = 0.5, θᵦ = 30/100 = 0.3, θ𝒸 = 20/100 = 0.2")

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

print("\nStep 2.2: Calculate each entropy term")

# Calculate entropy terms for each category
entropy_terms = []

for i, (cat, prob) in enumerate(zip(categories, mle_probs)):
    log_val = np.log2(prob)
    entropy_term = -prob * log_val
    entropy_terms.append(entropy_term)
    
    print(f"\nFor category {cat}:")
    print(f"-P({cat}) × log₂(P({cat})) = -({prob}) × log₂({prob})")
    print(f"log₂({prob}) = {log_val:.4f}")
    print(f"-P({cat}) × log₂(P({cat})) = -({prob}) × ({log_val:.4f}) = {entropy_term:.4f} bits")

print("\nStep 2.3: Sum all entropy terms to get the total entropy")
total_entropy = sum(entropy_terms)
print(f"H(X) = {' + '.join([f'{term:.4f}' for term in entropy_terms])}")
print(f"H(X) = {total_entropy:.4f} bits")

print(f"\nTherefore, the entropy of the MLE distribution is approximately {total_entropy:.4f} bits per example.")

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
for i, (bar, p, ent_term) in enumerate(zip(bars, mle_probs, entropy_terms)):
    height = bar.get_height()
    plt.text(i, height + 0.02, f'P={p:.2f}\n{ent_term:.4f} bits', 
            ha='center', va='bottom', fontweight='bold')

# Print formula steps rather than putting them in the figure
print("\nDetailed entropy calculation formula:")
print(f"H(X) = -[{mle_probs[0]}log₂({mle_probs[0]}) + {mle_probs[1]}log₂({mle_probs[1]}) + {mle_probs[2]}log₂({mle_probs[2]})]")
print(f"     = -{mle_probs[0]}×({np.log2(mle_probs[0]):.4f}) - {mle_probs[1]}×({np.log2(mle_probs[1]):.4f}) - {mle_probs[2]}×({np.log2(mle_probs[2]):.4f})")
print(f"     = {entropy_terms[0]:.4f} + {entropy_terms[1]:.4f} + {entropy_terms[2]:.4f}")
print(f"     = {total_entropy:.4f} bits")

# Add total entropy annotation with a horizontal line
plt.axhline(y=0.1, color='red', linestyle='--', linewidth=2)
plt.text(len(categories)/2 - 0.5, 0.05, f'Total Entropy: {total_entropy:.4f} bits', 
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

# Breakdown by category for expected counts if we sample from MLE distribution
expected_counts = [p * total_examples for p in mle_probs]

# Visualization of encoding schemes
fig_encoding = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig_encoding, height_ratios=[1, 1])

# Top left: One-hot encoding matrix
ax1 = fig_encoding.add_subplot(gs[0, 0])
onehot_matrix = np.array(list(scheme1_onehot.values()))
sns.heatmap(onehot_matrix, annot=True, cmap="Blues", cbar=False, 
           xticklabels=['Bit 1', 'Bit 2', 'Bit 3'], 
           yticklabels=categories, ax=ax1)
ax1.set_title('One-Hot Encoding Matrix')

# Top right: Binary encoding matrix
ax2 = fig_encoding.add_subplot(gs[0, 1])
binary_matrix = np.array(list(scheme2_binary.values()))
sns.heatmap(binary_matrix, annot=True, cmap="Blues", cbar=False, 
           xticklabels=['Bit 1', 'Bit 2'], 
           yticklabels=categories, ax=ax2)
ax2.set_title('Binary Encoding Matrix')

# Bottom: Comparison chart
ax3 = fig_encoding.add_subplot(gs[1, :])
schemes = ['One-Hot Encoding', 'Binary Encoding', 'Entropy Limit']
total_bits = [total_bits_onehot, total_bits_binary, total_entropy * total_examples]
bits_per_ex = [bits_per_example_onehot, bits_per_example_binary, total_entropy]
colors = ['#ff9999', '#66b3ff', '#99ff99']

bars = ax3.bar(schemes, total_bits, color=colors)
ax3.set_title('Encoding Efficiency Comparison')
ax3.set_ylabel('Total Bits')
ax3.set_ylim(0, max(total_bits) * 1.2)

# Add detailed text on each bar
for i, (bar, bits, bits_per) in enumerate(zip(bars, total_bits, bits_per_ex)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
            f"{bits:.1f} bits total\n{bits_per:.4f} bits/example", 
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

# Add arrow showing the reduction
reduction_bits = total_bits_onehot - total_bits_binary
reduction_percentage = (reduction_bits / total_bits_onehot) * 100

y_pos = (total_bits_onehot + total_bits_binary) / 2
ax3.annotate(
    f"{reduction_percentage:.1f}% reduction\n({reduction_bits} bits)", 
    xy=(0, y_pos),
    xytext=(1, y_pos),
    ha='center', va='center',
    arrowprops=dict(arrowstyle='<->', color='green', lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
)

plt.tight_layout()
save_figure(fig_encoding, "step3_encoding_comparison.png")

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
print(f"For our dataset: log L(θ | data) = {counts[0]} log θₐ + {counts[1]} log θᵦ + {counts[2]} log θ𝒸")

# Define the empirical distribution q based on observed data
empirical_dist = {cat: count/total_examples for cat, count in zip(categories, counts)}
print("\nLet's define the empirical distribution q based on our observed data:")
for cat, prob in empirical_dist.items():
    print(f"q({cat}) = {counts[categories.index(cat)]}/{total_examples} = {prob}")

print("\nWe can rewrite the log-likelihood as:")
print("log L(θ | data) = n × ∑ q(i) log θᵢ")
print(f"log L(θ | data) = {total_examples} × [{empirical_dist['A']} log θₐ + {empirical_dist['B']} log θᵦ + {empirical_dist['C']} log θ𝒸]")

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
ax1.plot(constraint_x, constraint_y, 'w--', linewidth=2, label='Constraint: θₐ + θᵦ + θ𝒸 = 1')
ax1.legend(loc='upper right')

# Print the explanation of relationship instead of putting it in the figure
print("\nRelationship between MLE and Cross-Entropy:")
print("1. Log-Likelihood Formula: log L(θ | data) = ∑ nᵢ log θᵢ")
print("2. Cross-Entropy Formula: H(q, θ) = -∑ q(i) log θᵢ")
print("3. Relationship: log L(θ | data) = -n × H(q, θ)")
print("   where q is the empirical distribution and n is the sample size")
print("\nTherefore, maximizing log-likelihood is equivalent to minimizing cross-entropy between the empirical and model distributions")

# Bottom: Simple title
ax2 = fig_cross_entropy.add_subplot(gs[1])
ax2.axis('off')  # Hide axes
ax2.text(0.5, 0.5, 'Cross-Entropy Minimization', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
save_figure(fig_cross_entropy, "step5_cross_entropy_relation.png")

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

print("\nStep 6.2: Asymptotic Normality")
print("\nAsymptotic normality states that as sample size increases, the distribution of the MLE")
print("approaches a normal distribution centered at the true parameter value.")

print("\nFor a categorical distribution, the asymptotic distribution of the MLE is:")
print("√n(θ̂ - θ) → N(0, Σ) as n → ∞")
print("where Σ is the covariance matrix with elements:")
print("Σᵢᵢ = θᵢ(1-θᵢ) and Σᵢⱼ = -θᵢθⱼ for i≠j")

# Calculate standard error for the largest sample
n_large = sample_sizes[-1]
mle_large = estimates[-1]
se_theta_a = np.sqrt(true_probs[0] * (1 - true_probs[0]) / n_large)
print(f"\nFor n = {n_large}, the standard error of θ̂ₐ is approximately:")
print(f"SE(θ̂ₐ) = √(θₐ(1-θₐ)/n) = √({true_probs[0]}×{1-true_probs[0]}/{n_large}) = {se_theta_a:.6f}")

# Create visualization for MLE properties
fig_properties = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=fig_properties)

# Top left: Consistency plot
ax1 = fig_properties.add_subplot(gs[0, 0])
for i, cat in enumerate(categories):
    ax1.plot(sample_sizes, [est[i] for est in estimates], 'o-', label=f'MLE for {cat}')
    ax1.axhline(y=true_probs[i], linestyle='--', color=f'C{i}', alpha=0.5, 
                label=f'True P({cat})={true_probs[i]}')

ax1.set_title('MLE Consistency: Convergence to True Values')
ax1.set_xlabel('Sample Size (n)')
ax1.set_ylabel('Probability Estimate')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Asymptotic normality - error distribution
ax2 = fig_properties.add_subplot(gs[0, 1])

# For several large sample sizes, plot the distribution of errors
large_sample_sizes = [100, 1000, 10000]
n_simulations = 1000
results = []

for n in large_sample_sizes:
    errors_a = []
    for _ in range(n_simulations):
        sample = np.random.choice(3, size=n, p=true_probs)
        counts = [np.sum(sample == i) for i in range(3)]
        mle = [count/n for count in counts]
        # Calculate scaled error for category A
        scaled_error = np.sqrt(n) * (mle[0] - true_probs[0])
        errors_a.append(scaled_error)
    results.append(errors_a)

# Plot distributions
for i, n in enumerate(large_sample_sizes):
    sns.kdeplot(results[i], label=f'n = {n}', ax=ax2)

# Plot theoretical normal distribution
x = np.linspace(-4, 4, 1000)
sigma = np.sqrt(true_probs[0] * (1 - true_probs[0]))
y = stats.norm.pdf(x, 0, sigma)
ax2.plot(x, y, 'k--', label='Asymptotic Normal')

ax2.set_title('Asymptotic Normality: Distribution of √n(θ̂ₐ - θₐ)')
ax2.set_xlabel('Scaled Error: √n(θ̂ₐ - θₐ)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Variance reduction with sample size
ax3 = fig_properties.add_subplot(gs[1, 0])

# Calculate theoretical standard errors for different sample sizes
sizes = np.logspace(1, 5, 100)
se_values = [np.sqrt(true_probs[0] * (1 - true_probs[0]) / n) for n in sizes]

ax3.loglog(sizes, se_values, 'b-', linewidth=2)
ax3.set_title('Standard Error Reduction with Sample Size')
ax3.set_xlabel('Sample Size (n)')
ax3.set_ylabel('Standard Error of θ̂ₐ')
ax3.grid(True, which="both", alpha=0.3)

# Add annotations
for n in [100, 1000, 10000]:
    se = np.sqrt(true_probs[0] * (1 - true_probs[0]) / n)
    ax3.plot(n, se, 'ro')
    ax3.annotate(f'n = {n}\nSE = {se:.4f}',
               (n, se), xytext=(10, 0),
               textcoords='offset points',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Add line showing 1/√n relationship
ax3.annotate('SE ∝ 1/√n',
           (sizes[len(sizes)//2], se_values[len(sizes)//4]),
           xytext=(20, -20), textcoords='offset points',
           arrowprops=dict(arrowstyle='->'),
           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))

# Print MLE properties explanation
print("\nMLE Properties for Categorical Distribution:")
print("\n1. Consistency:")
print("   • MLE converges to the true parameter values as sample size increases")
print("   • θ̂ → θ in probability as n → ∞")
print("   • Ensures reliable estimation with sufficient data")
print("\n2. Asymptotic Normality:")
print("   • For large n, the distribution of MLE is approximately normal")
print("   • √n(θ̂ - θ) → N(0, Σ) as n → ∞")
print("   • Allows construction of confidence intervals")
print("   • Standard error of θ̂ᵢ: SE(θ̂ᵢ) = √(θᵢ(1-θᵢ)/n)")
print("\n3. Efficiency:")
print("   • MLE achieves the Cramér-Rao lower bound asymptotically")
print("   • No consistent estimator has smaller asymptotic variance")
print("   • MLE is the most efficient estimator for large samples")

# Bottom right: Simple title
ax4 = fig_properties.add_subplot(gs[1, 1])
ax4.axis('off')
ax4.text(0.5, 0.5, 'Properties of Maximum Likelihood Estimation', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
save_figure(fig_properties, "step6_MLE_properties.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Summary of question 29 on Maximum Likelihood Estimation for categorical data with information theory analysis:")
print(f"1. MLE for categorical distribution: P(A)={mle_probs[0]}, P(B)={mle_probs[1]}, P(C)={mle_probs[2]}")
print(f"2. Entropy of MLE distribution: {total_entropy:.4f} bits per example")
print(f"3. Encoding efficiency: One-hot ({total_bits_onehot} bits) vs Binary ({total_bits_binary} bits)")
print(f"4. Binary encoding is {reduction_percentage:.2f}% more efficient than one-hot")
print("5. MLE is equivalent to minimizing cross-entropy between empirical and model distributions")
print("6. MLE properties: consistency and asymptotic normality ensure reliable estimation with sufficient data")

print("\nKey insights:")
print("- MLE provides a principled way to estimate probability distributions from data")
print("- The relationship with cross-entropy reveals the information-theoretic interpretation of MLE")
print("- Binary encoding offers significant efficiency advantages over one-hot encoding") 