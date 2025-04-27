import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

# Create directory for saving images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
    plt.close(fig)

# Problem data
categories = ['Billing Issues', 'Technical Problems', 'Account Access', 'Feature Requests', 'General Inquiries']
short_categories = ['Billing', 'Technical', 'Account', 'Feature', 'General']  # For plots
training_counts = [100, 150, 80, 70, 100]
model_probabilities = [0.15, 0.35, 0.20, 0.10, 0.20]
threshold = 0.30
true_label = [0, 1, 0, 0, 0]  # One-hot encoded label for "Technical Problems"
prior_technical = 0.60  # Given prior that 60% of tickets are technical issues

# ==============================
# STEP 1: Relationship Between One-Hot Encoding and Multinomial Distribution
# ==============================
print_section_header("STEP 1: Relationship Between One-Hot Encoding and Multinomial Distribution")

print("One-hot encoding represents categorical variables as binary vectors where only one element is 1 (hot),")
print("and all others are 0 (cold). This encoding establishes a direct relationship with the multinomial")
print("distribution, which is used to model the probability of observing counts across multiple categories.")

print("\nIn this problem with 5 categories, the one-hot encodings are:")
for i, category in enumerate(categories):
    encoding = ["0"] * len(categories)
    encoding[i] = "1"
    print(f"{category.ljust(20)}: [{', '.join(encoding)}]")

print("\nDetailed explanation of the relationship:")
print("1. Multinomial Distribution Fundamentals:")
print("   - The multinomial distribution models the probability of observing specific counts across k categories")
print("   - Probability mass function: P(X₁=n₁, X₂=n₂, ..., X_k=n_k) = (n! / (n₁! × n₂! × ... × n_k!)) × p₁^n₁ × p₂^n₂ × ... × p_k^n_k")
print("   - Where n = n₁ + n₂ + ... + n_k is the total number of trials")
print("   - And p₁, p₂, ..., p_k are the probabilities for each category (∑p_i = 1)")

print("\n2. Connection to One-Hot Encoding:")
print("   - Each support ticket is one trial in the multinomial distribution")
print("   - One-hot encoding vectors [1,0,0,0,0], [0,1,0,0,0], etc. represent which category a ticket belongs to")
print("   - The position of the '1' in each vector corresponds to a specific category")
print("   - The sum of all positions equals 1 (mutual exclusivity property)")

print("\n3. Mathematical Connection:")
print("   - For a single ticket, let y = [y₁, y₂, ..., y₅] be the one-hot encoded vector")
print("   - Only one y_i equals 1, and all others are 0")
print("   - Let p = [p₁, p₂, ..., p₅] be the predicted probabilities from the model")
print("   - The probability of observing y is: P(y) = p₁^y₁ × p₂^y₂ × ... × p₅^y₅ = p_i (where y_i = 1)")
print("   - For multiple tickets, the joint probability follows the multinomial distribution")

print("\n4. Connection to Logistic Regression:")
print("   - Multinomial logistic regression uses softmax activation to output probabilities:")
print("     P(category i) = exp(z_i) / ∑exp(z_j) where z_i are the model's raw outputs")
print("   - These probabilities sum to 1 across all categories")
print("   - The model's output directly represents the parameters of the multinomial distribution")
print("   - Cross-entropy loss (-∑y_i log(p_i)) is derived from the negative log-likelihood of the multinomial distribution")

# Create visualization for one-hot encoding with more detail
fig1 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 1.2])

# Top-left: One-hot encoding visualization
ax1 = fig1.add_subplot(gs[0, 0])
one_hot_matrix = np.eye(len(categories))
sns.heatmap(one_hot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=[f'Bit {i+1}' for i in range(len(categories))], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding Matrix')
ax1.set_xlabel('Position')
ax1.set_ylabel('Category')

# Top-right: Multinomial distribution visualization
ax2 = fig1.add_subplot(gs[0, 1])
proportions = np.array(training_counts) / sum(training_counts)
colors = sns.color_palette("pastel", len(categories))
bars = ax2.bar(categories, proportions, color=colors)
ax2.set_title('Category Distribution in Training Data')
ax2.set_xlabel('Category')
ax2.set_ylabel('Proportion')
ax2.set_xticklabels(short_categories, rotation=45)
# Add detailed labels
for bar, prop, count in zip(bars, proportions, training_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, 
            f'{count}/{sum(training_counts)}\n{prop:.3f}', 
            ha='center', va='center', color='black', fontweight='bold')

# Bottom: Diagram showing connection between one-hot and multinomial
ax3 = fig1.add_subplot(gs[1, :])
ax3.axis('off')

# Title for connection diagram
ax3.text(0.5, 0.95, "Connection between One-Hot Encoding and Multinomial Distribution", 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Create a conceptual diagram
y_positions = [0.75, 0.6, 0.45, 0.3, 0.15]

# Draw example for each category
for i, (cat, y_pos) in enumerate(zip(categories, y_positions)):
    # One-hot vector
    one_hot = ["0"] * len(categories)
    one_hot[i] = "1"
    
    # Draw one-hot vector
    ax3.text(0.15, y_pos, f"{cat}:", ha='right', va='center', fontweight='bold')
    ax3.text(0.25, y_pos, f"[{', '.join(one_hot)}]", ha='left', va='center',
             bbox=dict(facecolor=colors[i], alpha=0.3, boxstyle='round'))
    
    # Draw arrow
    ax3.annotate('', xy=(0.45, y_pos), xytext=(0.35, y_pos),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Draw probability
    p_i = proportions[i]
    ax3.text(0.55, y_pos, f"p_{i+1} = {p_i:.3f}", ha='center', va='center',
             bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round'))

# Add multinomial formula
ax3.text(0.8, 0.5, "Multinomial PMF:\nP(X₁=n₁,...,X₅=n₅) =\n$\\frac{n!}{n₁!...n₅!}p₁^{n₁}...p₅^{n₅}$", 
         ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig1, "step1_one_hot_multinomial.png")

# ==============================
# STEP 2: MLE for Prior Probabilities
# ==============================
print_section_header("STEP 2: MLE for Prior Probabilities")

print("The Maximum Likelihood Estimate (MLE) for the prior probabilities of each category")
print("is the proportion of training examples that belong to each category. We'll derive this")
print("step-by-step using the multinomial likelihood function.")

# Calculate MLE for prior probabilities with detailed steps
total_training = sum(training_counts)
print(f"\nTotal number of training tickets: {total_training}")

print("\nDetailed derivation of the MLE for multinomial parameters:")
print("1. Likelihood Function:")
print("   For the multinomial distribution with k=5 categories, we have:")
print("   L(p₁, p₂, ..., p₅) = (n! / (n₁! × n₂! × ... × n₅!)) × p₁^n₁ × p₂^n₂ × ... × p₅^n₅")
print("   where n = n₁ + n₂ + ... + n₅ is the total number of tickets")

print("\n2. Log-Likelihood Function (easier to work with):")
print("   ln(L) = ln(n! / (n₁! × n₂! × ... × n₅!)) + n₁ln(p₁) + n₂ln(p₂) + ... + n₅ln(p₅)")
print("   The first term is constant with respect to p_i, so we can focus on:")
print("   ln(L) ∝ n₁ln(p₁) + n₂ln(p₂) + ... + n₅ln(p₅)")

print("\n3. Constraint: The probabilities must sum to 1")
print("   p₁ + p₂ + ... + p₅ = 1")

print("\n4. Lagrangian Formulation:")
print("   We need to maximize ln(L) subject to the constraint.")
print("   Lagrangian: ℒ = n₁ln(p₁) + n₂ln(p₂) + ... + n₅ln(p₅) - λ(p₁ + p₂ + ... + p₅ - 1)")

print("\n5. Taking Partial Derivatives:")
print("   ∂ℒ/∂p_i = n_i/p_i - λ")
print("   Setting all ∂ℒ/∂p_i = 0:")
print("   n_i/p_i = λ  for all i")
print("   p_i = n_i/λ  for all i")

print("\n6. Using the Constraint:")
print("   p₁ + p₂ + ... + p₅ = 1")
print("   n₁/λ + n₂/λ + ... + n₅/λ = 1")
print("   (n₁ + n₂ + ... + n₅)/λ = 1")
print("   n/λ = 1")
print("   λ = n (where n is the total count)")

print("\n7. Substituting Back:")
print("   p_i = n_i/λ = n_i/n")

print("\nCalculating MLE prior probability for each category:")
prior_probabilities = []
for i, category in enumerate(categories):
    prior = training_counts[i] / total_training
    prior_probabilities.append(prior)
    print(f"\n{category}:")
    print(f"  Count (n_{i+1}): {training_counts[i]}")
    print(f"  Total count (n): {total_training}")
    print(f"  MLE calculation: p_{i+1} = n_{i+1}/n = {training_counts[i]} / {total_training} = {prior:.4f}")
    print(f"  Percentage: {prior*100:.1f}%")

# Verify sum equals 1
print(f"\nVerification: Sum of all probabilities = {sum(prior_probabilities):.4f} (should equal 1)")

# Visualization of prior probabilities - Enhanced
fig2 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 1])

# Top-left: Pie chart of training data distribution
ax1 = fig2.add_subplot(gs[0, 0])
wedges, texts, autotexts = ax1.pie(training_counts, labels=short_categories, autopct='%1.1f%%', 
        startangle=90, colors=colors)
ax1.set_title('Training Data Distribution')
# Enhance the pie chart text
for autotext in autotexts:
    autotext.set_fontweight('bold')

# Top-right: Bar chart of prior probabilities
ax2 = fig2.add_subplot(gs[0, 1])
bars = ax2.bar(short_categories, prior_probabilities, color=colors)
ax2.set_title('MLE Prior Probabilities')
ax2.set_xlabel('Category')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, max(prior_probabilities) * 1.2)
# Add detailed labels
for bar, prob, count in zip(bars, prior_probabilities, training_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, 
            f'{count}/{total_training}\n{prob:.4f}', 
            ha='center', va='center', color='black', fontweight='bold')

# Bottom: Formula and explanation
ax3 = fig2.add_subplot(gs[1, :])
ax3.axis('off')

# Title
ax3.text(0.5, 0.95, "MLE Derivation for Multinomial Distribution", 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Create a step-by-step explanation with formulas
steps = [
    "1. Likelihood: $L(p_1,...,p_5) = \\frac{n!}{n_1!...n_5!} p_1^{n_1}...p_5^{n_5}$",
    "2. Log-likelihood: $\\ln(L) \\propto n_1\\ln(p_1) + ... + n_5\\ln(p_5)$",
    "3. Constraint: $p_1 + p_2 + ... + p_5 = 1$",
    "4. Partial derivatives: $\\frac{\\partial \\ln(L)}{\\partial p_i} = \\frac{n_i}{p_i} - \\lambda$",
    "5. Set to zero: $\\frac{n_i}{p_i} = \\lambda$ for all $i$",
    "6. Solve for $\\lambda$: $\\lambda = n = n_1 + ... + n_5 = 500$",
    f"7. MLE result: $p_i = \\frac{{n_i}}{{n}} = \\frac{{n_i}}{{{total_training}}}$"
]

for i, step in enumerate(steps):
    y_pos = 0.85 - 0.12*i
    ax3.text(0.1, y_pos, step, ha='left', va='center', fontsize=12)

# Example calculation for one category
example_idx = 1  # Technical Problems
example_text = f"Example calculation for {categories[example_idx]}:\n" \
               f"$p_{example_idx+1} = \\frac{{n_{example_idx+1}}}{{n}} = " \
               f"\\frac{{{training_counts[example_idx]}}}{{{total_training}}} = " \
               f"{prior_probabilities[example_idx]:.4f}$"
               
ax3.text(0.7, 0.5, example_text, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor=colors[example_idx], alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig2, "step2_prior_probabilities.png")

# ==============================
# STEP 3: Bayes' Theorem for Posterior Probability
# ==============================
print_section_header("STEP 3: Bayes' Theorem for Posterior Probability")

print("We need to calculate the posterior probability that the ticket belongs to 'Technical Problems'")
print("given that we know 60% of all support tickets are about technical issues. This requires using")
print("Bayes' theorem to combine the model's prediction with this prior knowledge.")

print("\nBayes' theorem formal definition:")
print("P(A|B) = P(B|A) × P(A) / P(B)")
print("In our context:")
print("P(Technical|Model) = P(Model|Technical) × P(Technical) / P(Model)")

# Define variables for clarity
model_prob_technical = model_probabilities[1]  # P(Model predicts Technical | Ticket)
prior_technical = 0.60  # P(Technical) = 0.60
prior_non_technical = 1 - prior_technical  # P(Non-Technical) = 0.40

print("\nDetailed step-by-step calculation:")

print("Step 1: Identify the values we know")
print(f"   • Model probability for Technical Problems: {model_prob_technical:.2f}")
print(f"     This is P(Model|Technical) - the likelihood of the model giving this output for a technical issue")
print(f"   • Prior probability P(Technical): {prior_technical:.2f}")
print(f"     This is our prior knowledge that 60% of tickets are technical issues")
print(f"   • Prior probability P(Non-Technical): {prior_non_technical:.2f} = 1 - {prior_technical:.2f}")
print(f"     All other categories combined")

print("\nStep 2: Calculate the model's behavior for non-technical tickets")
print("   We need P(Model|Non-Technical) - how the model behaves for non-technical tickets")
print("   Non-technical probabilities from model output:")
non_technical_probs = [model_probabilities[i] for i in range(len(model_probabilities)) if i != 1]
print(f"   Non-technical categories: {[categories[i] for i in range(len(categories)) if i != 1]}")
print(f"   Associated probabilities: {non_technical_probs}")
total_non_technical_prob = sum(non_technical_probs)
p_model_given_non_technical = total_non_technical_prob / (len(categories) - 1)  # Average across non-technical classes

print(f"   Sum of non-technical probabilities: {non_technical_probs[0]} + {non_technical_probs[1]} + {non_technical_probs[2]} + {non_technical_probs[3]} = {total_non_technical_prob:.4f}")
print(f"   Average across {len(categories)-1} non-technical categories: {total_non_technical_prob:.4f} / {len(categories)-1} = {p_model_given_non_technical:.4f}")
print(f"   This gives us P(Model|Non-Technical) = {p_model_given_non_technical:.4f}")

print("\nStep 3: Calculate P(Model) using the law of total probability")
print("   P(Model) = P(Model|Technical) × P(Technical) + P(Model|Non-Technical) × P(Non-Technical)")
p_model_tech_term = model_prob_technical * prior_technical
p_model_nontech_term = p_model_given_non_technical * prior_non_technical
p_model = p_model_tech_term + p_model_nontech_term

print(f"   P(Model) = {model_prob_technical:.2f} × {prior_technical:.2f} + {p_model_given_non_technical:.4f} × {prior_non_technical:.2f}")
print(f"   P(Model) = {p_model_tech_term:.4f} + {p_model_nontech_term:.4f}")
print(f"   P(Model) = {p_model:.4f}")

print("\nStep 4: Apply Bayes' theorem to calculate the posterior probability")
print("   P(Technical|Model) = P(Model|Technical) × P(Technical) / P(Model)")
posterior_technical = (model_prob_technical * prior_technical) / p_model
print(f"   P(Technical|Model) = ({model_prob_technical:.2f} × {prior_technical:.2f}) / {p_model:.4f}")
print(f"   P(Technical|Model) = {p_model_tech_term:.4f} / {p_model:.4f}")
print(f"   P(Technical|Model) = {posterior_technical:.4f} or {posterior_technical*100:.1f}%")

print("\nStep 5: Interpret the result")
print(f"   • Original model probability for Technical Problems: {model_prob_technical:.2f} or {model_prob_technical*100:.1f}%")
print(f"   • Posterior probability after incorporating prior knowledge: {posterior_technical:.4f} or {posterior_technical*100:.1f}%")
print(f"   • This represents a {(posterior_technical-model_prob_technical)*100:.1f} percentage point increase")
print(f"   • Or a {((posterior_technical/model_prob_technical)-1)*100:.1f}% relative increase in probability")
print("   • This demonstrates the significant impact of incorporating prior knowledge via Bayes' theorem")

# Alternative binary approach for comparison
print("\nAlternative approach (treating as binary classification - Technical vs. Non-Technical):")
# In binary classification, we have:
# P(Technical|Model) = P(Model|Technical)P(Technical) / [P(Model|Technical)P(Technical) + P(Model|Non-Technical)P(Non-Technical)]
binary_model_nontech = 1 - model_prob_technical  # Probability of model assigning to any non-technical category
binary_posterior = (model_prob_technical * prior_technical) / (model_prob_technical * prior_technical + 
                                             binary_model_nontech * prior_non_technical)
print(f"   If we treat this as a binary problem (Technical vs. Non-Technical):")
print(f"   • P(Model|Technical) = {model_prob_technical:.2f} (probability model assigns to Technical)")
print(f"   • P(Model|Non-Technical) = {binary_model_nontech:.2f} = 1 - {model_prob_technical:.2f} (probability model assigns to any non-technical class)")
print(f"   • P(Technical|Model) = {model_prob_technical:.2f}×{prior_technical:.2f} / [{model_prob_technical:.2f}×{prior_technical:.2f} + {binary_model_nontech:.2f}×{prior_non_technical:.2f}]")
print(f"   • P(Technical|Model) = {model_prob_technical*prior_technical:.4f} / [{model_prob_technical*prior_technical:.4f} + {binary_model_nontech*prior_non_technical:.4f}]")
print(f"   • P(Technical|Model) = {model_prob_technical*prior_technical:.4f} / {model_prob_technical*prior_technical + binary_model_nontech*prior_non_technical:.4f}")
print(f"   • P(Technical|Model) = {binary_posterior:.4f} or {binary_posterior*100:.1f}%")
print(f"   • Note: This differs from our first approach because we're treating the problem differently")

# Enhanced visualization of Bayes' theorem application
fig3 = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# Top-left: Prior probabilities comparison
ax1 = fig3.add_subplot(gs[0, 0])
prior_labels = ['Training Data Prior', 'Known Population Prior']
prior_values = [prior_probabilities[1], prior_technical]
bars = ax1.bar(prior_labels, prior_values, color=['lightblue', 'lightgreen'])
ax1.set_title('Prior Probability for Technical Problems')
ax1.set_ylabel('Probability')
ax1.set_ylim(0, max(prior_values) * 1.2)
# Add detailed labels
for bar, val, label in zip(bars, prior_values, prior_labels):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.4f}\n({val*100:.1f}%)', 
            ha='center', va='center', color='black', fontweight='bold')

# Top-right: Likelihood visualization
ax2 = fig3.add_subplot(gs[0, 1])
likelihood_labels = ['P(Model|Technical)', 'P(Model|Non-Technical)']
likelihood_values = [model_prob_technical, p_model_given_non_technical]
bars = ax2.bar(likelihood_labels, likelihood_values, color=['salmon', 'lightsalmon'])
ax2.set_title('Model Likelihood')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, max(likelihood_values) * 1.2)
# Add detailed labels
for bar, val in zip(bars, likelihood_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.4f}', 
            ha='center', va='center', color='black', fontweight='bold')

# Middle-left: Evidence calculation
ax3 = fig3.add_subplot(gs[1, 0])
evidence_labels = ['P(Technical)×P(Model|Technical)', 'P(Non-Technical)×P(Model|Non-Technical)', 'P(Model) (Sum)']
evidence_values = [p_model_tech_term, p_model_nontech_term, p_model]
bars = ax3.bar(evidence_labels, evidence_values, color=['lightblue', 'lightsalmon', 'purple'])
ax3.set_title('Evidence Calculation')
ax3.set_ylabel('Probability')
ax3.set_xticklabels(evidence_labels, rotation=45, ha='right')
ax3.set_ylim(0, max(evidence_values) * 1.2)
# Add detailed labels
for bar, val in zip(bars, evidence_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.4f}', 
            ha='center', va='center', color='black', fontweight='bold')

# Middle-right: Result comparison
ax4 = fig3.add_subplot(gs[1, 1])
result_labels = ['Model Output', 'Posterior Probability']
result_values = [model_prob_technical, posterior_technical]
bars = ax4.bar(result_labels, result_values, color=['lightgray', 'lightgreen'])
ax4.set_title('Probability Comparison')
ax4.set_ylabel('Probability')
ax4.set_ylim(0, max(result_values) * 1.2)
# Add detailed labels
for bar, val in zip(bars, result_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.4f}\n({val*100:.1f}%)', 
            ha='center', va='center', color='black', fontweight='bold')

# Bottom: Bayes' theorem formula and application
ax5 = fig3.add_subplot(gs[2, :])
ax5.axis('off')

# Create a visual flow of Bayes' theorem calculation
ax5.text(0.5, 0.95, "Bayes' Theorem Calculation", ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Formula
formula = r"$P(\text{Technical}|\text{Model}) = \frac{P(\text{Model}|\text{Technical}) \times P(\text{Technical})}{P(\text{Model})}$"
ax5.text(0.5, 0.85, formula, ha='center', va='center', fontsize=12)

# Values
values = [
    f"P(Model|Technical) = {model_prob_technical:.2f}",
    f"P(Technical) = {prior_technical:.2f}",
    f"P(Model) = {p_model:.4f}",
    f"P(Technical|Model) = {posterior_technical:.4f}"
]

positions = [0.2, 0.4, 0.6, 0.8]
colors = ['salmon', 'lightgreen', 'lightblue', 'purple']

for i, (text, pos, color) in enumerate(zip(values, positions, colors)):
    ax5.text(pos, 0.7, text, ha='center', va='center', fontsize=12,
             bbox=dict(facecolor=color, alpha=0.3, boxstyle='round'))

# Arrow pointing to the calculation
calculation = f"$P(\text{{Technical}}|\text{{Model}}) = \\frac{{{model_prob_technical:.2f} \\times {prior_technical:.2f}}}{{{p_model:.4f}}} = \\frac{{{p_model_tech_term:.4f}}}{{{p_model:.4f}}} = {posterior_technical:.4f}$"
ax5.text(0.5, 0.5, calculation, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.7'))

# Interpretation
interpretation = f"The posterior probability ({posterior_technical:.4f}) is significantly higher than\nthe original model output ({model_prob_technical:.2f}) due to our prior knowledge\nthat 60% of tickets are about technical issues."
ax5.text(0.5, 0.3, interpretation, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig3, "step3_bayes_theorem.png")

# ==============================
# STEP 4: Classification with Threshold
# ==============================
print_section_header("STEP 4: Classification with Threshold")

print(f"Using a classification threshold of {threshold:.2f}, we determine which categories should be assigned")
print("to the ticket based on the model's probability outputs.")

print("\nDetailed threshold-based classification procedure:")
print("Step 1: Compare each category's probability to the threshold")
print(f"   We classify a category as positive if its probability exceeds {threshold:.2f}")
categories_above_threshold = []
for i, category in enumerate(categories):
    probability = model_probabilities[i]
    exceeds = probability > threshold
    result = "✓ Exceeds threshold" if exceeds else "✗ Below threshold"
    print(f"   • {category}: {probability:.4f} - {result}")
    relation = ">" if exceeds else "≤"
    print(f"     {probability:.4f} {relation} {threshold:.2f}")
    
    if exceeds:
        categories_above_threshold.append((category, probability))

print("\nStep 2: Determine final classification decision")
if not categories_above_threshold:
    print("   Result: No category exceeds the threshold.")
    print("   Classification decision: Ticket remains unclassified.")
    print(f"   Mathematical condition: max(probabilities) = {max(model_probabilities):.4f} ≤ {threshold:.2f}")
elif len(categories_above_threshold) == 1:
    assigned_category, prob = categories_above_threshold[0]
    print(f"   Result: Only '{assigned_category}' exceeds the threshold with probability {prob:.4f}.")
    print(f"   Classification decision: Ticket is classified as '{assigned_category}'.")
else:
    print("   Result: Multiple categories exceed the threshold:")
    for category, prob in categories_above_threshold:
        print(f"     • {category}: {prob:.4f}")
    
    # Find category with highest probability
    highest_prob_index = np.argmax(model_probabilities)
    highest_prob_category = categories[highest_prob_index]
    highest_prob_value = model_probabilities[highest_prob_index]
    
    print(f"   Classification decision: Choose the category with the highest probability.")
    print(f"   Highest probability category: '{highest_prob_category}' with probability {highest_prob_value:.4f}")

print("\nStep 3: Understand potential issues with fixed thresholds")
print("Using a fixed threshold of 0.30 for multinomial classification has several potential issues:")
issues = [
    "Loss of probability information - discards model confidence levels",
    "Ambiguity when multiple categories exceed the threshold",
    "Ambiguity when no category exceeds the threshold",
    "Different categories may need different optimal thresholds",
    "Doesn't account for class imbalance in the data",
    "Ignores varying costs of misclassification between categories"
]
for i, issue in enumerate(issues, 1):
    print(f"   {i}. {issue}")

print("\nStep 4: Alternative approaches to threshold-based classification")
print("Better alternatives include:")
print("   1. Always selecting the highest probability category (argmax)")
print("   2. Using category-specific thresholds")
print("   3. Incorporating prior probabilities through Bayes' theorem (as we did in Step 3)")
print("   4. Using a reject option for low-confidence predictions")
print("   5. Applying calibration techniques to improve probability estimates")

# Enhanced visualization of threshold-based classification
fig4 = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, height_ratios=[1.2, 1])

# Top: Detailed bar chart with threshold line
ax1 = fig4.add_subplot(gs[0, :])
bars = ax1.bar(categories, model_probabilities, color='lightgray')
ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
ax1.set_title('Category Probabilities vs. Classification Threshold')
ax1.set_xlabel('Category')
ax1.set_ylabel('Probability')
ax1.set_ylim(0, max(model_probabilities) * 1.3)
ax1.legend()

# Color the bars based on threshold
for i, bar in enumerate(bars):
    if model_probabilities[i] > threshold:
        bar.set_color('green')
    # Add a slight outline to make bars more visible
    bar.set_edgecolor('black')
    bar.set_linewidth(0.5)
    
# Add detailed labels
for i, (bar, prob) in enumerate(zip(bars, model_probabilities)):
    height = bar.get_height()
    status = "✓" if prob > threshold else "✗"
    relation = ">" if prob > threshold else "≤"
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
            f'{prob:.2f} {relation} {threshold:.2f}\n{status}', 
            ha='center', va='bottom', color='black', fontweight='bold')

# Bottom-left: Pie chart of model probabilities
ax2 = fig4.add_subplot(gs[1, 0])
wedges, texts, autotexts = ax2.pie(model_probabilities, labels=short_categories, autopct='%1.1f%%', 
                                  startangle=90, colors=sns.color_palette("pastel", len(categories)))
ax2.set_title('Model Probability Distribution')
# Highlight the category that exceeds threshold
for i, wedge in enumerate(wedges):
    if model_probabilities[i] > threshold:
        wedge.set_edgecolor('green')
        wedge.set_linewidth(2)
# Make the text more readable
for autotext in autotexts:
    autotext.set_fontweight('bold')

# Bottom-right: Decision flowchart for classification
ax3 = fig4.add_subplot(gs[1, 1])
ax3.axis('off')

# Create a simple decision flowchart
ax3.text(0.5, 0.95, "Classification Decision Process", ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Check if any category exceeds threshold
if max(model_probabilities) > threshold:
    # At least one category exceeds
    exceeding_indices = [i for i, p in enumerate(model_probabilities) if p > threshold]
    if len(exceeding_indices) == 1:
        # One category exceeds
        idx = exceeding_indices[0]
        decision_text = f"Only one category exceeds threshold:\n{categories[idx]} ({model_probabilities[idx]:.2f})"
        final_decision = f"Classify as: {categories[idx]}"
    else:
        # Multiple categories exceed
        decision_text = "Multiple categories exceed threshold:\n" + \
                       "\n".join([f"{categories[i]} ({model_probabilities[i]:.2f})" for i in exceeding_indices])
        max_idx = np.argmax(model_probabilities)
        final_decision = f"Select highest probability:\nClassify as {categories[max_idx]}"
else:
    # No category exceeds
    decision_text = "No category exceeds threshold"
    final_decision = "Result: Unclassified"

# Draw decision path
ax3.text(0.5, 0.7, decision_text, ha='center', va='center', fontsize=10,
         bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.5'))
ax3.text(0.5, 0.3, final_decision, ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(facecolor='lightgreen' if max(model_probabilities) > threshold else 'salmon', 
                  alpha=0.3, boxstyle='round,pad=0.5'))

# Draw arrow connecting the boxes
ax3.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.45),
            arrowprops=dict(arrowstyle='->', lw=1.5))

plt.tight_layout()
save_figure(fig4, "step4_classification_threshold.png")

# ==============================
# STEP 5: Cross-Entropy Loss
# ==============================
print_section_header("STEP 5: Cross-Entropy Loss")

print("The cross-entropy loss between the true one-hot encoded label [0,1,0,0,0] (Technical Problems)")
print("and the model's predicted probabilities is calculated using:")
print("H(y, ŷ) = -∑(i=1 to 5) y_i log(ŷ_i)")

print("\nStep 1: Understanding cross-entropy loss")
print("   • Cross-entropy loss measures the difference between two probability distributions")
print("   • It quantifies how well our predicted probabilities match the true labels")
print("   • For classification, one distribution is the one-hot encoded true label")
print("   • The other distribution is the model's predicted probabilities")
print("   • Lower values indicate better model performance")
print("   • Perfect predictions (probability 1.0 for correct class) would give loss = 0")

print("\nStep 2: Review the input values")
print(f"   • True label (y): {true_label} (one-hot encoding for Technical Problems)")
print(f"   • Predicted probabilities (ŷ): {[round(p, 2) for p in model_probabilities]}")
print("   • When using one-hot encoding, only the term for the true class contributes to the loss")
print("     because y_i = 0 for all other classes")

print("\nStep 3: Calculate each term in the summation")
cross_entropy = 0
terms = []
for i, (y, y_hat) in enumerate(zip(true_label, model_probabilities)):
    if y == 1:  # Only non-zero elements of the one-hot encoded vector contribute
        term_value = -y * math.log(y_hat)
        cross_entropy += term_value
        terms.append(term_value)
        print(f"\nTechnical Problems (y_{i+1} = {y}):")
        print(f"   • Term calculation: -y_{i+1} × log(ŷ_{i+1})")
        print(f"   • Substituting values: -{y} × log({y_hat:.4f})")
        print(f"   • Computing log: -{y} × ({math.log(y_hat):.6f})")
        print(f"   • Final value: {term_value:.6f}")
    else:
        print(f"\n{categories[i]} (y_{i+1} = {y}):")
        print(f"   • Term calculation: -y_{i+1} × log(ŷ_{i+1})")
        print(f"   • Substituting values: -{y} × log({y_hat:.4f})")
        print(f"   • Since y_{i+1} = 0, this term equals: 0")
        terms.append(0)

print(f"\nStep 4: Sum all terms to get the final cross-entropy loss")
print(f"   • H(y, ŷ) = {' + '.join([f'{term:.6f}' for term in terms])}")
print(f"   • H(y, ŷ) = {cross_entropy:.6f}")

print(f"\nStep 5: Interpret the cross-entropy loss value of {cross_entropy:.6f}")
print("   • This loss quantifies the 'surprise' of seeing the true label given the model's predictions")
print(f"   • The model assigned probability {model_probabilities[1]:.4f} to the correct class")
print("   • Perfect predictions would give a cross-entropy of 0")
print("   • Random guessing (probability 0.2 for each class) would give a cross-entropy of 1.609")
print(f"   • Our loss is {cross_entropy:.4f}, indicating the model performs better than random")
print("   • But there's still room for improvement (loss > 0)")

print("\nStep 6: Understanding how to improve cross-entropy loss")
print("   • The loss would be lower if the model assigned higher probability to the correct class")
print("   • For example, if P(Technical) = 0.5, loss would be 0.693")
print("   • If P(Technical) = 0.9, loss would be 0.105")
print("   • Cross-entropy loss thus encourages high confidence in correct predictions")
print("   • It heavily penalizes being confidently wrong (if true class gets very low probability)")

# Enhanced visualization of cross-entropy loss
fig5 = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1.2])

# Top-left: True label vs predicted probabilities
ax1 = fig5.add_subplot(gs[0, 0])
x = np.arange(len(categories))
width = 0.35
ax1.bar(x - width/2, true_label, width, label='True Label', color='lightgreen')
ax1.bar(x + width/2, model_probabilities, width, label='Predicted', color='lightblue')
ax1.set_title('True Label vs. Predicted Probabilities')
ax1.set_xticks(x)
ax1.set_xticklabels(short_categories, rotation=45)
ax1.set_ylabel('Value')
ax1.legend()

# Add details to bars
for i, (true, pred) in enumerate(zip(true_label, model_probabilities)):
    # Label for true values
    if true > 0:
        ax1.text(i - width/2, true/2, f"{true:.0f}", ha='center', va='center', 
                color='black', fontweight='bold')
    # Label for predictions
    ax1.text(i + width/2, pred/2, f"{pred:.2f}", ha='center', va='center', 
            color='black', fontweight='bold')
    
    # Add calculation for each category
    if true > 0:
        ax1.text(i, -0.1, f"-{true}×log({pred:.2f})={terms[i]:.3f}", ha='center', va='center', 
                color='red', fontsize=8)
    else:
        ax1.text(i, -0.1, f"-{true}×log({pred:.2f})=0", ha='center', va='center', 
                color='gray', fontsize=8)

# Top-right: Cross-entropy function curve
ax2 = fig5.add_subplot(gs[0, 1])
# Create a range of probabilities for the true class
p_range = np.linspace(0.01, 1, 100)
# Calculate cross-entropy for each probability value
ce_values = [-math.log(p) for p in p_range]

# Plot the cross-entropy function
ax2.plot(p_range, ce_values, 'b-', label='Cross-entropy: -log(p)')
ax2.set_title('Cross-Entropy Loss as Function of True Class Probability')
ax2.set_xlabel('Probability Assigned to True Class')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 5)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Highlight some important points on the curve
highlight_points = [
    (0.1, -math.log(0.1), "Very low confidence\nLoss = 2.30"),
    (0.2, -math.log(0.2), "Random guessing\nLoss = 1.61"),
    (model_probabilities[1], -math.log(model_probabilities[1]), f"Our model\nLoss = {cross_entropy:.2f}"),
    (0.5, -math.log(0.5), "50% confidence\nLoss = 0.69"),
    (0.9, -math.log(0.9), "High confidence\nLoss = 0.11")
]

for i, (p, loss, label) in enumerate(highlight_points):
    marker_color = 'red' if abs(p - model_probabilities[1]) < 0.01 else 'green'
    marker_size = 100 if abs(p - model_probabilities[1]) < 0.01 else 60
    ax2.scatter([p], [loss], color=marker_color, s=marker_size, zorder=3)
    
    # Position the label text to avoid overlap
    if i % 2 == 0:
        xytext = (p-0.15, loss-0.2)
    else:
        xytext = (p+0.05, loss+0.2)
        
    ax2.annotate(label, 
                xy=(p, loss),
                xytext=xytext,
                arrowprops=dict(arrowstyle='->'))

# Bottom: Detailed calculation and formula explanation
ax3 = fig5.add_subplot(gs[1, :])
ax3.axis('off')

# Create a visual explanation of cross-entropy calculation
ax3.text(0.5, 0.95, "Cross-Entropy Loss Calculation", ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Cross-entropy formula
formula = r"$H(y, \hat{y}) = -\sum_{i=1}^{5} y_i \log(\hat{y}_i)$"
ax3.text(0.5, 0.85, formula, ha='center', va='center', fontsize=12)

# Detailed calculation showing only the non-zero term
calculation = (
    r"$H(y, \hat{y}) = -y_1 \log(\hat{y}_1) - y_2 \log(\hat{y}_2) - y_3 \log(\hat{y}_3) - y_4 \log(\hat{y}_4) - y_5 \log(\hat{y}_5)$" + "\n" +
    fr"$H(y, \hat{y}) = -0 \times \log({model_probabilities[0]:.2f}) - 1 \times \log({model_probabilities[1]:.2f}) - 0 \times \log({model_probabilities[2]:.2f}) - 0 \times \log({model_probabilities[3]:.2f}) - 0 \times \log({model_probabilities[4]:.2f})$" + "\n" +
    fr"$H(y, \hat{y}) = 0 - 1 \times \log({model_probabilities[1]:.2f}) - 0 - 0 - 0$" + "\n" +
    fr"$H(y, \hat{y}) = -1 \times ({math.log(model_probabilities[1]):.6f}) = {cross_entropy:.6f}$"
)
ax3.text(0.5, 0.7, calculation, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.7'))

# Interpretation and insights
insights = [
    "• Cross-entropy loss measures the difference between predicted and true distributions",
    "• Lower values indicate better performance (0 is perfect prediction)",
    f"• Only the term for the true class contributes due to one-hot encoding",
    f"• Model assigns {model_probabilities[1]:.2f} probability to the correct class (Technical Problems)",
    f"• Loss could be improved by increasing the probability for the correct class",
    "• Cross-entropy loss is widely used as a training objective for classification models"
]

# Create a box with insights
insight_text = "\n".join(insights)
ax3.text(0.5, 0.4, insight_text, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.7'))

# Show some additional examples
examples = [
    f"Random guessing (p=0.2): Loss = -log(0.2) = 1.61",
    f"Our model (p={model_probabilities[1]:.2f}): Loss = -log({model_probabilities[1]:.2f}) = {cross_entropy:.2f}",
    f"Better model (p=0.5): Loss = -log(0.5) = 0.69",
    f"Excellent model (p=0.9): Loss = -log(0.9) = 0.11",
    f"Perfect model (p=1.0): Loss = -log(1.0) = 0.00"
]

example_text = "\n".join(examples)
ax3.text(0.5, 0.15, example_text, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightyellow', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig5, "step5_cross_entropy_loss.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this question on one-hot encoding and multinomial classification:")

print("\n1. One-Hot Encoding and Multinomial Distribution:")
print("   • One-hot encoding represents categories as binary vectors: [1,0,0,0,0], [0,1,0,0,0], etc.")
print("   • Each support ticket belongs to exactly one of the five categories")
print("   • The multinomial distribution directly models the probabilities across multiple categories")
print("   • Softmax activation outputs probabilities that align with multinomial distribution parameters")
print("   • The mathematical connection: P(y) = p₁^y₁ × p₂^y₂ × ... × p₅^y₅ (where y is one-hot encoded)")

print("\n2. MLE Prior Probabilities:")
for i, category in enumerate(categories):
    print(f"   • {category}: {prior_probabilities[i]:.4f} ({prior_probabilities[i]*100:.1f}%)")
print("   • These probabilities maximize the likelihood function: L(p₁,...,p₅) ∝ p₁^n₁ × ... × p₅^n₅")
print("   • The MLE formula is: p̂_i = n_i/n (count in category / total count)")
print(f"   • Sum of all probabilities = {sum(prior_probabilities):.4f} (verifying they sum to 1)")

print("\n3. Bayes' Theorem (Posterior Probability):")
print(f"   • Model probability for Technical Problems: {model_probabilities[1]:.4f}")
print(f"   • Prior probability for Technical Problems: {prior_technical:.4f}")
print(f"   • Calculation: P(Tech|Model) = P(Model|Tech)×P(Tech) / P(Model)")
print(f"   • P(Model) = P(Model|Tech)×P(Tech) + P(Model|Non-Tech)×P(Non-Tech) = {p_model:.4f}")
print(f"   • Posterior probability for Technical Problems: {posterior_technical:.4f} ({posterior_technical*100:.1f}%)")
print(f"   • Impact: Using prior knowledge increases probability from {model_probabilities[1]:.4f} to {posterior_technical:.4f}")
print(f"   • Relative increase: {((posterior_technical/model_probabilities[1])-1)*100:.1f}%")

print("\n4. Classification Threshold:")
if categories_above_threshold:
    print(f"   • Using threshold {threshold:.2f}, classified as: {categories_above_threshold[0][0]}")
else:
    print(f"   • Using threshold {threshold:.2f}, no category qualifies")
print("   • Fixed thresholds introduce issues like ambiguity and information loss")
print("   • Better alternatives: argmax selection, category-specific thresholds, Bayes' theorem")
print("   • Classification decisions should consider both probability values and domain context")

print("\n5. Cross-Entropy Loss:")
print(f"   • Formula: H(y, ŷ) = -∑(i=1 to 5) y_i log(ŷ_i)")
print(f"   • Calculation: H(y, ŷ) = -1 × log({model_probabilities[1]:.4f}) = {cross_entropy:.4f}")
print("   • Only the probability assigned to the true class affects the loss (due to one-hot encoding)")
print("   • Interpretation: Lower values indicate better model performance (0 is perfect)")
print("   • For reference: Random guessing would give loss ≈ 1.61")
print("   • Cross-entropy loss serves as a natural training objective for classification models")
print("   • Improving the model would increase probability for the correct class, reducing loss")

print("\nThis question demonstrates how statistical concepts like multinomial distributions, maximum likelihood")
print("estimation, Bayes' theorem, and information theory are foundational to machine learning classification,")
print("particularly in natural language processing applications like support ticket categorization.") 