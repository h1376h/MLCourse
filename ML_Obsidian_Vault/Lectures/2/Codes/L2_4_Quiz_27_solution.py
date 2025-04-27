import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Create directory to save figures if it doesn't exist
os.makedirs("images", exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join("images", filename)
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
print("and all others are 0 (cold).")
print("\nIn this problem with 5 categories, the one-hot encodings are:")
for i, category in enumerate(categories):
    encoding = ["0"] * 5
    encoding[i] = "1"
    print(f"{category.ljust(20)}: [{', '.join(encoding)}]")

print("\nThis encoding scheme directly relates to the multinomial distribution:")
print("- Each support ticket represents one trial")
print("- Each ticket must belong to exactly one of the five categories")
print("- One-hot encoding maps each category to a unique binary vector")
print("- The multinomial distribution models the probability of observing counts across categories")
print("\nFor a logistic regression model with softmax activation (multinomial logistic regression),")
print("the model outputs probabilities that sum to 1 across all categories.")

# Create visualization for one-hot encoding - SIMPLIFIED
fig1 = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1.2])

# Left plot: One-hot encoding visualization (simplified)
ax1 = fig1.add_subplot(gs[0, 0])
one_hot_matrix = np.eye(5)
sns.heatmap(one_hot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['1', '2', '3', '4', '5'], 
            yticklabels=short_categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding')
ax1.set_xlabel('Position')
ax1.set_ylabel('Category')

# Right plot: Multinomial distribution bar chart (simplified)
ax2 = fig1.add_subplot(gs[0, 1])
# Calculate proportions directly from training data
proportions = np.array(training_counts) / sum(training_counts)
colors = sns.color_palette("pastel", 5)
bars = ax2.bar(short_categories, proportions, color=colors)
ax2.set_title('Category Distribution')
ax2.set_xlabel('Category')
ax2.set_ylabel('Proportion')
# Add simplified labels
for bar, prop in zip(bars, proportions):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, f'{prop:.2f}', 
            ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
save_figure(fig1, "step1_one_hot_multinomial.png")

# ==============================
# STEP 2: MLE for Prior Probabilities
# ==============================
print_section_header("STEP 2: MLE for Prior Probabilities")

print("The Maximum Likelihood Estimate (MLE) for the prior probabilities of each category")
print("is the proportion of training examples that belong to each category.")

# Calculate MLE for prior probabilities with detailed steps
total_training = sum(training_counts)
print(f"\nTotal number of training tickets: {total_training}")

print("\nCalculating MLE prior probability for each category:")
prior_probabilities = []
for i, category in enumerate(categories):
    prior = training_counts[i] / total_training
    prior_probabilities.append(prior)
    print(f"{category}:")
    print(f"  Count: {training_counts[i]}")
    print(f"  Calculation: {training_counts[i]} / {total_training} = {prior:.4f}")
    print(f"  Percentage: {prior*100:.1f}%")

print("\nThe formula for the MLE of multinomial probabilities is:")
print("P(Category i) = Count of Category i / Total Count")
print("\nThis follows from maximizing the likelihood function:")
print("L(p₁, p₂, ..., p₅) = n! / (n₁!n₂!...n₅!) × p₁^n₁ × p₂^n₂ × ... × p₅^n₅")
print("subject to the constraint that Σp_i = 1")

# Visualization of prior probabilities - SIMPLIFIED
fig2 = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Left plot: Pie chart of training data distribution (simplified)
ax1 = fig2.add_subplot(gs[0, 0])
ax1.pie(training_counts, labels=short_categories, autopct='%1.1f%%', 
        startangle=90, colors=colors)
ax1.set_title('Training Data Distribution')

# Right plot: Bar chart of prior probabilities (simplified)
ax2 = fig2.add_subplot(gs[0, 1])
bars = ax2.bar(short_categories, prior_probabilities, color=colors)
ax2.set_title('MLE Prior Probabilities')
ax2.set_xlabel('Category')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, max(prior_probabilities) * 1.2)
# Add simplified labels
for bar, prob in zip(bars, prior_probabilities):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, f'{prob:.2f}', 
            ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
save_figure(fig2, "step2_prior_probabilities.png")

# ==============================
# STEP 3: Bayes' Theorem for Posterior Probability
# ==============================
print_section_header("STEP 3: Bayes' Theorem for Posterior Probability")

print("We need to calculate the posterior probability that the ticket belongs to 'Technical Problems'")
print("if we know that 60% of all support tickets are about technical issues.")

# Define variables for clarity
model_prob_technical = model_probabilities[1]  # P(Model predicts Technical | Ticket)
prior_technical = 0.60  # P(Technical) = 0.60
prior_non_technical = 1 - prior_technical  # P(Non-Technical) = 0.40

print("\nStep-by-step calculation using Bayes' theorem:")
print("P(Technical | Model) = P(Model | Technical) × P(Technical) / P(Model)")

print("\n1. Identify the values we know:")
print(f"   - Model probability for Technical: {model_prob_technical:.2f}")
print(f"   - Prior probability P(Technical): {prior_technical:.2f}")
print(f"   - Prior probability P(Non-Technical): {prior_non_technical:.2f}")

print("\n2. For P(Model | Technical), we use the model output directly:")
print(f"   P(Model | Technical) = {model_prob_technical:.2f}")

print("\n3. Calculate P(Model | Non-Technical):")
non_technical_probs = [model_probabilities[i] for i in range(len(model_probabilities)) if i != 1]
total_non_technical_prob = sum(non_technical_probs)
p_model_given_non_technical = total_non_technical_prob / 4  # Average across the 4 non-technical classes

print(f"   Non-technical probabilities: {non_technical_probs}")
print(f"   Sum of non-technical probabilities: {total_non_technical_prob:.4f}")
print(f"   Average (P(Model | Non-Technical)): {total_non_technical_prob:.4f} / 4 = {p_model_given_non_technical:.4f}")

print("\n4. Calculate P(Model) using the law of total probability:")
print("   P(Model) = P(Model | Technical) × P(Technical) + P(Model | Non-Technical) × P(Non-Technical)")
p_model = model_prob_technical * prior_technical + p_model_given_non_technical * prior_non_technical
print(f"   P(Model) = {model_prob_technical:.2f} × {prior_technical:.2f} + {p_model_given_non_technical:.4f} × {prior_non_technical:.2f}")
print(f"   P(Model) = {model_prob_technical * prior_technical:.4f} + {p_model_given_non_technical * prior_non_technical:.4f}")
print(f"   P(Model) = {p_model:.4f}")

print("\n5. Finally, calculate the posterior probability:")
print("   P(Technical | Model) = P(Model | Technical) × P(Technical) / P(Model)")
posterior_technical = (model_prob_technical * prior_technical) / p_model
print(f"   P(Technical | Model) = ({model_prob_technical:.2f} × {prior_technical:.2f}) / {p_model:.4f}")
print(f"   P(Technical | Model) = {model_prob_technical * prior_technical:.4f} / {p_model:.4f}")
print(f"   P(Technical | Model) = {posterior_technical:.4f} or {posterior_technical*100:.1f}%")

# Simplified calculation (binary approach)
print("\n6. Alternative simplified calculation (treating it as a binary problem):")
simplified_posterior = model_prob_technical * prior_technical / (model_prob_technical * prior_technical + 
                                                 (1-model_prob_technical) * (1-prior_technical))
print("   For binary classification (Technical vs. Non-Technical):")
print(f"   P(Technical | Model) = {model_prob_technical:.2f} × {prior_technical:.2f} / [{model_prob_technical:.2f} × {prior_technical:.2f} + (1-{model_prob_technical:.2f}) × (1-{prior_technical:.2f})]")
print(f"   P(Technical | Model) = {model_prob_technical * prior_technical:.4f} / [{model_prob_technical * prior_technical:.4f} + {(1-model_prob_technical) * (1-prior_technical):.4f}]")
print(f"   P(Technical | Model) = {model_prob_technical * prior_technical:.4f} / {model_prob_technical * prior_technical + (1-model_prob_technical) * (1-prior_technical):.4f}")
print(f"   P(Technical | Model) = {simplified_posterior:.4f} or {simplified_posterior*100:.1f}%")

# Visualization of Bayes' theorem application - SIMPLIFIED
fig3 = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2, height_ratios=[1, 1.2])

# Plot 1: Prior probabilities (simplified)
ax1 = fig3.add_subplot(gs[0, 0])
prior_labels = ['Original Prior', 'Adjusted Prior']
prior_values = [prior_probabilities[1], prior_technical]
bars = ax1.bar(prior_labels, prior_values, color=['lightblue', 'lightgreen'])
ax1.set_title('Prior Probability')
ax1.set_ylabel('Probability')
# Add simplified labels
for bar, val in zip(bars, prior_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.2f}', 
            ha='center', va='center', color='black', fontweight='bold')

# Plot 2: Likelihood (simplified)
ax2 = fig3.add_subplot(gs[0, 1])
ax2.bar(['Model Probability'], [model_prob_technical], color='salmon')
ax2.set_title('Likelihood')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, 1.0)
ax2.text(0, model_prob_technical/2, f'{model_prob_technical:.2f}', 
        ha='center', va='center', color='black', fontweight='bold')

# Plot 3: Calculation diagram (simplified)
ax3 = fig3.add_subplot(gs[1, :])
# Create a simple diagram showing the calculation
components = [
    f"Prior: {prior_technical:.2f}",
    "×",
    f"Likelihood: {model_prob_technical:.2f}",
    "÷",
    f"Evidence: {p_model:.4f}",
    "=",
    f"Posterior: {posterior_technical:.4f}"
]
positions = np.linspace(0.1, 0.9, len(components))
ax3.axis('off')
for i, (pos, text) in enumerate(zip(positions, components)):
    color = 'black'
    if i == 0:  # Prior
        color = 'green'
    elif i == 2:  # Likelihood
        color = 'red'
    elif i == 4:  # Evidence
        color = 'blue'
    elif i == 6:  # Posterior
        color = 'purple'
    
    ax3.text(pos, 0.5, text, ha='center', va='center', fontsize=14, 
             color=color, fontweight='bold')

plt.tight_layout()
save_figure(fig3, "step3_bayes_theorem.png")

# ==============================
# STEP 4: Classification with Threshold
# ==============================
print_section_header("STEP 4: Classification with Threshold")

print(f"Using a classification threshold of {threshold:.2f}, we determine if a category's probability")
print("exceeds this threshold.")

print("\nStep 1: Compare each category's probability to the threshold:")
categories_above_threshold = []
for i, category in enumerate(categories):
    probability = model_probabilities[i]
    exceeds = probability > threshold
    result = "Exceeds threshold" if exceeds else "Below threshold"
    print(f"{category}: {probability:.2f} - {result}")
    
    if exceeds:
        categories_above_threshold.append(category)

print("\nStep 2: Determine the classification result:")
if not categories_above_threshold:
    print("Result: No category exceeds the threshold. The ticket remains unclassified.")
elif len(categories_above_threshold) == 1:
    assigned_category = categories_above_threshold[0]
    print(f"Result: Only '{assigned_category}' exceeds the threshold. The ticket is classified as '{assigned_category}'.")
else:
    print("Result: Multiple categories exceed the threshold:")
    for category in categories_above_threshold:
        print(f"  - {category}")
    highest_prob_index = np.argmax(model_probabilities)
    highest_prob_category = categories[highest_prob_index]
    print(f"In this case, we would typically choose the category with the highest probability: '{highest_prob_category}'")

print("\nPotential issues with fixed thresholds in multinomial classification:")
issues = [
    "Loss of probability information (discards confidence levels)",
    "Ambiguity when multiple categories exceed the threshold",
    "Ambiguity when no category exceeds the threshold",
    "Different categories may need different optimal thresholds",
    "Doesn't account for class imbalance",
    "Ignores varying costs of misclassification between categories"
]
for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue}")

# Visualization of threshold-based classification - SIMPLIFIED
fig4 = plt.figure(figsize=(10, 5))

# Bar chart with threshold line (simplified)
bars = plt.bar(short_categories, model_probabilities, color='lightgray')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
plt.title('Probabilities vs. Threshold')
plt.xlabel('Category')
plt.ylabel('Probability')
plt.ylim(0, max(model_probabilities) * 1.3)
plt.legend()

# Color the bars based on threshold
for i, bar in enumerate(bars):
    if model_probabilities[i] > threshold:
        bar.set_color('green')
        
# Add simplified labels
for bar, prob in zip(bars, model_probabilities):
    height = bar.get_height()
    status = "✓" if height > threshold else "✗"
    plt.text(bar.get_x() + bar.get_width()/2., height/2, f'{prob:.2f}\n{status}', 
            ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
save_figure(fig4, "step4_classification_threshold.png")

# ==============================
# STEP 5: Cross-Entropy Loss
# ==============================
print_section_header("STEP 5: Cross-Entropy Loss")

print("The cross-entropy loss between the true one-hot encoded label [0,1,0,0,0] (Technical Problems)")
print("and the model's predicted probabilities is calculated using:")
print("H(y, ŷ) = -∑(i=1 to 5) y_i log(ŷ_i)")

print("\nStep-by-step calculation:")
# Detailed calculation of cross-entropy loss
print("\nFirst, let's identify our values:")
print(f"True label (y): {true_label} (one-hot encoding for Technical Problems)")
print(f"Predicted probabilities (ŷ): {[round(p, 2) for p in model_probabilities]}")

print("\nNow, let's calculate each term in the summation:")
cross_entropy = 0
for i, (y, y_hat) in enumerate(zip(true_label, model_probabilities)):
    term_value = 0
    if y == 1:  # Only non-zero elements of the one-hot encoded vector contribute
        term_value = -y * math.log(y_hat)
        cross_entropy += term_value
        print(f"Category {i+1} ({categories[i]}):")
        print(f"  y_i = {y}, ŷ_i = {y_hat:.2f}")
        print(f"  -y_i × log(ŷ_i) = -{y} × log({y_hat:.2f})")
        print(f"  -y_i × log(ŷ_i) = -{y} × ({math.log(y_hat):.4f})")
        print(f"  -y_i × log(ŷ_i) = {term_value:.4f}")
    else:
        print(f"Category {i+1} ({categories[i]}):")
        print(f"  y_i = {y}, ŷ_i = {y_hat:.2f}")
        print(f"  -y_i × log(ŷ_i) = -{y} × log({y_hat:.2f}) = 0 (since y_i = 0)")

print(f"\nFinal cross-entropy loss = {cross_entropy:.4f}")

print("\nInterpretation of the result:")
print("- The cross-entropy loss measures how well predicted probabilities match true labels")
print("- Lower values indicate better model performance")
print("- Only the term for the true class (Technical Problems) contributes")
print(f"- The model assigned probability {model_probabilities[1]:.2f} to the correct class")
print(f"- If the model had been more confident (closer to 1.0), the loss would be lower")
print(f"- If the model had been less confident (closer to 0), the loss would be higher")

# Visualization of cross-entropy loss - SIMPLIFIED
fig5 = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Left plot: True label vs predicted probabilities (simplified)
ax1 = fig5.add_subplot(gs[0, 0])
x = np.arange(len(categories))
width = 0.35
ax1.bar(x - width/2, true_label, width, label='True Label', color='gray')
ax1.bar(x + width/2, model_probabilities, width, label='Predicted', color='lightblue')
ax1.set_title('True vs. Predicted')
ax1.set_xticks(x)
ax1.set_xticklabels(short_categories, rotation=45)
ax1.set_ylabel('Value')
ax1.legend()

# Right plot: Cross-entropy function (simplified)
ax2 = fig5.add_subplot(gs[0, 1])
# Plot the cross-entropy function curve
x_range = np.linspace(0.01, 1, 100)
y_range = [-math.log(x) for x in x_range]
ax2.plot(x_range, y_range, 'b-', label='-log(p)')
# Highlight the actual value
true_class_prob = model_probabilities[1]  # Technical Problems probability
loss_value = -math.log(true_class_prob)
ax2.scatter([true_class_prob], [loss_value], color='red', s=100, zorder=3)
ax2.annotate(f'Loss: {loss_value:.2f}', 
            xy=(true_class_prob, loss_value),
            xytext=(true_class_prob+0.1, loss_value-0.5),
            arrowprops=dict(arrowstyle='->'))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 5)
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Loss')
ax2.set_title('Cross-Entropy Loss')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
save_figure(fig5, "step5_cross_entropy_loss.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this question on one-hot encoding and multinomial classification:")

print("\n1. One-Hot Encoding and Multinomial Distribution:")
print("   - One-hot encoding represents categories as binary vectors: [1,0,0,0,0], [0,1,0,0,0], etc.")
print("   - Each support ticket belongs to exactly one of the five categories")
print("   - Multinomial distribution models probabilities across multiple categories")

print("\n2. MLE Prior Probabilities:")
for i, category in enumerate(categories):
    print(f"   - {category}: {prior_probabilities[i]:.4f} ({prior_probabilities[i]*100:.1f}%)")

print("\n3. Bayes' Theorem (Posterior Probability):")
print(f"   - Model probability for Technical Problems: {model_probabilities[1]:.2f}")
print(f"   - Prior probability for Technical Problems: {prior_technical:.2f}")
print(f"   - Posterior probability for Technical Problems: {posterior_technical:.4f}")
print(f"   - Using prior knowledge increases probability from {model_probabilities[1]:.2f} to {posterior_technical:.4f}")

print("\n4. Classification Threshold:")
if categories_above_threshold:
    print(f"   - Using threshold {threshold:.2f}, classified as: {categories_above_threshold[0]}")
else:
    print(f"   - Using threshold {threshold:.2f}, no category qualifies")
print("   - Fixed thresholds can lead to ambiguity and information loss")

print("\n5. Cross-Entropy Loss:")
print(f"   - Cross-entropy loss: {cross_entropy:.4f}")
print("   - Measures how well the model's probabilities match the true label")
print("   - Only the probability assigned to the true class affects the loss") 