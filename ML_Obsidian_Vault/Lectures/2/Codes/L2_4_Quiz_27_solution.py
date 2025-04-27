import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 12})

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

# Simplified visualization for one-hot encoding - Improved figure
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# One-hot encoding visualization (simplified)
one_hot_matrix = np.eye(len(categories))
sns.heatmap(one_hot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=[f'Pos {i+1}' for i in range(len(categories))], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding Matrix', fontsize=14)

# Category distribution visualization (simplified)
proportions = np.array(training_counts) / sum(training_counts)
bars = ax2.bar(short_categories, proportions, color=sns.color_palette("pastel", len(categories)))
ax2.set_title('Category Distribution', fontsize=14)
ax2.set_ylabel('Proportion', fontsize=12)
for bar, prop in zip(bars, proportions):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{prop:.2f}', 
            ha='center', va='bottom')

plt.tight_layout()
save_figure(fig1, "step1_one_hot_multinomial.png")

print("\nKey points about the figure:")
print("- Left panel: One-hot encoding matrix showing how each category is represented as a binary vector")
print("- Right panel: Distribution of categories in the training data, representing the multinomial parameters")
print("- These proportions will be used as the MLE estimates for prior probabilities in the next step")

print("\nMultinomial PMF Formula:")
print("P(X₁=n₁,...,X₅=n₅) = (n! / (n₁!...n₅!)) × p₁^n₁ × ... × p₅^n₅")
print("\nExample with one-hot encoding [0,1,0,0,0] (Technical Problems) and model probabilities:")
print(f"P(y) = {model_probabilities[0]}^0 × {model_probabilities[1]}^1 × {model_probabilities[2]}^0 × {model_probabilities[3]}^0 × {model_probabilities[4]}^0 = {model_probabilities[1]:.4f}")

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
print("   This gives us our final MLE formula: p̂_i = n_i/n")

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

# Simplified visualization of prior probabilities
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart (simplified)
wedges, texts, autotexts = ax1.pie(training_counts, labels=short_categories, autopct='%1.1f%%', 
        startangle=90, colors=sns.color_palette("pastel", len(categories)))
ax1.set_title('Training Data Distribution', fontsize=14)
        
# Add center circle to make a donut chart for better visualization
centre_circle = plt.Circle((0, 0), 0.5, fc='white')
ax1.add_patch(centre_circle)

# Add total sample size in center
ax1.text(0, 0, f"n = {total_training}", ha='center', va='center', fontweight='bold', fontsize=12)

# Bar chart of probabilities (simplified)
bars = ax2.bar(short_categories, prior_probabilities, 
              color=sns.color_palette("pastel", len(categories)))
ax2.set_title('MLE Prior Probabilities', fontsize=14)
ax2.set_ylabel('Probability', fontsize=12)
for bar, prob, count in zip(bars, prior_probabilities, training_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
            f'{prob:.2f}', 
            ha='center', va='bottom')

plt.tight_layout()
save_figure(fig2, "step2_prior_probabilities.png")

print("\nKey points about the figure:")
print("- Left panel: Pie chart showing the distribution of training data across categories")
print("- Right panel: Bar chart showing the MLE prior probabilities")
print("- The MLE formula is: p̂_i = n_i/n (count in category / total count)")
print("\nExample calculation for Technical Problems:")
i = 1  # Index for Technical Problems
print(f"p̂_{i+1} = n_{i+1}/n = {training_counts[i]}/{total_training} = {prior_probabilities[i]:.4f}")

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

# Simplified visualization of Bayes' theorem application
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Prior vs Posterior comparison (simplified)
labels = ['Model Output', 'After Bayes']
values = [model_prob_technical, posterior_technical]
bars = ax1.bar(labels, values, color=['lightblue', 'lightgreen'])
ax1.set_title("Bayes' Theorem Effect", fontsize=14)
ax1.set_ylabel('Probability', fontsize=12)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.2f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.0)

# Components of Bayes calculation (simplified)
terms = ['Term 1', 'Term 2', 'Total']
component_values = [p_model_tech_term, p_model_nontech_term, p_model]
colors = ['lightblue', 'lightcoral', 'lightgreen']
bars = ax2.bar(terms, component_values, color=colors)
ax2.set_title('Components of Bayes Calculation', fontsize=14)
ax2.set_ylabel('Probability', fontsize=12)

# Add value labels to each bar
for bar, val in zip(bars, component_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{val:.4f}', 
            ha='center', va='bottom')

plt.tight_layout()
save_figure(fig3, "step3_bayes_theorem.png")

print("\nKey points about the figure:")
print("- Left panel: Comparing the original model probability with the updated posterior probability after Bayes' theorem")
print("- Right panel: Components of the calculation - Term 1 = P(Model|Tech)×P(Tech), Term 2 = P(Model|Non-Tech)×P(Non-Tech), Total = P(Model)")
print("- Bayes' theorem formula: P(Technical|Model) = P(Model|Technical) × P(Technical) / P(Model)")
print(f"- Calculation: P(Technical|Model) = ({model_prob_technical:.2f} × {prior_technical:.2f}) / {p_model:.4f} = {posterior_technical:.4f}")

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

# Simplified visualization of threshold-based classification
fig4, ax = plt.subplots(figsize=(10, 6))

# Bar chart with threshold line (simplified)
bars = ax.bar(short_categories, model_probabilities, color='lightgray')
ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
ax.set_title('Probabilities vs. Classification Threshold', fontsize=14)
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_ylim(0, max(model_probabilities) * 1.3)
ax.legend(fontsize=12)

# Color the bars based on threshold
for i, bar in enumerate(bars):
    if model_probabilities[i] > threshold:
        bar.set_color('green')
    else:
        bar.set_color('lightgray')
        
# Add clear decision labels
for i, (bar, prob) in enumerate(zip(bars, model_probabilities)):
    height = bar.get_height()
    decision = "SELECTED" if prob > threshold else ""
    color = "green" if prob > threshold else "black"
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
            f'{prob:.2f}', 
            ha='center', va='bottom', color=color, fontweight='bold')

plt.tight_layout()
save_figure(fig4, "step4_classification_threshold.png")

print("\nKey points about the figure:")
print(f"- Bar chart showing model probabilities for each category with a threshold line at {threshold:.2f}")
print("- Green bars exceed the threshold and are selected; gray bars are below the threshold")
print("- In this example, only 'Technical Problems' exceeds the threshold with probability 0.35")
print("- Classification with threshold formula: If P(category) > threshold, then assign to that category")
print("\nIn this example, we have:")
for i, category in enumerate(categories):
    comparison = ">" if model_probabilities[i] > threshold else "≤"
    result = "Assign" if model_probabilities[i] > threshold else "Don't assign"
    print(f"P({short_categories[i]}) = {model_probabilities[i]:.2f} {comparison} {threshold:.2f} → {result}")

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

# Simplified visualization of cross-entropy loss
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# True label vs predicted probabilities (simplified)
x = np.arange(len(categories))
width = 0.35
ax1.bar(x - width/2, true_label, width, label='True Label', color='lightgreen')
ax1.bar(x + width/2, model_probabilities, width, label='Predicted', color='lightblue')
ax1.set_title('True Label vs. Predicted Probabilities', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(short_categories, rotation=45)
ax1.set_ylabel('Value', fontsize=12)
ax1.legend(fontsize=10)

# Highlight the critical value with rectangular outline
tech_index = 1  # Index for Technical Problems
rect = plt.Rectangle((tech_index + width/2 - width*0.6, 0), width*1.2, model_probabilities[tech_index], 
                    facecolor='none', edgecolor='red', linestyle='--', linewidth=2)
ax1.add_patch(rect)

# Cross-entropy function curve (simplified)
p_range = np.linspace(0.01, 1, 100)
ce_values = [-math.log(p) for p in p_range]
ax2.plot(p_range, ce_values, 'b-', linewidth=2, label='Cross-entropy: -log(p)')
ax2.set_title('Cross-Entropy Loss Function', fontsize=14)
ax2.set_xlabel('Probability for True Class', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 5)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Mark our model's position
ax2.scatter([model_probabilities[1]], [-math.log(model_probabilities[1])], 
           color='red', s=100, zorder=3)

# Add reference points
reference_points = [
    (0.2, 'Random'),
    (0.5, 'Better'),
    (1.0, 'Perfect')
]
for p, label in reference_points:
    if p > 0:  # Avoid log(0)
        loss_val = -math.log(p)
        ax2.scatter([p], [loss_val], color='green', s=60, zorder=3)

plt.tight_layout()
save_figure(fig5, "step5_cross_entropy_loss.png")

print("\nKey points about the figure:")
print("- Left panel: Comparison between the true one-hot encoded label and model's predicted probabilities")
print("- Right panel: Cross-entropy loss curve showing how the loss decreases as the probability for the true class increases")
print("- The red dot represents our model's position (p=0.35, loss≈1.05)")
print("- Green reference points show random guessing (0.2), better performance (0.5), and perfect prediction (1.0)")

print("\nCross-entropy loss formula:")
print("H(y, ŷ) = -∑(i=1 to 5) y_i log(ŷ_i)")
print("\nWith one-hot encoding [0,1,0,0,0], this simplifies to:")
print("H(y, ŷ) = -1 × log(P(Technical))")
print(f"H(y, ŷ) = -1 × log({model_probabilities[1]:.2f})")
print(f"H(y, ŷ) = -1 × ({math.log(model_probabilities[1]):.6f})")
print(f"H(y, ŷ) = {cross_entropy:.6f}")

print("\nExample cross-entropy values for different probabilities:")
examples = [0.1, 0.2, 0.35, 0.5, 0.8, 0.9, 1.0]
print("| Probability | Loss     | Notes           |")
print("|------------|----------|-----------------|")
for p in examples:
    loss = -math.log(p)
    highlight = " (our model)" if abs(p - model_probabilities[1]) < 0.01 else ""
    note = highlight.strip() if highlight else "Random guessing" if abs(p - 0.2) < 0.01 else "Perfect" if p == 1.0 else ""
    print(f"| {p:.2f}        | {loss:.4f}   | {note:<15} |")

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