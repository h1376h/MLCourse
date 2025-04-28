import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Function to save figures
def save_figure(fig, filename):
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")

# Function to print step headers for better organization in output
def print_step_header(step_number, step_title):
    print(f"\n{'='*80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'='*80}")

# Function to print substeps
def print_substep(substep_title):
    print(f"\n{'-'*40}")
    print(f"{substep_title}")
    print(f"{'-'*40}")

# Function to print mathematical derivations in detail
def print_derivation(title, steps):
    print(f"\n{'-'*20} {title} {'-'*20}")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    print(f"{'-'*50}")

# Problem setup
print_step_header(1, "Problem Definition")
print("Classification problem: Will a student pass (class 1) or fail (class 0) an exam based on study hours.")
print("Model produces a probability p that the student will pass.")
print("\nLoss functions:")
print("1. 0-1 Loss: L(y, ŷ) = 1 if y ≠ ŷ, 0 otherwise (equal penalty for all errors)")
print("2. Asymmetric Loss:")
print("   - L(1, 0) = 2 (missed opportunity when student would pass but predicted fail)")
print("   - L(0, 1) = 1 (wasted effort when student would fail but predicted pass)")
print("   - L(y, y) = 0 (correct prediction)")

# Task 1: 0-1 Loss Function Analysis
print_step_header(2, "0-1 Loss Function Analysis")

# Define the 0-1 loss function
def zero_one_loss(true_class, predicted_class):
    return 1 if true_class != predicted_class else 0

print_substep("Detailed Derivation of Decision Rule for 0-1 Loss")

# More detailed calculation of expected loss for 0-1 loss
zero_one_loss_table = [
    ["True Class (y)", "Predicted Class (ŷ)", "Loss L(y,ŷ)"],
    ["0 (fail)", "0 (fail)", "0 (correct prediction)"],
    ["0 (fail)", "1 (pass)", "1 (incorrect prediction)"],
    ["1 (pass)", "0 (fail)", "1 (incorrect prediction)"],
    ["1 (pass)", "1 (pass)", "0 (correct prediction)"]
]

print("0-1 Loss Function Table:")
for row in zero_one_loss_table:
    print(f"  {row[0]:<15} {row[1]:<18} {row[2]}")

zero_one_derivation_steps = [
    "We need to calculate the expected loss for each possible action (predict pass or predict fail)",
    "Expected loss = sum(probability of each case × loss for that case)",
    "Let p = P(y=1) be the probability that the student will pass",
    "Then 1-p = P(y=0) is the probability that the student will fail",
    "",
    "For predicting pass (ŷ=1):",
    "Case 1: Student fails (y=0) but we predicted pass (ŷ=1)",
    "     Probability: P(y=0) = 1-p",
    "     Loss: L(0,1) = 1",
    "Case 2: Student passes (y=1) and we predicted pass (ŷ=1)",
    "     Probability: P(y=1) = p",
    "     Loss: L(1,1) = 0",
    "Expected loss for predicting pass = (1-p)×1 + p×0 = 1-p",
    "",
    "For predicting fail (ŷ=0):",
    "Case 1: Student fails (y=0) and we predicted fail (ŷ=0)",
    "     Probability: P(y=0) = 1-p",
    "     Loss: L(0,0) = 0",
    "Case 2: Student passes (y=1) but we predicted fail (ŷ=0)",
    "     Probability: P(y=1) = p",
    "     Loss: L(1,0) = 1",
    "Expected loss for predicting fail = (1-p)×0 + p×1 = p",
    "",
    "To minimize expected loss, we compare the two options:",
    "Choose 'predict pass' if: 1-p < p",
    "Choose 'predict fail' if: p < 1-p",
    "",
    "Solving 1-p < p:",
    "1-p < p",
    "1 < 2p",
    "0.5 < p",
    "",
    "Therefore, we should predict pass (ŷ=1) when p > 0.5, and predict fail (ŷ=0) when p < 0.5."
]

print_derivation("Mathematical Derivation for 0-1 Loss", zero_one_derivation_steps)

print("\nDecision Rule for 0-1 Loss:")
print("  * Predict pass (ŷ=1) if p > 0.5")
print("  * Predict fail (ŷ=0) if p < 0.5")
print("  * At p = 0.5, both decisions have equal expected loss (0.5)")

# Plot the expected loss for 0-1 loss function, but simpler with less annotations
print_substep("Visualizing expected loss for 0-1 loss")
p_values = np.linspace(0, 1, 1000)
exp_loss_pred_pass = 1 - p_values  # Expected loss when predicting pass
exp_loss_pred_fail = p_values      # Expected loss when predicting fail

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p_values, exp_loss_pred_pass, 'b-', linewidth=2)
ax.plot(p_values, exp_loss_pred_fail, 'r-', linewidth=2)
ax.axvline(x=0.5, color='k', linestyle='--')

# Fill regions to show optimal decisions
ax.fill_between(p_values[p_values <= 0.5], 0, exp_loss_pred_fail[p_values <= 0.5], color='r', alpha=0.2)
ax.fill_between(p_values[p_values >= 0.5], 0, exp_loss_pred_pass[p_values >= 0.5], color='b', alpha=0.2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Probability of passing (p)', fontsize=12)
ax.set_ylabel('Expected Loss', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

save_figure(fig, "zero_one_loss.png")
plt.close(fig)

print("\nGraph Interpretation:")
print("  * Blue line: Expected loss when predicting pass (ŷ=1) = 1-p")
print("  * Red line: Expected loss when predicting fail (ŷ=0) = p")
print("  * Vertical dashed line: Decision threshold at p = 0.5")
print("  * Red shaded region (left): Optimal to predict fail when p < 0.5")
print("  * Blue shaded region (right): Optimal to predict pass when p > 0.5")

# Task 2: Asymmetric Loss Function Analysis
print_step_header(3, "Asymmetric Loss Function Analysis")

# Define the asymmetric loss function
def asymmetric_loss(true_class, predicted_class):
    if true_class == 1 and predicted_class == 0:  # Missed opportunity
        return 2
    elif true_class == 0 and predicted_class == 1:  # Wasted effort
        return 1
    else:  # Correct prediction
        return 0

print_substep("Detailed Derivation of Decision Rule for Asymmetric Loss")

# More detailed calculation of expected loss for asymmetric loss
asymmetric_loss_table = [
    ["True Class (y)", "Predicted Class (ŷ)", "Loss L(y,ŷ)", "Interpretation"],
    ["0 (fail)", "0 (fail)", "0", "Correct prediction"],
    ["0 (fail)", "1 (pass)", "1", "Wasted effort"],
    ["1 (pass)", "0 (fail)", "2", "Missed opportunity"],
    ["1 (pass)", "1 (pass)", "0", "Correct prediction"]
]

print("Asymmetric Loss Function Table:")
for row in asymmetric_loss_table:
    if len(row) > 3:
        print(f"  {row[0]:<15} {row[1]:<18} {row[2]:<10} {row[3]}")
    else:
        print(f"  {row[0]:<15} {row[1]:<18} {row[2]}")

asymmetric_derivation_steps = [
    "We need to calculate the expected loss for each possible action (predict pass or predict fail)",
    "Expected loss = sum(probability of each case × loss for that case)",
    "Let p = P(y=1) be the probability that the student will pass",
    "Then 1-p = P(y=0) is the probability that the student will fail",
    "",
    "For predicting pass (ŷ=1):",
    "Case 1: Student fails (y=0) but we predicted pass (ŷ=1)",
    "     Probability: P(y=0) = 1-p",
    "     Loss: L(0,1) = 1 (wasted effort)",
    "Case 2: Student passes (y=1) and we predicted pass (ŷ=1)",
    "     Probability: P(y=1) = p",
    "     Loss: L(1,1) = 0",
    "Expected loss for predicting pass = (1-p)×1 + p×0 = 1-p",
    "",
    "For predicting fail (ŷ=0):",
    "Case 1: Student fails (y=0) and we predicted fail (ŷ=0)",
    "     Probability: P(y=0) = 1-p",
    "     Loss: L(0,0) = 0",
    "Case 2: Student passes (y=1) but we predicted fail (ŷ=0)",
    "     Probability: P(y=1) = p",
    "     Loss: L(1,0) = 2 (missed opportunity)",
    "Expected loss for predicting fail = (1-p)×0 + p×2 = 2p",
    "",
    "To minimize expected loss, we compare the two options:",
    "Choose 'predict pass' if: 1-p < 2p",
    "Choose 'predict fail' if: 2p < 1-p",
    "",
    "Solving 1-p < 2p:",
    "1-p < 2p",
    "1 < 2p + p",
    "1 < 3p",
    "1/3 < p",
    "",
    "Therefore, we should predict pass (ŷ=1) when p > 1/3, and predict fail (ŷ=0) when p < 1/3."
]

print_derivation("Mathematical Derivation for Asymmetric Loss", asymmetric_derivation_steps)

print("\nDecision Rule for Asymmetric Loss:")
print("  * Predict pass (ŷ=1) if p > 1/3")
print("  * Predict fail (ŷ=0) if p < 1/3")
print("  * At p = 1/3, both decisions have equal expected loss (2/3)")

# Plot the expected loss for asymmetric loss function - simpler with less annotations
print_substep("Visualizing expected loss for asymmetric loss")
p_values = np.linspace(0, 1, 1000)
asymm_exp_loss_pred_pass = 1 - p_values    # Expected loss when predicting pass
asymm_exp_loss_pred_fail = 2 * p_values    # Expected loss when predicting fail

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(p_values, asymm_exp_loss_pred_pass, 'b-', linewidth=2)
ax.plot(p_values, asymm_exp_loss_pred_fail, 'r-', linewidth=2)
ax.axvline(x=1/3, color='k', linestyle='--')

# Fill regions to show optimal decisions
ax.fill_between(p_values[p_values <= 1/3], 0, asymm_exp_loss_pred_fail[p_values <= 1/3], color='r', alpha=0.2)
ax.fill_between(p_values[p_values >= 1/3], 0, asymm_exp_loss_pred_pass[p_values >= 1/3], color='b', alpha=0.2)

ax.set_xlim(0, 1)
ax.set_ylim(0, 2)
ax.set_xlabel('Probability of passing (p)', fontsize=12)
ax.set_ylabel('Expected Loss', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

save_figure(fig, "asymmetric_loss.png")
plt.close(fig)

print("\nGraph Interpretation:")
print("  * Blue line: Expected loss when predicting pass (ŷ=1) = 1-p")
print("  * Red line: Expected loss when predicting fail (ŷ=0) = 2p")
print("  * Vertical dashed line: Decision threshold at p = 1/3")
print("  * Red shaded region (left): Optimal to predict fail when p < 1/3")
print("  * Blue shaded region (right): Optimal to predict pass when p > 1/3")

# Task 3: Comparison of loss functions
print_step_header(4, "Comparison of Loss Functions")
print("The decision thresholds differ because of the asymmetric nature of the second loss function.")
print("- With 0-1 loss, the threshold is p = 0.5 (predict the more likely class)")
print("- With asymmetric loss, the threshold is p = 1/3 (reflect the higher cost of missed opportunities)")
print("\nPractical interpretation:")
print("With asymmetric loss, we're more willing to predict 'pass' even with lower probabilities (p > 1/3).")
print("This reflects the fact that a missed opportunity (predicting fail when student would pass) is costlier (L=2)")
print("than a wasted effort (predicting pass when student would fail) (L=1).")

print_substep("Mathematical explanation of threshold shift")
threshold_derivation = [
    "For a general binary classification with cost C_FP for false positives and C_FN for false negatives:",
    "Expected loss for predicting class 1 = (1-p) × C_FP",
    "Expected loss for predicting class 0 = p × C_FN",
    "",
    "The optimal threshold p* is where these expected losses are equal:",
    "(1-p*) × C_FP = p* × C_FN",
    "C_FP - p* × C_FP = p* × C_FN",
    "C_FP = p* × (C_FP + C_FN)",
    "p* = C_FP / (C_FP + C_FN)",
    "",
    "For 0-1 loss: C_FP = C_FN = 1, so p* = 1/(1+1) = 0.5",
    "For asymmetric loss: C_FP = 1, C_FN = 2, so p* = 1/(1+2) = 1/3"
]
print_derivation("General Formula for Optimal Threshold", threshold_derivation)

# Create a comparison visualization - simpler with more printing than annotations
print_substep("Visualizing both loss functions together")
fig, ax = plt.subplots(figsize=(8, 6))

# Plot 0-1 loss function
ax.plot(p_values, exp_loss_pred_pass, 'b-', linewidth=2)
ax.plot(p_values, exp_loss_pred_fail, 'r-', linewidth=2)
ax.axvline(x=0.5, color='k', linestyle='--')

# Plot asymmetric loss function (dashed)
ax.plot(p_values, asymm_exp_loss_pred_pass, 'b--', linewidth=2)
ax.plot(p_values, asymm_exp_loss_pred_fail, 'r--', linewidth=2)
ax.axvline(x=1/3, color='k', linestyle=':')

# Add vertical line for p=0.4 example
ax.axvline(x=0.4, color='g', linestyle='-', linewidth=2, alpha=0.7)

# Calculate expected losses for p=0.4
p_example = 0.4
zero_one_pass = 1 - p_example
zero_one_fail = p_example
asymm_pass = 1 - p_example
asymm_fail = 2 * p_example

# Mark the expected losses for p=0.4
ax.plot(p_example, zero_one_pass, 'bo', markersize=8)
ax.plot(p_example, zero_one_fail, 'ro', markersize=8)
ax.plot(p_example, asymm_pass, 'bs', markersize=8)
ax.plot(p_example, asymm_fail, 'rs', markersize=8)

# Decision region shading
# For 0-1 loss
ax.axvspan(0, 0.5, alpha=0.1, color='red')
ax.axvspan(0.5, 1, alpha=0.1, color='blue')

# For asymmetric loss (with lighter shading and hatching)
ax.axvspan(0, 1/3, alpha=0.05, color='red', hatch='///')
ax.axvspan(1/3, 1, alpha=0.05, color='blue', hatch='///')

ax.set_xlim(0, 1)
ax.set_ylim(0, 2)
ax.set_xlabel('Probability of passing (p)', fontsize=12)
ax.set_ylabel('Expected Loss', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

plt.tight_layout()
save_figure(fig, "loss_function_comparison.png")
plt.close(fig)

print("\nGraph Interpretation:")
print("  * Solid blue line: Expected loss when predicting pass (0-1 loss)")
print("  * Solid red line: Expected loss when predicting fail (0-1 loss)")
print("  * Dashed blue line: Expected loss when predicting pass (asymmetric loss)")
print("  * Dashed red line: Expected loss when predicting fail (asymmetric loss)")
print("  * Vertical dashed line: 0-1 loss threshold at p = 0.5")
print("  * Vertical dotted line: Asymmetric loss threshold at p = 1/3")
print("  * Vertical green line: Example case with p = 0.4")
print("  * Red shaded regions: Areas where predicting fail minimizes expected loss")
print("  * Blue shaded regions: Areas where predicting pass minimizes expected loss")
print("  * Markers at p = 0.4: Expected losses for each combination of loss function and decision")

# Task 4: Example with p=0.4
print_step_header(5, "Example: p = 0.4")
p_example = 0.4
print(f"Given that our model produces a probability p = {p_example} of the student passing:")

print("\nDetailed calculations for 0-1 loss function:")
print(f"Expected loss if predict pass (ŷ=1):")
print(f"  = P(y=0) × L(0,1) + P(y=1) × L(1,1)")
print(f"  = (1-p) × 1 + p × 0")
print(f"  = (1-{p_example}) × 1 + {p_example} × 0")
print(f"  = {1-p_example:.4f}")

print(f"\nExpected loss if predict fail (ŷ=0):")
print(f"  = P(y=0) × L(0,0) + P(y=1) × L(1,0)")
print(f"  = (1-p) × 0 + p × 1")
print(f"  = (1-{p_example}) × 0 + {p_example} × 1")
print(f"  = {p_example:.4f}")

print(f"\nComparing: {p_example:.4f} < {1-p_example:.4f}")
print(f"Since {p_example:.4f} < {1-p_example:.4f}, the expected loss is lower when predicting fail")
print(f"Decision: Predict fail (ŷ=0)")

print("\nDetailed calculations for asymmetric loss function:")
print(f"Expected loss if predict pass (ŷ=1):")
print(f"  = P(y=0) × L(0,1) + P(y=1) × L(1,1)")
print(f"  = (1-p) × 1 + p × 0")
print(f"  = (1-{p_example}) × 1 + {p_example} × 0")
print(f"  = {1-p_example:.4f}")

print(f"\nExpected loss if predict fail (ŷ=0):")
print(f"  = P(y=0) × L(0,0) + P(y=1) × L(1,0)")
print(f"  = (1-p) × 0 + p × 2")
print(f"  = (1-{p_example}) × 0 + {p_example} × 2")
print(f"  = {2*p_example:.4f}")

print(f"\nComparing: {1-p_example:.4f} < {2*p_example:.4f}")
print(f"Since {1-p_example:.4f} < {2*p_example:.4f}, the expected loss is lower when predicting pass")
print(f"Decision: Predict pass (ŷ=1)")

print("\nConclusion:")
print("The decisions differ because the asymmetric loss function penalizes missed opportunities (L=2) more")
print("than wasted effort (L=1), making us more willing to predict 'pass' with lower probability values.")

# Create a visual summary
print_step_header(6, "Decision Regions Visualization")

fig, ax = plt.subplots(figsize=(8, 3))

# Create horizontal bars to represent decision regions
ax.barh(0, 1, left=0, height=0.5, color='lightcoral', alpha=0.6)
ax.barh(0, 0.5, left=0.5, height=0.5, color='lightblue', alpha=0.6)
ax.barh(1, 1/3, left=0, height=0.5, color='indianred', alpha=0.6)
ax.barh(1, 2/3, left=1/3, height=0.5, color='royalblue', alpha=0.6)

# Add threshold markers
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
ax.axvline(x=1/3, color='black', linestyle=':', alpha=0.7)

# Add p=0.4 marker
ax.axvline(x=0.4, color='green', linewidth=2, alpha=0.7)

# Configure the plot
ax.set_yticks([0, 1])
ax.set_yticklabels(['0-1 Loss', 'Asymmetric Loss'])
ax.set_xlabel('Probability of passing (p)', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 1.5)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add annotations for the p=0.4 decisions
if p_example < 0.5:  # 0-1 Loss: Predict Fail
    ax.plot(p_example, 0, 'ro', markersize=8, alpha=0.8)
else:  # 0-1 Loss: Predict Pass
    ax.plot(p_example, 0, 'bo', markersize=8, alpha=0.8)
    
if p_example < 1/3:  # Asymmetric Loss: Predict Fail
    ax.plot(p_example, 1, 'ro', markersize=8, alpha=0.8)
else:  # Asymmetric Loss: Predict Pass
    ax.plot(p_example, 1, 'bo', markersize=8, alpha=0.8)

plt.tight_layout()
save_figure(fig, "decision_regions.png")
plt.close(fig)

print("\nDecision Regions Visualization Explanation:")
print("  * Top bar: Decision regions for 0-1 loss")
print("    - Red region (0 to 0.5): Predict fail")
print("    - Blue region (0.5 to 1): Predict pass")
print("  * Bottom bar: Decision regions for asymmetric loss")
print("    - Red region (0 to 1/3): Predict fail")
print("    - Blue region (1/3 to 1): Predict pass")
print("  * Vertical green line: Our example case p = 0.4")
print("  * Red circle on top bar: With 0-1 loss and p = 0.4, we predict fail")
print("  * Blue circle on bottom bar: With asymmetric loss and p = 0.4, we predict pass")

print("\nSummary:")
print("For 0-1 Loss: The decision threshold is p = 0.5")
print("For Asymmetric Loss: The decision threshold is p = 1/3")
print("\nWith p = 0.4:")
print("- Using 0-1 loss: Predict fail (ŷ=0)")
print("- Using asymmetric loss: Predict pass (ŷ=1)") 