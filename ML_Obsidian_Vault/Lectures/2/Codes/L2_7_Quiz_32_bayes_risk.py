import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the images directory"""
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_step_header(step_number, step_title):
    """Print a formatted header for each step"""
    print(f"\n{'='*80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'='*80}")

def print_substep(substep_title):
    """Print a formatted header for each substep"""
    print(f"\n{'-'*80}")
    print(f"{substep_title}")
    print(f"{'-'*80}")

def print_derivation(title, steps):
    """Print a derivation with multiple steps"""
    print(f"\n{title}:")
    for i, step in enumerate(steps):
        print(f"  {i+1}. {step}")

def zero_one_loss(true_class, predicted_class):
    """
    Calculate zero-one loss
    L(y, ŷ) = 1 if y ≠ ŷ, 0 if y = ŷ
    """
    return 0 if true_class == predicted_class else 1

def asymmetric_loss(true_class, predicted_class):
    """
    Calculate asymmetric loss for the cancer diagnosis problem
    
    True class:
    - C1: Benign
    - C2: Malignant
    
    Predicted class:
    - a1: Classify as Benign
    - a2: Classify as Malignant
    
    Loss matrix:
    | Loss | Classify as Benign (a1) | Classify as Malignant (a2) |
    | :--: | :---------------------: | :------------------------: |
    | Benign (C1) | 0 | 2 |
    | Malignant (C2) | 10 | 0 |
    """
    loss_matrix = {
        "C1": {"a1": 0, "a2": 2},
        "C2": {"a1": 10, "a2": 0}
    }
    return loss_matrix[true_class][predicted_class]

def calculate_expected_loss(loss_function, action, posteriors):
    """
    Calculate the expected loss (Bayes risk) for a given action
    R(a_i) = Σ L(a_i, C_j) × P(C_j|x)
    """
    expected_loss = 0
    for class_label, probability in posteriors.items():
        expected_loss += loss_function(class_label, action) * probability
    return expected_loss

def find_optimal_action(loss_function, actions, posteriors):
    """
    Find the action that minimizes the expected loss
    a* = argmin_a R(a)
    """
    expected_losses = {}
    for action in actions:
        expected_losses[action] = calculate_expected_loss(loss_function, action, posteriors)
    
    optimal_action = min(expected_losses, key=expected_losses.get)
    return optimal_action, expected_losses

def calculate_posterior_with_prior(likelihood_ratio, prior_malignant):
    """
    Calculate posterior probabilities using Bayes theorem with a prior
    P(C_j|x) = P(x|C_j)P(C_j) / P(x)
    
    Given:
    - likelihood_ratio = P(x|C2) / P(x|C1)
    - prior_malignant = P(C2)
    - prior_benign = P(C1) = 1 - P(C2)
    """
    prior_benign = 1 - prior_malignant
    
    # We need to solve for posteriors given likelihood ratio
    # P(C2|x) / P(C1|x) = [P(x|C2)/P(x|C1)] × [P(C2)/P(C1)]
    posterior_ratio = likelihood_ratio * (prior_malignant / prior_benign)
    
    # Since P(C1|x) + P(C2|x) = 1
    posterior_benign = 1 / (1 + posterior_ratio)
    posterior_malignant = posterior_ratio / (1 + posterior_ratio)
    
    return {"C1": posterior_benign, "C2": posterior_malignant}

def plot_decision_boundaries(threshold_01, threshold_asymmetric):
    """
    Plot decision boundaries for both loss functions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot probability range from 0 to 1
    x = np.linspace(0, 1, 1000)
    y = np.zeros_like(x)
    
    # Add rectangles showing decision regions for 0-1 loss
    ax.add_patch(patches.Rectangle((0, 0.05), threshold_01, 0.15, 
                                  facecolor='red', alpha=0.3))
    ax.add_patch(patches.Rectangle((threshold_01, 0.05), 1-threshold_01, 0.15, 
                                  facecolor='blue', alpha=0.3))
    
    # Add rectangles showing decision regions for asymmetric loss
    ax.add_patch(patches.Rectangle((0, -0.15), threshold_asymmetric, 0.15, 
                                  facecolor='red', alpha=0.3))
    ax.add_patch(patches.Rectangle((threshold_asymmetric, -0.15), 1-threshold_asymmetric, 0.15, 
                                  facecolor='blue', alpha=0.3))
    
    # Add probability markers
    ax.plot([0.3, 0.3], [-0.25, 0.25], 'g--', alpha=0.7)
    ax.text(0.31, 0.21, 'P(Malignant|x) = 0.7', va='center')
    
    # Add legend and annotations
    plt.text(0.5, 0.12, '0-1 Loss Decision Regions', ha='center')
    plt.text(0.5, -0.08, 'Asymmetric Loss Decision Regions', ha='center')
    plt.text(0.12, 0.12, 'Benign', ha='center', color='darkred')
    plt.text(0.12, -0.08, 'Benign', ha='center', color='darkred')
    plt.text(0.88, 0.12, 'Malignant', ha='center', color='darkblue')
    plt.text(0.88, -0.08, 'Malignant', ha='center', color='darkblue')
    
    # Add threshold labels
    plt.text(threshold_01, 0.22, f'Threshold = {threshold_01}', ha='center')
    plt.text(threshold_asymmetric, -0.22, f'Threshold = {threshold_asymmetric:.3f}', ha='center')
    
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.25, 0.25)
    plt.xlabel('Probability of Malignant - P(C2|x)')
    plt.title('Decision Boundaries for Different Loss Functions')
    plt.axis('off')
    
    return fig

def plot_bayes_risk_curves():
    """
    Plot Bayes risk curves for both loss functions to show how expected loss
    varies with the probability of malignancy
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Create probability range
    p_malignant = np.linspace(0, 1, 1000)
    
    # Calculate expected losses for 0-1 loss
    risk_benign_01 = [p for p in p_malignant]  # Risk of classifying as benign: P(Malignant|x)
    risk_malignant_01 = [1-p for p in p_malignant]  # Risk of classifying as malignant: P(Benign|x) = 1-P(Malignant|x)
    
    # Calculate expected losses for asymmetric loss
    risk_benign_asymm = [10*p for p in p_malignant]  # Risk of classifying as benign: 10*P(Malignant|x)
    risk_malignant_asymm = [2*(1-p) for p in p_malignant]  # Risk of classifying as malignant: 2*P(Benign|x)
    
    # Plot 0-1 loss curves
    ax1.plot(p_malignant, risk_benign_01, 'r-', label='Classify as Benign')
    ax1.plot(p_malignant, risk_malignant_01, 'b-', label='Classify as Malignant')
    threshold_01 = 0.5
    ax1.axvline(x=threshold_01, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.3, color='g', linestyle='--', alpha=0.7)
    ax1.fill_between(p_malignant, 0, 1, where=(p_malignant < threshold_01), color='r', alpha=0.1)
    ax1.fill_between(p_malignant, 0, 1, where=(p_malignant >= threshold_01), color='b', alpha=0.1)
    ax1.set_title('Expected Loss with 0-1 Loss Function')
    ax1.set_xlabel('Probability of Malignant - P(C2|x)')
    ax1.set_ylabel('Expected Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot asymmetric loss curves
    ax2.plot(p_malignant, risk_benign_asymm, 'r-', label='Classify as Benign')
    ax2.plot(p_malignant, risk_malignant_asymm, 'b-', label='Classify as Malignant')
    threshold_asymm = 2/12  # Analytical solution
    ax2.axvline(x=threshold_asymm, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.3, color='g', linestyle='--', alpha=0.7)
    ax2.fill_between(p_malignant, 0, 10, where=(p_malignant < threshold_asymm), color='r', alpha=0.1)
    ax2.fill_between(p_malignant, 0, 10, where=(p_malignant >= threshold_asymm), color='b', alpha=0.1)
    ax2.set_title('Expected Loss with Asymmetric Loss Function')
    ax2.set_xlabel('Probability of Malignant - P(C2|x)')
    ax2.set_ylabel('Expected Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_prior_impact():
    """
    Plot how the prior probability impacts the posterior probabilities and decisions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create prior range
    prior_malignant_range = np.linspace(0.01, 0.99, 1000)
    likelihood_ratio = 14  # Given in the problem
    
    # Calculate posterior for each prior
    posterior_malignant = []
    for prior in prior_malignant_range:
        posteriors = calculate_posterior_with_prior(likelihood_ratio, prior)
        posterior_malignant.append(posteriors["C2"])
    
    # Plot posterior vs prior
    ax.plot(prior_malignant_range, posterior_malignant, 'b-', label='Posterior P(Malignant|x)')
    
    # Add threshold lines for both loss functions
    threshold_01 = 0.5
    threshold_asymm = 2/12
    ax.axhline(y=threshold_01, color='r', linestyle='--', alpha=0.7, label='0-1 Loss Threshold')
    ax.axhline(y=threshold_asymm, color='g', linestyle='--', alpha=0.7, label='Asymmetric Loss Threshold')
    
    # Mark the specific prior of 0.05
    prior_005 = 0.05
    posteriors_005 = calculate_posterior_with_prior(likelihood_ratio, prior_005)
    posterior_005 = posteriors_005["C2"]
    ax.plot([prior_005], [posterior_005], 'ro', markersize=8)
    ax.axvline(x=prior_005, color='k', linestyle='--', alpha=0.5)
    
    # Annotations
    ax.annotate(f'Prior=0.05, Posterior={posterior_005:.3f}', 
                xy=(prior_005, posterior_005), xytext=(0.1, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=9)
    
    ax.set_title('Impact of Prior on Posterior Probability with Likelihood Ratio = 14')
    ax.set_xlabel('Prior Probability of Malignant - P(C2)')
    ax.set_ylabel('Posterior Probability of Malignant - P(C2|x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    # Define the problem parameters
    print_step_header(0, "PROBLEM DEFINITION")
    print("Biopsies classification problem:")
    print("- Classes: C1 (Benign), C2 (Malignant)")
    print("- Actions: a1 (Classify as Benign), a2 (Classify as Malignant)")
    print("- Given posteriors: P(C1|x) = 0.3, P(C2|x) = 0.7")
    
    # Original posterior probabilities
    posteriors = {"C1": 0.3, "C2": 0.7}
    actions = ["a1", "a2"]
    
    # Step 1: Zero-One Loss Calculation
    print_step_header(1, "ZERO-ONE LOSS FUNCTION CALCULATION")
    
    print_substep("Zero-One Loss Matrix")
    print("| Loss | Classify as Benign (a1) | Classify as Malignant (a2) |")
    print("| :--: | :---------------------: | :------------------------: |")
    print("| Benign (C1) | 0 | 1 |")
    print("| Malignant (C2) | 1 | 0 |")
    
    print_substep("Expected Loss Calculation for Zero-One Loss")
    # Calculate expected loss for each action using zero-one loss
    print("For action a1 (Classify as Benign):")
    loss_a1_c1 = zero_one_loss("C1", "a1")
    loss_a1_c2 = zero_one_loss("C2", "a1")
    print(f"L(a1, C1) × P(C1|x) = {loss_a1_c1} × 0.3 = {loss_a1_c1 * posteriors['C1']}")
    print(f"L(a1, C2) × P(C2|x) = {loss_a1_c2} × 0.7 = {loss_a1_c2 * posteriors['C2']}")
    risk_a1_01 = loss_a1_c1 * posteriors['C1'] + loss_a1_c2 * posteriors['C2']
    print(f"R(a1) = {loss_a1_c1 * posteriors['C1']} + {loss_a1_c2 * posteriors['C2']} = {risk_a1_01}")
    
    print("\nFor action a2 (Classify as Malignant):")
    loss_a2_c1 = zero_one_loss("C1", "a2")
    loss_a2_c2 = zero_one_loss("C2", "a2")
    print(f"L(a2, C1) × P(C1|x) = {loss_a2_c1} × 0.3 = {loss_a2_c1 * posteriors['C1']}")
    print(f"L(a2, C2) × P(C2|x) = {loss_a2_c2} × 0.7 = {loss_a2_c2 * posteriors['C2']}")
    risk_a2_01 = loss_a2_c1 * posteriors['C1'] + loss_a2_c2 * posteriors['C2']
    print(f"R(a2) = {loss_a2_c1 * posteriors['C1']} + {loss_a2_c2 * posteriors['C2']} = {risk_a2_01}")
    
    optimal_action_01, expected_losses_01 = find_optimal_action(zero_one_loss, actions, posteriors)
    print("\nExpected losses for each action:")
    print(f"R(a1) = {expected_losses_01['a1']}")
    print(f"R(a2) = {expected_losses_01['a2']}")
    
    print(f"\nOptimal decision: {optimal_action_01} (Classify as {'Benign' if optimal_action_01 == 'a1' else 'Malignant'})")
    if expected_losses_01['a1'] < expected_losses_01['a2']:
        print(f"Since R(a1) = {expected_losses_01['a1']} < R(a2) = {expected_losses_01['a2']}")
    else:
        print(f"Since R(a2) = {expected_losses_01['a2']} < R(a1) = {expected_losses_01['a1']}")
    
    # Step 2: Asymmetric Loss Calculation
    print_step_header(2, "ASYMMETRIC LOSS FUNCTION CALCULATION")
    
    print_substep("Asymmetric Loss Matrix")
    print("| Loss | Classify as Benign (a1) | Classify as Malignant (a2) |")
    print("| :--: | :---------------------: | :------------------------: |")
    print("| Benign (C1) | 0 | 2 |")
    print("| Malignant (C2) | 10 | 0 |")
    
    print_substep("Expected Loss Calculation for Asymmetric Loss")
    # Calculate expected loss for each action using asymmetric loss
    print("For action a1 (Classify as Benign):")
    loss_a1_c1_asym = asymmetric_loss("C1", "a1")
    loss_a1_c2_asym = asymmetric_loss("C2", "a1")
    print(f"L(a1, C1) × P(C1|x) = {loss_a1_c1_asym} × 0.3 = {loss_a1_c1_asym * posteriors['C1']}")
    print(f"L(a1, C2) × P(C2|x) = {loss_a1_c2_asym} × 0.7 = {loss_a1_c2_asym * posteriors['C2']}")
    risk_a1_asym = loss_a1_c1_asym * posteriors['C1'] + loss_a1_c2_asym * posteriors['C2']
    print(f"R(a1) = {loss_a1_c1_asym * posteriors['C1']} + {loss_a1_c2_asym * posteriors['C2']} = {risk_a1_asym}")
    
    print("\nFor action a2 (Classify as Malignant):")
    loss_a2_c1_asym = asymmetric_loss("C1", "a2")
    loss_a2_c2_asym = asymmetric_loss("C2", "a2")
    print(f"L(a2, C1) × P(C1|x) = {loss_a2_c1_asym} × 0.3 = {loss_a2_c1_asym * posteriors['C1']}")
    print(f"L(a2, C2) × P(C2|x) = {loss_a2_c2_asym} × 0.7 = {loss_a2_c2_asym * posteriors['C2']}")
    risk_a2_asym = loss_a2_c1_asym * posteriors['C1'] + loss_a2_c2_asym * posteriors['C2']
    print(f"R(a2) = {loss_a2_c1_asym * posteriors['C1']} + {loss_a2_c2_asym * posteriors['C2']} = {risk_a2_asym}")
    
    optimal_action_asym, expected_losses_asym = find_optimal_action(asymmetric_loss, actions, posteriors)
    print("\nExpected losses for each action:")
    print(f"R(a1) = {expected_losses_asym['a1']}")
    print(f"R(a2) = {expected_losses_asym['a2']}")
    
    print(f"\nOptimal decision: {optimal_action_asym} (Classify as {'Benign' if optimal_action_asym == 'a1' else 'Malignant'})")
    if expected_losses_asym['a1'] < expected_losses_asym['a2']:
        print(f"Since R(a1) = {expected_losses_asym['a1']} < R(a2) = {expected_losses_asym['a2']}")
    else:
        print(f"Since R(a2) = {expected_losses_asym['a2']} < R(a1) = {expected_losses_asym['a1']}")
    
    # Step 3: Simplifying Bayes Decision Rule for Zero-One Loss
    print_step_header(3, "SIMPLIFYING BAYES DECISION RULE FOR ZERO-ONE LOSS")
    
    print_derivation("Simplification of the Bayes decision rule with 0-1 loss", [
        "Start with the Bayes decision rule: α̂(x) = argmin_i Σ L_ij P(C_j|x)",
        "For 0-1 loss, L_ij = 1 - δ_ij, where δ_ij = 1 if i=j and 0 otherwise",
        "Substituting: α̂(x) = argmin_i Σ (1 - δ_ij) P(C_j|x)",
        "Expanding: α̂(x) = argmin_i [Σ P(C_j|x) - Σ δ_ij P(C_j|x)]",
        "Since Σ P(C_j|x) = 1 (sum of probabilities): α̂(x) = argmin_i [1 - Σ δ_ij P(C_j|x)]",
        "The only term with δ_ij = 1 is when j = i: α̂(x) = argmin_i [1 - P(C_i|x)]",
        "Since we want to minimize [1 - P(C_i|x)], this is equivalent to maximizing P(C_i|x)",
        "Therefore: α̂(x) = argmax_i P(C_i|x)"
    ])
    
    print("\nThis shows that the Bayes minimum risk decision with 0-1 loss is equivalent to the MAP estimate.")
    print(f"For our problem: argmax_i P(C_i|x) = argmax({posteriors['C1']}, {posteriors['C2']}) = C2 (Malignant)")
    print("Therefore, the optimal action is a2 (Classify as Malignant), which matches our result in Step 1.")
    
    # Step 4: Deriving the Decision Threshold for Asymmetric Loss
    print_step_header(4, "DERIVING DECISION THRESHOLD FOR ASYMMETRIC LOSS")
    
    print_derivation("Derivation of the decision threshold", [
        "We classify as malignant (a2) when R(a2) < R(a1)",
        "R(a2) = L(a2,C1)P(C1|x) + L(a2,C2)P(C2|x) = 2×P(C1|x) + 0×P(C2|x) = 2×P(C1|x)",
        "R(a1) = L(a1,C1)P(C1|x) + L(a1,C2)P(C2|x) = 0×P(C1|x) + 10×P(C2|x) = 10×P(C2|x)",
        "For a2 to be optimal: 2×P(C1|x) < 10×P(C2|x)",
        "Using P(C1|x) = 1 - P(C2|x): 2×(1-P(C2|x)) < 10×P(C2|x)",
        "Expanding: 2 - 2×P(C2|x) < 10×P(C2|x)",
        "Rearranging: 2 < 10×P(C2|x) + 2×P(C2|x) = 12×P(C2|x)",
        "Solving for P(C2|x): 2/12 < P(C2|x)",
        "Therefore, the decision threshold is t = 2/12 = 1/6 = 0.1667"
    ])
    
    threshold_asymmetric = 2/12
    print(f"\nDecision rule: Classify as malignant when P(C2|x) > {threshold_asymmetric}")
    print(f"Since P(C2|x) = 0.7 > {threshold_asymmetric}, the optimal action is a2 (Classify as Malignant)")
    print("This matches our result in Step 2.")
    
    # Step 5: Updating with a Prior
    print_step_header(5, "UPDATING WITH A PRIOR DISTRIBUTION")
    
    print_substep("Calculating Posterior Probabilities with Prior")
    print("Given:")
    print("- Prior probability of malignant: P(C2) = 0.05")
    print("- Likelihood ratio: P(x|C2)/P(x|C1) = 14")
    
    # Calculate new posteriors using the prior
    prior_malignant = 0.05
    likelihood_ratio = 14
    new_posteriors = calculate_posterior_with_prior(likelihood_ratio, prior_malignant)
    
    print("\nCalculation:")
    print(f"P(C2|x)/P(C1|x) = [P(x|C2)/P(x|C1)] × [P(C2)/P(C1)] = 14 × [0.05/(1-0.05)] = 14 × 0.0526 = {14 * 0.0526:.4f}")
    print(f"Let's denote P(C2|x) as p. Then P(C1|x) = 1-p.")
    print(f"We have: p/(1-p) = {14 * 0.0526:.4f}")
    print(f"Solving for p: p = {14 * 0.0526:.4f} × (1-p)")
    print(f"p = {14 * 0.0526:.4f} - {14 * 0.0526:.4f}p")
    print(f"p + {14 * 0.0526:.4f}p = {14 * 0.0526:.4f}")
    print(f"p × (1 + {14 * 0.0526:.4f}) = {14 * 0.0526:.4f}")
    print(f"p = {14 * 0.0526:.4f} / (1 + {14 * 0.0526:.4f}) = {14 * 0.0526 / (1 + 14 * 0.0526):.4f}")
    
    print(f"\nUpdated posterior probabilities:")
    print(f"P(C1|x) = {new_posteriors['C1']:.4f}")
    print(f"P(C2|x) = {new_posteriors['C2']:.4f}")
    
    print_substep("Determining Optimal Decisions with Updated Posteriors")
    
    # Zero-One Loss with Updated Posteriors
    optimal_action_01_new, expected_losses_01_new = find_optimal_action(zero_one_loss, actions, new_posteriors)
    print("Zero-One Loss with Updated Posteriors:")
    print(f"R(a1) = {expected_losses_01_new['a1']:.4f}")
    print(f"R(a2) = {expected_losses_01_new['a2']:.4f}")
    print(f"Optimal decision: {optimal_action_01_new} (Classify as {'Benign' if optimal_action_01_new == 'a1' else 'Malignant'})")
    
    # Asymmetric Loss with Updated Posteriors
    optimal_action_asym_new, expected_losses_asym_new = find_optimal_action(asymmetric_loss, actions, new_posteriors)
    print("\nAsymmetric Loss with Updated Posteriors:")
    print(f"R(a1) = {expected_losses_asym_new['a1']:.4f}")
    print(f"R(a2) = {expected_losses_asym_new['a2']:.4f}")
    print(f"Optimal decision: {optimal_action_asym_new} (Classify as {'Benign' if optimal_action_asym_new == 'a1' else 'Malignant'})")
    
    print_substep("Explaining the Relationship between Prior and Expected Loss")
    print("The prior probability influences the posterior probability, which in turn affects the expected loss calculation:")
    print("1. Even with a high likelihood ratio of 14 (strong evidence for malignancy), the low prior of 0.05 results in a lower posterior probability of malignancy.")
    print(f"2. This changes our decision for the 0-1 loss function from 'Malignant' to 'Benign', because P(C2|x) = {new_posteriors['C2']:.4f} < 0.5.")
    print(f"3. For the asymmetric loss, our decision is still 'Malignant' because P(C2|x) = {new_posteriors['C2']:.4f} > {threshold_asymmetric}, the asymmetric threshold.")
    print("4. This demonstrates how the prior encodes our belief about class distribution, while the loss function encodes the costs of different decision errors.")
    
    # Step 6: Extending to More Complex Problems
    print_step_header(6, "EXTENDING TO MORE COMPLEX PROBLEMS")
    
    print("The Bayes risk minimization framework can be extended to more complex problems as follows:")
    print("\n1. Multi-class extension:")
    print("   - For K classes, use a K×K loss matrix L where L_ij is the cost of predicting class i when true class is j")
    print("   - The Bayes decision rule remains: α̂(x) = argmin_i Σ_j L_ij P(C_j|x)")
    
    print("\n2. Custom loss functions:")
    print("   - Any arbitrary loss matrix can be used, reflecting domain-specific costs of different error types")
    print("   - For example, in medical diagnostics, missing different diseases may have different costs")
    print("   - The framework automatically balances these different costs with class probabilities")
    
    print("\n3. Decision regions:")
    print("   - The decision boundaries in feature space are determined by both the posterior probabilities and the loss function")
    print("   - The optimal decision for a given example depends on where it falls in these decision regions")
    
    print("\n4. Regression extension:")
    print("   - For continuous outputs, the sum becomes an integral over the conditional density")
    print("   - The optimal decision minimizes: α̂(x) = argmin_a ∫ L(a,y) p(y|x) dy")
    print("   - For squared error loss, this gives the conditional mean E[Y|X=x]")
    print("   - For absolute error loss, this gives the conditional median")
    
    print("\n5. Cost-sensitive learning:")
    print("   - Machine learning algorithms can be modified to directly optimize the expected loss")
    print("   - This is especially important when class distributions are imbalanced or errors have asymmetric costs")
    
    # Generate and save figures
    print_step_header(7, "GENERATING VISUALIZATIONS")
    
    # Decision boundaries figure
    threshold_01 = 0.5
    threshold_asymmetric = 2/12
    fig_boundaries = plot_decision_boundaries(threshold_01, threshold_asymmetric)
    save_figure(fig_boundaries, "decision_boundaries.png")
    print("Generated visualization: decision_boundaries.png")
    
    # Bayes risk curves
    fig_risk = plot_bayes_risk_curves()
    save_figure(fig_risk, "bayes_risk_curves.png")
    print("Generated visualization: bayes_risk_curves.png")
    
    # Prior impact
    fig_prior = plot_prior_impact()
    save_figure(fig_prior, "prior_impact.png")
    print("Generated visualization: prior_impact.png")
    
    print("\nAll visualizations saved to ../Images/L2_7_Quiz_32/")
    
if __name__ == "__main__":
    main() 