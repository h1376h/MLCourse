import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_medical_diagnostics_probabilities():
    """Calculate all probabilities for the medical diagnostics problem"""
    # Given values
    p_a = 0.05  # P(A) - probability that patient has disease X
    tpr = 0.92  # P(B|A) - true positive rate/sensitivity
    fpr = 0.08  # P(B|not A) - false positive rate
    
    # Calculate P(not A)
    p_not_a = 1 - p_a
    
    # Calculate P(B) using the law of total probability
    p_b = (tpr * p_a) + (fpr * p_not_a)
    
    # Calculate P(A|B) using Bayes' theorem
    p_a_given_b = (tpr * p_a) / p_b
    
    return {
        'P(A)': p_a,
        'P(not A)': p_not_a,
        'P(B|A)': tpr,
        'P(B|not A)': fpr,
        'P(B)': p_b,
        'P(A|B)': p_a_given_b,
        'Reliability': p_a_given_b > 0.5
    }

def plot_confusion_matrix(probs, save_path=None):
    """Create a visual representation of the confusion matrix"""
    # Extract values for the confusion matrix
    p_a = probs['P(A)']
    p_not_a = probs['P(not A)']
    tpr = probs['P(B|A)']
    fpr = probs['P(B|not A)']
    tnr = 1 - fpr
    fnr = 1 - tpr
    
    # Calculate the four cells of confusion matrix
    true_positive = p_a * tpr
    false_positive = p_not_a * fpr
    false_negative = p_a * fnr
    true_negative = p_not_a * tnr
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the confusion matrix visualization
    confusion_matrix = np.array([
        [true_positive, false_positive],
        [false_negative, true_negative]
    ])
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Probability', rotation=-90, va="bottom")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actually Positive', 'Actually Negative'])
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.4f}',
                          ha="center", va="center", 
                          color="white" if confusion_matrix[i, j] > 0.1 else "black")
    
    # Set title
    ax.set_title("Confusion Matrix Probabilities")
    fig.tight_layout()
    
    # Add custom annotations
    plt.figtext(0.2, 0.02, f"True Positive Rate (Sensitivity): {tpr:.4f}", ha="center")
    plt.figtext(0.8, 0.02, f"False Positive Rate: {fpr:.4f}", ha="center")
    plt.figtext(0.2, 0.98, f"P(Disease): {p_a:.4f}", ha="center")
    plt.figtext(0.8, 0.98, f"P(No Disease): {p_not_a:.4f}", ha="center")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_bayes_theorem(probs, save_path=None):
    """Create a visualization of Bayes' theorem application"""
    # Extract relevant probabilities
    p_a = probs['P(A)']
    p_b_given_a = probs['P(B|A)']
    p_b = probs['P(B)']
    p_a_given_b = probs['P(A|B)']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    bar_width = 0.35
    r1 = np.arange(4)
    
    # Create bars
    values = [p_a, p_b_given_a, p_b, p_a_given_b]
    bars = ax.bar(r1, values, bar_width, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    
    # Add labels
    ax.set_xticks(r1)
    ax.set_xticklabels(['P(A)', 'P(B|A)', 'P(B)', 'P(A|B)'])
    ax.set_ylabel('Probability')
    ax.set_title('Bayes\' Theorem Visualization')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add formula for Bayes' theorem
    formula = r"$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$"
    computation = f"= {p_b_given_a:.4f} × {p_a:.4f} / {p_b:.4f} = {p_a_given_b:.4f}"
    
    plt.figtext(0.5, 0.01, formula + "\n" + computation, ha="center", 
                fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Add grid
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_positive_predictive_value(save_path=None):
    """Create a visualization showing how P(A|B) changes with disease prevalence"""
    # Define range of disease prevalence values
    prevalence_values = np.linspace(0.001, 0.2, 100)
    
    # Fixed sensitivity and specificity
    sensitivity = 0.92  # P(B|A)
    specificity = 0.92  # P(not B|not A) = 1 - FPR
    
    # Calculate PPV for each prevalence value
    ppv_values = []
    for prev in prevalence_values:
        p_b = (sensitivity * prev) + ((1 - specificity) * (1 - prev))
        ppv = (sensitivity * prev) / p_b
        ppv_values.append(ppv)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PPV curve
    ax.plot(prevalence_values, ppv_values, 'b-', linewidth=2)
    
    # Add point for our specific case
    probs = calculate_medical_diagnostics_probabilities()
    ax.plot(probs['P(A)'], probs['P(A|B)'], 'ro', markersize=8)
    ax.annotate(f"Current case: P(A)={probs['P(A)']:.4f}, P(A|B)={probs['P(A|B)']:.4f}",
                xy=(probs['P(A)'], probs['P(A|B)']),
                xytext=(probs['P(A)']+0.03, probs['P(A|B)']-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Add a reference line for 0.5 probability
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    ax.text(0.15, 0.51, "PPV = 0.5 (50-50 chance)", fontsize=10)
    
    # Add labels and title
    ax.set_xlabel('Disease Prevalence P(A)')
    ax.set_ylabel('Positive Predictive Value P(A|B)')
    ax.set_title('How Positive Predictive Value Changes with Disease Prevalence')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 5"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_5")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 5 of the L2.1 Probability quiz...")
    
    # Calculate probabilities
    probs = calculate_medical_diagnostics_probabilities()
    print("\nGiven Information:")
    print(f"Disease prevalence P(A): {probs['P(A)']:.4f}")
    print(f"True positive rate (sensitivity) P(B|A): {probs['P(B|A)']:.4f}")
    print(f"False positive rate P(B|not A): {probs['P(B|not A)']:.4f}")
    
    print("\nCalculated Probabilities:")
    print(f"Probability of positive test P(B): {probs['P(B)']:.4f}")
    print(f"Positive predictive value P(A|B): {probs['P(A|B)']:.4f}")
    print(f"Is the model reliable? {probs['Reliability']} (PPV > 0.5)")
    
    # Generate visualizations
    plot_confusion_matrix(probs, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    print("1. Confusion matrix visualization created")
    
    plot_bayes_theorem(probs, save_path=os.path.join(save_dir, "bayes_theorem.png"))
    print("2. Bayes' theorem visualization created")
    
    plot_positive_predictive_value(save_path=os.path.join(save_dir, "ppv_curve.png"))
    print("3. Positive predictive value curve created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 