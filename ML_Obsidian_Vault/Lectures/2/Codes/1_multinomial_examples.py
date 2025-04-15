import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import os
from math import factorial

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

def plot_multinomial_distribution(counts, probs, labels, title, save_path=None):
    """Plot the multinomial distribution with observed counts and probabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Observed counts
    ax1.bar(labels, counts, color='skyblue', edgecolor='navy')
    ax1.set_title(f"{title} - Observed Counts")
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Count')
    
    # Label bars with actual values
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, str(v), ha='center')
    
    # Plot 2: Probability distribution
    ax2.bar(labels, probs, color='lightgreen', edgecolor='darkgreen')
    ax2.set_title(f"{title} - Probability Distribution")
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Probability')
    
    # Label bars with probability values
    for i, v in enumerate(probs):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks

def calculate_multinomial_probability(n, counts, probs):
    """Calculate the probability of observing specific counts in a multinomial distribution"""
    # Calculate the multinomial coefficient
    multinomial_coeff = factorial(n) / np.prod([factorial(x) for x in counts])
    
    # Calculate the probability component
    prob_component = np.prod([p**x for p, x in zip(probs, counts)])
    
    # Total probability
    total_prob = multinomial_coeff * prob_component
    
    return total_prob, multinomial_coeff, prob_component

def analyze_multinomial_example(name, n, counts, probs, labels, context, save_dir=None):
    """Analyze a multinomial distribution example with detailed steps"""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Verify parameters
    print("\nStep 1: Verify Parameters")
    print(f"- Total trials (n): {n}")
    print(f"- Observed counts: {dict(zip(labels, counts))}")
    print(f"- Probabilities: {dict(zip(labels, probs))}")
    print(f"- Sum of counts: {sum(counts)} (should equal n)")
    print(f"- Sum of probabilities: {sum(probs):.2f} (should equal 1)")
    
    # Step 2: Calculate probability
    print("\nStep 2: Calculate Probability")
    total_prob, multinomial_coeff, prob_component = calculate_multinomial_probability(n, counts, probs)
    print(f"- Multinomial coefficient: {multinomial_coeff}")
    print(f"- Probability component: {prob_component}")
    print(f"- Total probability: {total_prob:.6f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"multinomial_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 3: Visualize
    print("\nStep 3: Visualization")
    plot_multinomial_distribution(counts, probs, labels, name, save_path)
    
    # Step 4: Expected values
    print("\nStep 4: Expected Values")
    expected_counts = [n * p for p in probs]
    print("Expected counts for each category:")
    for label, expected in zip(labels, expected_counts):
        print(f"- {label}: {expected:.2f}")
    
    return total_prob, save_path

def generate_multinomial_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze different scenarios using the Multinomial distribution!
    Each example will demonstrate how to calculate probabilities for categorical outcomes.
    """)

    # Example 1: Dice Rolling
    dice_n = 10
    dice_counts = [2, 3, 1, 2, 1, 1]  # Counts for faces 1-6
    dice_probs = [1/6] * 6  # Equal probabilities for fair die
    dice_labels = ["Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"]
    dice_context = """
    A fair six-sided die is rolled 10 times.
    - We observe: 2 ones, 3 twos, 1 three, 2 fours, 1 five, and 1 six
    - Each face has equal probability (1/6)
    """
    prob, path = analyze_multinomial_example("Dice Rolling", dice_n, dice_counts, dice_probs, dice_labels, dice_context, save_dir)
    results["Dice Rolling"] = {"probability": prob, "path": path}

    # Example 2: Text Analysis
    text_n = 100
    text_counts = [20, 12, 9, 4, 55]  # Counts for each word category
    text_probs = [0.15, 0.10, 0.08, 0.05, 0.62]  # Probabilities for each word category
    text_labels = ["computer", "software", "data", "digital", "other words"]
    text_context = """
    Analyzing word frequencies in a 100-word technology document.
    - Probabilities: computer(0.15), software(0.10), data(0.08), digital(0.05), other(0.62)
    - Observed counts: computer(20), software(12), data(9), digital(4), other(55)
    """
    prob, path = analyze_multinomial_example("Text Analysis", text_n, text_counts, text_probs, text_labels, text_context, save_dir)
    results["Text Analysis"] = {"probability": prob, "path": path}

    # Example 3: Topic Modeling
    topic_n = 50
    topic_counts = [28, 15, 7]  # Counts for each topic
    topic_probs = [0.6, 0.3, 0.1]  # Probabilities for each topic
    topic_labels = ["Science", "Politics", "Art"]
    topic_context = """
    Topic modeling with 50 words from 3 topics.
    - Probabilities: Science(0.6), Politics(0.3), Art(0.1)
    - Observed counts: Science(28), Politics(15), Art(7)
    """
    prob, path = analyze_multinomial_example("Topic Modeling", topic_n, topic_counts, topic_probs, topic_labels, topic_context, save_dir)
    results["Topic Modeling"] = {"probability": prob, "path": path}
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    save_dir = os.path.join(images_dir, "multinomial_examples")
    generate_multinomial_examples(save_dir) 