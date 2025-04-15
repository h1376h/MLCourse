import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import os

def calculate_dice_probabilities():
    """Calculate all probabilities for the dice problem"""
    # Generate all possible outcomes (36 total)
    dice_outcomes = list(itertools.product(range(1, 7), repeat=2))
    
    # Calculate probabilities
    # Event A: sum equals 7
    event_a = [sum(dice) == 7 for dice in dice_outcomes]
    p_a = sum(event_a) / len(dice_outcomes)
    
    # Event B: at least one die shows 6
    event_b = [6 in dice for dice in dice_outcomes]
    p_b = sum(event_b) / len(dice_outcomes)
    
    # Event A ∩ B: sum equals 7 AND at least one die shows 6
    event_a_and_b = [sum(dice) == 7 and 6 in dice for dice in dice_outcomes]
    p_a_and_b = sum(event_a_and_b) / len(dice_outcomes)
    
    # Event A ∪ B: sum equals 7 OR at least one die shows 6
    event_a_or_b = [sum(dice) == 7 or 6 in dice for dice in dice_outcomes]
    p_a_or_b = sum(event_a_or_b) / len(dice_outcomes)
    
    # Check independence
    is_independent = abs(p_a * p_b - p_a_and_b) < 1e-10
    
    return {
        'p_a': p_a,
        'p_b': p_b,
        'p_a_and_b': p_a_and_b,
        'p_a_or_b': p_a_or_b,
        'is_independent': is_independent
    }

def plot_dice_outcomes(save_path=None):
    """Create a visualization of all possible dice outcomes"""
    # Generate all possible outcomes
    dice_outcomes = list(itertools.product(range(1, 7), repeat=2))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each outcome
    for i, (die1, die2) in enumerate(dice_outcomes):
        x = i % 6
        y = i // 6
        
        # Color based on events
        color = 'lightgray'  # default
        if die1 + die2 == 7:
            color = 'lightblue'  # Event A
        if 6 in (die1, die2):
            color = 'lightgreen'  # Event B
        if die1 + die2 == 7 and 6 in (die1, die2):
            color = 'yellow'  # Event A ∩ B
        
        # Draw rectangle
        rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
        # Add text
        plt.text(x + 0.5, y + 0.5, f'({die1},{die2})', 
                ha='center', va='center', fontsize=10)
    
    # Set up the plot
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_xticklabels(range(1, 7))
    ax.set_yticklabels(range(1, 7))
    ax.set_xlabel('Die 1')
    ax.set_ylabel('Die 2')
    ax.set_title('All Possible Outcomes of Two Dice')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Sum = 7'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='At least one 6'),
        plt.Rectangle((0, 0), 1, 1, facecolor='yellow', edgecolor='black', label='Both conditions'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Neither')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_probability_distribution(save_path=None):
    """Plot the probability distribution of dice sums"""
    # Generate all possible outcomes
    dice_outcomes = list(itertools.product(range(1, 7), repeat=2))
    
    # Calculate sum frequencies
    sums = [sum(dice) for dice in dice_outcomes]
    sum_counts = Counter(sums)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    x = list(range(2, 13))
    y = [sum_counts[s] / len(dice_outcomes) for s in x]
    
    bars = ax.bar(x, y, color='skyblue', edgecolor='black')
    
    # Highlight sum of 7
    for i, bar in enumerate(bars):
        if x[i] == 7:
            bar.set_color('lightblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    ax.set_xlabel('Sum of Two Dice')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution of Dice Sums')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 1"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_1")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 1 of the L2.1 Probability quiz...")
    
    # Calculate probabilities
    probs = calculate_dice_probabilities()
    print("\nCalculated Probabilities:")
    print(f"P(A): {probs['p_a']:.4f}")
    print(f"P(B): {probs['p_b']:.4f}")
    print(f"P(A ∩ B): {probs['p_a_and_b']:.4f}")
    print(f"P(A ∪ B): {probs['p_a_or_b']:.4f}")
    print(f"Are A and B independent? {probs['is_independent']}")
    
    # Generate visualizations
    plot_dice_outcomes(save_path=os.path.join(save_dir, "dice_outcomes.png"))
    print("1. Dice outcomes visualization created")
    
    plot_probability_distribution(save_path=os.path.join(save_dir, "probability_distribution.png"))
    print("2. Probability distribution visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 