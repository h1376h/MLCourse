import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import os

def calculate_ball_probabilities():
    """Calculate all probabilities for the ball problem"""
    # Define the bag contents
    bag = ['R'] * 5 + ['B'] * 3 + ['G'] * 2  # 5 Red, 3 Blue, 2 Green
    
    # Generate all possible combinations of drawing 2 balls
    combinations = list(itertools.combinations(bag, 2))
    
    # Calculate probabilities
    # 1. Drawing 2 red balls
    two_red = [c == ('R', 'R') for c in combinations]
    p_two_red = sum(two_red) / len(combinations)
    
    # 2. Drawing exactly 1 blue ball
    one_blue = [c.count('B') == 1 for c in combinations]
    p_one_blue = sum(one_blue) / len(combinations)
    
    # 3. Drawing no green balls
    no_green = ['G' not in c for c in combinations]
    p_no_green = sum(no_green) / len(combinations)
    
    # 4. Drawing at least one ball of each color
    # This is impossible since we're only drawing 2 balls and there are 3 colors
    p_all_colors = 0
    
    return {
        'p_two_red': p_two_red,
        'p_one_blue': p_one_blue,
        'p_no_green': p_no_green,
        'p_all_colors': p_all_colors
    }

def plot_ball_combinations(save_path=None):
    """Create a visualization of all possible ball combinations"""
    # Define the bag contents
    bag = ['R'] * 5 + ['B'] * 3 + ['G'] * 2
    
    # Generate all possible combinations
    combinations = list(itertools.combinations(bag, 2))
    
    # Count each type of combination
    combo_counts = Counter(combinations)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different combinations
    colors = {
        ('R', 'R'): 'red',
        ('B', 'R'): 'purple',
        ('G', 'R'): 'orange',
        ('B', 'B'): 'blue',
        ('B', 'G'): 'green',
        ('G', 'G'): 'lightgreen'
    }
    
    # Plot bars
    x = np.arange(len(combo_counts))
    bars = []
    labels = []
    
    for i, (combo, count) in enumerate(combo_counts.items()):
        color = colors[tuple(sorted(combo))]
        bar = ax.bar(i, count, color=color, edgecolor='black')
        bars.append(bar)
        labels.append(f"{combo[0]}{combo[1]}")
    
    # Customize the plot
    ax.set_xlabel('Combination Type')
    ax.set_ylabel('Number of Ways')
    ax.set_title('Possible Combinations of Drawing 2 Balls')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value labels
    for bar in bars:
        height = bar[0].get_height()
        ax.text(bar[0].get_x() + bar[0].get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Two Red'),
        plt.Rectangle((0, 0), 1, 1, facecolor='purple', edgecolor='black', label='Red and Blue'),
        plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='black', label='Red and Green'),
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', label='Two Blue'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='Blue and Green'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Two Green')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_probability_breakdown(save_path=None):
    """Create a pie chart showing the probability breakdown"""
    # Calculate probabilities
    probs = calculate_ball_probabilities()
    
    # Define labels and values for mutually exclusive categories
    labels = ['Two Red', 'One Blue (not Red)', 'Two Blue', 'Contains Green', 'Other']
    
    # Calculate mutually exclusive probabilities
    bag = ['R'] * 5 + ['B'] * 3 + ['G'] * 2
    combinations = list(itertools.combinations(bag, 2))
    total = len(combinations)
    
    # Two Red
    two_red = sum(c == ('R', 'R') for c in combinations) / total
    
    # One Blue (not with Red)
    one_blue_not_red = sum((c.count('B') == 1 and 'R' not in c) for c in combinations) / total
    
    # Two Blue
    two_blue = sum(c == ('B', 'B') for c in combinations) / total
    
    # Contains Green
    contains_green = sum('G' in c for c in combinations) / total
    
    # Other (remaining combinations - Red-Blue pairs)
    other = 1 - (two_red + one_blue_not_red + two_blue + contains_green)
    
    sizes = [two_red, one_blue_not_red, two_blue, contains_green, other]
    colors = ['red', 'lightblue', 'blue', 'green', 'gray']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Probability Breakdown of Drawing 2 Balls')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_key_probabilities(save_path=None):
    """Create a horizontal bar chart showing the key probabilities mentioned in the markdown"""
    # Calculate probabilities
    probs = calculate_ball_probabilities()
    
    # Define events, values and colors for the key probabilities
    events = ['Two Red', 'One Blue', 'No Green']
    values = [probs['p_two_red'], probs['p_one_blue'], probs['p_no_green']]
    colors = ['red', 'blue', 'gray']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars
    y_pos = np.arange(len(events))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add labels and percentages
    ax.set_yticks(y_pos)
    ax.set_yticklabels(events, fontsize=12)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Probabilities of Different Events', fontsize=14)
    
    # Set x-axis to show percentages
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                va='center', fontsize=12, fontweight='bold')
    
    # Add grid and tight layout
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    
    # Add note about non-mutually exclusive events
    ax.text(0.5, -0.15, 
            "Note: These probabilities sum to more than 100% because the events are not mutually exclusive.",
            ha='center', va='center', transform=ax.transAxes, fontsize=10, fontstyle='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 2"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_2")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 2 of the L2.1 Probability quiz...")
    
    # Calculate probabilities
    probs = calculate_ball_probabilities()
    print("\nCalculated Probabilities:")
    print(f"P(Two Red): {probs['p_two_red']:.4f}")
    print(f"P(One Blue): {probs['p_one_blue']:.4f}")
    print(f"P(No Green): {probs['p_no_green']:.4f}")
    print(f"P(All Colors): {probs['p_all_colors']:.4f}")
    
    # Generate visualizations
    plot_ball_combinations(save_path=os.path.join(save_dir, "ball_combinations.png"))
    print("1. Ball combinations visualization created")
    
    plot_probability_breakdown(save_path=os.path.join(save_dir, "probability_breakdown.png"))
    print("2. Probability breakdown visualization created")
    
    plot_key_probabilities(save_path=os.path.join(save_dir, "key_probabilities.png"))
    print("3. Key probabilities visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 