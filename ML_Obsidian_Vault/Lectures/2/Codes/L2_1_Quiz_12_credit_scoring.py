import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

def calculate_credit_scoring_probabilities():
    """
    Calculate probabilities for the credit scoring system scenario.
    
    Returns:
        Dictionary with calculated probabilities
    """
    # Define given probabilities
    # Prior probabilities
    p_low = 0.20    # 20% of applicants have low income
    p_medium = 0.30 # 30% have medium income
    p_high = 0.50   # 50% have high income
    
    # Conditional probabilities of default given income level
    p_default_given_low = 0.25    # 25% default probability for low income
    p_default_given_medium = 0.10 # 10% default probability for medium income
    p_default_given_high = 0.05   # 5% default probability for high income
    
    # Calculate joint probabilities
    # P(Low, Default) - probability of low income and default
    p_low_and_default = p_low * p_default_given_low
    # P(Medium, Default) - probability of medium income and default
    p_medium_and_default = p_medium * p_default_given_medium
    # P(High, Default) - probability of high income and default
    p_high_and_default = p_high * p_default_given_high
    
    # P(Low, No Default) - probability of low income and no default
    p_low_and_no_default = p_low * (1 - p_default_given_low)
    # P(Medium, No Default) - probability of medium income and no default
    p_medium_and_no_default = p_medium * (1 - p_default_given_medium)
    # P(High, No Default) - probability of high income and no default
    p_high_and_no_default = p_high * (1 - p_default_given_high)
    
    # Task 1: Calculate the overall probability of default for a randomly selected loan applicant
    p_default = p_low_and_default + p_medium_and_default + p_high_and_default
    
    # Task 2: P(Low | Default) - probability borrower is from low income given they defaulted
    p_low_given_default = p_low_and_default / p_default
    
    # Task 3: P(High | Default) - probability borrower is from high income given they defaulted
    p_high_given_default = p_high_and_default / p_default
    
    # Task 4: Expected default rate if only medium and high income applicants are approved
    p_medium_or_high = p_medium + p_high
    p_default_medium_or_high = (p_medium_and_default + p_high_and_default) / p_medium_or_high
    
    # Calculate remaining probability P(Medium | Default) for completeness
    p_medium_given_default = p_medium_and_default / p_default
    
    # For visualizations - totals for each income category and overall
    total_default = p_default
    total_no_default = 1 - p_default
    
    return {
        # Prior probabilities
        'p_low': p_low,
        'p_medium': p_medium,
        'p_high': p_high,
        
        # Conditional probabilities
        'p_default_given_low': p_default_given_low,
        'p_default_given_medium': p_default_given_medium,
        'p_default_given_high': p_default_given_high,
        
        # Joint probabilities
        'p_low_and_default': p_low_and_default,
        'p_medium_and_default': p_medium_and_default,
        'p_high_and_default': p_high_and_default,
        'p_low_and_no_default': p_low_and_no_default,
        'p_medium_and_no_default': p_medium_and_no_default,
        'p_high_and_no_default': p_high_and_no_default,
        
        # Task results
        'p_default': p_default,
        'p_low_given_default': p_low_given_default,
        'p_medium_given_default': p_medium_given_default,
        'p_high_given_default': p_high_given_default,
        'p_default_medium_or_high': p_default_medium_or_high,
        
        # For visualizations
        'total_default': total_default,
        'total_no_default': total_no_default
    }

def create_joint_probability_visualization(probs, save_path=None):
    """
    Create a visual representation of the joint probability distribution.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    # Create a 3x2 joint probability matrix
    joint_matrix = np.array([
        [probs['p_low_and_default'], probs['p_low_and_no_default']],      # Low income
        [probs['p_medium_and_default'], probs['p_medium_and_no_default']], # Medium income
        [probs['p_high_and_default'], probs['p_high_and_no_default']]      # High income
    ])
    
    # Labels
    y_labels = ['Low\nIncome', 'Medium\nIncome', 'High\nIncome']
    x_labels = ['Default', 'No Default']
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a custom colormap
    cmap = plt.cm.Blues
    
    # Plot the heatmap
    ax = plt.gca()
    im = ax.imshow(joint_matrix, cmap=cmap)
    
    # Add text annotations showing probabilities
    for i in range(3):
        for j in range(2):
            text_color = "white" if joint_matrix[i, j] > 0.1 else "black"
            ax.text(j, i, f'{joint_matrix[i, j]:.4f}', 
                   ha="center", va="center", 
                   color=text_color,
                   fontsize=14, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_yticklabels(y_labels, fontsize=13)
    
    # Add title
    plt.title('Joint Probability Distribution\nIncome Level vs Loan Default', fontsize=16, pad=10)
    
    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Joint Probability', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Joint probability visualization saved to {save_path}")
    
    plt.close()

def create_conditional_probability_barchart(probs, save_path=None):
    """
    Create a bar chart showing conditional probabilities of default given income level.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Data for the bar chart
    categories = ['Low Income', 'Medium Income', 'High Income']
    default_probs = [probs['p_default_given_low'], probs['p_default_given_medium'], probs['p_default_given_high']]
    
    # Create bar chart
    bars = plt.bar(categories, default_probs, color=['#d73027', '#fee090', '#4575b4'])
    
    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Income Level', fontsize=14)
    plt.ylabel('Probability of Default', fontsize=14)
    plt.title('Conditional Probability of Default Given Income Level', fontsize=16)
    
    # Set y-axis limit to make sure the text annotations are visible
    plt.ylim(0, max(default_probs) + 0.05)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Conditional probability bar chart saved to {save_path}")
    
    plt.close()

def create_bayes_visualization(probs, save_path=None):
    """
    Create a visualization to illustrate Bayes' theorem application.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create data for stacked bar chart
    categories = ['All Defaults']
    low_portion = [probs['p_low_given_default']]
    medium_portion = [probs['p_medium_given_default']]
    high_portion = [probs['p_high_given_default']]
    
    # Create stacked bar chart
    plt.bar(categories, low_portion, label=f'Low Income ({low_portion[0]:.4f})', color='#d73027')
    plt.bar(categories, medium_portion, bottom=low_portion, label=f'Medium Income ({medium_portion[0]:.4f})', color='#fee090')
    plt.bar(categories, high_portion, bottom=[low_portion[0] + medium_portion[0]], label=f'High Income ({high_portion[0]:.4f})', color='#4575b4')
    
    # Add labels and title
    plt.ylabel('Posterior Probability', fontsize=14)
    plt.title('Posterior Probabilities of Income Categories Given Default', fontsize=16)
    plt.legend(loc='upper right')
    
    # Remove x-axis ticks
    plt.xticks(fontsize=14)
    
    # Add y-axis grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bayes visualization saved to {save_path}")
    
    plt.close()

def create_loan_approval_comparison(probs, save_path=None):
    """
    Create a visualization comparing default rates for different approval policies.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Data for the bar chart
    categories = ['All Applicants', 'Only Medium & High Income']
    default_rates = [probs['p_default'], probs['p_default_medium_or_high']]
    
    # Create bar chart
    bars = plt.bar(categories, default_rates, color=['#d73027', '#4575b4'], width=0.6)
    
    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=14)
    
    # Add labels and title
    plt.xlabel('Approval Policy', fontsize=14)
    plt.ylabel('Expected Default Rate', fontsize=14)
    plt.title('Comparison of Expected Default Rates Under Different Approval Policies', fontsize=16)
    
    # Set y-axis limit
    plt.ylim(0, max(default_rates) + 0.03)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percent improvement
    improvement = (1 - probs['p_default_medium_or_high'] / probs['p_default']) * 100
    plt.figtext(0.5, 0.2, f"Improvement: {improvement:.2f}%", fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loan approval comparison saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 12 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_12")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 12 of the L2.1 Probability quiz: Credit Scoring Probability...\n")
    
    # Calculate all probabilities
    probs = calculate_credit_scoring_probabilities()
    
    # Print results for all tasks
    print("Task 1: Overall probability of default")
    print(f"P(Default) = {probs['p_default']:.4f}\n")
    
    print("Task 2: Probability of low income given default")
    print(f"P(Low Income | Default) = {probs['p_low_given_default']:.4f}\n")
    
    print("Task 3: Probability of high income given default")
    print(f"P(High Income | Default) = {probs['p_high_given_default']:.4f}\n")
    
    print("Task 4: Expected default rate with medium and high income approvals only")
    print(f"P(Default | Medium or High Income) = {probs['p_default_medium_or_high']:.4f}\n")
    
    # Additional calculations for completeness
    print("Additional calculations:")
    print(f"P(Medium Income | Default) = {probs['p_medium_given_default']:.4f}")
    
    # Generate visualizations
    create_joint_probability_visualization(probs, save_path=os.path.join(save_dir, "joint_probability.png"))
    print("1. Joint probability visualization created")
    
    create_conditional_probability_barchart(probs, save_path=os.path.join(save_dir, "conditional_probability.png"))
    print("2. Conditional probability bar chart created")
    
    create_bayes_visualization(probs, save_path=os.path.join(save_dir, "bayes_theorem.png"))
    print("3. Bayes' theorem visualization created")
    
    create_loan_approval_comparison(probs, save_path=os.path.join(save_dir, "loan_approval.png"))
    print("4. Loan approval policy comparison created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 