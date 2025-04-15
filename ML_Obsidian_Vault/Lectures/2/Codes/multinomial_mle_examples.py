import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

def plot_multinomial_likelihood(data, categories, title, save_path=None):
    """
    Visualize multinomial MLE for categorical data with a bar chart comparison
    between observed proportions and counts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate MLE estimates (proportions)
    total_observations = sum(data)
    mle_estimates = [count / total_observations for count in data]
    
    # Plot 1: Observed counts
    ax1.bar(categories, data, color='skyblue', edgecolor='navy')
    ax1.set_title(f"{title} - Observed Counts")
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Count')
    
    # Label bars with actual values
    for i, v in enumerate(data):
        ax1.text(i, v + 0.5, str(v), ha='center')
    
    # Plot 2: MLE probability estimates
    ax2.bar(categories, mle_estimates, color='lightgreen', edgecolor='darkgreen')
    ax2.set_title(f"{title} - MLE Probability Estimates")
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Probability')
    
    # Label bars with probability values
    for i, v in enumerate(mle_estimates):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks
    
    return mle_estimates

def analyze_multinomial_data(name, data, categories, context, save_dir=None):
    """Analyze multinomial data with detailed steps using MLE."""
    print(f"\n{'='*50}")
    print(f"{name} Example")
    print(f"{'='*50}")
    print(f"Context: {context}")
    
    # Step 1: Data analysis
    print("\nStep 1: Data Analysis")
    total_observations = sum(data)
    print(f"- Data: {dict(zip(categories, data))}")
    print(f"- Total observations: {total_observations}")
    
    # Step 2: Calculate MLE
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- MLE is simply the proportion of each category in the data")
    mle_estimates = [count / total_observations for count in data]
    for i, category in enumerate(categories):
        print(f"- MLE estimate for {category} = {mle_estimates[i]:.3f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"multinomial_mle_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step 3: Visualize and confirm
    print("\nStep 3: Visualization")
    plot_multinomial_likelihood(data, categories, name, save_path)
    
    # Step 4: Confidence Intervals
    print("\nStep 4: Confidence Intervals")
    if total_observations >= 30:  # Enough samples for normal approximation
        # 95% confidence intervals using normal approximation
        z = 1.96  # 95% confidence level
        print("- 95% Confidence Intervals:")
        for i, category in enumerate(categories):
            p = mle_estimates[i]
            std_error = np.sqrt((p * (1 - p)) / total_observations)
            ci_lower = max(0, p - z * std_error)
            ci_upper = min(1, p + z * std_error)
            print(f"  {category}: [{ci_lower:.3f}, {ci_upper:.3f}]")
    else:
        print("- Sample size too small for reliable confidence intervals")
        
    # Step 5: Interpretation
    print("\nStep 5: Interpretation")
    print(f"- Based on our data alone, the most likely probabilities are:")
    for i, category in enumerate(categories):
        print(f"  {category}: {mle_estimates[i]:.1%}")
    print(f"- These estimates maximize the likelihood of observing our data")
    print(f"- Unlike MAP estimation, MLE does not incorporate prior beliefs")
    
    return mle_estimates, save_path

def generate_multinomial_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze categorical data using Maximum Likelihood Estimation for Multinomial distributions!
    Each example will show how we can estimate category probabilities using only the observed data.
    """)

    # Example: Dice Rolls
    dice_data = [7, 5, 9, 12, 8, 9]  # Count of each face (1-6)
    dice_categories = ["Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"]
    dice_context = """
    You want to test if a six-sided die is fair.
    - You roll the die 50 times
    - Count occurrences of each face
    - Using only the observed data (no prior assumptions)
    """
    mle_result, path = analyze_multinomial_data("Dice Rolls", dice_data, dice_categories, dice_context, save_dir)
    results["Dice Rolls"] = {"mle": mle_result, "path": path}

    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_multinomial_examples(save_dir) 