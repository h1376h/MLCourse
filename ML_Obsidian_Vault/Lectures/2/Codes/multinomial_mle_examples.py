import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import pandas as pd
import seaborn as sns
from scipy import stats

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

def plot_one_hot_encoding_example(data, categories, confidence_level=0.95, save_path=None):
    """
    Create visualizations for the one-hot encoding example including confidence intervals.
    """
    total = sum(data)
    mle_estimates = [count / total for count in data]
    
    # For 90% confidence intervals (example 3 asks for 90%)
    z = stats.norm.ppf((1 + confidence_level) / 2)  # z-score for confidence level
    
    # Calculate confidence intervals
    ci_lower = []
    ci_upper = []
    for p in mle_estimates:
        std_error = np.sqrt((p * (1 - p)) / total)
        ci_lower.append(max(0, p - z * std_error))
        ci_upper.append(min(1, p + z * std_error))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: MLE Estimates with Confidence Intervals
    ax1.bar(categories, mle_estimates, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Color Distribution - MLE Estimates')
    ax1.set_xlabel('Color')
    ax1.set_ylabel('Probability')
    
    # Add error bars for confidence intervals
    error_lower = [mle_estimates[i] - ci_lower[i] for i in range(len(mle_estimates))]
    error_upper = [ci_upper[i] - mle_estimates[i] for i in range(len(mle_estimates))]
    ax1.errorbar(categories, mle_estimates, yerr=[error_lower, error_upper], 
                fmt='o', color='black', capsize=5)
    
    # Add value labels
    for i, v in enumerate(mle_estimates):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Plot 2: One-hot encoding visualization
    # Create a mini-dataframe for demonstration
    sample_size = min(20, total // 4)  # Show a subset of data points
    one_hot_data = []
    
    color_map = {'Red': [1, 0, 0, 0], 
                 'Green': [0, 1, 0, 0], 
                 'Blue': [0, 0, 1, 0], 
                 'Yellow': [0, 0, 0, 1]}
    
    for i, color in enumerate(categories):
        n_samples = int(sample_size * mle_estimates[i])
        for _ in range(n_samples):
            one_hot_data.append(color_map[color])
    
    df = pd.DataFrame(one_hot_data, columns=categories)
    
    # Plot heatmap of one-hot encoded data
    sns.heatmap(df.sample(min(len(df), 15)), cmap="YlGnBu", cbar=False, ax=ax2)
    ax2.set_title("One-Hot Encoding Representation")
    ax2.set_ylabel("Sample Data Points")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_estimates, ci_lower, ci_upper

def entropy(probs):
    """Calculate Shannon entropy for a list of probabilities"""
    return -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

def conditional_entropy(joint_probs, marginal_probs):
    """Calculate conditional entropy H(Y|X) where joint_probs is p(x,y) and marginal_probs is p(x)"""
    cond_ent = 0
    for i, px in enumerate(marginal_probs):
        if px > 0:
            for p_xy in joint_probs[i]:
                if p_xy > 0:
                    cond_ent -= p_xy * np.log2(p_xy / px)
    return cond_ent

def plot_information_gain(season_data, save_path=None):
    """
    Create visualizations for the information gain example.
    """
    # Extract data
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    target_0 = [season_data[season][0] for season in seasons]
    target_1 = [season_data[season][1] for season in seasons]
    
    # Calculate totals
    season_totals = [sum(season_data[season]) for season in seasons]
    total_samples = sum(season_totals)
    
    # Calculate probabilities for visualization
    season_probs = [total / total_samples for total in season_totals]
    target_1_given_season = [season_data[season][1] / sum(season_data[season]) for season in seasons]
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Season distribution
    ax1.bar(seasons, season_probs, color='lightblue', edgecolor='navy')
    ax1.set_title('Season Distribution - P(Season)')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Probability')
    for i, v in enumerate(season_probs):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Plot 2: Target distribution given season
    ax2.bar(seasons, target_1_given_season, color='salmon', edgecolor='darkred')
    ax2.set_title('Conditional Probability - P(Target=1|Season)')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Probability')
    for i, v in enumerate(target_1_given_season):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Plot 3: Stacked bar chart showing the joint distribution
    width = 0.35
    ax3.bar(seasons, [season_data[s][0]/total_samples for s in seasons], 
            width, label='Target=0', color='lightblue')
    ax3.bar(seasons, [season_data[s][1]/total_samples for s in seasons], 
            width, bottom=[season_data[s][0]/total_samples for s in seasons], 
            label='Target=1', color='salmon')
    ax3.set_title('Joint Distribution - P(Season, Target)')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Probability')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return season_probs, target_1_given_season

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

def analyze_one_hot_encoding_example(save_dir=None):
    """Analyze the one-hot encoding example with detailed steps."""
    print(f"\n{'='*50}")
    print(f"One-Hot Encoding and Multinomial Distribution Example")
    print(f"{'='*50}")
    
    # Problem context
    context = """
    A machine learning engineer is analyzing a dataset with a categorical feature "Color" that
    has 4 possible values: Red, Green, Blue, and Yellow. After one-hot encoding, there are 120
    observations with the following counts:
    - Red: 30 observations
    - Green: 45 observations
    - Blue: 25 observations
    - Yellow: 20 observations
    """
    print(f"Context: {context}")
    
    # Data
    colors = ["Red", "Green", "Blue", "Yellow"]
    color_counts = [30, 45, 25, 20]
    total_observations = sum(color_counts)
    
    # Step 1: Explanation of one-hot encoding and multinomial distribution
    print("\nStep 1: One-Hot Encoding and Multinomial Distribution")
    print("- One-hot encoding transforms categorical variables into a binary vector representation:")
    print("  * Each category gets its own binary column (0 or 1)")
    print("  * Exactly one column is 1 ('hot'), all others are 0 for each observation")
    print("  * Example: Red → [1,0,0,0], Green → [0,1,0,0], etc.")
    print("\n- Connection to multinomial distribution:")
    print("  * The counts of each category follow a multinomial distribution")
    print("  * The true probabilities of each category are the parameters we estimate using MLE")
    print("  * One-hot encoding is essentially creating indicator variables for each category")
    
    # Step 2: Calculate MLE for each color probability
    print("\nStep 2: Maximum Likelihood Estimation")
    print("- MLE for multinomial is simply the proportion of each category in the data")
    mle_estimates = [count / total_observations for count in color_counts]
    for i, color in enumerate(colors):
        print(f"- MLE estimate for P({color}) = {color_counts[i]}/{total_observations} = {mle_estimates[i]:.3f}")
    
    # Create save path
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "multinomial_mle_one_hot_encoding.png"
        save_path = os.path.join(save_dir, filename)
    
    # Step A: Visualization with confidence intervals
    print("\nStep 3: Visualization with Confidence Intervals")
    mle_estimates, ci_lower, ci_upper = plot_one_hot_encoding_example(color_counts, colors, 
                                                                     confidence_level=0.90, # 90% as requested
                                                                     save_path=save_path)
    
    # Step 3: Probability of Red or Blue
    print("\nStep 4: Probability of Red or Blue")
    red_index = colors.index("Red")
    blue_index = colors.index("Blue")
    prob_red_or_blue = mle_estimates[red_index] + mle_estimates[blue_index]
    print(f"- P(Red or Blue) = P(Red) + P(Blue) = {mle_estimates[red_index]:.3f} + {mle_estimates[blue_index]:.3f} = {prob_red_or_blue:.3f}")
    print(f"- This means there's a {prob_red_or_blue:.1%} chance that a randomly selected observation will be either Red or Blue")
    
    # Step 4: 90% Confidence Interval for Green
    green_index = colors.index("Green")
    print("\nStep 5: 90% Confidence Interval for Green")
    print(f"- 90% Confidence Interval for P(Green):")
    print(f"  * Lower bound: {ci_lower[green_index]:.3f}")
    print(f"  * MLE estimate: {mle_estimates[green_index]:.3f}")
    print(f"  * Upper bound: {ci_upper[green_index]:.3f}")
    print(f"  * Interpretation: We are 90% confident that the true probability of Green is between {ci_lower[green_index]:.3f} and {ci_upper[green_index]:.3f}")
    
    return mle_estimates, save_path

def analyze_information_gain_example(save_dir=None):
    """Analyze the information gain example with detailed steps."""
    print(f"\n{'='*50}")
    print(f"Information Gain with One-Hot Encoded Features Example")
    print(f"{'='*50}")
    
    # Problem context
    context = """
    In a classification task, we have a categorical feature "Season" with values:
    Spring, Summer, Fall, and Winter. After one-hot encoding, we have binary features 
    for each season. In a dataset of 200 samples with a binary target variable (0 or 1),
    we observe the following:
    
    | Season  | Target=0 | Target=1 | Total |
    |---------|----------|----------|-------|
    | Spring  | 30       | 10       | 40    |
    | Summer  | 20       | 40       | 60    |
    | Fall    | 25       | 15       | 40    |
    | Winter  | 45       | 15       | 60    |
    | Total   | 120      | 80       | 200   |
    """
    print(f"Context: {context}")
    
    # Data
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    season_data = {
        "Spring": [30, 10],  # [Target=0, Target=1]
        "Summer": [20, 40],
        "Fall": [25, 15],
        "Winter": [45, 15]
    }
    
    # Calculate totals
    season_totals = [sum(season_data[season]) for season in seasons]
    total_samples = sum(season_totals)
    target_0_total = sum(season_data[season][0] for season in seasons)
    target_1_total = sum(season_data[season][1] for season in seasons)
    
    # Step 1: MLE for each season probability
    print("\nStep 1: MLE for Season Probabilities")
    season_probs = [total / total_samples for total in season_totals]
    for i, season in enumerate(seasons):
        print(f"- MLE estimate for P({season}) = {season_totals[i]}/{total_samples} = {season_probs[i]:.3f}")
    
    # Step 2: MLE for conditional probabilities
    print("\nStep 2: MLE for Conditional Probabilities P(Target=1|Season)")
    target_1_given_season = [season_data[season][1] / sum(season_data[season]) for season in seasons]
    for i, season in enumerate(seasons):
        print(f"- P(Target=1|{season}) = {season_data[season][1]}/{season_totals[i]} = {target_1_given_season[i]:.3f}")
    
    # Step 3: Entropy of target variable
    print("\nStep 3: Entropy of Target Variable")
    p_target_0 = target_0_total / total_samples
    p_target_1 = target_1_total / total_samples
    target_entropy = entropy([p_target_0, p_target_1])
    print(f"- P(Target=0) = {target_0_total}/{total_samples} = {p_target_0:.3f}")
    print(f"- P(Target=1) = {target_1_total}/{total_samples} = {p_target_1:.3f}")
    print(f"- Entropy H(Target) = -({p_target_0:.3f} × log₂({p_target_0:.3f}) + {p_target_1:.3f} × log₂({p_target_1:.3f}))")
    print(f"- Entropy H(Target) = {target_entropy:.4f} bits")
    
    # Step 4: Conditional entropy
    print("\nStep 4: Conditional Entropy of Target Given Season")
    # Calculate joint probabilities P(Season, Target)
    joint_probs = []
    for season in seasons:
        season_joint = [season_data[season][0] / total_samples, 
                        season_data[season][1] / total_samples]
        joint_probs.append(season_joint)
    
    # Calculate conditional entropy H(Target|Season)
    cond_entropy_value = 0
    for i, season in enumerate(seasons):
        p_season = season_probs[i]
        p_target_0_given_season = season_data[season][0] / season_totals[i]
        p_target_1_given_season = season_data[season][1] / season_totals[i]
        
        season_entropy = entropy([p_target_0_given_season, p_target_1_given_season])
        cond_entropy_value += p_season * season_entropy
        
        print(f"- H(Target|{season}) = -({p_target_0_given_season:.3f} × log₂({p_target_0_given_season:.3f}) + "
              f"{p_target_1_given_season:.3f} × log₂({p_target_1_given_season:.3f})) = {season_entropy:.4f}")
    
    print(f"- H(Target|Season) = Σ P(Season) × H(Target|Season) = {cond_entropy_value:.4f} bits")
    
    # Step 5: Calculate information gain
    print("\nStep 5: Information Gain")
    info_gain = target_entropy - cond_entropy_value
    print(f"- Information Gain = H(Target) - H(Target|Season) = {target_entropy:.4f} - {cond_entropy_value:.4f} = {info_gain:.4f} bits")
    print(f"- This means that knowing the Season reduces uncertainty about the Target by {info_gain:.4f} bits")
    
    # Create save path
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "multinomial_mle_information_gain.png"
        save_path = os.path.join(save_dir, filename)
    
    # Visualization
    print("\nStep 6: Visualization")
    season_probs, target_1_given_season = plot_information_gain(season_data, save_path)
    
    # Final interpretation
    print("\nStep 7: Interpretation")
    print("- Season provides valuable information for predicting the target variable")
    print(f"- The Season feature reduces uncertainty about the Target by {(info_gain/target_entropy)*100:.1f}%")
    print("- Summer shows the strongest positive association with Target=1")
    print("- Spring and Winter show the strongest negative association with Target=1")
    
    return {"season_probs": season_probs, "info_gain": info_gain}, save_path

def generate_multinomial_examples(save_dir=None):
    results = {}
    
    print("""
    Let's analyze categorical data using Maximum Likelihood Estimation for Multinomial distributions!
    Each example will show how we can estimate category probabilities using only the observed data.
    """)

    # Example 1: Dice Rolls
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
    
    # Example 2: Survey Responses (from markdown documentation)
    survey_data = [15, 25, 60, 70, 30]  # Count of each star rating (1-5)
    survey_categories = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    survey_context = """
    A marketing team conducted a survey asking customers to rate their product on a scale from 1 to 5 stars.
    - 200 total responses across 5 rating categories
    - Using only the observed data to estimate true rating distribution
    - No prior assumptions about ratings
    """
    mle_result, path = analyze_multinomial_data("Survey Responses", survey_data, survey_categories, survey_context, save_dir)
    results["Survey Responses"] = {"mle": mle_result, "path": path}
    
    # Example 3: One-Hot Encoding
    mle_result, path = analyze_one_hot_encoding_example(save_dir)
    results["One-Hot Encoding"] = {"mle": mle_result, "path": path}
    
    # Example 4: Information Gain
    mle_result, path = analyze_information_gain_example(save_dir)
    results["Information Gain"] = {"mle": mle_result, "path": path}

    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_multinomial_examples(save_dir) 