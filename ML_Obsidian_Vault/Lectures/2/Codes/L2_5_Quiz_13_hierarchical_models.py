import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Music Streaming Problem")

print("Given:")
print("- A music streaming service is analyzing user preferences across different genres")
print("- They want to apply a hierarchical Bayesian model to understand listening patterns")
print()
print("We need to:")
print("1. Explain how a two-level hierarchical Bayesian model could be structured for this scenario,")
print("   where individual users are grouped by geographical regions")
print("2. Discuss how empirical Bayes could be used as an alternative to full Bayesian inference")
print("   if the service has limited computational resources")
print("3. Describe a key difference between Bayesian credible intervals and frequentist")
print("   confidence intervals in interpreting user preference data")
print()

# Step 2: Simulate some music streaming preference data for illustration
print_step_header(2, "Simulating Music Streaming Data for Illustration")

# Define parameters for simulation
np.random.seed(42)
n_regions = 4
n_users_per_region = 30
n_genres = 5
genre_names = ['Pop', 'Rock', 'Hip-Hop', 'Classical', 'Electronic']
region_names = ['North America', 'Europe', 'Asia', 'Latin America']

# Create region-specific preferences (true parameters)
region_genre_preferences = {
    'North America': [0.35, 0.25, 0.20, 0.05, 0.15],
    'Europe': [0.20, 0.35, 0.15, 0.20, 0.10],
    'Asia': [0.40, 0.10, 0.15, 0.15, 0.20],
    'Latin America': [0.15, 0.25, 0.35, 0.05, 0.20]
}

# Generate individual user preferences with regional tendencies plus individual variation
all_users_data = []

for region_idx, region in enumerate(region_names):
    region_prefs = region_genre_preferences[region]
    
    for user_idx in range(n_users_per_region):
        # Add some individual variation to the regional preferences
        user_prefs = np.random.dirichlet(np.array(region_prefs) * 10)
        
        # Generate listening counts for each genre (100 songs per user)
        listening_counts = np.random.multinomial(100, user_prefs)
        
        # Store the data
        user_data = {
            'user_id': f"{region[:3]}_User_{user_idx+1}",
            'region': region,
            'region_idx': region_idx
        }
        
        # Add genre counts
        for genre_idx, genre in enumerate(genre_names):
            user_data[f'{genre}_count'] = listening_counts[genre_idx]
        
        all_users_data.append(user_data)

# Convert to DataFrame
user_df = pd.DataFrame(all_users_data)

# Display the first few rows of the data
print("Sample of simulated music streaming data:")
print(user_df.head())
print()

print("Summary statistics by region:")
for region in region_names:
    region_data = user_df[user_df['region'] == region]
    print(f"\nRegion: {region} (n={len(region_data)} users)")
    
    genre_means = []
    for genre in genre_names:
        mean_count = region_data[f'{genre}_count'].mean()
        genre_means.append(mean_count)
        print(f"  Mean {genre} plays: {mean_count:.2f}")
    
    # Show the highest genre
    max_genre_idx = np.argmax(genre_means)
    print(f"  Most popular genre: {genre_names[max_genre_idx]}")

print()

# Step 3: Visualize the data
print_step_header(3, "Visualizing Music Streaming Preferences")

# Plot 1: Regional differences in genre preferences
plt.figure(figsize=(14, 8))

# Prepare data for plotting
region_genre_means = []
for region in region_names:
    region_data = user_df[user_df['region'] == region]
    genre_means = [region_data[f'{genre}_count'].mean() for genre in genre_names]
    region_genre_means.append(genre_means)

region_genre_means = np.array(region_genre_means)

# Create a grouped bar plot
bar_width = 0.15
x = np.arange(len(genre_names))

for i, region in enumerate(region_names):
    plt.bar(x + i*bar_width, region_genre_means[i], width=bar_width, 
            label=region, alpha=0.7)

plt.xlabel('Genre', fontsize=14)
plt.ylabel('Average Plays per User', fontsize=14)
plt.title('Regional Differences in Music Genre Preferences', fontsize=16)
plt.xticks(x + bar_width * (len(region_names) - 1) / 2, genre_names, fontsize=12)
plt.legend(title='Region', fontsize=12)
plt.grid(True, axis='y', alpha=0.3)

# Save the figure
file_path = os.path.join(save_dir, "regional_preferences.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Plot 2: Individual variation within regions
plt.figure(figsize=(14, 10))

# Create a subplot for each genre
for i, genre in enumerate(genre_names):
    plt.subplot(2, 3, i+1)
    
    # Create boxplots for each region
    data = [user_df[user_df['region'] == region][f'{genre}_count'] for region in region_names]
    plt.boxplot(data, labels=region_names)
    
    plt.title(f'{genre}', fontsize=14)
    plt.ylabel('Plays per User', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "within_region_variation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Illustrate hierarchical Bayesian model structure
print_step_header(4, "Hierarchical Bayesian Model Structure")

print("A two-level hierarchical Bayesian model for music streaming preferences could be structured as follows:")
print()
print("Level 1 (Individual User Level):")
print("- Let θᵢⱼ be the preference parameter for genre j by user i")
print("- For each user i in region r, the observed play counts xᵢⱼ for genre j follow:")
print("  xᵢⱼ ~ Multinomial(n_total_plays, [θᵢ₁, θᵢ₂, ..., θᵢₖ])")
print("  where the preference parameters θᵢⱼ sum to 1 across all genres for each user")
print()
print("Level 2 (Regional Level):")
print("- The individual user preferences are drawn from a regional distribution:")
print("  [θᵢ₁, θᵢ₂, ..., θᵢₖ] ~ Dirichlet(α_r₁, α_r₂, ..., α_rₖ)")
print("  where α_rⱼ represents the regional preference strength for genre j in region r")
print()
print("Level 3 (Global Level):")
print("- The regional parameters themselves come from a global distribution:")
print("  [α_r₁, α_r₂, ..., α_rₖ] ~ some prior distribution")
print("  This allows sharing of information across regions")
print()

# Create a visual representation of the hierarchical model
plt.figure(figsize=(12, 8))

# Create a hierarchical structure visualization
def draw_node(ax, x, y, width, height, label, facecolor='lightblue', alpha=0.7):
    rect = plt.Rectangle((x, y), width, height, facecolor=facecolor, 
                        edgecolor='black', alpha=alpha, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=12)

fig, ax = plt.subplots(figsize=(12, 8))

# Global level
draw_node(ax, 4, 7, 4, 1, 'Global Hyperpriors', 'gold', 0.7)

# Regional level
region_colors = ['lightblue', 'lightgreen', 'salmon', 'plum']
for i, region in enumerate(region_names):
    x = 2 + i * 2
    draw_node(ax, x, 5, 1.6, 0.8, f'{region}\nParameters', region_colors[i])
    
    # Connect to global level
    plt.plot([x + 0.8, 6], [5.8, 7], 'k-', alpha=0.5)
    
    # Add user nodes
    for j in range(3):  # Just show 3 users per region
        user_x = x + j * 0.5 - 0.25
        draw_node(ax, user_x, 3, 0.4, 0.6, f'User\n{j+1}', region_colors[i], 0.5)
        
        # Connect to regional level
        plt.plot([user_x + 0.2, x + 0.8], [3.6, 5], 'k-', alpha=0.5)
        
        # Add data nodes
        draw_node(ax, user_x, 1.5, 0.4, 0.6, 'Data', 'white', 0.3)
        plt.plot([user_x + 0.2, user_x + 0.2], [2.1, 3], 'k-', alpha=0.5)

# Add annotations
plt.text(8.5, 7.5, 'Level 3: Global Hyperpriors\nShares information across all regions', fontsize=12)
plt.text(8.5, 5.5, 'Level 2: Regional Parameters\nCaptures regional trends and variations', fontsize=12)
plt.text(8.5, 3.5, 'Level 1: User Parameters\nReflects individual preferences\nwithin regional context', fontsize=12)
plt.text(8.5, 1.5, 'Observed Data\nActual listening counts\nper genre for each user', fontsize=12)

plt.axis('off')
plt.title('Hierarchical Bayesian Model for Music Streaming Preferences', fontsize=16)

# Save the figure
file_path = os.path.join(save_dir, "hierarchical_model_structure.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Demonstrate empirical Bayes approach
print_step_header(5, "Empirical Bayes Approach")

print("Empirical Bayes is a compromise between fully Bayesian and frequentist approaches.")
print("Instead of placing priors on all parameters and performing full Bayesian inference,")
print("some parameters are estimated from the data and then used as fixed values.")
print()
print("For the music streaming service with limited computational resources:")
print()
print("1. Full Hierarchical Bayesian Approach:")
print("   - Specify priors for all parameters at all levels")
print("   - Use MCMC or variational inference to sample from the full posterior")
print("   - Computationally intensive, especially with millions of users")
print()
print("2. Empirical Bayes Alternative:")
print("   - Estimate the global hyperparameters from the data (e.g., using maximum likelihood)")
print("   - Use these point estimates as fixed values for the regional-level priors")
print("   - Only compute user-level posteriors conditional on these fixed parameters")
print()
print("Benefits of Empirical Bayes for the streaming service:")
print("- Significantly reduced computational burden")
print("- Still maintains the hierarchical structure")
print("- Allows for individual user inference while borrowing strength across users and regions")
print("- Can be implemented using more scalable algorithms")
print()

# Demonstration of Empirical Bayes on a single genre
print("Let's demonstrate a simple empirical Bayes approach on the 'Pop' genre:")

# Extract Pop counts for all regions
pop_counts = []
region_labels = []

for region in region_names:
    region_data = user_df[user_df['region'] == region][f'Pop_count']
    pop_counts.extend(region_data.values)
    region_labels.extend([region] * len(region_data))

pop_counts = np.array(pop_counts)
region_indices = np.array([region_names.index(r) for r in region_labels])

# Fit a Beta-Binomial model (simplified for illustration)
# In practice, this would be a more complex model with multiple genres
n_trials = 100  # Total songs per user

# Function to compute Beta parameters from mean and variance
def get_beta_params(mean, var):
    # Ensure variance is valid for a Beta distribution
    max_var = mean * (1 - mean)
    if var >= max_var:
        var = max_var * 0.99
    
    common_term = (mean * (1 - mean) / var) - 1
    alpha = mean * common_term
    beta = (1 - mean) * common_term
    return alpha, beta

# Step 1: Calculate global mean and variance
global_mean = np.mean(pop_counts) / n_trials
global_var = np.var(pop_counts) / (n_trials**2)

# Step 2: Estimate global hyperparameters
global_alpha, global_beta = get_beta_params(global_mean, global_var)

print(f"\nGlobal estimates (all users):")
print(f"- Mean Pop genre preference: {global_mean:.4f}")
print(f"- Estimated Beta parameters: α={global_alpha:.2f}, β={global_beta:.2f}")

# Step 3: Calculate region-specific means and variances
region_means = []
region_vars = []
region_alphas = []
region_betas = []

for r_idx, region in enumerate(region_names):
    region_data = pop_counts[region_indices == r_idx] / n_trials
    r_mean = np.mean(region_data)
    r_var = np.var(region_data)
    r_alpha, r_beta = get_beta_params(r_mean, r_var)
    
    region_means.append(r_mean)
    region_vars.append(r_var)
    region_alphas.append(r_alpha)
    region_betas.append(r_beta)
    
    print(f"\nRegion: {region}")
    print(f"- Mean Pop genre preference: {r_mean:.4f}")
    print(f"- Estimated Beta parameters: α={r_alpha:.2f}, β={r_beta:.2f}")

# Visualize the empirical Bayes hierarchical structure
plt.figure(figsize=(12, 8))

# Plot Beta distributions for each region and the global distribution
x = np.linspace(0, 1, 1000)
global_beta_pdf = stats.beta.pdf(x, global_alpha, global_beta)

plt.plot(x, global_beta_pdf, 'k-', linewidth=3, label='Global Prior')

for i, region in enumerate(region_names):
    alpha, beta = region_alphas[i], region_betas[i]
    region_pdf = stats.beta.pdf(x, alpha, beta)
    plt.plot(x, region_pdf, linewidth=2, label=f'{region} (α={alpha:.1f}, β={beta:.1f})')

plt.title('Empirical Bayes: Estimated Prior Distributions for Pop Genre', fontsize=14)
plt.xlabel('Preference for Pop Genre (proportion)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
file_path = os.path.join(save_dir, "empirical_bayes_priors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Comparing Bayesian Credible Intervals with Frequentist Confidence Intervals
print_step_header(6, "Credible Intervals vs. Confidence Intervals")

print("A key difference between Bayesian credible intervals and frequentist confidence intervals:")
print()
print("Bayesian Credible Interval:")
print("- Directly expresses the probability that the parameter lies within the interval,")
print("  given the observed data and prior")
print("- For music preferences: 'We are 95% confident that users in Europe have a")
print("  preference for Rock music between 0.32 and 0.38'")
print("- Based on the posterior distribution of the parameter")
print("- Interpretable as a statement about the parameter itself")
print()
print("Frequentist Confidence Interval:")
print("- Doesn't express probability about the parameter, but about the procedure")
print("- For music preferences: 'If we repeated this sampling process many times,")
print("  95% of the resulting intervals would contain the true preference for Rock music")
print("  among European users'")
print("- Based on the sampling distribution of the estimator")
print("- Less directly interpretable for decision-making")
print()

# Demonstrate with a specific example using Rock genre in Europe
europe_data = user_df[user_df['region'] == 'Europe']
europe_rock_counts = europe_data['Rock_count'].values
n_users = len(europe_rock_counts)

# Calculate sample mean and standard error
sample_mean = np.mean(europe_rock_counts) / n_trials
sample_std = np.std(europe_rock_counts, ddof=1) / n_trials
standard_error = sample_std / np.sqrt(n_users)

# Frequentist 95% confidence interval
conf_interval = (
    sample_mean - 1.96 * standard_error,
    sample_mean + 1.96 * standard_error
)

# Bayesian approach with a Beta prior (using empirical Bayes)
prior_alpha, prior_beta = 10, 20  # Weakly informative prior
posterior_alpha = prior_alpha + np.sum(europe_rock_counts)
posterior_beta = prior_beta + n_users * n_trials - np.sum(europe_rock_counts)

# 95% credible interval
cred_interval = stats.beta.ppf([0.025, 0.975], posterior_alpha, posterior_beta)

print(f"Example: Rock genre preference among European users")
print(f"Sample size: {n_users} users")
print(f"Sample mean: {sample_mean:.4f}")
print()
print(f"Frequentist 95% Confidence Interval: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]")
print("Interpretation: If we were to repeat our study many times, 95% of the resulting")
print("                confidence intervals would contain the true preference value.")
print()
print(f"Bayesian 95% Credible Interval: [{cred_interval[0]:.4f}, {cred_interval[1]:.4f}]")
print("Interpretation: Given our data and prior beliefs, there is a 95% probability that")
print("                the true preference value lies within this interval.")
print()

# Visualize the difference
plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Plot 1: Frequentist perspective
ax1 = plt.subplot(gs[0])
x = np.linspace(sample_mean - 4*standard_error, sample_mean + 4*standard_error, 1000)
normal_pdf = stats.norm.pdf(x, sample_mean, standard_error)

ax1.plot(x, normal_pdf, 'b-', linewidth=2)
ax1.fill_between(x, 0, normal_pdf, 
                 where=(x >= conf_interval[0]) & (x <= conf_interval[1]), 
                 color='blue', alpha=0.3)

# Add vertical line for true value (for illustration)
true_value = region_genre_preferences['Europe'][1]  # Rock preference for Europe
ax1.axvline(x=true_value, color='red', linestyle='--', label='True Value')

ax1.set_title('Frequentist Approach: Sampling Distribution', fontsize=14)
ax1.set_xlabel('Preference for Rock Genre', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.text(0.05, 0.95, f"95% Confidence Interval:\n[{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]", 
         transform=ax1.transAxes, verticalalignment='top', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.legend()

# Plot 2: Bayesian perspective
ax2 = plt.subplot(gs[1])
x = np.linspace(0, 1, 1000)
prior_pdf = stats.beta.pdf(x, prior_alpha, prior_beta)
posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)

ax2.plot(x, prior_pdf, 'g--', linewidth=2, alpha=0.7, label='Prior')
ax2.plot(x, posterior_pdf, 'g-', linewidth=2, label='Posterior')
ax2.fill_between(x, 0, posterior_pdf, 
                 where=(x >= cred_interval[0]) & (x <= cred_interval[1]), 
                 color='green', alpha=0.3)

# Add vertical line for true value (for illustration)
ax2.axvline(x=true_value, color='red', linestyle='--', label='True Value')

ax2.set_title('Bayesian Approach: Parameter Distribution', fontsize=14)
ax2.set_xlabel('Preference for Rock Genre', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.text(0.05, 0.95, f"95% Credible Interval:\n[{cred_interval[0]:.4f}, {cred_interval[1]:.4f}]", 
         transform=ax2.transAxes, verticalalignment='top', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "credible_vs_confidence_intervals.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion
print_step_header(7, "Conclusion")

print("Key takeaways for the music streaming service:")
print()
print("1. Hierarchical Bayesian Model Structure:")
print("   - Individual user preferences are nested within regional patterns")
print("   - Regional patterns are informed by global hyperparameters")
print("   - This structure allows for personalized recommendations while")
print("     leveraging patterns from similar users within the same region")
print()
print("2. Empirical Bayes as a Practical Alternative:")
print("   - Estimate global parameters from data to reduce computational burden")
print("   - Still maintains the hierarchical structure's benefits")
print("   - Particularly valuable for the streaming service's large user base")
print("   - Can be implemented with more scalable techniques like variational inference")
print()
print("3. Interpretation of Uncertainty:")
print("   - Bayesian credible intervals provide direct probabilistic statements about user preferences")
print("   - This is more intuitive for decision-making about music recommendations")
print("   - Allows for more nuanced personalization by considering the full uncertainty")
print("     in preference estimates, not just point estimates")
print()
print("The hierarchical Bayesian approach is well-suited to the music streaming scenario")
print("because it naturally captures the nested structure of preferences (individuals within")
print("regions) while allowing appropriate sharing of information across users and regions.") 