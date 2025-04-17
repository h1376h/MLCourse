import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, invgamma

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Problem: Hierarchical Bayesian model for analyzing students' test scores across different schools")
print()
print("Tasks:")
print("1. Describe the basic structure of a two-level hierarchical Bayesian model for this scenario")
print("2. Explain one advantage of using a hierarchical model versus a non-hierarchical model")
print("3. Identify a scenario where empirical Bayes might be used instead of a fully Bayesian approach")
print()

# Step 2: Generating Example Data (in simplified form)
print_step_header(2, "Generating Example Data")

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for 10 schools
n_schools = 10
n_students_per_school = np.random.randint(20, 50, n_schools)  # Different number of students per school

# True parameters
mu_overall = 75  # Overall mean across all schools
sigma_between = 5  # Standard deviation between schools
sigma_within = 10  # Standard deviation within schools

# Generate school means from the distribution of school effects
school_means = np.random.normal(mu_overall, sigma_between, n_schools)

# Generate student scores within each school
all_scores = []
school_data = []

for i in range(n_schools):
    school_mean = school_means[i]
    n_students = n_students_per_school[i]
    
    # Generate student scores from the school-specific distribution
    student_scores = np.random.normal(school_mean, sigma_within, n_students)
    
    # Store the data
    all_scores.extend(student_scores)
    school_data.append(student_scores)

# Calculate statistics
overall_mean = np.mean(all_scores)
overall_std = np.std(all_scores)

# Print summary statistics
print("Generated data for student test scores across different schools:")
print(f"Number of schools: {n_schools}")
print(f"Total number of students: {len(all_scores)}")
print(f"Overall mean score: {overall_mean:.2f}")
print(f"Overall standard deviation: {overall_std:.2f}")
print()

# Calculate school-specific statistics
print("School-specific statistics:")
print(f"{'School':<10} {'Mean':<10} {'Std':<10} {'Count':<10}")
print("-" * 40)

school_means_sample = []
school_stds = []
for i, scores in enumerate(school_data):
    school_mean = np.mean(scores)
    school_std = np.std(scores)
    school_means_sample.append(school_mean)
    school_stds.append(school_std)
    print(f"{f'School {i+1}':<10} {school_mean:<10.2f} {school_std:<10.2f} {len(scores):<10}")
print()

# Visualize the data
plt.figure(figsize=(12, 6))

# Create a histogram of all scores
plt.hist(all_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=mu_overall, color='r', linestyle='--', label=f'True Overall Mean: {mu_overall}')
plt.title('Overall Distribution of Scores', fontsize=14)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(save_dir, "student_scores_data.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Hierarchical Bayesian Model Structure
print_step_header(3, "Hierarchical Bayesian Model Structure")

print("A two-level hierarchical Bayesian model for student test scores across schools has the following structure:")
print()
print("Level 1 (Student Level):")
print("For student j in school i:")
print("y_ij ~ Normal(μ_i, σ²_within)")
print("where:")
print("- y_ij is the test score of student j in school i")
print("- μ_i is the mean score for school i")
print("- σ²_within is the variance of scores within schools (assumed same across schools for simplicity)")
print()
print("Level 2 (School Level):")
print("For each school i:")
print("μ_i ~ Normal(μ_0, τ²)")
print("where:")
print("- μ_0 is the overall mean score across all schools")
print("- τ² is the between-school variance")
print()
print("Hyperpriors (optional in fully Bayesian approach):")
print("μ_0 ~ Normal(μ_prior, σ²_prior)  # Prior for overall mean")
print("τ² ~ InvGamma(a, b)             # Prior for between-school variance")
print("σ²_within ~ InvGamma(c, d)       # Prior for within-school variance")
print()

# Create a visual representation of the hierarchical model
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)

# Hide axes
ax.axis('off')

# Draw the hierarchical structure
levels = 3
level_height = 2
level_positions = [i*level_height for i in range(levels)]

# Level 3: Hyperpriors
hyper_x = [2, 4, 6]
hyper_y = [level_positions[2]]*3
hyper_labels = ['μ₀', 'τ²', 'σ²_within']
hyper_colors = ['#9FC5E8', '#B6D7A8', '#D5A6BD']

for i, (x, y, label, color) in enumerate(zip(hyper_x, hyper_y, hyper_labels, hyper_colors)):
    circle = plt.Circle((x, y), 0.4, facecolor=color, edgecolor='black', alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=12)

# Level 2: School-specific means
school_x = np.linspace(1, 7, 5)
school_y = [level_positions[1]]*5
school_labels = ['μ₁', 'μ₂', 'μ₃', 'μ₄', 'μ₅']

for i, (x, y, label) in enumerate(zip(school_x, school_y, school_labels)):
    circle = plt.Circle((x, y), 0.4, facecolor='#FFD966', edgecolor='black', alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=12)
    
    # Connect to hyperpriors
    for hx, hy, hl in zip(hyper_x, hyper_y, hyper_labels):
        if hl in ['μ₀', 'τ²']:  # Only connect to relevant hyperpriors
            ax.plot([x, hx], [y+0.4, hy-0.4], 'k-', alpha=0.5)

# Level 1: Student scores
student_groups = [np.linspace(x-0.5, x+0.5, 3) for x in school_x]
student_y = [level_positions[0]]*15

for i, group in enumerate(student_groups):
    for j, x in enumerate(group):
        circle = plt.Circle((x, student_y[0]), 0.3, facecolor='#F8CECC', edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, student_y[0], f'y_{i+1}{j+1}', ha='center', va='center', fontsize=10)
        
        # Connect to school mean
        ax.plot([x, school_x[i]], [student_y[0]+0.3, school_y[0]-0.4], 'k-', alpha=0.5)
        
        # Connect to within-school variance
        if j == 1:  # Only connect middle student to avoid clutter
            ax.plot([x, hyper_x[2]], [student_y[0]+0.3, hyper_y[2]-0.4], 'k-', alpha=0.3, linestyle='--')

# Add level labels
ax.text(0, level_positions[0], "Level 1:\nStudent Scores", fontsize=12, ha='right', va='center', fontweight='bold')
ax.text(0, level_positions[1], "Level 2:\nSchool Means", fontsize=12, ha='right', va='center', fontweight='bold')
ax.text(0, level_positions[2], "Level 3:\nHyperparameters", fontsize=12, ha='right', va='center', fontweight='bold')

# Set axes limits
ax.set_xlim(0, 8)
ax.set_ylim(-1, level_positions[-1]+1)

plt.title('Hierarchical Bayesian Model for Student Test Scores', fontsize=16)
plt.tight_layout()

file_path = os.path.join(save_dir, "hierarchical_model_structure.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Visualization of Partial Pooling Effect
print_step_header(4, "Partial Pooling in Hierarchical Models")

# Calculate estimates under different approaches
grand_mean = np.mean(all_scores)  # Complete pooling

# Simulate hierarchical (partial pooling) estimates
# Here we'll use a simplified approach based on empirical Bayes
overall_var = np.var(all_scores)
between_var_est = np.var(school_means_sample)
within_var_est = np.mean([std**2 for std in school_stds])

# Shrinkage factor for each school
shrinkage_factors = [within_var_est / (within_var_est + between_var_est/len(scores)) for scores in school_data]

# Partial pooling estimates
partial_pooling_means = [grand_mean + (1 - factor) * (sample_mean - grand_mean) 
                         for factor, sample_mean in zip(shrinkage_factors, school_means_sample)]

# Create a comparison table
print("Comparison of estimation approaches:")
print(f"{'School':<10} {'Sample Size':<15} {'No Pooling':<15} {'Complete Pooling':<20} {'Partial Pooling':<20} {'True Mean':<15}")
print("-" * 90)

for i in range(n_schools):
    print(f"{f'School {i+1}':<10} {len(school_data[i]):<15} {school_means_sample[i]:<15.2f} {grand_mean:<20.2f} {partial_pooling_means[i]:<20.2f} {school_means[i]:<15.2f}")
print()

print("Explanation of the approaches:")
print("1. No Pooling: Each school's mean is estimated independently using only data from that school")
print("2. Complete Pooling: All schools are assumed to have the same mean (the grand mean)")
print("3. Partial Pooling (Hierarchical): School means are shrunk toward the grand mean,")
print("   with the amount of shrinkage depending on the sample size and variance components")
print()

# Visualize the comparison
plt.figure(figsize=(12, 8))

# Scatter plot of school means vs. sample size
plt.scatter(n_students_per_school, school_means_sample, s=100, alpha=0.7, label='No Pooling', color='blue')
plt.scatter(n_students_per_school, partial_pooling_means, s=100, alpha=0.7, label='Partial Pooling', color='red')
plt.scatter(n_students_per_school, school_means, s=100, alpha=0.7, label='True Means', color='green', marker='X')

for i in range(n_schools):
    plt.plot([n_students_per_school[i], n_students_per_school[i]], 
             [school_means_sample[i], partial_pooling_means[i]], 'k-', alpha=0.5)

plt.axhline(y=grand_mean, color='purple', linestyle='--', label='Complete Pooling')
plt.xlabel('School Sample Size', fontsize=12)
plt.ylabel('Estimated Mean Score', fontsize=12)
plt.title('Effect of Sample Size on Shrinkage', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(save_dir, "pooling_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Advantages of Hierarchical Models
print_step_header(5, "Advantages of Hierarchical Models")

print("Advantages of using a hierarchical model over a non-hierarchical model in the context of student test scores:")
print()
print("1. Partial Pooling / Shrinkage Effect:")
print("   - Hierarchical models automatically implement a form of 'shrinkage' or 'partial pooling'")
print("   - School means with small sample sizes are pulled toward the overall mean")
print("   - Schools with larger sample sizes are less affected by shrinkage")
print("   - This balances the risk of overfitting in small schools while preserving distinctive patterns in large schools")
print()
print("2. Improved Accuracy and Precision:")
print("   - For schools with small sample sizes, hierarchical models typically provide more accurate estimates")
print("   - The borrowing of information across schools reduces estimation variance")
print("   - Reduces the impact of outliers or sampling variability in small schools")
print()
print("3. Modeling of Group-Level Variation:")
print("   - Explicitly models between-school variability (τ²)")
print("   - Provides insights into how much schools differ from each other")
print("   - Allows for examination of factors that might explain between-school differences")
print()
print("4. Simultaneous Inference at Multiple Levels:")
print("   - Can make inferences about individual students, schools, and the overall population")
print("   - Provides a coherent framework for answering different types of questions")
print("   - Properly accounts for uncertainty at each level of the hierarchy")
print()
print("5. Flexibility for Extension:")
print("   - Can be extended to include predictors at different levels (e.g., student characteristics, school resources)")
print("   - Can accommodate more complex structures (e.g., classrooms within schools)")
print("   - Allows for modeling of cross-level interactions")
print()

# Step 6: Empirical Bayes vs. Fully Bayesian
print_step_header(6, "Empirical Bayes vs. Fully Bayesian Approaches")

print("Empirical Bayes (EB) is a hybrid approach where hyperparameters (like μ₀ and τ²) are estimated from the data")
print("rather than given prior distributions as in a fully Bayesian approach.")
print()
print("Scenarios where empirical Bayes might be used instead of a fully Bayesian approach:")
print()
print("1. Computational Constraints:")
print("   - When working with very large datasets (e.g., thousands of schools, millions of students)")
print("   - When rapid analysis is required (e.g., real-time educational assessment systems)")
print("   - When computational resources are limited")
print()
print("2. Weak Prior Information:")
print("   - When there is little reliable prior information about hyperparameters")
print("   - When stakeholders are reluctant to incorporate subjective prior beliefs")
print("   - When the focus is on what the data reveals rather than prior knowledge")
print()
print("3. Routine Analysis in Educational Research:")
print("   - When analyzing standardized test scores across many districts for regular reporting")
print("   - When the goal is to rank or compare schools based on performance metrics")
print("   - When consistent methodology is needed across multiple analyses")
print()
print("4. Exploration Before Full Modeling:")
print("   - As an initial analysis before developing a more complex fully Bayesian model")
print("   - When exploring which hierarchical structure best fits the data")
print("   - For diagnostic purposes to identify potential modeling issues")
print()

# Create a visual comparison of approaches
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)

# Hide axes
ax.axis('off')

# Define regions
ebx, eby, ebw, ebh = 0.1, 0.05, 0.35, 0.42
fbx, fby, fbw, fbh = 0.55, 0.05, 0.35, 0.42
compare_x, compare_y, compare_w, compare_h = 0.1, 0.53, 0.8, 0.42

# Draw regions
eb_rect = plt.Rectangle((ebx, eby), ebw, ebh, facecolor='#E6F2FF', edgecolor='blue', alpha=0.5)
fb_rect = plt.Rectangle((fbx, fby), fbw, fbh, facecolor='#F9E6FF', edgecolor='purple', alpha=0.5)
compare_rect = plt.Rectangle((compare_x, compare_y), compare_w, compare_h, facecolor='#EFFFEF', edgecolor='green', alpha=0.5)

ax.add_patch(eb_rect)
ax.add_patch(fb_rect)
ax.add_patch(compare_rect)

# Add titles
ax.text(ebx + ebw/2, eby + ebh - 0.03, "Empirical Bayes", fontsize=14, fontweight='bold', ha='center')
ax.text(fbx + fbw/2, fby + fbh - 0.03, "Fully Bayesian", fontsize=14, fontweight='bold', ha='center')
ax.text(compare_x + compare_w/2, compare_y + compare_h - 0.03, "Comparison", fontsize=14, fontweight='bold', ha='center')

# Add content for Empirical Bayes
eb_content = [
    "• Hyperparameters estimated from data",
    "• Simplified computation",
    "• Point estimates for μ₀, τ², σ²",
    "• Often uses REML or MLE",
    "• Less uncertainty quantification",
    "• Faster implementation",
    "• Useful for large-scale applications"
]

for i, text in enumerate(eb_content):
    ax.text(ebx + 0.02, eby + ebh - 0.08 - i*0.05, text, fontsize=10, va='top')

# Add content for Fully Bayesian
fb_content = [
    "• Hyperpriors for all parameters",
    "• MCMC or variational inference",
    "• Full posterior distributions",
    "• Incorporates prior knowledge",
    "• Complete uncertainty propagation",
    "• Computationally intensive",
    "• More flexible modeling options"
]

for i, text in enumerate(fb_content):
    ax.text(fbx + 0.02, fby + fbh - 0.08 - i*0.05, text, fontsize=10, va='top')

# Add comparison content
compare_content = [
    "Computational Demand: EB < FB",
    "Uncertainty Quantification: FB > EB",
    "Flexibility in Modeling: FB > EB",
    "Speed of Implementation: EB > FB",
    "Incorporation of Prior Knowledge: FB > EB",
    "Ease of Interpretation: EB ≥ FB",
    "Suitability for Very Large Datasets: EB > FB"
]

for i, text in enumerate(compare_content):
    ax.text(compare_x + 0.05, compare_y + compare_h - 0.08 - i*0.05, text, fontsize=10, va='top')

# Add a small diagram showing the difference in how hyperparameters are treated
diag_x, diag_y, diag_w, diag_h = 0.55, 0.65, 0.3, 0.25
diag_rect = plt.Rectangle((diag_x, diag_y), diag_w, diag_h, facecolor='white', edgecolor='black', alpha=0.5)
ax.add_patch(diag_rect)

# Draw EB approach
ax.text(diag_x + 0.02, diag_y + diag_h - 0.03, "Hyperparameter Treatment:", fontsize=10, fontweight='bold', va='top')
ax.text(diag_x + 0.05, diag_y + diag_h - 0.07, "EB: μ₀, τ² = fixed estimates", fontsize=9, va='top', color='blue')
ax.text(diag_x + 0.05, diag_y + diag_h - 0.11, "FB: μ₀ ~ N(μ_prior, σ²_prior)", fontsize=9, va='top', color='purple')
ax.text(diag_x + 0.05, diag_y + diag_h - 0.15, "FB: τ² ~ InvGamma(a, b)", fontsize=9, va='top', color='purple')

# Set axes limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.title('Empirical Bayes vs. Fully Bayesian Approaches', fontsize=16, pad=20)
plt.tight_layout()

file_path = os.path.join(save_dir, "eb_vs_fb.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion
print_step_header(7, "Conclusion")

print("Key points about hierarchical Bayesian models for student test scores:")
print()
print("1. Structure: A two-level hierarchical model has student scores nested within schools, with")
print("   hyperparameters for the overall mean, between-school variance, and within-school variance.")
print()
print("2. Advantages: The main advantage is partial pooling, which improves estimation for schools")
print("   with small sample sizes by borrowing information from other schools, leading to more")
print("   accurate estimates and better predictive performance.")
print()
print("3. Empirical Bayes: This approach might be used instead of a fully Bayesian approach when")
print("   dealing with computational constraints, weak prior information, routine educational")
print("   analysis, or as an exploratory step before more complex modeling.")
print()
print("Hierarchical Bayesian models provide a powerful framework for analyzing educational data")
print("by accounting for the nested structure of students within schools, properly estimating")
print("school effects, and quantifying uncertainty at multiple levels.") 