import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_8")
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

print("Problem Statement: What is the fundamental difference between probability and likelihood?")
print()
print("Task: Explain in one sentence the key distinction between probability and likelihood in the context of statistical estimation.")
print()
print("To answer this, we need to understand both concepts and their usage in statistics.")
print()

# Step 2: Definition of Probability vs Likelihood
print_step_header(2, "Definition of Probability vs Likelihood")

print("Probability:")
print("Probability measures the chance of observing data given a fixed parameter value.")
print("P(data | parameter) = Probability of observing data given a specific parameter value")
print()
print("Likelihood:")
print("Likelihood measures the plausibility of a parameter value given fixed observed data.")
print("L(parameter | data) = Likelihood of a parameter value given the observed data")
print()
print("Key Distinction:")
print("- Probability: parameter is fixed, data is variable")
print("- Likelihood: data is fixed, parameter is variable")
print()

# Step 3: Examples with Coin Flips
print_step_header(3, "Coin Flip Example")

print("Example - Coin Flipping:")
print("- Parameter: p = probability of heads in a single coin flip")
print("- Data: observed sequence of heads (H) and tails (T)")
print()
print("Consider the following scenarios:")
print("1. Fixed parameter (p = 0.5), variable data: What's the probability of observing HHTHT?")
print("2. Fixed data (observed HHTHT), variable parameter: What's the likelihood of different p values?")
print()

# Create visualization of coin flip probability
def create_coin_flip_probability_plot():
    p = 0.5  # Fixed parameter (probability of heads)
    n = 5    # Number of coin flips
    
    plt.figure(figsize=(10, 6))
    
    # Calculate probabilities for different numbers of heads (0 to 5)
    k_values = np.arange(n + 1)  # 0 to 5 heads
    probabilities = stats.binom.pmf(k_values, n, p)
    
    # Plot the probability mass function
    plt.bar(k_values, probabilities, alpha=0.7, width=0.4, color='blue')
    
    # Add details
    plt.xlabel('Number of Heads in 5 Flips', fontsize=12)
    plt.ylabel('Probability P(data | p=0.5)', fontsize=12)
    plt.title('Probability of Observing k Heads in 5 Flips (p=0.5)', fontsize=14)
    plt.xticks(k_values)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Annotate the specific case of 3 heads (HHTHT)
    plt.annotate(f'P(3 heads in 5 flips | p=0.5) = {probabilities[3]:.4f}',
                xy=(3, probabilities[3]), 
                xytext=(3, probabilities[3] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, ha='center')
    
    # Save the figure
    file_path = os.path.join(save_dir, "coin_flip_probability.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Create visualization of coin flip likelihood
def create_coin_flip_likelihood_plot():
    p_values = np.linspace(0, 1, 100)  # Variable parameter values
    
    # Fixed data: 3 heads in 5 trials (e.g., HHTHT)
    n = 5
    k = 3
    
    # Calculate the likelihood for different p values
    likelihoods = stats.binom.pmf(k, n, p_values)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the likelihood function
    plt.plot(p_values, likelihoods, 'r-', linewidth=2.5)
    
    # Highlight the maximum likelihood estimate
    max_idx = np.argmax(likelihoods)
    max_p = p_values[max_idx]
    max_likelihood = likelihoods[max_idx]
    
    plt.scatter([max_p], [max_likelihood], color='blue', s=100, zorder=3)
    plt.annotate(f'Maximum Likelihood Estimate:\np = {max_p:.2f}',
                xy=(max_p, max_likelihood), 
                xytext=(max_p - 0.3, max_likelihood - 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Add details
    plt.xlabel('Parameter p (probability of heads)', fontsize=12)
    plt.ylabel('Likelihood L(p | 3 heads in 5 flips)', fontsize=12)
    plt.title('Likelihood Function for Different Values of p (Given 3 Heads in 5 Flips)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    file_path = os.path.join(save_dir, "coin_flip_likelihood.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the coin flip visualizations
create_coin_flip_probability_plot()
create_coin_flip_likelihood_plot()

# Step 4: Normal Distribution Example
print_step_header(4, "Normal Distribution Example")

print("Example - Normal Distribution:")
print("- Parameter: μ (mean of the distribution)")
print("- Data: observed values x₁, x₂, ..., xₙ")
print()
print("Consider the following scenarios:")
print("1. Fixed parameter (μ = 0), variable data: What's the probability density at different x values?")
print("2. Fixed data (x = 1.5), variable parameter: What's the likelihood for different μ values?")
print()

# Create visualization of probability density function
def create_normal_probability_plot():
    mu = 0  # Fixed parameter (mean)
    sigma = 1  # Standard deviation
    
    plt.figure(figsize=(10, 6))
    
    # Calculate probability density function for different x values
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)
    
    # Plot the PDF
    plt.plot(x, pdf, 'b-', linewidth=2.5)
    
    # Highlight a specific x value
    x_obs = 1.5
    p_x_obs = stats.norm.pdf(x_obs, mu, sigma)
    
    plt.scatter([x_obs], [p_x_obs], color='red', s=100, zorder=3)
    plt.annotate(f'P(x=1.5 | μ=0, σ=1) = {p_x_obs:.4f}',
                xy=(x_obs, p_x_obs), 
                xytext=(x_obs + 0.5, p_x_obs + 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Add vertical line at x=1.5
    plt.axvline(x=x_obs, color='red', linestyle='--', alpha=0.5)
    
    # Add details
    plt.xlabel('Data Value (x)', fontsize=12)
    plt.ylabel('Probability Density f(x | μ=0, σ=1)', fontsize=12)
    plt.title('Probability Density Function (Fixed μ=0, Variable x)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    file_path = os.path.join(save_dir, "normal_probability.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Create visualization of likelihood function
def create_normal_likelihood_plot():
    x_obs = 1.5  # Fixed observation
    sigma = 1  # Standard deviation
    
    plt.figure(figsize=(10, 6))
    
    # Calculate likelihood for different mu values
    mu_values = np.linspace(-3, 5, 1000)
    likelihoods = stats.norm.pdf(x_obs, mu_values, sigma)
    
    # Plot the likelihood function
    plt.plot(mu_values, likelihoods, 'r-', linewidth=2.5)
    
    # Highlight the maximum likelihood estimate
    max_idx = np.argmax(likelihoods)
    max_mu = mu_values[max_idx]
    max_likelihood = likelihoods[max_idx]
    
    plt.scatter([max_mu], [max_likelihood], color='blue', s=100, zorder=3)
    plt.annotate(f'Maximum Likelihood Estimate:\nμ = {max_mu:.2f}',
                xy=(max_mu, max_likelihood), 
                xytext=(max_mu - 2, max_likelihood - 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    # Add vertical line at MLE
    plt.axvline(x=max_mu, color='blue', linestyle='--', alpha=0.5)
    
    # Add details
    plt.xlabel('Parameter Value (μ)', fontsize=12)
    plt.ylabel('Likelihood L(μ | x=1.5, σ=1)', fontsize=12)
    plt.title('Likelihood Function (Fixed x=1.5, Variable μ)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    file_path = os.path.join(save_dir, "normal_likelihood.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the normal distribution visualizations
create_normal_probability_plot()
create_normal_likelihood_plot()

# Step 5: Unified Visualization
print_step_header(5, "Unified Visualization")

def create_unified_visualization():
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    
    # Create a title for the whole figure
    fig.suptitle('Probability vs. Likelihood: A Visual Comparison', fontsize=16, y=0.98)
    
    # 1. Probability: Fixed parameter, variable data (Coin Flip)
    ax1 = plt.subplot(gs[0, 0])
    
    p = 0.5  # Fixed parameter (probability of heads)
    n = 5    # Number of coin flips
    
    # Calculate probabilities for different numbers of heads (0 to 5)
    k_values = np.arange(n + 1)  # 0 to 5 heads
    probabilities = stats.binom.pmf(k_values, n, p)
    
    # Plot the probability mass function
    ax1.bar(k_values, probabilities, alpha=0.7, width=0.4, color='blue')
    
    # Add details
    ax1.set_xlabel('Number of Heads in 5 Flips', fontsize=10)
    ax1.set_ylabel('Probability P(data | p=0.5)', fontsize=10)
    ax1.set_title('Probability: Fixed p=0.5, Variable Data', fontsize=12)
    ax1.set_xticks(k_values)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add note about what probability represents
    ax1.text(0.5, -0.2, 'P(data | parameter) measures the chance of\ndifferent data outcomes for a fixed parameter',
             transform=ax1.transAxes, fontsize=9, ha='center')
    
    # 2. Likelihood: Fixed data, variable parameter (Coin Flip)
    ax2 = plt.subplot(gs[0, 1])
    
    p_values = np.linspace(0, 1, 100)  # Variable parameter values
    
    # Fixed data: 3 heads in 5 trials
    n = 5
    k = 3
    
    # Calculate the likelihood
    likelihoods = stats.binom.pmf(k, n, p_values)
    
    # Plot the likelihood function
    ax2.plot(p_values, likelihoods, 'r-', linewidth=2.5)
    
    # Add details
    ax2.set_xlabel('Parameter p (probability of heads)', fontsize=10)
    ax2.set_ylabel('Likelihood L(p | 3 heads in 5 flips)', fontsize=10)
    ax2.set_title('Likelihood: Fixed Data (3 Heads), Variable p', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add note about what likelihood represents
    ax2.text(0.5, -0.2, 'L(parameter | data) measures the plausibility of\ndifferent parameter values for fixed observed data',
             transform=ax2.transAxes, fontsize=9, ha='center')
    
    # 3. Probability: Fixed parameter, variable data (Normal)
    ax3 = plt.subplot(gs[1, 0])
    
    mu = 0  # Fixed parameter (mean)
    sigma = 1  # Standard deviation
    
    # Calculate probability density function for different x values
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)
    
    # Plot the PDF
    ax3.plot(x, pdf, 'b-', linewidth=2.5)
    
    # Add details
    ax3.set_xlabel('Data Value (x)', fontsize=10)
    ax3.set_ylabel('Probability Density f(x | μ=0, σ=1)', fontsize=10)
    ax3.set_title('Probability: Fixed μ=0, Variable x', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add note
    ax3.text(0.5, -0.2, 'Probability treats parameters as fixed and\ndata as random variables',
             transform=ax3.transAxes, fontsize=9, ha='center')
    
    # 4. Likelihood: Fixed data, variable parameter (Normal)
    ax4 = plt.subplot(gs[1, 1])
    
    x_obs = 1.5  # Fixed observation
    sigma = 1  # Standard deviation
    
    # Calculate likelihood for different mu values
    mu_values = np.linspace(-3, 5, 1000)
    likelihoods = stats.norm.pdf(x_obs, mu_values, sigma)
    
    # Plot the likelihood function
    ax4.plot(mu_values, likelihoods, 'r-', linewidth=2.5)
    
    # Add details
    ax4.set_xlabel('Parameter Value (μ)', fontsize=10)
    ax4.set_ylabel('Likelihood L(μ | x=1.5, σ=1)', fontsize=10)
    ax4.set_title('Likelihood: Fixed x=1.5, Variable μ', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add note
    ax4.text(0.5, -0.2, 'Likelihood treats data as fixed and\nparameters as variables to be estimated',
             transform=ax4.transAxes, fontsize=9, ha='center')
    
    # Add the key distinction at the bottom of the figure
    plt.figtext(0.5, 0.02, 
                'Key Distinction: Probability deals with the distribution of data given fixed parameters,\n'
                'while likelihood deals with the plausibility of parameters given fixed data.',
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    file_path = os.path.join(save_dir, "probability_vs_likelihood.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the unified visualization
create_unified_visualization()

# Step 6: Summary of Differences
print_step_header(6, "Summary of Differences")

print("Summary of Key Differences:")
print("-" * 80)
print("{:<25} {:<55}".format("Probability", "Likelihood"))
print("-" * 80)
print("{:<25} {:<55}".format("P(data | parameter)", "L(parameter | data)"))
print("{:<25} {:<55}".format("Parameter is fixed", "Data is fixed"))
print("{:<25} {:<55}".format("Data is variable", "Parameter is variable"))
print("{:<25} {:<55}".format("Sums/integrates to 1", "Does not need to sum/integrate to 1"))
print("{:<25} {:<55}".format("Used to make predictions", "Used for inference and estimation"))
print("{:<25} {:<55}".format("Forward problem", "Inverse problem"))
print("-" * 80)
print()

# Step 7: Create a graphical table of differences
def create_difference_table():
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    # Define the table data
    table_data = [
        ["Notation", "P(data | parameter)", "L(parameter | data)"],
        ["What's Fixed", "Parameter", "Data"],
        ["What's Variable", "Data", "Parameter"],
        ["Normalization", "Sums/integrates to 1", "No normalization requirement"],
        ["Used For", "Predictions (forward problem)", "Inference (inverse problem)"],
        ["Goal", "Determine data distribution", "Estimate parameters"]
    ]
    
    # Create the table with colors
    table = ax.table(cellText=table_data, 
                    colLabels=["Aspect", "Probability", "Likelihood"],
                    colColours=["#f2f2f2", "#d4e6f1", "#f5cba7"],
                    cellLoc='center',
                    loc='center',
                    cellColours=[["#f2f2f2"]*3] + [["#f2f2f2", "#d4e6f1", "#f5cba7"]]*5)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Add a title
    plt.title('Comparison of Probability and Likelihood', fontsize=14, pad=20)
    
    # Save the figure
    file_path = os.path.join(save_dir, "difference_table.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {file_path}")
    
    plt.close()

# Generate the difference table
create_difference_table()

# Step 8: Conclusion and Answer
print_step_header(7, "Conclusion and Answer")

print("Conclusion:")
print("The fundamental difference between probability and likelihood is that probability measures the chance of observing data given fixed parameters, while likelihood measures the plausibility of parameter values given fixed observed data.")
print()
print("One-sentence answer:")
print("Probability treats parameters as fixed and data as random (P(data|parameter)), while likelihood treats data as fixed and parameters as variables to be estimated (L(parameter|data)).")