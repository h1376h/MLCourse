import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import random
from matplotlib import cm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- A posterior distribution over θ that is a mixture of two normal distributions:")
print("- p(θ|D) = 0.7 · N(θ|2, 1) + 0.3 · N(θ|5, 0.5)")
print("\nTasks:")
print("1. Generate 5 samples from this posterior distribution using rejection sampling")
print("2. Briefly explain how Markov Chain Monte Carlo (MCMC) could be used to sample from this distribution")
print("3. Would importance sampling be effective for this distribution? Why or why not?")

# Step 2: Visualizing the Posterior Distribution
print_step_header(2, "Visualizing the Posterior Distribution")

# Define parameters for the mixture components
mu1, sigma1 = 2, 1  # First component: N(2, 1)
mu2, sigma2 = 5, 0.5  # Second component: N(5, 0.5)
w1, w2 = 0.7, 0.3  # Mixture weights

# Evaluate the mixture density on a grid
theta = np.linspace(-1, 8, 1000)
p1 = stats.norm.pdf(theta, mu1, sigma1)
p2 = stats.norm.pdf(theta, mu2, sigma2)
p_mixture = w1 * p1 + w2 * p2

# Plot the mixture distribution
plt.figure(figsize=(12, 8))
plt.plot(theta, p_mixture, 'b-', linewidth=2.5, label='Mixture Posterior')
plt.plot(theta, w1 * p1, 'r--', linewidth=1.5, label=f'Component 1: 0.7 · N({mu1}, {sigma1})')
plt.plot(theta, w2 * p2, 'g--', linewidth=1.5, label=f'Component 2: 0.3 · N({mu2}, {sigma2})')
plt.fill_between(theta, p_mixture, alpha=0.2)
plt.xlabel('θ', fontsize=14)
plt.ylabel('Density p(θ|D)', fontsize=14)
plt.title('Posterior Distribution: Mixture of Two Normals', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_mixture.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Find the maximum value of the mixture density
max_density = np.max(p_mixture)
print(f"Maximum density value: {max_density:.6f}")

# Step 3: Rejection Sampling Implementation
print_step_header(3, "Rejection Sampling Implementation")

print("Rejection sampling algorithm:")
print("1. Choose a proposal distribution q(θ) that is easy to sample from")
print("2. Find a constant M such that p(θ|D) ≤ M·q(θ) for all θ")
print("3. Sample θ' from q(θ)")
print("4. Sample u from Uniform(0, 1)")
print("5. If u ≤ p(θ'|D) / (M·q(θ')), accept θ'; otherwise reject and go back to step 3")

# Define a proposal distribution
# We'll use a uniform distribution over the range of interest
proposal_min, proposal_max = -2, 9
proposal_range = proposal_max - proposal_min

# Define the proposal distribution density function
def q_theta(theta):
    """Uniform proposal distribution."""
    if proposal_min <= theta <= proposal_max:
        return 1.0 / proposal_range
    else:
        return 0.0

# Define the target mixture distribution density function
def p_theta(theta):
    """Mixture of two normals."""
    p1_val = w1 * stats.norm.pdf(theta, mu1, sigma1)
    p2_val = w2 * stats.norm.pdf(theta, mu2, sigma2)
    return p1_val + p2_val

# Find a suitable M
# For a uniform proposal, M should be range * max_density
M = proposal_range * max_density * 1.1  # Adding a safety factor
print(f"Using scaling factor M = {M:.6f}")

# Implement rejection sampling
def rejection_sampling(n_samples):
    """Generate samples using rejection sampling."""
    samples = []
    accepted_count = 0
    rejected_count = 0
    
    while len(samples) < n_samples:
        # Step 3: Sample from proposal
        theta_proposal = np.random.uniform(proposal_min, proposal_max)
        
        # Step 4: Sample uniform for acceptance test
        u = np.random.uniform(0, 1)
        
        # Step 5: Accept/reject
        if u <= p_theta(theta_proposal) / (M * q_theta(theta_proposal)):
            samples.append(theta_proposal)
            accepted_count += 1
        else:
            rejected_count += 1
    
    acceptance_rate = accepted_count / (accepted_count + rejected_count)
    return np.array(samples), acceptance_rate

# Generate 5 samples
n_samples = 5
rejection_samples, acceptance_rate = rejection_sampling(n_samples)

print(f"\nGenerated {n_samples} samples using rejection sampling:")
for i, sample in enumerate(rejection_samples):
    print(f"Sample {i+1}: {sample:.4f}")
print(f"\nAcceptance rate: {acceptance_rate:.4f}")

# Let's generate more samples for visualization
n_vis_samples = 1000
vis_samples, _ = rejection_sampling(n_vis_samples)

# Plot the posterior with samples
plt.figure(figsize=(12, 8))
plt.plot(theta, p_mixture, 'b-', linewidth=2.5, label='Target Posterior')

# Plot histogram of samples
plt.hist(vis_samples, bins=30, density=True, alpha=0.5, color='green', label='Samples Histogram')

# Mark the 5 required samples
plt.scatter(rejection_samples, np.zeros_like(rejection_samples) + 0.01, 
           color='red', s=100, marker='o', label='5 Generated Samples')

plt.xlabel('θ', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Rejection Sampling from Mixture of Normals', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "rejection_sampling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: MCMC Approach
print_step_header(4, "MCMC Approach")

print("Markov Chain Monte Carlo (MCMC) can be used to sample from this posterior distribution.")
print("Specifically, we can use the Metropolis-Hastings algorithm:")
print("\n1. Initialize θ₀")
print("2. For t = 1 to T:")
print("   a. Propose θ' ~ q(θ'|θ_{t-1})")
print("   b. Compute acceptance ratio α = min(1, [p(θ'|D) · q(θ_{t-1}|θ')] / [p(θ_{t-1}|D) · q(θ'|θ_{t-1})])")
print("   c. With probability α, accept θ' and set θ_t = θ'; otherwise set θ_t = θ_{t-1}")
print("3. Return samples θ₁,...,θ_T")
print("\nFor this distribution, we could use a random walk proposal: θ' ~ N(θ_{t-1}, σ²)")
print("This would efficiently explore the posterior, handling the multimodal nature of the mixture.")

# Implement the Metropolis-Hastings algorithm for demonstration
def metropolis_hastings(n_samples, proposal_std=0.5, burn_in=100):
    """Generate samples using Metropolis-Hastings algorithm."""
    # Start with a random initialization
    current = np.random.normal(mu1, sigma1)
    samples = []
    accepted_count = 0
    
    # Run for burn-in plus required samples
    total_iterations = burn_in + n_samples
    
    for i in range(total_iterations):
        # Generate proposal
        proposal = np.random.normal(current, proposal_std)
        
        # Compute acceptance ratio
        acceptance_ratio = min(1, p_theta(proposal) / p_theta(current))
        
        # Accept or reject
        u = np.random.uniform(0, 1)
        if u <= acceptance_ratio:
            current = proposal
            if i >= burn_in:
                accepted_count += 1
        
        # Store sample after burn-in
        if i >= burn_in:
            samples.append(current)
    
    acceptance_rate = accepted_count / n_samples
    return np.array(samples), acceptance_rate

# Generate 1000 samples with MCMC
mcmc_samples, mcmc_acceptance_rate = metropolis_hastings(1000)

print(f"\nGenerated {len(mcmc_samples)} samples using MCMC (after burn-in):")
print(f"First 5 samples: {mcmc_samples[:5]}")
print(f"Acceptance rate: {mcmc_acceptance_rate:.4f}")

# Plot MCMC sampling results
plt.figure(figsize=(12, 8))
plt.plot(theta, p_mixture, 'b-', linewidth=2, label='Target Posterior')
plt.hist(mcmc_samples, bins=30, density=True, alpha=0.5, color='green', label='MCMC Samples')
plt.xlabel('θ', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('MCMC Sampling from Mixture of Normals', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mcmc_sampling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Plot MCMC trace to check convergence
plt.figure(figsize=(12, 8))
plt.plot(mcmc_samples, 'b-', alpha=0.6)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('θ', fontsize=14)
plt.title('MCMC Trace Plot (after burn-in)', fontsize=16)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mcmc_trace.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Importance Sampling
print_step_header(5, "Importance Sampling")

print("Importance sampling is a technique where we sample from a proposal distribution q(θ)")
print("and then reweight the samples to approximate expectations under p(θ|D).")
print("\nFor our mixture posterior, importance sampling could work if we choose a good proposal.")
print("A suitable proposal for this mixture would be another mixture that covers both modes.")
print("\nHowever, there are challenges:")
print("1. Finding a good proposal is non-trivial; we need to cover both modes adequately")
print("2. If the proposal doesn't match the target well, we get high variance in weights")
print("3. For multimodal distributions, some regions might be poorly represented")

# Implement importance sampling for demonstration
def importance_sampling(n_samples):
    """Generate weighted samples using importance sampling."""
    # Use a mixture of normals as the proposal
    # This should cover both modes reasonably well
    proposal_mu1, proposal_sigma1 = 2.5, 1.5  # Broader than the target's first component
    proposal_mu2, proposal_sigma2 = 5, 1.0    # Broader than the target's second component
    proposal_w1, proposal_w2 = 0.6, 0.4       # Slightly shifted weights
    
    # Function to sample from the proposal
    def sample_from_proposal():
        if np.random.random() < proposal_w1:
            return np.random.normal(proposal_mu1, proposal_sigma1)
        else:
            return np.random.normal(proposal_mu2, proposal_sigma2)
    
    # Function to evaluate the proposal density
    def q_proposal(x):
        p1_val = proposal_w1 * stats.norm.pdf(x, proposal_mu1, proposal_sigma1)
        p2_val = proposal_w2 * stats.norm.pdf(x, proposal_mu2, proposal_sigma2)
        return p1_val + p2_val
    
    # Generate samples and compute weights
    samples = []
    weights = []
    
    for _ in range(n_samples):
        sample = sample_from_proposal()
        weight = p_theta(sample) / q_proposal(sample)
        samples.append(sample)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)
    
    return np.array(samples), normalized_weights

# Generate 1000 samples with importance sampling
is_samples, is_weights = importance_sampling(1000)

# Compute effective sample size (ESS)
ess = 1 / np.sum(is_weights**2)
print(f"\nImportance Sampling:")
print(f"Effective Sample Size (ESS): {ess:.2f} out of 1000")
print(f"ESS Ratio: {ess/1000:.4f}")

# Plot the importance sampling results
plt.figure(figsize=(12, 8))
plt.plot(theta, p_mixture, 'b-', linewidth=2, label='Target Posterior')

# Plot weighted histogram
plt.hist(is_samples, bins=30, weights=is_weights*len(is_samples)/np.sum(is_weights), 
         density=True, alpha=0.5, color='red', label='Weighted IS Samples')

plt.xlabel('θ', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Importance Sampling from Mixture of Normals', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "importance_sampling.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Comparison of Sampling Methods
print_step_header(6, "Comparison of Sampling Methods")

# Generate more samples for better comparison
n_compare = 2000
rejection_samples_compare, _ = rejection_sampling(n_compare)
mcmc_samples_compare, _ = metropolis_hastings(n_compare)
is_samples_compare, is_weights_compare = importance_sampling(n_compare)

# Create a combined plot
plt.figure(figsize=(15, 10))

# Plot the target density
plt.subplot(2, 2, 1)
plt.plot(theta, p_mixture, 'k-', linewidth=2)
plt.fill_between(theta, p_mixture, alpha=0.2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Target Posterior: Mixture of Normals', fontsize=14)
plt.grid(True)

# Plot rejection sampling results
plt.subplot(2, 2, 2)
plt.hist(rejection_samples_compare, bins=30, density=True, alpha=0.6, color='blue')
plt.plot(theta, p_mixture, 'k-', linewidth=2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Rejection Sampling', fontsize=14)
plt.grid(True)

# Plot MCMC results
plt.subplot(2, 2, 3)
plt.hist(mcmc_samples_compare, bins=30, density=True, alpha=0.6, color='green')
plt.plot(theta, p_mixture, 'k-', linewidth=2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('MCMC (Metropolis-Hastings)', fontsize=14)
plt.grid(True)

# Plot importance sampling results
plt.subplot(2, 2, 4)
plt.hist(is_samples_compare, bins=30, weights=is_weights_compare*len(is_samples_compare)/np.sum(is_weights_compare), 
         density=True, alpha=0.6, color='red')
plt.plot(theta, p_mixture, 'k-', linewidth=2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Importance Sampling', fontsize=14)
plt.grid(True)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sampling_methods_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {file_path}")

# Summarize findings
print("\nComparison of Sampling Methods:")
print("1. Rejection Sampling:")
print("   + Simple to implement")
print("   + Independent samples")
print("   - Low acceptance rate (inefficient)")
print("   - Requires knowing an upper bound on the density")
print("\n2. MCMC (Metropolis-Hastings):")
print("   + Works well for complex distributions")
print("   + Only requires density up to a constant")
print("   - Correlated samples")
print("   - Requires tuning (proposal distribution, burn-in)")
print("\n3. Importance Sampling:")
print("   + Independent samples")
print("   + Can estimate multiple expectations efficiently")
print("   - Requires a good proposal distribution")
print("   - Can have high variance for multimodal distributions")
print("\nFor the given mixture distribution:")
print("- Rejection sampling works but may be inefficient")
print("- MCMC works well, especially with a well-tuned proposal")
print("- Importance sampling could work with a carefully designed proposal, but might struggle")
print("  with balancing coverage of both modes without introducing high variance") 