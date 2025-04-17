import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_5")
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

print("Given:")
print("- A fair coin toss with P(heads) = 0.5, P(tails) = 0.5")
print("- A biased coin with P(heads) = 0.8, P(tails) = 0.2")
print()
print("Tasks:")
print("1. Calculate the entropy of the fair coin distribution")
print("2. Calculate the entropy of the biased coin distribution")
print("3. Explain why the fair coin has higher entropy than the biased coin")
print()

# Step 2: Define the entropy function
print_step_header(2, "Defining the Entropy Function")

def entropy(p):
    """
    Calculate the entropy of a binary probability distribution.
    
    Parameters:
    p (float): Probability of one outcome (e.g., heads)
    
    Returns:
    float: Entropy in bits
    """
    # Ensure p is between 0 and 1
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # Calculate entropy: -p*log2(p) - (1-p)*log2(1-p)
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

print("Entropy formula for a binary distribution (e.g., coin toss):")
print("H(X) = -p * log₂(p) - (1-p) * log₂(1-p)")
print()
print("Where:")
print("- p is the probability of one outcome (e.g., heads)")
print("- 1-p is the probability of the other outcome (e.g., tails)")
print()

# Step 3: Calculate entropy for the fair coin
print_step_header(3, "Calculating Entropy for the Fair Coin")

p_fair = 0.5
h_fair = entropy(p_fair)

print(f"For a fair coin with P(heads) = {p_fair}, P(tails) = {1-p_fair}:")
print(f"H(X) = -({p_fair}) * log₂({p_fair}) - ({1-p_fair}) * log₂({1-p_fair})")
print(f"H(X) = -{p_fair} * ({np.log2(p_fair)}) - {1-p_fair} * ({np.log2(1-p_fair)})")
print(f"H(X) = {-p_fair * np.log2(p_fair):.6f} + {-p_fair * np.log2(p_fair):.6f}")
print(f"H(X) = {h_fair:.6f} bits")
print()
print(f"The entropy of a fair coin toss is {h_fair} bits.")
print("This means we need 1 bit of information on average to describe the outcome of a fair coin toss.")

# Visualize the entropy calculation for the fair coin
plt.figure(figsize=(8, 6))
labels = ['P(H) * (-log₂(P(H)))', 'P(T) * (-log₂(P(T)))', 'Total Entropy']
values = [-p_fair * np.log2(p_fair), -(1-p_fair) * np.log2(1-p_fair), h_fair]
colors = ['blue', 'green', 'red']

plt.bar(labels, values, color=colors, alpha=0.7)
plt.title('Entropy Calculation for Fair Coin')
plt.ylabel('Information (bits)')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
plt.tight_layout()
file_path = os.path.join(save_dir, "fair_coin_entropy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate entropy for the biased coin
print_step_header(4, "Calculating Entropy for the Biased Coin")

p_biased = 0.8
h_biased = entropy(p_biased)

print(f"For a biased coin with P(heads) = {p_biased}, P(tails) = {1-p_biased}:")
print(f"H(X) = -({p_biased}) * log₂({p_biased}) - ({1-p_biased}) * log₂({1-p_biased})")
print(f"H(X) = -{p_biased} * ({np.log2(p_biased)}) - {1-p_biased} * ({np.log2(1-p_biased)})")
print(f"H(X) = {-p_biased * np.log2(p_biased):.6f} + {-(1-p_biased) * np.log2(1-p_biased):.6f}")
print(f"H(X) = {h_biased:.6f} bits")
print()
print(f"The entropy of a biased coin toss with P(heads) = {p_biased} is {h_biased} bits.")
print(f"This is less than the entropy of a fair coin ({h_fair} bits).")

# Visualize the entropy calculation for the biased coin
plt.figure(figsize=(8, 6))
labels = ['P(H) * (-log₂(P(H)))', 'P(T) * (-log₂(P(T)))', 'Total Entropy']
values = [-p_biased * np.log2(p_biased), -(1-p_biased) * np.log2(1-p_biased), h_biased]
colors = ['blue', 'green', 'red']

plt.bar(labels, values, color=colors, alpha=0.7)
plt.title(f'Entropy Calculation for Biased Coin (P(heads) = {p_biased})')
plt.ylabel('Information (bits)')
plt.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
plt.tight_layout()
file_path = os.path.join(save_dir, "biased_coin_entropy.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Compare the entropies for different bias probabilities
print_step_header(5, "Comparing Entropies for Different Bias Probabilities")

# Calculate entropy for a range of probabilities
p_range = np.linspace(0.01, 0.99, 100)
h_range = [entropy(p) for p in p_range]

# Visualize how entropy changes with the bias probability
plt.figure(figsize=(10, 6))
plt.plot(p_range, h_range, 'b-', linewidth=2)

# Mark the fair and biased coins
plt.plot(p_fair, h_fair, 'ro', markersize=8, label=f'Fair Coin: p={p_fair}, H={h_fair:.4f}')
plt.plot(p_biased, h_biased, 'go', markersize=8, label=f'Biased Coin: p={p_biased}, H={h_biased:.4f}')
plt.plot(1-p_biased, h_biased, 'go', markersize=8)  # Mark the symmetric point as well

plt.xlabel('Probability of Heads (p)')
plt.ylabel('Entropy H(X) (bits)')
plt.title('Entropy vs. Bias Probability for a Coin Toss')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Add a horizontal line for fair coin entropy
plt.axhline(y=h_fair, color='r', linestyle='--', alpha=0.5)

# Add a vertical line for p = 0.5
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

file_path = os.path.join(save_dir, "entropy_vs_bias.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Explain why fair coin has higher entropy
print_step_header(6, "Explaining Why Fair Coin Has Higher Entropy")

print("Why fair coin has higher entropy than a biased coin:")
print()
print("1. Entropy measures uncertainty or information content of a random variable.")
print()
print("2. The fair coin has maximum uncertainty:")
print("   - P(heads) = P(tails) = 0.5")
print("   - Outcomes are equally likely, making prediction hardest")
print("   - Maximum entropy for a binary variable is 1 bit (achieved by fair coin)")
print()
print("3. The biased coin has more predictability:")
print(f"   - P(heads) = {p_biased}, P(tails) = {1-p_biased}")
print("   - Outcomes are not equally likely - we can make better predictions")
print("   - The more biased the coin, the more predictable the outcome")
print()
print("4. Mathematically, entropy H(X) = -Σp(x)log₂(p(x)) is maximized when all")
print("   outcomes are equally likely. Any deviation from equal probabilities")
print("   reduces entropy.")
print()
print("5. Intuitive explanation: If asked to guess the outcome of:")
print("   - A fair coin toss: Best strategy gives 50% accuracy")
print(f"   - A biased coin toss (P(heads) = {p_biased}): Best strategy gives {max(p_biased, 1-p_biased)*100:.0f}% accuracy")
print("   The biased coin requires less information to predict well.")

# Create a visualization to explain the entropy difference
plt.figure(figsize=(12, 10))

# Create a 2x2 grid
gs = GridSpec(2, 2, height_ratios=[1, 1])

# Top left: Fair coin probability distribution
ax1 = plt.subplot(gs[0, 0])
ax1.bar(['Heads', 'Tails'], [p_fair, 1-p_fair], color=['skyblue', 'skyblue'], alpha=0.7)
ax1.set_ylim(0, 1)
ax1.set_ylabel('Probability')
ax1.set_title('Fair Coin Distribution')
ax1.text(0.5, 0.8, f'Entropy = {h_fair:.4f} bits', transform=ax1.transAxes, 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Top right: Biased coin probability distribution
ax2 = plt.subplot(gs[0, 1])
ax2.bar(['Heads', 'Tails'], [p_biased, 1-p_biased], color=['skyblue', 'skyblue'], alpha=0.7)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Probability')
ax2.set_title(f'Biased Coin Distribution (p={p_biased})')
ax2.text(0.5, 0.8, f'Entropy = {h_biased:.4f} bits', transform=ax2.transAxes, 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Bottom: Entropy vs. probability curve
ax3 = plt.subplot(gs[1, :])
ax3.plot(p_range, h_range, 'b-', linewidth=2)
ax3.plot(p_fair, h_fair, 'ro', markersize=8)
ax3.plot(p_biased, h_biased, 'go', markersize=8)
ax3.plot(1-p_biased, h_biased, 'go', markersize=8)  # Plot symmetric point
ax3.axhline(y=h_fair, color='r', linestyle='--', alpha=0.5)
ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Probability of Heads (p)')
ax3.set_ylabel('Entropy H(X) (bits)')
ax3.set_title('Entropy vs. Bias Probability')
ax3.grid(True)
ax3.text(0.2, 0.9, 'Maximum entropy at p = 0.5', transform=ax3.transAxes, 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
file_path = os.path.join(save_dir, "entropy_explanation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Create a visual comparison of predictability
print_step_header(7, "Visualizing Predictability and Entropy")

# Set up simulation parameters
n_samples = 1000
rng = np.random.RandomState(42)  # For reproducibility

# Generate sequences of coin tosses
fair_tosses = rng.choice([0, 1], size=n_samples, p=[p_fair, 1-p_fair])
biased_tosses = rng.choice([0, 1], size=n_samples, p=[p_biased, 1-p_biased])

# Calculate best prediction accuracy
fair_best_pred = max(np.mean(fair_tosses), 1 - np.mean(fair_tosses))
biased_best_pred = max(np.mean(biased_tosses), 1 - np.mean(biased_tosses))

print(f"Simulation of {n_samples} coin tosses:")
print(f"- Fair coin: Best prediction accuracy = {fair_best_pred:.4f}")
print(f"- Biased coin: Best prediction accuracy = {biased_best_pred:.4f}")

# Visualize the first 20 tosses of each
n_display = 20
plt.figure(figsize=(12, 6))

# Plot fair coin tosses
plt.subplot(2, 1, 1)
plt.stem(range(n_display), fair_tosses[:n_display], linefmt='b-', markerfmt='bo', basefmt='none')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.yticks([0, 1], ['Heads', 'Tails'])
plt.ylim(-0.1, 1.1)
plt.title(f'Fair Coin Tosses (Entropy = {h_fair:.4f} bits, Predictability ≈ 50%)')
plt.xticks(range(n_display))
plt.grid(True, axis='y')

# Plot biased coin tosses
plt.subplot(2, 1, 2)
plt.stem(range(n_display), biased_tosses[:n_display], linefmt='g-', markerfmt='go', basefmt='none')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.yticks([0, 1], ['Heads', 'Tails'])
plt.ylim(-0.1, 1.1)
plt.title(f'Biased Coin Tosses (P(heads) = {p_biased}, Entropy = {h_biased:.4f} bits, Predictability ≈ {biased_best_pred*100:.0f}%)')
plt.xticks(range(n_display))
plt.xlabel('Toss Number')
plt.grid(True, axis='y')

plt.tight_layout()
file_path = os.path.join(save_dir, "coin_tosses_simulation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Create a visualization of required bits
print_step_header(8, "Visualizing Encoding Efficiency and Entropy")

print("Entropy also relates to the average number of bits needed to encode outcomes:")

# Create simulation for encoding length
def encode_tosses(tosses, p_heads):
    """Simulate encoding of coin tosses using optimal coding."""
    if p_heads == 0.5:  # Fair coin
        # For fair coin, 1 bit per toss is optimal
        return len(tosses)
    else:
        # For biased coin, approximate Shannon coding length
        p_tails = 1 - p_heads
        bits_heads = -np.log2(p_heads)
        bits_tails = -np.log2(p_tails)
        return sum(bits_heads if toss == 0 else bits_tails for toss in tosses)

# Calculate bits needed
n_tosses = 100
fair_bits = encode_tosses(fair_tosses[:n_tosses], 0.5)
biased_bits = encode_tosses(biased_tosses[:n_tosses], 1-p_biased)  # Using P(tails) for encoding

print(f"To encode {n_tosses} coin tosses:")
print(f"- Fair coin: Approximately {fair_bits:.2f} bits ({fair_bits/n_tosses:.2f} bits per toss)")
print(f"- Biased coin: Approximately {biased_bits:.2f} bits ({biased_bits/n_tosses:.2f} bits per toss)")
print(f"- Ratio: {biased_bits/fair_bits:.2f} (biased / fair)")
print()
print("This demonstrates that entropy is directly related to the optimal encoding length.")
print("The lower entropy of the biased coin allows for more efficient encoding.")

# Visualize encoding efficiency
plt.figure(figsize=(10, 6))
bits_per_p = []
for p in p_range:
    # Calculate theoretical bits per toss
    p_h = p
    p_t = 1 - p
    if p_h > 0 and p_t > 0:
        bits = p_h * (-np.log2(p_h)) + p_t * (-np.log2(p_t))
    else:
        bits = 0
    bits_per_p.append(bits)

plt.plot(p_range, bits_per_p, 'b-', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1 bit per toss')
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

plt.plot(p_fair, h_fair, 'ro', markersize=8, label=f'Fair Coin: {h_fair:.2f} bits/toss')
plt.plot(p_biased, h_biased, 'go', markersize=8, label=f'Biased Coin: {h_biased:.2f} bits/toss')

plt.xlabel('Probability of Heads (p)')
plt.ylabel('Average Bits Required per Toss')
plt.title('Encoding Efficiency vs. Bias Probability')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_path = os.path.join(save_dir, "encoding_efficiency.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Conclusion
print_step_header(9, "Conclusion and Answer Summary")

print("Question 5 Solution Summary:")
print("\n1. Entropy of a fair coin (P(heads) = 0.5, P(tails) = 0.5):")
print(f"   H(X) = {h_fair} bits")

print("\n2. Entropy of a biased coin (P(heads) = 0.8, P(tails) = 0.2):")
print(f"   H(X) = {h_biased} bits")

print("\n3. The fair coin has higher entropy because:")
print("   a) Entropy measures uncertainty or unpredictability of a random variable")
print("   b) The fair coin has maximum uncertainty with equal probabilities for heads and tails")
print("   c) The biased coin has more predictable outcomes, making it less uncertain")
print("   d) Mathematically, entropy H(X) = -Σp(x)log₂(p(x)) is maximized when all outcomes")
print("      are equally likely (which happens with the fair coin)")
print("   e) Intuitively, we can predict the biased coin's outcome with greater accuracy,")
print("      so it contains less information (lower entropy)")
print("   f) Information-theoretically, we need fewer bits on average to encode outcomes")
print("      from the biased coin compared to the fair coin") 