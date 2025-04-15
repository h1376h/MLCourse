import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binom
from scipy.special import comb

# Create Images directory if it doesn't exist
os.makedirs('../Images', exist_ok=True)

print("\n=== DISCRETE PROBABILITY SIMULATIONS: THEORETICAL VS EMPIRICAL ===\n")

# Example 1: Coin Flip Simulation
print("Simulating coin flips...")
n_trials = 3
p_success = 0.5
n_simulations = [100, 1000, 10000, 100000]
target_heads = 2

theoretical_prob = binom.pmf(target_heads, n_trials, p_success)
simulated_probs = []

plt.figure(figsize=(12, 6))
for i, n_sim in enumerate(n_simulations):
    # Simulate coin flips
    flips = np.random.binomial(n_trials, p_success, n_sim)
    empirical_prob = np.sum(flips == target_heads) / n_sim
    simulated_probs.append(empirical_prob)
    
plt.plot(range(len(n_simulations)), [theoretical_prob]*len(n_simulations), 'r--', 
         label=f'Theoretical P(X={target_heads}) = {theoretical_prob:.4f}')
plt.plot(range(len(n_simulations)), simulated_probs, 'bo-', 
         label='Simulated Probability')

plt.xticks(range(len(n_simulations)), [f'n={n:,}' for n in n_simulations])
plt.xlabel('Number of Simulations', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Convergence of Simulated Coin Flip Probabilities', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('../Images/coin_flip_simulation.png', dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Dice Roll Simulation
print("Simulating dice rolls...")
n_simulations = [100, 1000, 10000, 100000]
target_sum = 7
theoretical_prob = 6/36  # We calculated this earlier

simulated_probs = []
plt.figure(figsize=(12, 6))

for i, n_sim in enumerate(n_simulations):
    # Simulate dice rolls
    die1 = np.random.randint(1, 7, n_sim)
    die2 = np.random.randint(1, 7, n_sim)
    sums = die1 + die2
    empirical_prob = np.sum(sums == target_sum) / n_sim
    simulated_probs.append(empirical_prob)

plt.plot(range(len(n_simulations)), [theoretical_prob]*len(n_simulations), 'r--',
         label=f'Theoretical P(sum=7) = {theoretical_prob:.4f}')
plt.plot(range(len(n_simulations)), simulated_probs, 'bo-',
         label='Simulated Probability')

plt.xticks(range(len(n_simulations)), [f'n={n:,}' for n in n_simulations])
plt.xlabel('Number of Simulations', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Convergence of Simulated Dice Roll Probabilities', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('../Images/dice_roll_simulation.png', dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Card Drawing Simulation
print("Simulating card draws...")
n_simulations = [100, 1000, 10000, 100000]
deck_size = 52
hearts_in_deck = 13
hand_size = 5
target_hearts = 2

theoretical_prob = (comb(hearts_in_deck, target_hearts) * 
                   comb(deck_size - hearts_in_deck, hand_size - target_hearts)) / comb(deck_size, hand_size)

simulated_probs = []
plt.figure(figsize=(12, 6))

for i, n_sim in enumerate(n_simulations):
    hearts_count = []
    for _ in range(n_sim):
        # Create a deck: 1-13 are hearts
        deck = np.arange(deck_size)
        # Draw 5 cards
        hand = np.random.choice(deck, hand_size, replace=False)
        # Count hearts (cards 0-12)
        n_hearts = np.sum(hand < hearts_in_deck)
        hearts_count.append(n_hearts)
    
    empirical_prob = np.sum(np.array(hearts_count) == target_hearts) / n_sim
    simulated_probs.append(empirical_prob)

plt.plot(range(len(n_simulations)), [theoretical_prob]*len(n_simulations), 'r--',
         label=f'Theoretical P(2 hearts) = {theoretical_prob:.4f}')
plt.plot(range(len(n_simulations)), simulated_probs, 'bo-',
         label='Simulated Probability')

plt.xticks(range(len(n_simulations)), [f'n={n:,}' for n in n_simulations])
plt.xlabel('Number of Simulations', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Convergence of Simulated Card Drawing Probabilities', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('../Images/card_drawing_simulation.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nAll simulation visualizations completed successfully.")
print("Additional visualizations have been saved to the Images directory.") 