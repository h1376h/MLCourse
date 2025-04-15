import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binom
from scipy.special import comb

# Create Images directory if it doesn't exist
os.makedirs('../Images', exist_ok=True)

print("\n=== DISCRETE PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Example 1: Fair Coin Flipping
print("Example 1: Fair Coin Flipping")
print("Problem: Consider flipping a fair coin 3 times. What is the probability of getting exactly 2 heads?")

# Parameters
n = 3  # number of trials
p = 0.5  # probability of heads
k = 2  # number of successes (heads)

# Step 1: Identify the relevant probability model
print("\nStep 1: Identify the relevant probability model")
print(f"This is a binomial probability problem with n={n} trials and probability of success p={p}.")

# Step 2: Apply the binomial formula
print("\nStep 2: Apply the binomial formula")
print(f"The binomial formula for k={k} successes in n={n} trials with probability p={p} is:")
print(f"P(X = k) = C(n,k) * p^k * (1-p)^(n-k)")

# Step 3: Calculate the result
print("\nStep 3: Calculate the result")
# Calculate binomial coefficient
nCk = comb(n, k)
print(f"Binomial coefficient C({n},{k}) = {nCk}")

# Calculate probability
prob = nCk * (p**k) * ((1-p)**(n-k))
print(f"P(X = {k}) = {nCk} * ({p})^{k} * (1-{p})^({n-k}) = {prob:.4f}")

print(f"Therefore, the probability of getting exactly {k} heads when flipping a fair coin {n} times is {prob:.4f} or {prob*100:.1f}%.")

# Visualize the binomial distribution
plt.figure(figsize=(10, 6))
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)
plt.bar(x, pmf, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Heads', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(f'Binomial Distribution: n={n}, p={p}', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.5)

# Highlight the probability we calculated
plt.bar(k, pmf[k], color='red', alpha=0.7)
plt.text(k, pmf[k] + 0.02, f'P(X={k}) = {prob:.4f}', ha='center', fontsize=10)

# Add expectation line
expected = n * p
plt.axvline(x=expected, color='green', linestyle='--', label=f'E[X] = np = {expected:.1f}')
plt.legend()

plt.tight_layout()
plt.savefig('../Images/coin_flip_binomial.png', dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Rolling Dice
print("\n\nExample 2: Rolling Dice")
print("Problem: You roll two fair six-sided dice. What is the probability that the sum of the dice equals 7?")

# Step 1: Identify the sample space
print("\nStep 1: Identify the sample space")
print("When rolling two dice, each die can show a value from 1 to 6.")
print("The total number of possible outcomes is 6 × 6 = 36, and each outcome is equally likely with probability 1/36.")

# Step 2: Count favorable outcomes
print("\nStep 2: Count favorable outcomes")
print("We need to count how many ways we can get a sum of 7:")
favorable_outcomes = [(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)]
for outcome in favorable_outcomes:
    print(f"- ({outcome[0]},{outcome[1]}): First die shows {outcome[0]}, second die shows {outcome[1]}")

num_favorable = len(favorable_outcomes)
print(f"There are {num_favorable} favorable outcomes.")

# Step 3: Calculate the probability
print("\nStep 3: Calculate the probability")
prob = num_favorable / 36
print(f"The probability is the number of favorable outcomes divided by the total number of possible outcomes:")
print(f"P(sum = 7) = {num_favorable}/36 = {prob:.4f}")

print(f"Therefore, the probability of rolling a sum of 7 with two fair dice is {prob:.4f} or approximately {prob*100:.1f}%.")

# Visualize the probability distribution of dice sums
plt.figure(figsize=(12, 6))
sums = np.arange(2, 13)
counts = np.zeros(11)
for i in range(1, 7):
    for j in range(1, 7):
        counts[i+j-2] += 1
probs = counts / 36

plt.bar(sums, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Sum of Dice', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability Distribution of Dice Sums', fontsize=14)
plt.xticks(sums)
plt.ylim(0, 0.2)

# Highlight the probability we calculated
sum_index = 7 - 2  # Index for sum of 7
plt.bar(7, probs[sum_index], color='red', alpha=0.7)
plt.text(7, probs[sum_index] + 0.01, f'P(sum=7) = {prob:.4f}', ha='center', fontsize=10)

# Add expectation line
expected = 7  # Expected value for sum of two fair dice
plt.axvline(x=expected, color='green', linestyle='--', label=f'E[Sum] = {expected}')
plt.legend()

plt.tight_layout()
plt.savefig('../Images/dice_sum_probability.png', dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Drawing Cards
print("\n\nExample 3: Drawing Cards")
print("Problem: A standard deck has 52 cards with 4 suits (hearts, diamonds, clubs, spades), each with 13 cards.")
print("If you draw 5 cards randomly without replacement, what is the probability of getting exactly 2 hearts?")

# Step 1: Identify the total number of possible 5-card hands
print("\nStep 1: Identify the total number of possible 5-card hands")
total_hands = comb(52, 5)
print(f"The total number of ways to draw 5 cards from a 52-card deck is:")
print(f"C(52,5) = {total_hands}")

# Step 2: Count the number of favorable outcomes
print("\nStep 2: Count the number of favorable outcomes")
hearts_ways = comb(13, 2)  # Ways to choose 2 hearts from 13
non_hearts_ways = comb(39, 3)  # Ways to choose 3 non-hearts from 39
favorable_hands = hearts_ways * non_hearts_ways

print(f"To get exactly 2 hearts:")
print(f"- We need to choose 2 cards from the 13 hearts: C(13,2) = {hearts_ways} ways")
print(f"- We need to choose 3 cards from the 39 non-hearts: C(39,3) = {non_hearts_ways} ways")
print(f"The total number of favorable outcomes is the product:")
print(f"C(13,2) × C(39,3) = {hearts_ways} × {non_hearts_ways} = {favorable_hands}")

# Step 3: Calculate the probability
print("\nStep 3: Calculate the probability")
prob = favorable_hands / total_hands
print(f"The probability is:")
print(f"P(exactly 2 hearts) = C(13,2) × C(39,3) / C(52,5) = {prob:.4f}")

print(f"Therefore, the probability of drawing exactly 2 hearts when drawing 5 cards from a standard deck is approximately {prob:.4f} or {prob*100:.2f}%.")

# Visualize the hypergeometric distribution
plt.figure(figsize=(10, 6))
x = np.arange(0, 6)
probs = np.zeros(6)
for i in range(6):
    hearts_ways = comb(13, i)
    non_hearts_ways = comb(39, 5-i)
    probs[i] = (hearts_ways * non_hearts_ways) / total_hands

plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Hearts', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Hypergeometric Distribution: Drawing 5 Cards from a Deck', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.4)

# Highlight the probability we calculated
plt.bar(2, probs[2], color='red', alpha=0.7)
plt.text(2, probs[2] + 0.02, f'P(X=2) = {prob:.4f}', ha='center', fontsize=10)

# Add expectation line
expected = 5 * (13/52)  # Expected value for hypergeometric distribution
plt.axvline(x=expected, color='green', linestyle='--', label=f'E[X] = {expected:.1f}')
plt.legend()

plt.tight_layout()
plt.savefig('../Images/card_drawing_hypergeometric.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nAll discrete probability examples completed successfully.")
print("Visualizations have been saved to the Images directory.") 