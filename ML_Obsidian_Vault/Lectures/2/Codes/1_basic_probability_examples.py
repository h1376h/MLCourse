import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os
from scipy import stats
import seaborn as sns

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

print("\n=== BASIC PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Example 1: Coin Toss Probability
print("\nExample 1: Coin Toss Probability")
print("What is the probability of getting exactly 2 heads when tossing a fair coin 3 times?")

# Generate all possible outcomes
coin_tosses = list(product(['H', 'T'], repeat=3))
print("\nAll possible outcomes:")
for outcome in coin_tosses:
    print(''.join(outcome))

# Count outcomes with exactly 2 heads
favorable_outcomes = [outcome for outcome in coin_tosses if outcome.count('H') == 2]
print("\nFavorable outcomes (exactly 2 heads):")
for outcome in favorable_outcomes:
    print(''.join(outcome))

probability = len(favorable_outcomes) / len(coin_tosses)
print(f"\nProbability = Number of favorable outcomes / Total number of outcomes")
print(f"Probability = {len(favorable_outcomes)} / {len(coin_tosses)} = {probability}")

# Create visualization
plt.figure(figsize=(10, 6))
outcomes = [''.join(outcome) for outcome in coin_tosses]
counts = [outcome.count('H') for outcome in coin_tosses]
colors = ['red' if count == 2 else 'blue' for count in counts]

plt.bar(outcomes, counts, color=colors)
plt.axhline(y=2, color='green', linestyle='--', label='Exactly 2 Heads')
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Number of Heads', fontsize=12)
plt.title('Coin Toss Outcomes (3 tosses)', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_toss_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Class Imbalance in ML Dataset
print("\n\nExample 2: Class Imbalance in ML Dataset")
print("In a machine learning dataset with 1000 samples, there are two classes:")
print("- Class A: 800 samples")
print("- Class B: 200 samples")

# Define the counts
total_samples = 1000
class_a = 800
class_b = 200

# Calculate probabilities
p_class_a = class_a / total_samples
p_class_b = class_b / total_samples
p_same_class = p_class_a**2 + p_class_b**2

print("\nStep-by-step calculation:")
print(f"1. P(Class A) = {class_a}/{total_samples} = {p_class_a:.2f}")
print(f"2. P(Class B) = {class_b}/{total_samples} = {p_class_b:.2f}")
print(f"3. P(same class) = P(both Class A) + P(both Class B)")
print(f"   = ({class_a}/{total_samples})² + ({class_b}/{total_samples})²")
print(f"   = {p_class_a:.2f}² + {p_class_b:.2f}²")
print(f"   = {p_class_a**2:.2f} + {p_class_b**2:.2f} = {p_same_class:.2f}")

# Create visualization
plt.figure(figsize=(12, 6))

# Create a 2x2 grid for the visualization
plt.subplot(1, 2, 1)
# Bar plot for class distribution
classes = ['Class A', 'Class B']
counts = [class_a, class_b]
colors = ['blue', 'red']
plt.bar(classes, counts, color=colors)
plt.title('Class Distribution')
plt.ylabel('Number of Samples')

# Pie chart for probabilities
plt.subplot(1, 2, 2)
labels = ['Class A', 'Class B']
sizes = [p_class_a, p_class_b]
colors = ['lightblue', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Class Probabilities')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'class_imbalance_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Card Drawing Probability
print("\n\nExample 3: Card Drawing Probability")
print("What is the probability of drawing a heart or a king from a standard deck of 52 cards?")

# Define the counts
total_cards = 52
hearts = 13
kings = 4
heart_kings = 1  # King of Hearts

print("\nUsing the inclusion-exclusion principle:")
print(f"Number of hearts: {hearts}")
print(f"Number of kings: {kings}")
print(f"Number of cards that are both hearts and kings: {heart_kings}")

probability = (hearts + kings - heart_kings) / total_cards
print(f"\nP(Heart or King) = P(Heart) + P(King) - P(Heart and King)")
print(f"P(Heart or King) = {hearts}/{total_cards} + {kings}/{total_cards} - {heart_kings}/{total_cards}")
print(f"P(Heart or King) = {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
labels = ['Hearts', 'Kings', 'Heart Kings']
sizes = [hearts, kings, heart_kings]
colors = ['red', 'blue', 'purple']
explode = (0.1, 0.1, 0.2)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.title('Card Drawing Probabilities', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'card_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Birthday Problem
print("\n\nExample 4: Birthday Problem")
print("What is the probability that in a group of 23 people, at least two people share the same birthday?")

def birthday_probability(n):
    return 1 - np.prod([(365 - i) / 365 for i in range(n)])

n_people = 23
probability = birthday_probability(n_people)
print(f"\nFor {n_people} people:")
print(f"Probability that all birthdays are different: {1 - probability:.4f}")
print(f"Probability that at least two share a birthday: {probability:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
n_values = range(1, 60)
probabilities = [birthday_probability(n) for n in n_values]
plt.plot(n_values, probabilities, 'b-', linewidth=2)
plt.axhline(y=0.5, color='red', linestyle='--', label='50% Probability')
plt.axvline(x=23, color='green', linestyle='--', label='23 People')
plt.xlabel('Number of People', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Birthday Problem Probability', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'birthday_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Monty Hall Problem
print("\n\nExample 5: Monty Hall Problem")
print("In the Monty Hall problem, you pick one of three doors. The host, who knows what's behind each door,")
print("opens one of the remaining doors, revealing a goat. Should you switch your choice to the other unopened door?")

# Simulate the Monty Hall problem
def monty_hall_simulation(n_simulations=10000):
    stay_wins = 0
    switch_wins = 0
    
    for _ in range(n_simulations):
        # Randomly place the car behind one of three doors
        doors = ['goat', 'goat', 'car']
        np.random.shuffle(doors)
        
        # Player's initial choice
        initial_choice = np.random.randint(0, 3)
        
        # Host opens a door with a goat
        for i in range(3):
            if i != initial_choice and doors[i] == 'goat':
                host_opens = i
                break
        
        # Determine the switch door
        for i in range(3):
            if i != initial_choice and i != host_opens:
                switch_door = i
                break
        
        # Count wins
        if doors[initial_choice] == 'car':
            stay_wins += 1
        if doors[switch_door] == 'car':
            switch_wins += 1
    
    return stay_wins/n_simulations, switch_wins/n_simulations

stay_prob, switch_prob = monty_hall_simulation()
print("\nSimulation results (10,000 trials):")
print(f"Probability of winning by staying: {stay_prob:.4f}")
print(f"Probability of winning by switching: {switch_prob:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
strategies = ['Stay', 'Switch']
probabilities = [stay_prob, switch_prob]
colors = ['red', 'green']

plt.bar(strategies, probabilities, color=colors)
plt.axhline(y=1/3, color='red', linestyle='--', label='Initial Probability (1/3)')
plt.axhline(y=2/3, color='green', linestyle='--', label='Theoretical Switch Probability (2/3)')
plt.xlabel('Strategy', fontsize=12)
plt.ylabel('Probability of Winning', fontsize=12)
plt.title('Monty Hall Problem Simulation Results', fontsize=14)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'monty_hall_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll basic probability example images created successfully.") 