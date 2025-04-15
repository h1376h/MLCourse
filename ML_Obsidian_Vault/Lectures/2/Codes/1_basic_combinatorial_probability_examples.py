import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
import os
from fractions import Fraction

# Function to calculate factorial
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# Function to calculate combinations: C(n,r)
def combination(n, r):
    return factorial(n) // (factorial(r) * factorial(n-r))

# Function to calculate permutations: P(n,r)
def permutation(n, r):
    return factorial(n) // factorial(n-r)

print("\n=== BASIC COMBINATORIAL PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Book Selection
print("Example 1: Book Selection")
print("Problem: A student has 8 math books and 6 computer science books on a shelf.")
print("If the student randomly selects 3 books, what is the probability that:")
print("a) All 3 books are math books?")
print("b) Exactly 2 books are computer science books?")

# Setup
math_books = 8
cs_books = 6
total_books = math_books + cs_books
books_to_select = 3

# Step 1: Calculate the total number of possible selections
total_ways = combination(total_books, books_to_select)
print(f"\nStep 1: Calculate the total number of possible selections")
print(f"Total ways to select {books_to_select} books from {total_books} books:")
print(f"C({total_books},{books_to_select}) = {total_books}! / ({books_to_select}! × ({total_books}-{books_to_select})!)")
print(f"C({total_books},{books_to_select}) = {total_books}! / ({books_to_select}! × {total_books-books_to_select}!)")
print(f"C({total_books},{books_to_select}) = {total_books**(total_books-books_to_select+1)} / ({books_to_select}! × {total_books-books_to_select}!)")
print(f"C({total_books},{books_to_select}) = {total_books} × {total_books-1} × {total_books-2} / ({books_to_select} × {books_to_select-1} × {books_to_select-2})")
print(f"C({total_books},{books_to_select}) = {total_books} × {total_books-1} × {total_books-2} / {books_to_select * (books_to_select-1) * (books_to_select-2)}")
print(f"C({total_books},{books_to_select}) = {total_books * (total_books-1) * (total_books-2)} / {books_to_select * (books_to_select-1) * (books_to_select-2)}")
print(f"C({total_books},{books_to_select}) = {total_ways}")

# Step 2: Calculate the number of favorable outcomes for part (a)
ways_all_math = combination(math_books, books_to_select)
print(f"\nStep 2: Calculate the number of favorable outcomes for part (a)")
print(f"Ways to select {books_to_select} math books from {math_books} math books:")
print(f"C({math_books},{books_to_select}) = {math_books}! / ({books_to_select}! × ({math_books}-{books_to_select})!)")
print(f"C({math_books},{books_to_select}) = {math_books}! / ({books_to_select}! × {math_books-books_to_select}!)")
print(f"C({math_books},{books_to_select}) = {math_books} × {math_books-1} × {math_books-2} / ({books_to_select} × {books_to_select-1} × {books_to_select-2})")
print(f"C({math_books},{books_to_select}) = {math_books} × {math_books-1} × {math_books-2} / {books_to_select * (books_to_select-1) * (books_to_select-2)}")
print(f"C({math_books},{books_to_select}) = {math_books * (math_books-1) * (math_books-2)} / {books_to_select * (books_to_select-1) * (books_to_select-2)}")
print(f"C({math_books},{books_to_select}) = {ways_all_math}")

# Step 3: Calculate the probability for part (a)
prob_all_math = ways_all_math / total_ways
prob_all_math_fraction = Fraction(ways_all_math, total_ways).limit_denominator()
print(f"\nStep 3: Calculate the probability for part (a)")
print(f"P(all 3 books are math books) = C({math_books},{books_to_select}) / C({total_books},{books_to_select})")
print(f"P(all 3 books are math books) = {ways_all_math} / {total_ways}")
print(f"P(all 3 books are math books) = {prob_all_math_fraction} ≈ {prob_all_math:.4f} or about {prob_all_math*100:.2f}%")

# Step 4: Calculate the number of favorable outcomes for part (b)
ways_2_cs_1_math = combination(cs_books, 2) * combination(math_books, 1)
print(f"\nStep 4: Calculate the number of favorable outcomes for part (b)")
print(f"Ways to select 2 CS books from {cs_books} CS books: C({cs_books},2) = {combination(cs_books, 2)}")
print(f"Ways to select 1 math book from {math_books} math books: C({math_books},1) = {combination(math_books, 1)}")
print(f"By the multiplication principle, total number of favorable outcomes:")
print(f"C({cs_books},2) × C({math_books},1) = {combination(cs_books, 2)} × {combination(math_books, 1)} = {ways_2_cs_1_math}")

# Step 5: Calculate the probability for part (b)
prob_2_cs = ways_2_cs_1_math / total_ways
prob_2_cs_fraction = Fraction(ways_2_cs_1_math, total_ways).limit_denominator()
print(f"\nStep 5: Calculate the probability for part (b)")
print(f"P(exactly 2 CS books) = (C({cs_books},2) × C({math_books},1)) / C({total_books},{books_to_select})")
print(f"P(exactly 2 CS books) = {ways_2_cs_1_math} / {total_ways}")
print(f"P(exactly 2 CS books) = {prob_2_cs_fraction} ≈ {prob_2_cs:.4f} or about {prob_2_cs*100:.2f}%")

# Create visualization for Example 1
plt.figure(figsize=(12, 8))

# Create a bar chart for the different outcomes
outcomes = ['All 3 Math', '2 CS, 1 Math', '2 Math, 1 CS', 'All 3 CS']
probabilities = [
    prob_all_math,
    prob_2_cs,
    combination(math_books, 2) * combination(cs_books, 1) / total_ways,
    combination(cs_books, 3) / total_ways
]

# Add colors to distinguish the outcomes we calculated
colors = ['lightgreen', 'lightblue', 'lightgray', 'lightgray']

plt.bar(outcomes, probabilities, color=colors, edgecolor='black', alpha=0.7)
plt.title('Book Selection Probabilities', fontsize=15)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, max(probabilities) * 1.2)  # Add some space for annotations

# Add probability values on top of each bar
for i, prob in enumerate(probabilities):
    plt.text(i, prob + 0.01, f'{prob:.4f}', ha='center', fontsize=10)

# Add annotations for the first two calculated probabilities
plt.annotate(f'P(all 3 math) = {ways_all_math}/{total_ways} = {prob_all_math_fraction}',
            xy=(0, prob_all_math + 0.04),
            xytext=(0, prob_all_math + 0.04),
            ha='center',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate(f'P(2 CS, 1 math) = {ways_2_cs_1_math}/{total_ways} = {prob_2_cs_fraction}',
            xy=(1, prob_2_cs + 0.04),
            xytext=(1, prob_2_cs + 0.04),
            ha='center',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'book_selection_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Selecting Balls from an Urn
print("\n\nExample 2: Selecting Balls from an Urn")
print("Problem: An urn contains 10 balls: 4 red, 3 blue, and 3 green.")
print("If 2 balls are randomly drawn without replacement, what is the probability of getting:")
print("a) 2 balls of the same color?")
print("b) 1 red ball and 1 blue ball?")

# Setup
red_balls = 4
blue_balls = 3
green_balls = 3
total_balls = red_balls + blue_balls + green_balls
balls_to_draw = 2

# Step 1: Calculate the total number of possible outcomes
total_outcomes = combination(total_balls, balls_to_draw)
print(f"\nStep 1: Calculate the total number of possible outcomes")
print(f"Total ways to select {balls_to_draw} balls from {total_balls} balls:")
print(f"C({total_balls},{balls_to_draw}) = {total_balls}! / ({balls_to_draw}! × ({total_balls}-{balls_to_draw})!)")
print(f"C({total_balls},{balls_to_draw}) = {total_balls} × {total_balls-1} / 2 × 1")
print(f"C({total_balls},{balls_to_draw}) = {total_balls * (total_balls-1) // 2}")
print(f"C({total_balls},{balls_to_draw}) = {total_outcomes}")

# Step 2: Calculate the number of favorable outcomes for part (a)
red_same_color = combination(red_balls, balls_to_draw)
blue_same_color = combination(blue_balls, balls_to_draw)
green_same_color = combination(green_balls, balls_to_draw)
total_same_color = red_same_color + blue_same_color + green_same_color

print(f"\nStep 2: Calculate the number of favorable outcomes for part (a)")
print(f"For 2 balls of the same color, we need 2 red OR 2 blue OR 2 green balls.")
print(f"Ways to select 2 red balls from {red_balls} red balls: C({red_balls},2) = {red_balls} × {red_balls-1} / 2 = {red_same_color}")
print(f"Ways to select 2 blue balls from {blue_balls} blue balls: C({blue_balls},2) = {blue_balls} × {blue_balls-1} / 2 = {blue_same_color}")
print(f"Ways to select 2 green balls from {green_balls} green balls: C({green_balls},2) = {green_balls} × {green_balls-1} / 2 = {green_same_color}")
print(f"Total number of favorable outcomes: {red_same_color} + {blue_same_color} + {green_same_color} = {total_same_color}")

# Step 3: Calculate the probability for part (a)
prob_same_color = total_same_color / total_outcomes
prob_same_color_fraction = Fraction(total_same_color, total_outcomes).limit_denominator()
print(f"\nStep 3: Calculate the probability for part (a)")
print(f"P(2 balls of same color) = {total_same_color} / {total_outcomes}")
print(f"P(2 balls of same color) = {prob_same_color_fraction} ≈ {prob_same_color:.4f} or about {prob_same_color*100:.2f}%")

# Step 4: Calculate the number of favorable outcomes for part (b)
ways_1_red_1_blue = combination(red_balls, 1) * combination(blue_balls, 1)
print(f"\nStep 4: Calculate the number of favorable outcomes for part (b)")
print(f"Ways to select 1 red ball from {red_balls} red balls: C({red_balls},1) = {red_balls}")
print(f"Ways to select 1 blue ball from {blue_balls} blue balls: C({blue_balls},1) = {blue_balls}")
print(f"By the multiplication principle, total number of favorable outcomes:")
print(f"C({red_balls},1) × C({blue_balls},1) = {red_balls} × {blue_balls} = {ways_1_red_1_blue}")

# Step 5: Calculate the probability for part (b)
prob_1_red_1_blue = ways_1_red_1_blue / total_outcomes
prob_1_red_1_blue_fraction = Fraction(ways_1_red_1_blue, total_outcomes).limit_denominator()
print(f"\nStep 5: Calculate the probability for part (b)")
print(f"P(1 red and 1 blue) = (C({red_balls},1) × C({blue_balls},1)) / C({total_balls},{balls_to_draw})")
print(f"P(1 red and 1 blue) = {ways_1_red_1_blue} / {total_outcomes}")
print(f"P(1 red and 1 blue) = {prob_1_red_1_blue_fraction} ≈ {prob_1_red_1_blue:.4f} or about {prob_1_red_1_blue*100:.2f}%")

# Create visualization for Example 2
plt.figure(figsize=(14, 6))

# Create a schematic of the urn with colored balls - FIXED VISUALIZATION
ax1 = plt.subplot(1, 2, 1)
plt.title('Urn with 10 Balls (4 Red, 3 Blue, 3 Green)', fontsize=14)

# Create a more structured layout for the balls
# Define better positions for balls - arranged in a circular pattern
radius = 0.3
center = [0.5, 0.5]
angles = np.linspace(0, 2*np.pi, total_balls, endpoint=False)
ball_size = 0.08  # Uniform size for all balls

# Use consistent colors with better visibility
face_colors = ['#FF5555'] * red_balls + ['#5555FF'] * blue_balls + ['#55AA55'] * green_balls

# Draw balls in a circle pattern
for i, angle in enumerate(angles):
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    circle = plt.Circle((x, y), ball_size, facecolor=face_colors[i], edgecolor='black', linewidth=1, alpha=0.8)
    ax1.add_patch(circle)

# Create a legend for clarity
red_patch = plt.Circle((0, 0), 0.1, facecolor=face_colors[0], edgecolor='black', alpha=0.8)
blue_patch = plt.Circle((0, 0), 0.1, facecolor=face_colors[red_balls], edgecolor='black', alpha=0.8)
green_patch = plt.Circle((0, 0), 0.1, facecolor=face_colors[red_balls + blue_balls], edgecolor='black', alpha=0.8)
ax1.legend([red_patch, blue_patch, green_patch], ['Red', 'Blue', 'Green'], loc='upper left')

# Set equal aspect ratio so circles look like circles
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('equal')
plt.axis('off')

# Create a bar chart of the outcomes and probabilities
ax2 = plt.subplot(1, 2, 2)
outcomes = ['2 Same Color', '1 Red, 1 Blue', '1 Red, 1 Green', '1 Blue, 1 Green']
probs = [
    prob_same_color,
    prob_1_red_1_blue,
    combination(red_balls, 1) * combination(green_balls, 1) / total_outcomes,
    combination(blue_balls, 1) * combination(green_balls, 1) / total_outcomes
]
bar_colors = ['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFFFAA']

plt.bar(outcomes, probs, color=bar_colors, edgecolor='black', alpha=0.7)
plt.title('Ball Selection Probabilities', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 0.33)  # Fixed height for better appearance
plt.xticks(rotation=10, ha='right')

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i, prob + 0.01, f'{prob:.4f}', ha='center', fontsize=10)

# Add single annotation box showing both calculations
annotation_text = f'P(same color) = 12/45 = 4/15\nP(1 red, 1 blue) = 12/45 = 4/15'
plt.annotate(annotation_text,
            xy=(1.5, 0.30),
            xytext=(1.5, 0.30),
            ha='center',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'ball_selection_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Password Creation
print("\n\nExample 3: Password Creation")
print("Problem: A system requires creating a 4-digit PIN code using digits 0-9, where digits can be repeated.")
print("a) What is the probability of creating a PIN with all different digits?")
print("b) What is the probability of creating a PIN with at least one repeated digit?")

# Setup
digits = 10  # 0-9
pin_length = 4

# Step 1: Calculate the total number of possible PINs
total_pins = digits ** pin_length  # With replacement
print(f"\nStep 1: Calculate the total number of possible PINs")
print(f"Using the multiplication principle, with repetition allowed:")
print(f"Total number of {pin_length}-digit PINs = {digits}^{pin_length} = {digits} × {digits} × {digits} × {digits} = {total_pins}")

# Step 2: Calculate the number of favorable outcomes for part (a)
# This is a permutation: P(10,4) = 10 × 9 × 8 × 7
pins_no_repetition = permutation(digits, pin_length)
print(f"\nStep 2: Calculate the number of favorable outcomes for part (a)")
print(f"To create a PIN with all different digits:")
print(f"First position: {digits} choices (0-9)")
print(f"Second position: {digits-1} choices (any digit except the first)")
print(f"Third position: {digits-2} choices (any digit except the first two)")
print(f"Fourth position: {digits-3} choices (any digit except the first three)")
print(f"Total number of PINs with no repeated digits: {digits} × {digits-1} × {digits-2} × {digits-3} = {pins_no_repetition}")

# Step 3: Calculate the probability for part (a)
prob_all_different = pins_no_repetition / total_pins
prob_all_different_fraction = Fraction(pins_no_repetition, total_pins).limit_denominator()
print(f"\nStep 3: Calculate the probability for part (a)")
print(f"P(all different digits) = {pins_no_repetition} / {total_pins}")
print(f"P(all different digits) = {prob_all_different_fraction} ≈ {prob_all_different:.4f} or {prob_all_different*100:.2f}%")

# Step 4: Calculate the probability for part (b)
prob_at_least_one_repeat = 1 - prob_all_different
print(f"\nStep 4: Calculate the probability for part (b)")
print(f"Using the complement rule:")
print(f"P(at least one repeated digit) = 1 - P(all different digits)")
print(f"P(at least one repeated digit) = 1 - {prob_all_different:.4f} = {prob_at_least_one_repeat:.4f} or {prob_at_least_one_repeat*100:.2f}%")

# Create visualization for Example 3
plt.figure(figsize=(10, 8))

# Create a visual representation of different PIN types
# First, create a grid of digits to represent PIN possibilities
ax1 = plt.subplot(2, 1, 1)
plt.title('Examples of 4-digit PINs', fontsize=14)

# Examples of PINs with all different digits
different_pins = [
    "1234", "9876", "5073", "2580"
]
pin_positions_different = np.array([
    [0.2, 0.7], [0.4, 0.7], [0.6, 0.7], [0.8, 0.7]
])

# Examples of PINs with at least one repeated digit
repeated_pins = [
    "1123", "7755", "8888", "2020"
]
pin_positions_repeated = np.array([
    [0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3]
])

# Draw the PINs
for i, pin in enumerate(different_pins):
    plt.text(pin_positions_different[i, 0], pin_positions_different[i, 1], pin,
             fontsize=16, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    plt.text(pin_positions_different[i, 0], pin_positions_different[i, 1] + 0.1, "All Different",
             fontsize=10, ha='center', va='center')

for i, pin in enumerate(repeated_pins):
    plt.text(pin_positions_repeated[i, 0], pin_positions_repeated[i, 1], pin,
             fontsize=16, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
    plt.text(pin_positions_repeated[i, 0], pin_positions_repeated[i, 1] + 0.1, "With Repetition",
             fontsize=10, ha='center', va='center')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Create a pie chart showing the probability distribution
ax2 = plt.subplot(2, 1, 2)
labels = ['All Different Digits', 'At Least One Repeated Digit']
sizes = [prob_all_different, prob_at_least_one_repeat]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # Explode the 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, wedgeprops={'alpha': 0.8})
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'pin_code_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Random Seating Arrangement
print("\n\nExample 4: Random Seating Arrangement")
print("Problem: 5 students (Alice, Bob, Charlie, David, and Emma) need to be randomly seated in a row of 5 chairs.")
print("a) What is the probability that Alice and Bob sit next to each other?")
print("b) What is the probability that Alice sits at one end and Bob at the other end?")

# Setup
students = ["Alice", "Bob", "Charlie", "David", "Emma"]
num_students = len(students)

# Step 1: Calculate the total number of possible seating arrangements
total_arrangements = factorial(num_students)  # 5!
print(f"\nStep 1: Calculate the total number of possible seating arrangements")
print(f"Total number of ways to arrange {num_students} students in {num_students} chairs:")
print(f"P({num_students},{num_students}) = {num_students}!")
print(f"P({num_students},{num_students}) = {num_students} × {num_students-1} × {num_students-2} × {num_students-3} × {num_students-4}")
print(f"P({num_students},{num_students}) = {total_arrangements}")

# Step 2: Calculate the number of favorable outcomes for part (a)
# For Alice and Bob to sit together, we first treat them as one unit
# So we have 4 units (AliceBob, C, D, E) to arrange, which is 4!
# Then Alice and Bob can be arranged in 2! ways within their unit
adjacent_arrangements = factorial(num_students - 1) * factorial(2)
print(f"\nStep 2: Calculate the number of favorable outcomes for part (a)")
print(f"To have Alice and Bob sit together:")
print(f"First, consider Alice and Bob as one unit, giving us 4 units to arrange")
print(f"Number of ways to arrange 4 units: 4! = {factorial(num_students-1)}")
print(f"Alice and Bob can be arranged in 2! = {factorial(2)} ways within their unit")
print(f"Total favorable outcomes: {factorial(num_students-1)} × {factorial(2)} = {adjacent_arrangements}")

# Step 3: Calculate the probability for part (a)
prob_adjacent = adjacent_arrangements / total_arrangements
prob_adjacent_fraction = Fraction(adjacent_arrangements, total_arrangements).limit_denominator()
print(f"\nStep 3: Calculate the probability for part (a)")
print(f"P(Alice and Bob sit together) = {adjacent_arrangements} / {total_arrangements}")
print(f"P(Alice and Bob sit together) = {prob_adjacent_fraction} = {prob_adjacent:.4f} or {prob_adjacent*100:.2f}%")

# Step 4: Calculate the number of favorable outcomes for part (b)
# Alice can be at first or last position (2 choices)
# Bob must be at the opposite end (1 choice)
# Remaining 3 students can be arranged in 3! ways
opposite_ends_arrangements = 2 * 1 * factorial(num_students - 2)
print(f"\nStep 4: Calculate the number of favorable outcomes for part (b)")
print(f"For Alice and Bob to sit at opposite ends:")
print(f"Alice can sit at first chair or last chair: 2 ways")
print(f"Once Alice's position is fixed, Bob must be at the opposite end: 1 way")
print(f"Remaining {num_students-2} students can be arranged in {num_students-2}! = {factorial(num_students-2)} ways")
print(f"Total favorable outcomes: 2 × 1 × {factorial(num_students-2)} = {opposite_ends_arrangements}")

# Step 5: Calculate the probability for part (b)
prob_opposite_ends = opposite_ends_arrangements / total_arrangements
prob_opposite_ends_fraction = Fraction(opposite_ends_arrangements, total_arrangements).limit_denominator()
print(f"\nStep 5: Calculate the probability for part (b)")
print(f"P(Alice and Bob at opposite ends) = {opposite_ends_arrangements} / {total_arrangements}")
print(f"P(Alice and Bob at opposite ends) = {prob_opposite_ends_fraction} = {prob_opposite_ends:.4f} or {prob_opposite_ends*100:.2f}%")

# Create visualization for Example 4
plt.figure(figsize=(12, 8))

# Create a visual representation of the seating arrangements
ax1 = plt.subplot(1, 2, 1)
plt.title('Alice and Bob Sitting Together', fontsize=14)

# Draw 5 chairs in a row
chair_positions = np.linspace(0.1, 0.9, 5)
for pos in chair_positions:
    rect = plt.Rectangle((pos-0.05, 0.4), 0.1, 0.1, facecolor='grey', alpha=0.3)
    ax1.add_patch(rect)

# Draw an example arrangement with Alice and Bob together
positions = np.random.permutation(5)
names = students.copy()
# Ensure Alice and Bob are adjacent
alice_pos = np.random.randint(0, 4)  # Position 0-3, so Bob can be at alice_pos+1
bob_pos = alice_pos + 1
positions[alice_pos], positions[0] = positions[0], positions[alice_pos]
positions[bob_pos], positions[1] = positions[1], positions[bob_pos]

for i, pos in enumerate(positions):
    plt.text(chair_positions[pos], 0.45, names[i],
             fontsize=10, ha='center', va='center')
    
    # Highlight Alice and Bob
    if names[i] == "Alice" or names[i] == "Bob":
        rect = plt.Rectangle((chair_positions[pos]-0.05, 0.4), 0.1, 0.1, facecolor='lightgreen', alpha=0.7)
        ax1.add_patch(rect)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Add an explanation of the scenario
plt.text(0.5, 0.8, "Alice and Bob can sit together in various positions\nalong the row, with the other students arranged\naround them.", 
         ha='center', fontsize=10)

# Create a visual representation for opposite ends
ax2 = plt.subplot(1, 2, 2)
plt.title('Alice and Bob at Opposite Ends', fontsize=14)

# Draw 5 chairs in a row
for pos in chair_positions:
    rect = plt.Rectangle((pos-0.05, 0.4), 0.1, 0.1, facecolor='grey', alpha=0.3)
    ax2.add_patch(rect)

# Draw an example arrangement with Alice and Bob at opposite ends
positions = np.random.permutation(5)
# Ensure Alice is at one end and Bob at the other
positions[0], positions[0] = positions[0], positions[0]  # Alice at first chair
positions[4], positions[1] = positions[1], positions[4]  # Bob at last chair

names_opposite = students.copy()
for i, pos in enumerate(positions):
    plt.text(chair_positions[pos], 0.45, names_opposite[i],
             fontsize=10, ha='center', va='center')
    
    # Highlight Alice and Bob
    if names_opposite[i] == "Alice" or names_opposite[i] == "Bob":
        rect = plt.Rectangle((chair_positions[pos]-0.05, 0.4), 0.1, 0.1, facecolor='lightcoral', alpha=0.7)
        ax2.add_patch(rect)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Add an explanation
plt.text(0.5, 0.8, "Alice and Bob must sit at opposite ends,\nwith the other students arranged in the middle.", 
         ha='center', fontsize=10)

# Add a bar graph comparing probabilities
ax3 = plt.subplot(2, 1, 2)
scenarios = ['Alice and Bob Adjacent', 'Alice and Bob at Opposite Ends', 'Other Arrangements']
probs = [
    prob_adjacent,
    prob_opposite_ends,
    1 - (prob_adjacent + prob_opposite_ends)
]
colors = ['lightgreen', 'lightcoral', 'lightgray']

plt.bar(scenarios, probs, color=colors, edgecolor='black', alpha=0.7)
plt.title('Seating Arrangement Probabilities', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, max(probs) * 1.2)  # Add some space for annotations

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i, prob + 0.01, f'{prob:.4f}', ha='center', fontsize=10)

# Add annotations with calculations
plt.annotate(f'P(adjacent) = {adjacent_arrangements}/{total_arrangements}\n= {prob_adjacent_fraction}',
            xy=(0, prob_adjacent + 0.05),
            xytext=(0, prob_adjacent + 0.05),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate(f'P(opposite ends) = {opposite_ends_arrangements}/{total_arrangements}\n= {prob_opposite_ends_fraction}',
            xy=(1, prob_opposite_ends + 0.05),
            xytext=(1, prob_opposite_ends + 0.05),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'seating_arrangement_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Random Assignment of Tasks
print("\n\nExample 5: Random Assignment of Tasks")
print("Problem: A data science team needs to assign 7 different tasks to 7 team members.")
print("If the tasks are assigned randomly, what is the probability that:")
print("a) A specific team member gets a specific task?")
print("b) Each of 3 specific team members gets one of 3 specific tasks (without regard to which team member gets which of the three tasks)?")

# Setup
num_tasks = 7
num_members = 7  # Same as tasks; one task per member

# Step 1: Calculate the total number of possible assignments
total_assignments = factorial(num_tasks)  # 7!
print(f"\nStep 1: Calculate the total number of possible assignments")
print(f"Total number of ways to assign {num_tasks} tasks to {num_members} team members:")
print(f"This is a permutation of {num_tasks} tasks: {num_tasks}!")
print(f"Total possible assignments: {num_tasks}! = {total_assignments}")

# Step 2: Calculate the probability for part (a)
# For a specific member to get a specific task, the remaining (n-1) tasks 
# must be assigned to the remaining (n-1) members, which can be done in (n-1)! ways
favorable_specific = factorial(num_tasks - 1)
prob_specific = favorable_specific / total_assignments
prob_specific_fraction = Fraction(favorable_specific, total_assignments).limit_denominator()

print(f"\nStep 2: Calculate the probability for part (a)")
print(f"For a specific team member to get a specific task:")
print(f"The remaining {num_tasks-1} tasks can be assigned to the remaining {num_members-1} members in {num_tasks-1}! ways")
print(f"Number of favorable outcomes: {num_tasks-1}! = {favorable_specific}")
print(f"P(specific member gets specific task) = {favorable_specific} / {total_assignments}")
print(f"P(specific member gets specific task) = {prob_specific_fraction} ≈ {prob_specific:.4f} or {prob_specific*100:.2f}%")

# Step 3: Calculate the probability for part (b)
# For 3 specific members to get 3 specific tasks (without regard to which gets which):
# Number of ways to assign 3 specific tasks to 3 specific members: 3! = 6 ways
# Number of ways to assign the remaining 4 tasks to the remaining 4 members: 4! = 24 ways
favorable_three_specific = factorial(3) * factorial(4)
prob_three_specific = favorable_three_specific / total_assignments
prob_three_specific_fraction = Fraction(favorable_three_specific, total_assignments).limit_denominator()

print(f"\nStep 3: Calculate the probability for part (b)")
print(f"For 3 specific team members to get 3 specific tasks (without regard to which member gets which task):")
print(f"Number of ways to assign 3 specific tasks to 3 specific members: 3! = {factorial(3)}")
print(f"Number of ways to assign the remaining 4 tasks to the remaining 4 members: 4! = {factorial(4)}")
print(f"Total favorable outcomes: 3! × 4! = {factorial(3)} × {factorial(4)} = {favorable_three_specific}")
print(f"P(3 specific members get 3 specific tasks) = {favorable_three_specific} / {total_assignments}")
print(f"P(3 specific members get 3 specific tasks) = {prob_three_specific_fraction} ≈ {prob_three_specific:.4f} or {prob_three_specific*100:.2f}%")

# Create visualization for Example 5
plt.figure(figsize=(12, 8))

# Create a visualization of task assignments
ax1 = plt.subplot(2, 1, 1)
plt.title('Data Science Task Assignment', fontsize=14)

# Create a visual representation of members and tasks
member_positions = np.linspace(0.2, 0.8, num_members)
task_positions = np.linspace(0.2, 0.8, num_tasks)

# Plot members on left and tasks on right
for i in range(num_members):
    plt.text(0.1, member_positions[i], f"Member {i+1}", fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.9, task_positions[i], f"Task {i+1}", fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Highlight a specific member and task (for part a)
plt.text(0.1, member_positions[0], "Member 1", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="coral", alpha=0.7))
plt.text(0.9, task_positions[0], "Task 1", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="coral", alpha=0.7))

# Draw a line connecting them
plt.plot([0.15, 0.85], [member_positions[0], task_positions[0]], 'r-', alpha=0.5)

# Add text explanation
plt.text(0.5, 0.9, "For part (a), what is the probability that\nMember 1 gets assigned Task 1?", 
         ha='center', fontsize=10)

# Highlight 3 specific members and tasks (for part b)
for i in range(3):
    if i > 0:  # Don't re-highlight Member 1 and Task 1
        plt.text(0.1, member_positions[i], f"Member {i+1}", fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.7))
        plt.text(0.9, task_positions[i], f"Task {i+1}", fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.7))

# Draw lines connecting them (showing multiple possibilities)
plt.text(0.5, 0.05, "For part (b), what is the probability that Members 1, 2, and 3\nget Tasks 1, 2, and the 3 (in any order)?", 
         ha='center', fontsize=10)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Create bar chart of the probabilities
ax2 = plt.subplot(2, 1, 2)
scenarios = [
    'Specific Member\ngets Specific Task', 
    '3 Specific Members\nget 3 Specific Tasks',
    'Other\nArrangements'
]
probs = [
    prob_specific,
    prob_three_specific,
    1 - (prob_specific + prob_three_specific)
]
colors = ['coral', 'lightsalmon', 'lightgray']

plt.bar(scenarios, probs, color=colors, edgecolor='black', alpha=0.7)
plt.title('Task Assignment Probabilities', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, max(max(probs) * 1.2, 0.2))  # Add some space for annotations

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i, prob + 0.01, f'{prob:.4f}', ha='center', fontsize=10)

# Add annotations with calculations
plt.annotate(f'P(specific) = {favorable_specific}/{total_assignments}\n= {prob_specific_fraction}',
            xy=(0, prob_specific + 0.02),
            xytext=(0, prob_specific + 0.02),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate(f'P(3 specific) = {favorable_three_specific}/{total_assignments}\n= {prob_three_specific_fraction}',
            xy=(1, prob_three_specific + 0.02),
            xytext=(1, prob_three_specific + 0.02),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'task_assignment_probability.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll combinatorial probability example images created successfully.")
print("Images saved to:", images_dir) 