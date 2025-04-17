import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import Circle

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('ggplot')

print("# Statement 1: If two events A and B are independent, then P(A ∩ B) = P(A) × P(B).")

# Define probabilities
p_a = 0.5  # Probability of event A
p_b = 0.4  # Probability of event B
p_intersection = p_a * p_b  # Probability of intersection for independent events

# ---------- IMAGE 1: VENN DIAGRAM ----------
plt.figure(figsize=(6, 5))

# Draw a rectangle representing the sample space
ax = plt.gca()
sample_space = patches.Rectangle((0, 0), 1, 1, fill=True, color='#f0f0f0', alpha=0.5)
ax.add_patch(sample_space)

# Create and add circles for events A and B with proper overlap
circle_a = Circle((0.4, 0.5), 0.25, alpha=0.7, fc='#3498db', ec='#2980b9', linewidth=1.5, label='Event A')
circle_b = Circle((0.6, 0.5), 0.25, alpha=0.7, fc='#e74c3c', ec='#c0392b', linewidth=1.5, label='Event B')

# Add the circles to the plot
ax.add_patch(circle_a)
ax.add_patch(circle_b)

# Add minimal labels
ax.text(0.3, 0.5, "A", ha='center', va='center', fontsize=18, fontweight='bold', color='white')
ax.text(0.7, 0.5, "B", ha='center', va='center', fontsize=18, fontweight='bold', color='white')
ax.text(0.5, 0.5, "A∩B", ha='center', va='center', fontsize=16, fontweight='bold', color='white')

# Set the axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

# Adjust layout
plt.tight_layout()

# Save the figure
venn_img_path = os.path.join(save_dir, "statement1_venn.png")
plt.savefig(venn_img_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------- IMAGE 2: PROBABILITY TREE ----------
plt.figure(figsize=(7, 5))
ax = plt.gca()

# Set background color
ax.set_facecolor('#f9f9f9')

# Starting point for the tree
start_x, start_y = 0.1, 0.6

# Draw the main branches
# A branch
ax.plot([start_x, start_x+0.3], [start_y, start_y+0.2], 'k-', lw=2)
ax.plot([start_x, start_x+0.3], [start_y, start_y-0.2], 'k-', lw=2)

# B branches from A
ax.plot([start_x+0.3, start_x+0.6], [start_y+0.2, start_y+0.25], 'k-', lw=2)
ax.plot([start_x+0.3, start_x+0.6], [start_y+0.2, start_y+0.1], 'k-', lw=2)
ax.plot([start_x+0.3, start_x+0.6], [start_y-0.2, start_y-0.1], 'k-', lw=2)
ax.plot([start_x+0.3, start_x+0.6], [start_y-0.2, start_y-0.25], 'k-', lw=2)

# Add nodes
node_size = 400
plt.scatter(start_x, start_y, s=node_size, c='white', edgecolors='black', zorder=10)
plt.scatter(start_x+0.3, start_y+0.2, s=node_size, c='#3498db', edgecolors='black', zorder=10)
plt.scatter(start_x+0.3, start_y-0.2, s=node_size, c='white', edgecolors='black', zorder=10)
plt.scatter(start_x+0.6, start_y+0.25, s=node_size, c='#e74c3c', edgecolors='black', zorder=10)
plt.scatter(start_x+0.6, start_y+0.1, s=node_size, c='white', edgecolors='black', zorder=10)
plt.scatter(start_x+0.6, start_y-0.1, s=node_size, c='#e74c3c', edgecolors='black', zorder=10)
plt.scatter(start_x+0.6, start_y-0.25, s=node_size, c='white', edgecolors='black', zorder=10)

# Add event labels
ax.text(start_x+0.3, start_y+0.2, "A", ha='center', va='center', fontsize=16, color='white', fontweight='bold')
ax.text(start_x+0.3, start_y-0.2, "Aᶜ", ha='center', va='center', fontsize=16, color='black', fontweight='bold')
ax.text(start_x+0.6, start_y+0.25, "B", ha='center', va='center', fontsize=16, color='white', fontweight='bold')
ax.text(start_x+0.6, start_y+0.1, "Bᶜ", ha='center', va='center', fontsize=16, color='black', fontweight='bold')
ax.text(start_x+0.6, start_y-0.1, "B", ha='center', va='center', fontsize=16, color='white', fontweight='bold')
ax.text(start_x+0.6, start_y-0.25, "Bᶜ", ha='center', va='center', fontsize=16, color='black', fontweight='bold')

# Add probability labels on branches
ax.text(start_x+0.15, start_y+0.15, f"P(A) = {p_a}", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.15, start_y-0.15, f"P(Aᶜ) = {1-p_a}", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.45, start_y+0.25, f"P(B) = {p_b}", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.45, start_y+0.12, f"P(Bᶜ) = {1-p_b}", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.45, start_y-0.12, f"P(B) = {p_b}", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.45, start_y-0.25, f"P(Bᶜ) = {1-p_b}", ha='center', va='center', fontsize=14, fontweight='bold')

# Add outcome labels at the end
ax.text(start_x+0.75, start_y+0.25, f"A∩B: {p_a*p_b}", ha='left', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.75, start_y+0.1, f"A∩Bᶜ: {p_a*(1-p_b)}", ha='left', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.75, start_y-0.1, f"Aᶜ∩B: {(1-p_a)*p_b}", ha='left', va='center', fontsize=14, fontweight='bold')
ax.text(start_x+0.75, start_y-0.25, f"Aᶜ∩Bᶜ: {(1-p_a)*(1-p_b)}", ha='left', va='center', fontsize=14, fontweight='bold')

# Set axis
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Adjust layout
plt.tight_layout()

# Save the figure
tree_img_path = os.path.join(save_dir, "statement1_tree.png")
plt.savefig(tree_img_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------- IMAGE 3: SIMPLE CONCRETE EXAMPLE WITH CARDS ----------
plt.figure(figsize=(8, 5))
ax = plt.gca()

# Create a visual representation of a deck of cards example
# Divide the plot into regions: Full deck, Hearts, Face cards, Heart face cards
deck_height = 4
deck_width = 13

# Create grid for full deck representation
full_deck = np.ones((deck_height, deck_width))

# Create masks for different categories
hearts = np.zeros((deck_height, deck_width), dtype=bool)
hearts[0, :] = True  # First row represents hearts

face_cards = np.zeros((deck_height, deck_width), dtype=bool)
face_cards[:, 10:13] = True  # Last 3 columns represent face cards (J, Q, K)

# Calculate intersection
heart_face_cards = hearts & face_cards

# Calculate probabilities
p_heart = np.sum(hearts) / (deck_height * deck_width)
p_face = np.sum(face_cards) / (deck_height * deck_width)
p_heart_face = np.sum(heart_face_cards) / (deck_height * deck_width)

# Fill the grid with colors
grid = np.zeros((deck_height, deck_width, 3))
# Background (light gray)
grid.fill(0.95)

# Hearts (red)
grid[hearts & ~face_cards] = [0.9, 0.2, 0.2]  # Hearts but not face cards

# Face cards (blue)
grid[face_cards & ~hearts] = [0.2, 0.4, 0.8]  # Face cards but not hearts

# Heart face cards (purple)
grid[heart_face_cards] = [0.8, 0.2, 0.8]  # Both hearts and face cards

# Plot the grid
ax.imshow(grid, aspect='equal')

# Add grid lines
for i in range(deck_width + 1):
    ax.axvline(i - 0.5, color='black', linewidth=1)
for i in range(deck_height + 1):
    ax.axhline(i - 0.5, color='black', linewidth=1)

# Add card labels
symbols = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
suits = ['♥', '♠', '♦', '♣']

for i in range(deck_height):
    for j in range(deck_width):
        color = 'white' if (heart_face_cards[i, j] or hearts[i, j] or face_cards[i, j]) else 'black'
        ax.text(j, i, f"{symbols[j]}{suits[i]}", ha='center', va='center', fontsize=10, color=color)

# Set ticks
ax.set_xticks(np.arange(deck_width))
ax.set_yticks(np.arange(deck_height))
ax.set_xticklabels(symbols)
ax.set_yticklabels(suits)

# Set title
ax.set_title('Card Example: Independence of Suits and Ranks', fontsize=16, pad=10)

# Adjust layout
plt.tight_layout()

# Save the figure
cards_img_path = os.path.join(save_dir, "statement1_cards.png")
plt.savefig(cards_img_path, dpi=300, bbox_inches='tight')
plt.close()

# Print all the explanations
print("\n#### Key Elements of Independence Visualization:")
print("1. Venn Diagram (Image 1):")
print("   - Circle A represents event A with P(A) = 0.5")
print("   - Circle B represents event B with P(B) = 0.4")
print("   - The intersection A∩B has P(A∩B) = P(A) × P(B) = 0.5 × 0.4 = 0.2")
print("")
print("2. Probability Tree (Image 2):")
print("   - Shows how events branch, with probabilities at each step")
print("   - For independent events, the probability of B is the same (0.4) whether A occurred or not")
print("   - The probability of path A→B equals P(A) × P(B) = 0.5 × 0.4 = 0.2")
print("")
print("3. Card Example (Image 3):")
print("   - In a standard deck of 52 cards:")
print("   - P(Heart) = 13/52 = 0.25 (13 hearts in 52 cards)")
print("   - P(Face Card) = 12/52 = 0.23 (12 face cards in 52 cards)")
print("   - P(Heart and Face Card) = 3/52 = 0.06 (3 heart face cards)")
print("   - P(Heart) × P(Face Card) = 0.25 × 0.23 = 0.06")
print("   - This confirms that P(A∩B) = P(A) × P(B) for independent events")
print("")
print("#### Simple Explanation of Independence:")
print("Two events A and B are independent if knowing one occurred doesn't change the probability of the other occurring.")
print("For example, drawing a heart card doesn't affect whether you draw a face card.")
print("Mathematically, this means that P(A∩B) = P(A) × P(B).")
print("")
print("#### Visual Verification:")
print(f"Venn diagram: {venn_img_path}")
print(f"Probability tree: {tree_img_path}")
print(f"Card example: {cards_img_path}")
print("")
print("#### Conclusion:")
print("We have shown through both mathematical definitions and concrete examples that for independent events A and B,")
print("the probability of their intersection is exactly the product of their individual probabilities.")
print("")
print("Therefore, Statement 1 is TRUE.") 