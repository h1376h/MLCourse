import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Set a clean, simple style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
})

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# Create directory for saving images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    plt.close(fig)

# Problem data
categories = ['A', 'B', 'C']
counts = [50, 30, 20]
total_examples = sum(counts)
probabilities = [count / total_examples for count in counts]

# Encoding schemes
scheme1_onehot = {
    'A': [1, 0, 0],
    'B': [0, 1, 0],
    'C': [0, 0, 1]
}

scheme2_binary = {
    'A': [0, 0],
    'B': [0, 1],
    'C': [1, 0]
}

# ==============================
# STEP 1: Calculate the Entropy of the Class Distribution
# ==============================
print_section_header("STEP 1: Calculate the Entropy")

print("Formula: H(X) = -∑ P(x_i) log₂ P(x_i))")

print("\nClass distribution:")
for i, (category, count) in enumerate(zip(categories, counts)):
    percentage = (count / total_examples) * 100
    print(f"Category {category}: {count} instances ({percentage:.1f}%)")

# Calculate entropy (in bits) with detailed steps
print("\nDetailed entropy calculation:")

# First explain the probability calculation
print("Step 1: Calculate the probability of each category")
for i, (category, count) in enumerate(zip(categories, counts)):
    p = count / total_examples
    print(f"P({category}) = {count}/{total_examples} = {p:.4f}")

# Then calculate log values
print("\nStep 2: Calculate the log₂ of each probability")
log_values = []
for i, (category, p) in enumerate(zip(categories, probabilities)):
    log_val = math.log2(p)
    log_values.append(log_val)
    print(f"log₂(P({category})) = log₂({p:.4f}) = {log_val:.4f}")

# Calculate individual entropy terms
print("\nStep 3: Calculate each entropy term: -P(x_i) × log₂(P(x_i))")
entropy_terms = []
for i, (category, p, log_val) in enumerate(zip(categories, probabilities, log_values)):
    entropy_term = -p * log_val
    entropy_terms.append(entropy_term)
    print(f"For category {category}:")
    print(f"  -P({category}) × log₂(P({category})) = -({p:.4f}) × ({log_val:.4f})")
    print(f"  -P({category}) × log₂(P({category})) = {entropy_term:.4f} bits")

# Sum up to get total entropy
entropy = sum(entropy_terms)
print(f"\nStep 4: Sum all entropy terms to get total entropy")
print(f"H(X) = {' + '.join([f'{term:.4f}' for term in entropy_terms])}")
print(f"H(X) = {entropy:.4f} bits")

print("\nInterpretation:")
print(f"- An entropy of {entropy:.4f} bits is the theoretical minimum bits needed per example")
print(f"- For 3 categories, fixed-length encoding requires ceil(log2(3)) = 2 bits")
print("- Any encoding using fewer than entropy bits would lose information")

# Determine bits needed for fixed-length code
fixed_length_bits = math.ceil(math.log2(len(categories)))
print(f"\nCalculating minimum bits for fixed-length encoding:")
print(f"- Number of categories = {len(categories)}")
print(f"- log₂({len(categories)}) = {math.log2(len(categories)):.4f}")
print(f"- ceil(log₂({len(categories)})) = {fixed_length_bits}")
print(f"- Therefore, we need at least {fixed_length_bits} bits per example for fixed-length encoding")

# Simple visualization of entropy
fig1 = plt.figure(figsize=(10, 6))
# Bar chart showing distribution and entropy contribution
bars = plt.bar(categories, probabilities, color=sns.color_palette("pastel", 3), alpha=0.8, width=0.6)
plt.title('Class Distribution and Entropy Contribution')
plt.xlabel('Category')
plt.ylabel('Probability')
plt.ylim(0, max(probabilities) * 1.3)

# Add labels on each bar
for i, (bar, p, ent_term) in enumerate(zip(bars, probabilities, entropy_terms)):
    height = bar.get_height()
    plt.text(i, height + 0.02, f'P={p:.2f}\n{ent_term:.4f} bits', 
            ha='center', va='bottom', fontweight='bold')

# Add total entropy annotation - FIX POSITIONING
# Move the line to bottom part of the chart, away from other text
plt.axhline(y=0.1, color='red', linestyle='--', linewidth=2)
# Place the text below the line with better padding
plt.text(len(categories)/2 - 0.5, 0.05, f'Total Entropy: {entropy:.4f} bits', 
         ha='center', color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, pad=3))

plt.tight_layout()
save_figure(fig1, "step1_entropy_calculation.png")

# ==============================
# STEP 2: Calculate Bits Required for Scheme 1 (One-hot)
# ==============================
print_section_header("STEP 2: One-hot Encoding (Scheme 1)")

print("One-hot encoding scheme:")
for category, encoding in scheme1_onehot.items():
    print(f"- Category {category}: {encoding}")

bits_per_example_onehot = len(next(iter(scheme1_onehot.values())))
total_bits_onehot = bits_per_example_onehot * total_examples

print(f"\nCalculating storage requirements for one-hot encoding:")
print(f"Step 1: Determine bits needed per example")
print(f"- One-hot encoding uses one bit per possible category")
print(f"- Number of categories = {len(categories)}")
print(f"- Therefore, bits per example = {bits_per_example_onehot}")

print(f"\nStep 2: Calculate total bits for all examples")
print(f"- Total bits = bits per example × number of examples")
print(f"- Total bits = {bits_per_example_onehot} × {total_examples} = {total_bits_onehot} bits")

# Breakdown by category
print(f"\nStep 3: Calculate storage for each category")
category_bits_onehot = []
for category, count in zip(categories, counts):
    bits = bits_per_example_onehot * count
    category_bits_onehot.append(bits)
    print(f"- Category {category}: {count} examples × {bits_per_example_onehot} bits = {bits} bits")

print(f"\nVerification: {' + '.join([str(bits) for bits in category_bits_onehot])} = {sum(category_bits_onehot)} bits")

# Comparison with entropy
print(f"\nStep 4: Compare with entropy (theoretical minimum)")
onehot_overhead_bits = bits_per_example_onehot - entropy
onehot_overhead_percent = (onehot_overhead_bits / entropy) * 100
print(f"- Entropy: {entropy:.4f} bits per example")
print(f"- One-hot: {bits_per_example_onehot} bits per example")
print(f"- Overhead: {bits_per_example_onehot} - {entropy:.4f} = {onehot_overhead_bits:.4f} bits per example")
print(f"- Relative overhead: ({onehot_overhead_bits:.4f} / {entropy:.4f}) × 100% = {onehot_overhead_percent:.1f}%")

# Visualization of one-hot encoding
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left: One-hot matrix visualization
onehot_matrix = np.array(list(scheme1_onehot.values()))
sns.heatmap(onehot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2', 'Bit 3'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding Matrix')

# Right: Storage breakdown pie chart
pie_colors = sns.color_palette("pastel", 3)
wedges, texts = ax2.pie(
    category_bits_onehot, 
    labels=categories, 
    startangle=90, 
    colors=pie_colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
ax2.set_title(f'Storage: {total_bits_onehot} bits total')

# Add bit values to pie chart - improved positioning
for i, (wedge, cat_bits) in enumerate(zip(wedges, category_bits_onehot)):
    # Get angular position in the middle of the wedge (in radians)
    theta = np.pi/180 * (wedge.theta1 + wedge.theta2) / 2
    
    # Calculate radius (0.7 of the pie radius for positioning inside the wedge)
    radius = 0.7
    
    # Convert to cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Create clear white background for text
    ax2.text(
        x, y, f"{cat_bits} bits",
        ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='none')
    )

plt.tight_layout()
save_figure(fig2, "step2_onehot_encoding.png")

# ==============================
# STEP 3: Calculate Bits Required for Scheme 2 (Binary)
# ==============================
print_section_header("STEP 3: Binary Encoding (Scheme 2)")

print("Binary encoding scheme:")
for category, encoding in scheme2_binary.items():
    print(f"- Category {category}: {encoding}")

bits_per_example_binary = len(next(iter(scheme2_binary.values())))
total_bits_binary = bits_per_example_binary * total_examples

print(f"\nCalculating storage requirements for binary encoding:")
print(f"Step 1: Determine bits needed per example")
print(f"- For {len(categories)} distinct categories, we need log₂({len(categories)}) bits")
print(f"- log₂({len(categories)}) = {math.log2(len(categories)):.4f}")
print(f"- Since we need a whole number of bits, we use ceil(log₂({len(categories)})) = {math.ceil(math.log2(len(categories)))}")
print(f"- Therefore, bits per example = {bits_per_example_binary}")

print(f"\nStep 2: Calculate total bits for all examples")
print(f"- Total bits = bits per example × number of examples")
print(f"- Total bits = {bits_per_example_binary} × {total_examples} = {total_bits_binary} bits")

# Breakdown by category
print(f"\nStep 3: Calculate storage for each category")
category_bits_binary = []
for category, count in zip(categories, counts):
    bits = bits_per_example_binary * count
    category_bits_binary.append(bits)
    print(f"- Category {category}: {count} examples × {bits_per_example_binary} bits = {bits} bits")

print(f"\nVerification: {' + '.join([str(bits) for bits in category_bits_binary])} = {sum(category_bits_binary)} bits")

# Comparison with entropy
print(f"\nStep 4: Compare with entropy (theoretical minimum)")
binary_overhead_bits = bits_per_example_binary - entropy
binary_overhead_percent = (binary_overhead_bits / entropy) * 100
print(f"- Entropy: {entropy:.4f} bits per example")
print(f"- Binary: {bits_per_example_binary} bits per example")
print(f"- Overhead: {bits_per_example_binary} - {entropy:.4f} = {binary_overhead_bits:.4f} bits per example")
print(f"- Relative overhead: ({binary_overhead_bits:.4f} / {entropy:.4f}) × 100% = {binary_overhead_percent:.1f}%")

# Visualization of binary encoding
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left: Binary encoding matrix
binary_matrix = np.array(list(scheme2_binary.values()))
sns.heatmap(binary_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('Binary Encoding Matrix')

# Right: Storage breakdown pie chart
pie_colors = sns.color_palette("pastel", 3)
wedges, texts = ax2.pie(
    category_bits_binary, 
    labels=categories, 
    startangle=90, 
    colors=pie_colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
ax2.set_title(f'Storage: {total_bits_binary} bits total')

# Add bit values to pie chart - improved positioning
for i, (wedge, cat_bits) in enumerate(zip(wedges, category_bits_binary)):
    # Get angular position in the middle of the wedge (in radians)
    theta = np.pi/180 * (wedge.theta1 + wedge.theta2) / 2
    
    # Calculate radius (0.7 of the pie radius for positioning inside the wedge)
    radius = 0.7
    
    # Convert to cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Create clear white background for text
    ax2.text(
        x, y, f"{cat_bits} bits",
        ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='none')
    )

plt.tight_layout()
save_figure(fig3, "step3_binary_encoding.png")

# ==============================
# STEP 4: Compare the Efficiency of Both Encoding Schemes
# ==============================
print_section_header("STEP 4: Efficiency Comparison")

# Calculate percentage reduction
reduction_bits = total_bits_onehot - total_bits_binary
reduction_percentage = (reduction_bits / total_bits_onehot) * 100

print(f"Step 1: Compare storage requirements")
print(f"- One-hot encoding: {total_bits_onehot} bits total ({bits_per_example_onehot} bits per example)")
print(f"- Binary encoding: {total_bits_binary} bits total ({bits_per_example_binary} bits per example)")

print(f"\nStep 2: Calculate absolute savings")
print(f"- Absolute savings = One-hot bits - Binary bits")
print(f"- Absolute savings = {total_bits_onehot} - {total_bits_binary} = {reduction_bits} bits")

print(f"\nStep 3: Calculate percentage reduction")
print(f"- Percentage reduction = (Absolute savings / One-hot bits) × 100%")
print(f"- Percentage reduction = ({reduction_bits} / {total_bits_onehot}) × 100% = {reduction_percentage:.2f}%")

# Calculate comparison with theoretical minimum
theoretical_min_bits = entropy * total_examples
print(f"\nStep 4: Compare both schemes with theoretical minimum")
print(f"- Theoretical minimum (based on entropy): {entropy:.4f} bits/example × {total_examples} examples = {theoretical_min_bits:.2f} bits")

# Calculate overhead percentages
onehot_overhead_total = total_bits_onehot - theoretical_min_bits
binary_overhead_total = total_bits_binary - theoretical_min_bits
onehot_overhead_percent_total = (onehot_overhead_total / theoretical_min_bits) * 100
binary_overhead_percent_total = (binary_overhead_total / theoretical_min_bits) * 100

print(f"- One-hot overhead: {total_bits_onehot} - {theoretical_min_bits:.2f} = {onehot_overhead_total:.2f} bits")
print(f"- One-hot overhead percentage: ({onehot_overhead_total:.2f} / {theoretical_min_bits:.2f}) × 100% = {onehot_overhead_percent_total:.2f}%")
print(f"- Binary overhead: {total_bits_binary} - {theoretical_min_bits:.2f} = {binary_overhead_total:.2f} bits")
print(f"- Binary overhead percentage: ({binary_overhead_total:.2f} / {theoretical_min_bits:.2f}) × 100% = {binary_overhead_percent_total:.2f}%")

print("\nKey insights:")
print(f"- Binary encoding is {reduction_percentage:.1f}% more efficient than one-hot encoding")
print(f"- However, it still uses {binary_overhead_percent_total:.1f}% more bits than the theoretical minimum")
print(f"- This is because fixed-length codes must use whole numbers of bits per example")
print(f"- To approach the entropy limit of {entropy:.4f} bits, variable-length codes would be needed")

# Simple bar chart comparison
fig4 = plt.figure(figsize=(10, 6))
labels = ['One-Hot\nEncoding', 'Binary\nEncoding', 'Theoretical\nMinimum']
values = [total_bits_onehot, total_bits_binary, theoretical_min_bits]
bars = plt.bar(labels, values, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Storage Requirements Comparison')
plt.ylabel('Total Bits')
plt.grid(axis='y', alpha=0.3)

# Add text labels with details
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    bits_per_example = val / total_examples if i < 2 else entropy
    plt.text(i, height/2, 
            f"{val:.1f} bits\n{bits_per_example:.2f} bits/example", 
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
    )

# Add arrow showing the reduction between one-hot and binary
plt.annotate(
    f"{reduction_percentage:.1f}% reduction\n({reduction_bits} bits)", 
    xy=(0, total_bits_onehot - reduction_bits/2),
    xytext=(1, total_bits_onehot - reduction_bits/2),
    ha='center', va='center',
    arrowprops=dict(arrowstyle='<->', color='green', lw=2),
    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
)

plt.tight_layout()
save_figure(fig4, "step4_efficiency_comparison.png")

# ==============================
# STEP 5: Analyze Whether Binary Encoding is Lossless
# ==============================
print_section_header("STEP 5: Lossless Analysis")

print("Step 1: Define what makes an encoding lossless")
print("A lossless encoding must maintain a perfect one-to-one mapping between categories and codes.")
print("Each category must have a unique code that can be unambiguously decoded.")

print("\nStep 2: Examine binary encoding scheme")
for cat in categories:
    print(f"- Category {cat}: {scheme2_binary[cat]}")

# Check if binary encoding is lossless
binary_codes = list(scheme2_binary.values())
unique_codes = set(tuple(code) for code in binary_codes)
is_lossless = len(unique_codes) == len(categories)

print(f"\nStep 3: Analyze uniqueness of codes")
print(f"- Number of unique binary codes: {len(unique_codes)}")
print(f"- Number of categories: {len(categories)}")
print(f"- Are all codes unique? {'Yes' if is_lossless else 'No'}")

print(f"\nStep 4: Verify decodability")
print("- Can we recover the original category from each code?")
# Create a dictionary mapping binary codes to categories for decoding
decode_dict = {tuple(code): cat for cat, code in scheme2_binary.items()}
for code_tuple, cat in decode_dict.items():
    code_str = str(list(code_tuple))
    print(f"  Code {code_str} → Category {cat}")

print(f"\nConclusion: The binary encoding is{' ' if is_lossless else ' not '}lossless.")

if is_lossless:
    print("\nExplanation:")
    print("- Each category has a unique binary code (no ambiguity)")
    print("- There is a one-to-one mapping between categories and codes")
    print("- We can perfectly reconstruct the original category from its binary code")
    print("- No information is lost in the encoding process")

# Simple visualization of lossless property
fig5 = plt.figure(figsize=(8, 6))
plt.axis('off')
plt.title('Binary Encoding: One-to-One Mapping (Lossless)', fontsize=14)

# Draw a simple mapping diagram
y_positions = [0.7, 0.5, 0.3]
for i, cat in enumerate(categories):
    # Category on left
    plt.text(0.2, y_positions[i], f"Category {cat}", ha='center', va='center', fontsize=12,
           bbox=dict(facecolor=sns.color_palette("pastel", 3)[i], boxstyle='round', alpha=0.7))
    
    # Binary code on right
    code_str = f"[{', '.join(map(str, scheme2_binary[cat]))}]"
    plt.text(0.8, y_positions[i], f"Code: {code_str}", ha='center', va='center', fontsize=12,
           bbox=dict(facecolor='lightblue', boxstyle='round', alpha=0.7))
    
    # Draw arrows with labels
    plt.annotate('', xy=(0.65, y_positions[i]), xytext=(0.35, y_positions[i]),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    plt.text(0.5, y_positions[i]+0.03, "Encode", ha='center', va='bottom', fontsize=10, color='blue')
    
    plt.annotate('', xy=(0.35, y_positions[i]-0.05), xytext=(0.65, y_positions[i]-0.05),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    plt.text(0.5, y_positions[i]-0.08, "Decode", ha='center', va='top', fontsize=10, color='green')

plt.tight_layout()
save_figure(fig5, "step5_lossless_analysis.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this encoding schemes and information theory problem:")
print(f"1. Entropy: {entropy:.4f} bits per example (theoretical minimum)")
print(f"2. One-hot encoding: {bits_per_example_onehot} bits per example, {total_bits_onehot} bits total")
print(f"3. Binary encoding: {bits_per_example_binary} bits per example, {total_bits_binary} bits total")
print(f"4. Efficiency: Binary encoding saves {reduction_percentage:.2f}% compared to one-hot")
print(f"5. Binary encoding is lossless while being more efficient")

print("\nImportant concepts demonstrated:")
print("- Entropy as a measure of information content")
print("- Fixed-length vs. variable-length encoding")
print("- Lossless vs. lossy compression")
print("- Theoretical limits in information representation")
print("- Trade-offs between simplicity and efficiency in encoding schemes") 