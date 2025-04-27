import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Create directory to save figures if it doesn't exist
os.makedirs("images", exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join("images", filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
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
print_section_header("STEP 1: Calculate the Entropy of the Class Distribution")

print("The entropy measures the average information content of a probability distribution.")
print("For a discrete distribution, the entropy in bits is calculated as:")
print("H(X) = -∑ P(x_i) log₂ P(x_i)")

print("\nGiven dataset distribution:")
for i, (category, count) in enumerate(zip(categories, counts)):
    percentage = (count / total_examples) * 100
    print(f"Category {category}: {count} instances ({percentage:.1f}%)")

print(f"\nTotal instances: {total_examples}")

# Calculate entropy (in bits) with detailed steps
print("\nStep-by-step entropy calculation:")
entropy_terms = []
for i, (category, p) in enumerate(zip(categories, probabilities)):
    entropy_term = -p * math.log2(p)
    entropy_terms.append(entropy_term)
    print(f"Category {category}:")
    print(f"  P({category}) = {count}/{total_examples} = {p:.4f}")
    print(f"  -P({category}) × log₂(P({category})) = -({p:.4f}) × log₂({p:.4f}) = {entropy_term:.4f} bits")

entropy = sum(entropy_terms)
print(f"\nTotal entropy = sum of all terms = {entropy_terms[0]:.4f} + {entropy_terms[1]:.4f} + {entropy_terms[2]:.4f} = {entropy:.4f} bits")

print("\nInterpretation:")
print(f"An entropy of {entropy:.4f} bits means:")
print("- This is the theoretical minimum number of bits needed per example on average")
print("- Any encoding using fewer than this many bits would be lossy")
print("- Practical encodings typically use more bits due to implementation constraints")

# Visualization of entropy - SIMPLIFIED
fig1 = plt.figure(figsize=(10, 5))

# Create a single plot showing both distribution and information content
plt.bar(categories, probabilities, color=sns.color_palette("pastel", 3), alpha=0.7, width=0.4)
plt.title('Class Distribution and Information Content')
plt.xlabel('Category')
plt.ylabel('Probability')

# Add entropy labels on top of each bar
for i, (cat, p, ent_term) in enumerate(zip(categories, probabilities, entropy_terms)):
    plt.text(i, p/2, f'P={p:.2f}\nH={ent_term:.2f}', ha='center', va='center', color='black', fontweight='bold')

# Add total entropy line
plt.axhline(y=entropy/10, color='red', linestyle='-', linewidth=2)
plt.text(1.5, entropy/10 + 0.02, f'Total Entropy: {entropy:.4f} bits', ha='center', color='red', fontweight='bold')

plt.tight_layout()
save_figure(fig1, "step1_entropy_calculation.png")

# ==============================
# STEP 2: Calculate Bits Required for Scheme 1 (One-hot)
# ==============================
print_section_header("STEP 2: Calculate Bits Required for Scheme 1 (One-hot)")

print("In the one-hot encoding scheme (Scheme 1), each category is represented as:")
for category, encoding in scheme1_onehot.items():
    print(f"Category {category}: {encoding}")

print("\nStep-by-step calculation of storage requirements:")

# Calculate bits for one-hot encoding with detailed steps
bits_per_example_onehot = len(next(iter(scheme1_onehot.values())))
print(f"1. Each example requires {bits_per_example_onehot} bits to encode (one bit for each category)")

total_bits_onehot = bits_per_example_onehot * total_examples
print(f"2. Total bits required:")
print(f"   {bits_per_example_onehot} bits/example × {total_examples} examples = {total_bits_onehot} bits")

# Breakdown by category
print("\n3. Breakdown by category:")
category_bits_onehot = []
for category, count in zip(categories, counts):
    bits = bits_per_example_onehot * count
    category_bits_onehot.append(bits)
    print(f"   Category {category}: {count} examples × {bits_per_example_onehot} bits/example = {bits} bits")

print(f"\nVerification: {sum(category_bits_onehot)} total bits (should equal {total_bits_onehot})")

# Visualization of one-hot encoding - SIMPLIFIED
fig2 = plt.figure(figsize=(10, 5))

# Left: One-hot matrix
ax1 = plt.subplot(121)
onehot_matrix = np.array(list(scheme1_onehot.values()))
sns.heatmap(onehot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2', 'Bit 3'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding (3 bits/example)')

# Right: Storage breakdown
ax2 = plt.subplot(122)
ax2.pie(category_bits_onehot, labels=[f"{cat}: {bits} bits" for cat, bits in zip(categories, category_bits_onehot)], 
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", 3))
ax2.set_title(f'Total Storage: {total_bits_onehot} bits')

plt.tight_layout()
save_figure(fig2, "step2_onehot_encoding.png")

# ==============================
# STEP 3: Calculate Bits Required for Scheme 2 (Binary)
# ==============================
print_section_header("STEP 3: Calculate Bits Required for Scheme 2 (Binary)")

print("In the binary encoding scheme (Scheme 2), each category is represented as:")
for category, encoding in scheme2_binary.items():
    print(f"Category {category}: {encoding}")

print("\nStep-by-step calculation of storage requirements:")

# Calculate bits for binary encoding with detailed steps
bits_per_example_binary = len(next(iter(scheme2_binary.values())))
print(f"1. Each example requires {bits_per_example_binary} bits to encode (using binary encoding)")

total_bits_binary = bits_per_example_binary * total_examples
print(f"2. Total bits required:")
print(f"   {bits_per_example_binary} bits/example × {total_examples} examples = {total_bits_binary} bits")

# Breakdown by category
print("\n3. Breakdown by category:")
category_bits_binary = []
for category, count in zip(categories, counts):
    bits = bits_per_example_binary * count
    category_bits_binary.append(bits)
    print(f"   Category {category}: {count} examples × {bits_per_example_binary} bits/example = {bits} bits")

print(f"\nVerification: {sum(category_bits_binary)} total bits (should equal {total_bits_binary})")

# Visualization of binary encoding - SIMPLIFIED
fig3 = plt.figure(figsize=(10, 5))

# Left: Binary matrix
ax1 = plt.subplot(121)
binary_matrix = np.array(list(scheme2_binary.values()))
sns.heatmap(binary_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('Binary Encoding (2 bits/example)')

# Right: Storage breakdown
ax2 = plt.subplot(122)
ax2.pie(category_bits_binary, labels=[f"{cat}: {bits} bits" for cat, bits in zip(categories, category_bits_binary)], 
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", 3))
ax2.set_title(f'Total Storage: {total_bits_binary} bits')

plt.tight_layout()
save_figure(fig3, "step3_binary_encoding.png")

# ==============================
# STEP 4: Compare the Efficiency of Both Encoding Schemes
# ==============================
print_section_header("STEP 4: Compare the Efficiency of Both Encoding Schemes")

print("Now we'll compare the efficiency of both encoding schemes:")

# Calculate percentage reduction
reduction_bits = total_bits_onehot - total_bits_binary
reduction_percentage = (reduction_bits / total_bits_onehot) * 100

print(f"Scheme 1 (One-hot): {bits_per_example_onehot} bits/example × {total_examples} examples = {total_bits_onehot} bits")
print(f"Scheme 2 (Binary): {bits_per_example_binary} bits/example × {total_examples} examples = {total_bits_binary} bits")

print("\nStep-by-step calculation of savings:")
print(f"1. Absolute reduction in bits: {total_bits_onehot} - {total_bits_binary} = {reduction_bits} bits")
print(f"2. Percentage reduction: ({reduction_bits} / {total_bits_onehot}) × 100% = {reduction_percentage:.2f}%")

# Calculate comparison with theoretical minimum (based on entropy)
theoretical_min_bits = entropy * total_examples
print(f"\nComparison with theoretical minimum (based on entropy):")
print(f"Theoretical minimum: {entropy:.4f} bits/example × {total_examples} examples = {theoretical_min_bits:.2f} bits")

# Calculate overhead percentages
onehot_overhead = (total_bits_onehot - theoretical_min_bits) / theoretical_min_bits * 100
binary_overhead = (total_bits_binary - theoretical_min_bits) / theoretical_min_bits * 100

print(f"Scheme 1 (One-hot) overhead: {onehot_overhead:.2f}% above theoretical minimum")
print(f"Scheme 2 (Binary) overhead: {binary_overhead:.2f}% above theoretical minimum")

print(f"\nConclusion: Scheme 2 (Binary) is {reduction_percentage:.2f}% more efficient than Scheme 1 (One-hot)")

# Visualization of efficiency comparison - SIMPLIFIED
fig4 = plt.figure(figsize=(10, 5))

# Plot bar chart comparison
plt.bar(['One-Hot', 'Binary', 'Theoretical\nMinimum'], 
        [total_bits_onehot, total_bits_binary, theoretical_min_bits],
        color=['#ff9999', '#66b3ff', '#99ff99'])

plt.title('Storage Requirements Comparison')
plt.ylabel('Total Bits')
plt.grid(axis='y', alpha=0.3)

# Add text labels
for i, (val, label) in enumerate(zip(
    [total_bits_onehot, total_bits_binary, theoretical_min_bits],
    [f"{total_bits_onehot} bits", f"{total_bits_binary} bits", f"{theoretical_min_bits:.1f} bits"]
)):
    plt.text(i, val/2, label, ha='center', va='center', fontweight='bold')

# Add overhead percentages
plt.text(0, total_bits_onehot + 5, f"+{onehot_overhead:.1f}%", ha='center', color='red')
plt.text(1, total_bits_binary + 5, f"+{binary_overhead:.1f}%", ha='center', color='red')
plt.text(0.5, 150, f"{reduction_percentage:.1f}% reduction", ha='center', 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
save_figure(fig4, "step4_efficiency_comparison.png")

# ==============================
# STEP 5: Analyze Whether Binary Encoding is Lossless
# ==============================
print_section_header("STEP 5: Analyze Whether Binary Encoding is Lossless")

print("To determine if binary encoding is lossless, we need to check if each category")
print("can be uniquely identified from its binary code.")

print("\nEncoding table - comparing both schemes:")
for i, cat in enumerate(categories):
    print(f"Category {cat}:")
    print(f"  One-hot encoding: {scheme1_onehot[cat]}")
    print(f"  Binary encoding:  {scheme2_binary[cat]}")

# Check if binary encoding is lossless
binary_codes = list(scheme2_binary.values())
unique_codes = set(tuple(code) for code in binary_codes)
is_lossless = len(unique_codes) == len(categories)

print(f"\nAnalysis of binary encoding losslessness:")
print(f"1. Number of unique binary codes: {len(unique_codes)}")
print(f"2. Number of categories: {len(categories)}")
print(f"3. Is every category uniquely represented? {'Yes' if is_lossless else 'No'}")
print(f"\nConclusion: The binary encoding is{' ' if is_lossless else ' not '}lossless.")

if is_lossless:
    print("\nReasons why the binary encoding is lossless:")
    print("- Each category has a unique binary code")
    print("- There is a one-to-one mapping between categories and codes")
    print("- We can perfectly reconstruct the original category from its binary code")
    print("- No information is lost in the encoding process")

# Theoretical analysis
print(f"\nTheoretical analysis:")
print(f"1. Entropy of the distribution: {entropy:.4f} bits per example")
print(f"2. Bits per example in binary encoding: {bits_per_example_binary} bits")
extra_bits = bits_per_example_binary - entropy
print(f"3. Extra bits per example: {bits_per_example_binary} - {entropy:.4f} = {extra_bits:.4f} bits")
print(f"4. Percentage overhead: {binary_overhead:.2f}%")

# Theoretical minimum number of bits needed
min_bits_needed = math.ceil(math.log2(len(categories)))
print(f"\nMinimum bits needed to represent {len(categories)} categories: {min_bits_needed}")
print(f"Binary encoding uses {bits_per_example_binary} bits, which is the theoretical minimum possible")
print(f"for a fixed-length binary code that can represent {len(categories)} distinct categories.")

# Visualization of lossless encoding - SIMPLIFIED
fig5 = plt.figure(figsize=(10, 5))

# Create a simplified diagram showing encoding and decoding
plt.axis('off')

# Draw a diagram showing mapping between categories and binary codes
y_positions = [0.7, 0.5, 0.3]
for i, cat in enumerate(categories):
    # Draw category node on left
    plt.text(0.2, y_positions[i], f"Category {cat}", ha='center', va='center',
             bbox=dict(facecolor=sns.color_palette("pastel", 3)[i], boxstyle='round', alpha=0.7))
    
    # Draw binary code on right
    plt.text(0.8, y_positions[i], f"Binary: {scheme2_binary[cat]}", ha='center', va='center',
             bbox=dict(facecolor='lightblue', boxstyle='round', alpha=0.7))
    
    # Draw arrows
    plt.annotate('', xy=(0.65, y_positions[i]), xytext=(0.35, y_positions[i]),
                arrowprops=dict(arrowstyle='->', color='blue'))
    plt.annotate('', xy=(0.35, y_positions[i]), xytext=(0.65, y_positions[i]),
                arrowprops=dict(arrowstyle='->', color='green'))

# Add explanatory text
plt.text(0.5, 0.9, "Binary Encoding is Lossless" if is_lossless else "Binary Encoding is Lossy", 
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.text(0.5, 0.1, f"Entropy: {entropy:.4f} bits | Binary code: {bits_per_example_binary} bits | Overhead: {binary_overhead:.1f}%", 
         ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
save_figure(fig5, "step5_lossless_analysis.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this question on one-hot encoding and information theory:")

print("\n1. Entropy of the Class Distribution:")
print(f"   - Entropy: {entropy:.4f} bits per example (theoretical minimum)")
print(f"   - Class breakdown: Category A: {entropy_terms[0]:.4f} bits, B: {entropy_terms[1]:.4f} bits, C: {entropy_terms[2]:.4f} bits")

print("\n2. Scheme 1 (One-hot Encoding):")
print(f"   - Bits per example: {bits_per_example_onehot} bits")
print(f"   - Total storage required: {total_bits_onehot} bits")
print(f"   - Overhead vs. theoretical minimum: {onehot_overhead:.2f}%")

print("\n3. Scheme 2 (Binary Encoding):")
print(f"   - Bits per example: {bits_per_example_binary} bits")
print(f"   - Total storage required: {total_bits_binary} bits")
print(f"   - Overhead vs. theoretical minimum: {binary_overhead:.2f}%")

print("\n4. Efficiency Comparison:")
print(f"   - Binary encoding reduces storage by {reduction_bits} bits ({reduction_percentage:.2f}%)")
print(f"   - Scheme 2 uses {total_bits_binary} bits instead of {total_bits_onehot} bits (Scheme 1)")

print("\n5. Lossless Analysis:")
print(f"   - Binary encoding is {'lossless' if is_lossless else 'lossy'}")
print(f"   - Each category can be uniquely identified from its binary code")
print(f"   - Binary encoding uses {binary_overhead:.2f}% more bits than the theoretical minimum")
print(f"   - No practical encoding can use fewer than {min_bits_needed} bits per example for {len(categories)} categories") 