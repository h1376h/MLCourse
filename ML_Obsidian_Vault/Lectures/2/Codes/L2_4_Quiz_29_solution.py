import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

# Create directory for saving images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
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

print("Entropy measures the average information content or uncertainty in a probability distribution.")
print("It represents the theoretical minimum number of bits needed per example to encode the information.")
print("For a discrete distribution with categories {A, B, C}, the entropy in bits is calculated as:")
print("H(X) = -∑ P(x_i) log₂ P(x_i)")

print("\nGiven dataset distribution:")
for i, (category, count) in enumerate(zip(categories, counts)):
    percentage = (count / total_examples) * 100
    print(f"Category {category}: {count} instances ({percentage:.1f}%)")

print(f"\nTotal instances: {total_examples}")

# Calculate entropy (in bits) with detailed steps
print("\nDetailed entropy calculation for each category:")
entropy_terms = []
for i, (category, p) in enumerate(zip(categories, probabilities)):
    entropy_term = -p * math.log2(p)
    entropy_terms.append(entropy_term)
    print(f"Category {category}:")
    print(f"  P({category}) = {count}/{total_examples} = {p:.4f}")
    print(f"  -P({category}) × log₂(P({category})) = -({p:.4f}) × log₂({p:.4f})")
    print(f"  -P({category}) × log₂(P({category})) = -({p:.4f}) × ({math.log2(p):.6f})")
    print(f"  -P({category}) × log₂(P({category})) = {entropy_term:.6f} bits")

entropy = sum(entropy_terms)
print(f"\nTotal entropy = sum of all terms = {' + '.join([f'{term:.6f}' for term in entropy_terms])}")
print(f"Total entropy = {entropy:.6f} bits")

print("\nInterpretation:")
print(f"An entropy of {entropy:.6f} bits means:")
print("- This is the theoretical minimum number of bits needed per example on average")
print("- Any encoding using fewer than this many bits would necessarily be lossy")
print("- Practical encodings typically use more bits due to fixed-length encoding constraints")
print(f"- For {len(categories)} = 3 categories, the minimum number of bits needed is ceiling(log₂(3)) = {math.ceil(math.log2(len(categories)))}")

# Enhanced visualization of entropy
fig1 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Top plot: Bar chart showing distribution and entropy contribution
ax1 = fig1.add_subplot(gs[0])
bars = ax1.bar(categories, probabilities, color=sns.color_palette("pastel", 3), alpha=0.7, width=0.6)
ax1.set_title('Class Distribution and Information Content', fontsize=14)
ax1.set_xlabel('Category')
ax1.set_ylabel('Probability')
ax1.set_ylim(0, max(probabilities) * 1.3)

# Add detailed labels on top of each bar
for i, (bar, p, ent_term) in enumerate(zip(bars, probabilities, entropy_terms)):
    height = bar.get_height()
    ax1.text(i, height + 0.02, f'Count: {counts[i]}\nP={p:.4f}\nEntropy={ent_term:.4f} bits', 
            ha='center', va='bottom', fontweight='bold')

# Add total entropy annotation
ax1.axhline(y=entropy/5, color='red', linestyle='-', linewidth=2)
ax1.text(len(categories)/2 - 0.5, entropy/5 + 0.08, f'Total Entropy: {entropy:.4f} bits', 
         ha='center', color='red', fontweight='bold')

# Bottom: Detailed explanation
ax2 = fig1.add_subplot(gs[1])
ax2.axis('off')

# Create a visual explanation of entropy calculation
ax2.text(0.5, 0.95, "Entropy Calculation", ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Formula
formula = r"$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$"
ax2.text(0.5, 0.85, formula, ha='center', va='center', fontsize=12)

# Detailed calculation showing each term
calculation = [
    f"$H(X) = -P(A) \\log_2 P(A) - P(B) \\log_2 P(B) - P(C) \\log_2 P(C)$",
    f"$H(X) = -{probabilities[0]:.4f} \\times \\log_2({probabilities[0]:.4f}) - {probabilities[1]:.4f} \\times \\log_2({probabilities[1]:.4f}) - {probabilities[2]:.4f} \\times \\log_2({probabilities[2]:.4f})$",
    f"$H(X) = -{probabilities[0]:.4f} \\times ({math.log2(probabilities[0]):.4f}) - {probabilities[1]:.4f} \\times ({math.log2(probabilities[1]):.4f}) - {probabilities[2]:.4f} \\times ({math.log2(probabilities[2]):.4f})$",
    f"$H(X) = {entropy_terms[0]:.4f} + {entropy_terms[1]:.4f} + {entropy_terms[2]:.4f} = {entropy:.4f}$ bits"
]

for i, calc_line in enumerate(calculation):
    ax2.text(0.5, 0.75 - 0.1*i, calc_line, ha='center', va='center', fontsize=12)

# Add interpretation
interpretation = [
    f"• Entropy = {entropy:.4f} bits is the theoretical minimum bits per example",
    f"• Category A: {entropy_terms[0]:.4f} bits ({entropy_terms[0]/entropy*100:.1f}% of total entropy)",
    f"• Category B: {entropy_terms[1]:.4f} bits ({entropy_terms[1]/entropy*100:.1f}% of total entropy)",
    f"• Category C: {entropy_terms[2]:.4f} bits ({entropy_terms[2]/entropy*100:.1f}% of total entropy)",
    f"• Any encoding scheme using fewer than {entropy:.4f} bits per example would be lossy",
    f"• For 3 categories, we need at least ceiling(log₂(3)) = {math.ceil(math.log2(len(categories)))} bits per example"
]

interpretation_text = "\n".join(interpretation)
ax2.text(0.5, 0.3, interpretation_text, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightyellow', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig1, "step1_entropy_calculation.png")

# ==============================
# STEP 2: Calculate Bits Required for Scheme 1 (One-hot)
# ==============================
print_section_header("STEP 2: Calculate Bits Required for Scheme 1 (One-hot)")

print("One-hot encoding represents each category using a binary vector where only one element is 1,")
print("and all others are 0. This creates a direct one-to-one mapping between categories and vectors.")

print("\nOne-hot encoding scheme (Scheme 1):")
for category, encoding in scheme1_onehot.items():
    print(f"Category {category}: {encoding}")

print("\nDetailed calculation of storage requirements:")

# Calculate bits for one-hot encoding with detailed steps
bits_per_example_onehot = len(next(iter(scheme1_onehot.values())))
print(f"1. Number of bits per example:")
print(f"   • Each example requires {bits_per_example_onehot} bits (one bit for each possible category)")
print(f"   • This is equal to the number of categories ({len(categories)})")

total_bits_onehot = bits_per_example_onehot * total_examples
print(f"\n2. Total bits required for all {total_examples} examples:")
print(f"   • {bits_per_example_onehot} bits/example × {total_examples} examples = {total_bits_onehot} bits")

# Breakdown by category
print("\n3. Storage breakdown by category:")
category_bits_onehot = []
for category, count in zip(categories, counts):
    bits = bits_per_example_onehot * count
    category_bits_onehot.append(bits)
    print(f"   • Category {category}: {count} examples × {bits_per_example_onehot} bits/example = {bits} bits")

print(f"\nVerification: {' + '.join([str(bits) for bits in category_bits_onehot])} = {sum(category_bits_onehot)} total bits")
print(f"This equals {total_bits_onehot} bits as expected.")

# Enhanced visualization of one-hot encoding
fig2 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 1.2])

# Top-left: One-hot matrix visualization
ax1 = fig2.add_subplot(gs[0, 0])
onehot_matrix = np.array(list(scheme1_onehot.values()))
sns.heatmap(onehot_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2', 'Bit 3'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('One-Hot Encoding Matrix', fontsize=14)
ax1.set_xlabel('Bit Position')
ax1.set_ylabel('Category')

# Top-right: Storage breakdown pie chart
ax2 = fig2.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(
    category_bits_onehot, 
    labels=[f"{cat}" for cat in categories], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=sns.color_palette("pastel", 3)
)
ax2.set_title(f'Storage Breakdown: {total_bits_onehot} bits', fontsize=14)

# Enhance pie chart labels
for i, (autotext, cat_bits) in enumerate(zip(autotexts, category_bits_onehot)):
    autotext.set_fontweight('bold')
    # Add the actual bit values
    ax2.text(
        wedges[i].center[0]*0.6, 
        wedges[i].center[1]*0.6, 
        f"{cat_bits} bits",
        ha='center', va='center', fontweight='bold'
    )

# Bottom: Detailed calculation and explanation
ax3 = fig2.add_subplot(gs[1, :])
ax3.axis('off')

# Create a visual explanation of one-hot encoding
ax3.text(0.5, 0.95, "One-Hot Encoding Storage Calculation", ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Detail the calculation
explanation = [
    f"1. One-hot encoding requires {bits_per_example_onehot} bits per example (equal to the number of categories)",
    f"2. For category A ({counts[0]} examples): {counts[0]} × {bits_per_example_onehot} bits = {category_bits_onehot[0]} bits",
    f"3. For category B ({counts[1]} examples): {counts[1]} × {bits_per_example_onehot} bits = {category_bits_onehot[1]} bits",
    f"4. For category C ({counts[2]} examples): {counts[2]} × {bits_per_example_onehot} bits = {category_bits_onehot[2]} bits",
    f"5. Total bits: {' + '.join([str(bits) for bits in category_bits_onehot])} = {total_bits_onehot} bits"
]

for i, line in enumerate(explanation):
    ax3.text(0.1, 0.8 - 0.1*i, line, ha='left', va='center', fontsize=12)

# Compare with entropy
comparison = f"• The entropy of the distribution is {entropy:.4f} bits per example" + "\n" + \
             f"• One-hot encoding uses {bits_per_example_onehot} bits per example" + "\n" + \
             f"• Overhead: {bits_per_example_onehot - entropy:.4f} bits per example ({(bits_per_example_onehot/entropy - 1)*100:.1f}% more than the theoretical minimum)"

ax3.text(0.5, 0.25, comparison, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig2, "step2_onehot_encoding.png")

# ==============================
# STEP 3: Calculate Bits Required for Scheme 2 (Binary)
# ==============================
print_section_header("STEP 3: Calculate Bits Required for Scheme 2 (Binary)")

print("Binary encoding uses a more compact representation with fewer bits per example.")
print("Instead of using one bit per category, it uses log₂(n) bits, where n is the number of categories.")
print("This allows representing all categories uniquely while minimizing the number of bits.")

print("\nBinary encoding scheme (Scheme 2):")
for category, encoding in scheme2_binary.items():
    print(f"Category {category}: {encoding}")

print("\nDetailed calculation of storage requirements:")

# Calculate bits for binary encoding with detailed steps
bits_per_example_binary = len(next(iter(scheme2_binary.values())))
print(f"1. Number of bits per example:")
print(f"   • Each example requires {bits_per_example_binary} bits using binary encoding")
print(f"   • This is close to the theoretical minimum of log₂({len(categories)}) = {math.log2(len(categories)):.4f} bits")
print(f"   • Since we need a whole number of bits, we use ceiling(log₂({len(categories)})) = {math.ceil(math.log2(len(categories)))} bits")

total_bits_binary = bits_per_example_binary * total_examples
print(f"\n2. Total bits required for all {total_examples} examples:")
print(f"   • {bits_per_example_binary} bits/example × {total_examples} examples = {total_bits_binary} bits")

# Breakdown by category
print("\n3. Storage breakdown by category:")
category_bits_binary = []
for category, count in zip(categories, counts):
    bits = bits_per_example_binary * count
    category_bits_binary.append(bits)
    print(f"   • Category {category}: {count} examples × {bits_per_example_binary} bits/example = {bits} bits")

print(f"\nVerification: {' + '.join([str(bits) for bits in category_bits_binary])} = {sum(category_bits_binary)} total bits")
print(f"This equals {total_bits_binary} bits as expected.")

# Enhanced visualization of binary encoding
fig3 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, height_ratios=[1, 1.2])

# Top-left: Binary encoding matrix visualization
ax1 = fig3.add_subplot(gs[0, 0])
binary_matrix = np.array(list(scheme2_binary.values()))
sns.heatmap(binary_matrix, annot=True, cmap="Blues", cbar=False, 
            xticklabels=['Bit 1', 'Bit 2'], 
            yticklabels=categories,
            ax=ax1)
ax1.set_title('Binary Encoding Matrix', fontsize=14)
ax1.set_xlabel('Bit Position')
ax1.set_ylabel('Category')

# Top-right: Storage breakdown pie chart
ax2 = fig3.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(
    category_bits_binary, 
    labels=[f"{cat}" for cat in categories], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=sns.color_palette("pastel", 3)
)
ax2.set_title(f'Storage Breakdown: {total_bits_binary} bits', fontsize=14)

# Enhance pie chart labels
for i, (autotext, cat_bits) in enumerate(zip(autotexts, category_bits_binary)):
    autotext.set_fontweight('bold')
    # Add the actual bit values
    ax2.text(
        wedges[i].center[0]*0.6, 
        wedges[i].center[1]*0.6, 
        f"{cat_bits} bits",
        ha='center', va='center', fontweight='bold'
    )

# Bottom: Detailed calculation and explanation
ax3 = fig3.add_subplot(gs[1, :])
ax3.axis('off')

# Create a visual explanation of binary encoding
ax3.text(0.5, 0.95, "Binary Encoding Storage Calculation", ha='center', va='center', 
        fontsize=14, fontweight='bold')

# Detail the calculation
explanation = [
    f"1. Binary encoding requires {bits_per_example_binary} bits per example (ceiling(log₂({len(categories)})))",
    f"2. For category A ({counts[0]} examples): {counts[0]} × {bits_per_example_binary} bits = {category_bits_binary[0]} bits",
    f"3. For category B ({counts[1]} examples): {counts[1]} × {bits_per_example_binary} bits = {category_bits_binary[1]} bits",
    f"4. For category C ({counts[2]} examples): {counts[2]} × {bits_per_example_binary} bits = {category_bits_binary[2]} bits",
    f"5. Total bits: {' + '.join([str(bits) for bits in category_bits_binary])} = {total_bits_binary} bits"
]

for i, line in enumerate(explanation):
    ax3.text(0.1, 0.8 - 0.1*i, line, ha='left', va='center', fontsize=12)

# Compare with entropy
comparison = f"• The entropy of the distribution is {entropy:.4f} bits per example" + "\n" + \
             f"• Binary encoding uses {bits_per_example_binary} bits per example" + "\n" + \
             f"• Overhead: {bits_per_example_binary - entropy:.4f} bits per example ({(bits_per_example_binary/entropy - 1)*100:.1f}% more than the theoretical minimum)"

ax3.text(0.5, 0.25, comparison, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightblue', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig3, "step3_binary_encoding.png")

# ==============================
# STEP 4: Compare the Efficiency of Both Encoding Schemes
# ==============================
print_section_header("STEP 4: Compare the Efficiency of Both Encoding Schemes")

print("Now we'll compare the efficiency of both encoding schemes and quantify the savings.")

# Calculate percentage reduction
reduction_bits = total_bits_onehot - total_bits_binary
reduction_percentage = (reduction_bits / total_bits_onehot) * 100

print(f"Detailed comparison of storage requirements:")
print(f"1. Scheme 1 (One-hot encoding):")
print(f"   • {bits_per_example_onehot} bits/example × {total_examples} examples = {total_bits_onehot} bits")
print(f"2. Scheme 2 (Binary encoding):")
print(f"   • {bits_per_example_binary} bits/example × {total_examples} examples = {total_bits_binary} bits")

print(f"\nStep-by-step calculation of savings:")
print(f"1. Absolute reduction in bits:")
print(f"   • {total_bits_onehot} - {total_bits_binary} = {reduction_bits} bits")
print(f"2. Percentage reduction:")
print(f"   • ({reduction_bits} / {total_bits_onehot}) × 100% = {reduction_percentage:.2f}%")

# Calculate comparison with theoretical minimum (based on entropy)
theoretical_min_bits = entropy * total_examples
print(f"\nComparison with theoretical minimum (based on entropy):")
print(f"• Theoretical minimum: {entropy:.4f} bits/example × {total_examples} examples = {theoretical_min_bits:.2f} bits")

# Calculate overhead percentages
onehot_overhead = (total_bits_onehot - theoretical_min_bits) / theoretical_min_bits * 100
binary_overhead = (total_bits_binary - theoretical_min_bits) / theoretical_min_bits * 100

print(f"• Scheme 1 (One-hot) overhead: {total_bits_onehot - theoretical_min_bits:.2f} bits total ({onehot_overhead:.2f}% above theoretical minimum)")
print(f"• Scheme 2 (Binary) overhead: {total_bits_binary - theoretical_min_bits:.2f} bits total ({binary_overhead:.2f}% above theoretical minimum)")

print(f"\nConclusion:")
print(f"• Scheme 2 (Binary) is {reduction_percentage:.2f}% more efficient than Scheme 1 (One-hot)")
print(f"• Binary encoding saves {reduction_bits} bits compared to one-hot encoding")
print(f"• However, even binary encoding uses {binary_overhead:.2f}% more bits than the theoretical minimum")
print(f"• This is because we need to use a whole number of bits per example ({bits_per_example_binary}), while")
print(f"  the theoretical minimum ({entropy:.4f} bits) can be fractional when using variable-length codes")

# Enhanced visualization of efficiency comparison
fig4 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1.2, 1])

# Top: Bar chart comparison
ax1 = fig4.add_subplot(gs[0])
labels = ['One-Hot\nEncoding', 'Binary\nEncoding', 'Theoretical\nMinimum']
values = [total_bits_onehot, total_bits_binary, theoretical_min_bits]
bars = ax1.bar(labels, values, color=['#ff9999', '#66b3ff', '#99ff99'])
ax1.set_title('Storage Requirements Comparison', fontsize=14)
ax1.set_ylabel('Total Bits')
ax1.grid(axis='y', alpha=0.3)

# Add text labels with more details
for i, (bar, val, label) in enumerate(zip(bars, values, labels)):
    height = bar.get_height()
    bits_per_example = val / total_examples
    ax1.text(i, height/2, 
            f"{val:.1f} bits total\n{bits_per_example:.2f} bits/example", 
            ha='center', va='center', fontweight='bold')

# Add overhead annotations
ax1.text(0, total_bits_onehot + 5, 
        f"+{onehot_overhead:.1f}% overhead\ncompared to minimum", 
        ha='center', va='bottom', color='red')
ax1.text(1, total_bits_binary + 5, 
        f"+{binary_overhead:.1f}% overhead\ncompared to minimum", 
        ha='center', va='bottom', color='red')

# Add arrow showing the reduction
ax1.annotate(
    f"{reduction_percentage:.1f}% reduction\n({reduction_bits} bits saved)", 
    xy=(0, total_bits_onehot - reduction_bits/2),
    xytext=(1, total_bits_onehot - reduction_bits/2),
    ha='center', va='center',
    arrowprops=dict(arrowstyle='<->', color='green', lw=2),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
)

# Bottom: Detailed explanation
ax2 = fig4.add_subplot(gs[1])
ax2.axis('off')

# Create textual explanation
explanation = [
    f"• One-hot encoding requires {bits_per_example_onehot} bits per example = {total_bits_onehot} bits total",
    f"• Binary encoding requires {bits_per_example_binary} bits per example = {total_bits_binary} bits total",
    f"• The theoretical minimum based on entropy is {entropy:.4f} bits per example = {theoretical_min_bits:.2f} bits total",
    f"• Binary encoding saves {reduction_bits} bits ({reduction_percentage:.2f}%) compared to one-hot encoding",
    f"• Binary encoding still uses {binary_overhead:.2f}% more bits than the theoretical minimum",
    f"• This is because we need whole bits; the theoretical minimum of {entropy:.4f} bits/example",
    f"  could only be achieved with variable-length codes"
]

explanation_text = "\n".join(explanation)
ax2.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightyellow', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig4, "step4_efficiency_comparison.png")

# ==============================
# STEP 5: Analyze Whether Binary Encoding is Lossless
# ==============================
print_section_header("STEP 5: Analyze Whether Binary Encoding is Lossless")

print("A lossless encoding scheme allows perfect reconstruction of the original data without any")
print("information loss. To determine if binary encoding is lossless, we need to check if each")
print("category can be uniquely identified from its binary code.")

print("\nEncoding table comparing both schemes:")
for i, cat in enumerate(categories):
    print(f"Category {cat}:")
    print(f"  • One-hot encoding: {scheme1_onehot[cat]}")
    print(f"  • Binary encoding:  {scheme2_binary[cat]}")

# Check if binary encoding is lossless
binary_codes = list(scheme2_binary.values())
unique_codes = set(tuple(code) for code in binary_codes)
is_lossless = len(unique_codes) == len(categories)

print(f"\nDetailed analysis of binary encoding losslessness:")
print(f"1. Number of unique binary codes: {len(unique_codes)}")
print(f"2. Number of categories: {len(categories)}")
print(f"3. Is every category uniquely represented? {'Yes' if is_lossless else 'No'}")
print(f"\nConclusion: The binary encoding is{' ' if is_lossless else ' not '}lossless.")

if is_lossless:
    print("\nDetailed explanation of why the binary encoding is lossless:")
    print("• Each category has a unique binary code:")
    for cat, code in scheme2_binary.items():
        print(f"  - Category {cat}: {code}")
    print("• There is a one-to-one mapping between categories and codes")
    print("• We can perfectly reconstruct the original category from its binary code")
    print("• No information is lost in the encoding process")

# Theoretical analysis
print(f"\nTheoretical analysis of encoding efficiency:")
print(f"1. Entropy of the distribution: {entropy:.4f} bits per example")
print(f"2. Bits per example in binary encoding: {bits_per_example_binary} bits")
extra_bits = bits_per_example_binary - entropy
print(f"3. Extra bits per example: {bits_per_example_binary} - {entropy:.4f} = {extra_bits:.4f} bits")
print(f"4. Percentage overhead: {binary_overhead:.2f}%")

# Theoretical minimum number of bits needed for fixed-length code
min_bits_needed = math.ceil(math.log2(len(categories)))
print(f"\nMinimum bits needed for fixed-length code:")
print(f"• To represent {len(categories)} distinct categories, we need ceiling(log₂({len(categories)})) = {min_bits_needed} bits")
print(f"• Binary encoding uses {bits_per_example_binary} bits, which equals this theoretical minimum")
print(f"• This is the most efficient fixed-length binary encoding possible for {len(categories)} categories")
print(f"• To approach the entropy limit of {entropy:.4f} bits, we would need variable-length codes")
print(f"  (such as Huffman coding) that assign shorter codes to more frequent categories")

# Enhanced visualization of lossless encoding analysis
fig5 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Top: Encoding mapping diagram
ax1 = fig5.add_subplot(gs[0])
ax1.axis('off')

# Title
ax1.text(0.5, 1.0, "Binary Encoding is Lossless: Perfect One-to-One Mapping", 
        ha='center', va='top', fontsize=14, fontweight='bold')

# Draw a diagram showing mapping between categories and binary codes
y_positions = [0.7, 0.5, 0.3]
for i, cat in enumerate(categories):
    # Draw category node on left
    ax1.text(0.2, y_positions[i], f"Category {cat}", ha='center', va='center', fontsize=12,
            bbox=dict(facecolor=sns.color_palette("pastel", 3)[i], boxstyle='round', alpha=0.7))
    
    # Draw binary code on right
    ax1.text(0.8, y_positions[i], f"Binary: {scheme2_binary[cat]}", ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='lightblue', boxstyle='round', alpha=0.7))
    
    # Draw bidirectional arrows
    ax1.annotate('', xy=(0.65, y_positions[i]), xytext=(0.35, y_positions[i]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax1.text(0.5, y_positions[i]+0.03, "Encode", ha='center', va='bottom', fontsize=10, color='blue')
    
    ax1.annotate('', xy=(0.35, y_positions[i]-0.05), xytext=(0.65, y_positions[i]-0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.text(0.5, y_positions[i]-0.08, "Decode", ha='center', va='top', fontsize=10, color='green')

# Bottom: Detailed theoretical analysis
ax2 = fig5.add_subplot(gs[1])
ax2.axis('off')

# Create a comprehensive explanation
explanation = [
    "A lossless encoding scheme must satisfy these conditions:",
    "1. Each category must have a unique code (no ambiguity)",
    "2. It must be possible to decode every valid code back to its original category",
    "3. There must be a one-to-one mapping between categories and their codes",
    "",
    f"Binary encoding with {bits_per_example_binary} bits for {len(categories)} categories is lossless because:",
    f"• Each of the {len(categories)} categories has a unique binary code",
    "• We can perfectly reconstruct the original category from the code",
    "• No information is lost during encoding or decoding",
    "",
    "Theoretical analysis:",
    f"• Distribution entropy: {entropy:.4f} bits/example (theoretical information content)",
    f"• Binary encoding: {bits_per_example_binary} bits/example (actual storage used)",
    f"• Overhead: {extra_bits:.4f} bits/example ({binary_overhead:.2f}% more than entropy)",
    f"• For {len(categories)} categories, minimum fixed-length code: ceiling(log₂({len(categories)})) = {min_bits_needed} bits",
    f"• Binary encoding achieves this minimum for fixed-length codes"
]

# Format as a nice multi-paragraph explanation
formatted_explanation = ""
for line in explanation:
    if line == "":
        formatted_explanation += "\n\n"
    else:
        formatted_explanation += line + "\n"

ax2.text(0.5, 0.5, formatted_explanation, ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round,pad=0.7'))

plt.tight_layout()
save_figure(fig5, "step5_lossless_analysis.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this question on encoding schemes and information theory:")

print("\n1. Entropy of the Class Distribution:")
print(f"   • Entropy: {entropy:.4f} bits per example (theoretical minimum)")
print(f"   • Category breakdown: A: {entropy_terms[0]:.4f} bits, B: {entropy_terms[1]:.4f} bits, C: {entropy_terms[2]:.4f} bits")
print(f"   • This represents the theoretical minimum number of bits needed per example")

print("\n2. Scheme 1 (One-hot Encoding):")
print(f"   • Bits per example: {bits_per_example_onehot} bits")
print(f"   • Total storage required: {total_bits_onehot} bits")
print(f"   • Overhead vs. theoretical minimum: {onehot_overhead:.2f}%")

print("\n3. Scheme 2 (Binary Encoding):")
print(f"   • Bits per example: {bits_per_example_binary} bits")
print(f"   • Total storage required: {total_bits_binary} bits")
print(f"   • Overhead vs. theoretical minimum: {binary_overhead:.2f}%")

print("\n4. Efficiency Comparison:")
print(f"   • Binary encoding reduces storage by {reduction_bits} bits ({reduction_percentage:.2f}%)")
print(f"   • Scheme 2 uses {total_bits_binary} bits instead of {total_bits_onehot} bits (Scheme 1)")

print("\n5. Lossless Analysis:")
print(f"   • Binary encoding is {'lossless' if is_lossless else 'lossy'}")
print(f"   • Each category can be uniquely identified from its binary code")
print(f"   • Binary encoding is the most efficient fixed-length code possible for {len(categories)} categories")
print(f"   • To achieve the entropy limit of {entropy:.4f} bits, we would need variable-length codes")

print("\nThis problem demonstrates important principles in information theory and data encoding:")
print("• Entropy represents the theoretical minimum bits needed to encode information")
print("• One-hot encoding is simple but less efficient than binary encoding")
print("• Binary encoding achieves the minimum possible for fixed-length codes")
print("• Lossless encoding ensures perfect reconstruction of the original data")
print("• The efficiency of encoding schemes depends on the underlying data distribution") 