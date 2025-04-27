# Question 29: Encoding Schemes and Information Theory

## Problem Statement
Consider a dataset of 100 examples with three possible categories: A, B, and C. Two encoding schemes are proposed:

**Scheme 1 (One-hot):** 
- A = $[1,0,0]$
- B = $[0,1,0]$
- C = $[0,0,1]$

**Scheme 2 (Binary):**
- A = $[0,0]$
- B = $[0,1]$
- C = $[1,0]$

The dataset contains: 50 instances of A, 30 instances of B, and 20 instances of C.

### Task
1. Calculate the entropy of the class distribution in bits using:
   $$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$

2. How many bits are required to store the entire dataset using Scheme 1?

3. How many bits are required to store the entire dataset using Scheme 2?

4. Which encoding is more efficient, and by how much? Calculate the percentage reduction in bits:
   $$\text{Reduction} = \frac{\text{Bits}_{\text{Scheme 1}} - \text{Bits}_{\text{Scheme 2}}}{\text{Bits}_{\text{Scheme 1}}} \times 100\%$$

5. Is the binary encoding scheme lossless? Explain why or why not.

## Step-by-Step Solution

### Step 1: Calculate the Entropy of the Class Distribution

First, we calculate the entropy of the given distribution, which represents the theoretical minimum bits needed per example.

**Class distribution:**
- Category A: 50 instances (50.0%)
- Category B: 30 instances (30.0%)
- Category C: 20 instances (20.0%)
- Total instances: 100

**Entropy calculation:**
The entropy is calculated as H(X) = -∑ P(x_i) log₂ P(x_i)

- Category A:
  - P(A) = 50/100 = 0.5000
  - -P(A) × log₂(P(A)) = -(0.5000) × log₂(0.5000) = 0.5000 bits
- Category B:
  - P(B) = 30/100 = 0.3000
  - -P(B) × log₂(P(B)) = -(0.3000) × log₂(0.3000) = 0.5211 bits
- Category C:
  - P(C) = 20/100 = 0.2000
  - -P(C) × log₂(P(C)) = -(0.2000) × log₂(0.2000) = 0.4644 bits

Total entropy = 0.5000 + 0.5211 + 0.4644 = **1.4855 bits**

This means the theoretical minimum number of bits needed per example on average is 1.4855 bits. Any encoding using fewer than this many bits would necessarily be lossy.

![Entropy Calculation](../Codes/images/step1_entropy_calculation.png)

### Step 2: Calculate Bits Required for Scheme 1 (One-hot Encoding)

In one-hot encoding, each category is represented by a vector with a 1 in the position corresponding to that category and 0s elsewhere.

**One-hot encoding scheme:**
- Category A: [1, 0, 0]
- Category B: [0, 1, 0]
- Category C: [0, 0, 1]

**Storage requirements:**
1. Each example requires 3 bits to encode (one bit for each possible category)
2. Total bits required: 3 bits/example × 100 examples = 300 bits
3. Breakdown by category:
   - Category A: 50 examples × 3 bits/example = 150 bits (50.0%)
   - Category B: 30 examples × 3 bits/example = 90 bits (30.0%)
   - Category C: 20 examples × 3 bits/example = 60 bits (20.0%)

![One-Hot Encoding](../Codes/images/step2_onehot_encoding.png)

### Step 3: Calculate Bits Required for Scheme 2 (Binary Encoding)

Binary encoding uses a more compact representation with fewer bits per example.

**Binary encoding scheme:**
- Category A: [0, 0]
- Category B: [0, 1]
- Category C: [1, 0]

**Storage requirements:**
1. Each example requires 2 bits to encode
2. Total bits required: 2 bits/example × 100 examples = 200 bits
3. Breakdown by category:
   - Category A: 50 examples × 2 bits/example = 100 bits (50.0%)
   - Category B: 30 examples × 2 bits/example = 60 bits (30.0%)
   - Category C: 20 examples × 2 bits/example = 40 bits (20.0%)

![Binary Encoding](../Codes/images/step3_binary_encoding.png)

### Step 4: Compare the Efficiency of Both Encoding Schemes

Now we compare the efficiency of both encoding schemes:

**Comparison:**
- Scheme 1 (One-hot): 3 bits/example × 100 examples = 300 bits
- Scheme 2 (Binary): 2 bits/example × 100 examples = 200 bits

**Calculation of savings:**
1. Absolute reduction in bits: 300 - 200 = 100 bits
2. Percentage reduction: (100 / 300) × 100% = 33.33%

**Comparison with theoretical minimum:**
- Theoretical minimum: 1.4855 bits/example × 100 examples = 148.55 bits
- Scheme 1 (One-hot) overhead: 101.95% above theoretical minimum
- Scheme 2 (Binary) overhead: 34.63% above theoretical minimum

**Conclusion:**
Scheme 2 (Binary) is 33.33% more efficient than Scheme 1 (One-hot), but still uses more bits than the theoretical minimum based on entropy.

![Efficiency Comparison](../Codes/images/step4_efficiency_comparison.png)

### Step 5: Analyze Whether Binary Encoding is Lossless

To determine if binary encoding is lossless, we check if each category can be uniquely identified from its binary code.

**Encoding table:**
- Category A:
  - One-hot encoding: [1, 0, 0]
  - Binary encoding:  [0, 0]
- Category B:
  - One-hot encoding: [0, 1, 0]
  - Binary encoding:  [0, 1]
- Category C:
  - One-hot encoding: [0, 0, 1]
  - Binary encoding:  [1, 0]

**Analysis:**
1. Number of unique binary codes: 3
2. Number of categories: 3
3. Is every category uniquely represented? Yes

**Conclusion:**
The binary encoding is lossless because:
- Each category has a unique binary code
- There is a one-to-one mapping between categories and codes
- We can perfectly reconstruct the original category from its binary code
- No information is lost in the encoding process

**Theoretical analysis:**
1. Entropy of the distribution: 1.4855 bits per example
2. Bits per example in binary encoding: 2 bits
3. Extra bits per example: 2 - 1.4855 = 0.5145 bits
4. Percentage overhead: 34.63%

The minimum bits needed to represent 3 categories is log₂(3) = 1.58 bits, which rounds up to 2 bits for a fixed-length code. So the binary encoding uses the theoretical minimum possible for a fixed-length binary code.

![Lossless Analysis](../Codes/images/step5_lossless_analysis.png)

## Summary and Insights

### Key Findings
1. **Entropy of the Class Distribution:**
   - Entropy: 1.4855 bits per example (theoretical minimum)
   - Class breakdown: Category A: 0.5000 bits, B: 0.5211 bits, C: 0.4644 bits

2. **Scheme 1 (One-hot Encoding):**
   - Bits per example: 3 bits
   - Total storage required: 300 bits
   - Overhead vs. theoretical minimum: 101.95%

3. **Scheme 2 (Binary Encoding):**
   - Bits per example: 2 bits
   - Total storage required: 200 bits
   - Overhead vs. theoretical minimum: 34.63%

4. **Efficiency Comparison:**
   - Binary encoding reduces storage by 100 bits (33.33%)
   - Scheme 2 uses 200 bits instead of 300 bits (Scheme 1)

5. **Lossless Analysis:**
   - Binary encoding is lossless
   - Each category can be uniquely identified from its binary code
   - Binary encoding uses 34.63% more bits than the theoretical minimum
   - No practical encoding can use fewer than 2 bits per example for 3 categories

### Theoretical Foundations
- **Entropy**: Measures the average information content or uncertainty in a probability distribution. It represents the theoretical minimum number of bits needed to encode information.
- **Information Theory**: Provides a mathematical framework for quantifying information and determining the most efficient encoding schemes.
- **Lossless vs. Lossy Encoding**: A lossless encoding allows perfect reconstruction of the original data, while a lossy encoding sacrifices some information for better compression.

### Practical Applications
- **Data Compression**: Efficient encoding schemes can significantly reduce storage requirements and computational costs in machine learning models.
- **Feature Engineering**: Choosing the right encoding scheme can impact model performance and efficiency.
- **Memory Optimization**: In resource-constrained environments (e.g., edge devices), optimized encodings can be crucial.

### Common Pitfalls
- **Overlooking Entropy**: Not considering the theoretical limits can lead to inefficient design choices.
- **Fixed vs. Variable Length Codes**: Fixed-length codes are simpler but may be less efficient than variable-length codes like Huffman coding.
- **Ignoring Distribution**: The efficiency of encoding schemes depends on the underlying data distribution.

### Conclusions
This problem demonstrates that binary encoding is both more efficient and still lossless compared to one-hot encoding. While it doesn't reach the theoretical minimum bits required by entropy, it is the most efficient fixed-length binary code possible for representing three categories. This illustrates the important trade-off between simplicity (fixed-length codes) and efficiency (approaching the entropy limit) in information theory and machine learning. 