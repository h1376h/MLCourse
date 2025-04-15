# Entropy

## Definition
Entropy is a fundamental concept in information theory that quantifies the amount of uncertainty or randomness in a random variable. For a discrete random variable X, entropy H(X) is defined as:

$$H(X) = -\sum_{i} p(x_i) \log p(x_i)$$

where p(x_i) is the probability of event x_i.

## Properties
- Non-negative: H(X) ≥ 0
- Maximum entropy occurs with uniform distribution
- Entropy is additive for independent random variables
- Entropy is concave with respect to the probability distribution

## Information Content
The individual term -log p(x_i) represents the information content or "surprise" associated with observing outcome x_i.

## Examples
- A fair coin toss has entropy of 1 bit
- A biased coin with p(heads) = 0.9, p(tails) = 0.1 has entropy of approximately 0.47 bits
- Entropy of English text is approximately 1.1 bits per character

## Applications in ML
- Decision trees use entropy for optimal splitting criteria
- Maximum entropy models for classification
- Foundation for defining loss functions

## Related Concepts
- [[L2_2_Information_Theory|Information Theory]]
- [[L2_2_Cross_Entropy|Cross Entropy]]
- [[L2_2_KL_Divergence|KL Divergence]]
- [[L2_2_Mutual_Information|Mutual Information]] 