# Cross Entropy

## Definition
Cross entropy is a measure from information theory that quantifies the difference between two probability distributions. For discrete probability distributions P and Q, the cross entropy H(P,Q) is defined as:

$$H(P,Q) = -\sum_{i} p(x_i) \log q(x_i)$$

where p(x_i) and q(x_i) are the probabilities of event x_i under distributions P and Q respectively.

## Properties
- Non-negative: H(P,Q) ≥ 0
- Cross entropy is minimized when P = Q, in which case it equals the entropy of P
- Not symmetric: H(P,Q) ≠ H(Q,P) in general
- Related to KL divergence: H(P,Q) = H(P) + D_KL(P||Q)

## Intuition
Cross entropy measures the average number of bits needed to encode data from distribution P when using an optimal code for distribution Q.

## Applications in Machine Learning
- Loss function for classification problems (cross-entropy loss)
- Training neural networks with softmax outputs
- Evaluating probabilistic models
- Model selection and comparison

## Related Concepts
- [[L2_2_Information_Theory|Information Theory]]
- [[L2_2_Entropy|Entropy]]
- [[L2_2_KL_Divergence|KL Divergence]]
- [[L2_2_Mutual_Information|Mutual Information]] 