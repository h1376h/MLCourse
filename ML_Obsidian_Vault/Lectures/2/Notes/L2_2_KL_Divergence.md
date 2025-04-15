# Kullback-Leibler Divergence

## Definition
The Kullback-Leibler (KL) divergence, also known as relative entropy, measures how one probability distribution diverges from a second reference probability distribution. For discrete probability distributions P and Q, the KL divergence D_KL(P||Q) is defined as:

$$D_{KL}(P||Q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)}$$

## Properties
- Non-negative: D_KL(P||Q) ≥ 0 (Gibbs' inequality)
- D_KL(P||Q) = 0 if and only if P = Q
- Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P) in general
- Not a true metric (doesn't satisfy triangle inequality)
- Relation to cross entropy: D_KL(P||Q) = H(P,Q) - H(P)

## Intuition
KL divergence measures the extra bits required to encode samples from P using a code optimized for Q, rather than using a code optimized for P.

## Applications in Machine Learning
- Variational inference (VI) and variational autoencoders (VAEs)
- Model selection and comparison
- Regularization term in various algorithms
- Information bottleneck methods
- Measuring overfitting in probabilistic models

## Related Concepts
- [[L2_2_Information_Theory|Information Theory]]
- [[L2_2_Entropy|Entropy]]
- [[L2_2_Cross_Entropy|Cross Entropy]]
- [[L2_2_Mutual_Information|Mutual Information]]
- [[L2_2_Information_Theory_Applications|Information Theory Applications]] 