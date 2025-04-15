# Information Theory

## Overview
Information theory provides the mathematical framework for quantifying information content, uncertainty, and the efficiency of information transfer. It serves as a foundation for many machine learning concepts and algorithms.

## Key Concepts

### Information Content
- Measurement of surprise or uncertainty reduction

### Entropy
- **Definition**: Entropy $H(X)$ measures the uncertainty or information content of a random variable $X$
- For a discrete random variable with PMF $p(x)$:
  $$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$
- For a continuous random variable with PDF $f(x)$:
  $$H(X) = -\int_{\mathcal{X}} f(x) \log f(x) dx$$

#### Properties
- **Non-negativity**: $H(X) \geq 0$
- **Maximum Entropy**: For a discrete random variable with $n$ outcomes, maximum entropy is $\log n$ (achieved by uniform distribution)
- **Additivity**: For independent random variables $X$ and $Y$:
  $$H(X,Y) = H(X) + H(Y)$$

### Conditional Entropy
- **Definition**: Conditional Entropy $H(X|Y)$ measures the uncertainty in $X$ given knowledge of $Y$
- For discrete random variables:
  $$H(X|Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x|y)$$

#### Properties
- **Chain Rule**: $H(X,Y) = H(X) + H(Y|X)$
- **Reduction**: $H(X|Y) \leq H(X)$

### Cross-Entropy
- **Definition**: Cross-Entropy $H(P,Q)$ measures the average number of bits needed to encode data from distribution $P$ using distribution $Q$
- For discrete distributions:
  $$H(P,Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)$$
- For continuous distributions:
  $$H(P,Q) = -\int_{\mathcal{X}} p(x) \log q(x) dx$$

#### Relationship to KL Divergence
$$H(P,Q) = H(P) + D_{KL}(P||Q)$$

### KL Divergence
- **Definition**: KL Divergence $D_{KL}(P||Q)$ measures how one probability distribution $P$ differs from another $Q$
- For discrete distributions:
  $$D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$
- For continuous distributions:
  $$D_{KL}(P||Q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} dx$$

#### Properties
- **Non-negativity**: $D_{KL}(P||Q) \geq 0$ with equality iff $P = Q$
- **Asymmetry**: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$
- **Not a Metric**: Does not satisfy triangle inequality

### Mutual Information
- **Definition**: Mutual Information $I(X;Y)$ measures the amount of information shared between two random variables
- Can be expressed in terms of entropy:
  $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$
- In terms of KL divergence:
  $$I(X;Y) = D_{KL}(P_{X,Y}||P_X \times P_Y)$$

#### Properties
- **Non-negativity**: $I(X;Y) \geq 0$
- **Symmetry**: $I(X;Y) = I(Y;X)$
- **Independence**: $I(X;Y) = 0$ if and only if $X$ and $Y$ are independent

## Applications in ML
- Feature selection and dimensionality reduction
- Model evaluation and comparison
- Optimal coding and compression
- Decision trees and random forests
- Variational inference
- Generative models
- Neural network training
- Information bottleneck method
- Neural network compression
- Bayesian neural networks
- Variational autoencoders
- Reinforcement learning

## Related Concepts
- [[L2_2_Entropy|Entropy]]
- [[L2_2_Cross_Entropy|Cross Entropy]]
- [[L2_2_KL_Divergence|KL Divergence]]
- [[L2_2_Mutual_Information|Mutual Information]]
- [[L2_2_Information_Theory_Applications|Information Theory Applications]]
- [[L2_1_Basic_Probability|Basic Probability]]
- [[L2_1_Conditional_Probability|Conditional Probability]]
- [[L2_1_Independence|Independence]]
- [[L2_1_Examples|Probability Examples]]

## References
- Cover, T.M. and Thomas, J.A. (2006). Elements of Information Theory
- Bishop, C.M. (2006). Pattern Recognition and Machine Learning 