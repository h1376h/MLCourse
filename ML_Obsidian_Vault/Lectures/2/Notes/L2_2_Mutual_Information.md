# Mutual Information

## Definition
Mutual Information (MI) quantifies the amount of information obtained about one random variable through observing another random variable. For discrete random variables X and Y, the mutual information I(X;Y) is defined as:

$$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

where p(x,y) is the joint probability, and p(x) and p(y) are the marginal probabilities.

## Alternative Definitions
Mutual information can also be expressed in terms of entropy:

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

where H(X) and H(Y) are marginal entropies, H(X|Y) is conditional entropy, and H(X,Y) is joint entropy.

## Properties
- Non-negative: I(X;Y) ≥ 0
- Symmetric: I(X;Y) = I(Y;X)
- I(X;Y) = 0 if and only if X and Y are independent
- Related to KL divergence: I(X;Y) = D_KL(p(x,y) || p(x)p(y))
- Data processing inequality: If X → Y → Z forms a Markov chain, then I(X;Z) ≤ I(X;Y)

## Applications in Machine Learning
- Feature selection and dimensionality reduction
- Measuring dependence between variables
- Information bottleneck method
- Clustering and unsupervised learning
- Analyzing neural network representations

## Related Concepts
- [[L2_2_Information_Theory|Information Theory]]
- [[L2_2_Entropy|Entropy]]
- [[L2_2_Cross_Entropy|Cross Entropy]]
- [[L2_2_KL_Divergence|KL Divergence]]
- [[L2_2_Information_Theory_Applications|Information Theory Applications]] 