# Independence in Probability

Independence is a fundamental concept in probability theory that describes situations where the occurrence of one event does not affect the probability of another event.

## Independent Events

### Definition
- Events $A$ and $B$ are independent if and only if $P(A \cap B) = P(A) \times P(B)$
- Intuitively, the knowledge that one event occurred doesn't change the probability of the other event
- Alternative definition: $P(A|B) = P(A)$ or $P(B|A) = P(B)$

### Properties
- If $A$ and $B$ are independent, then:
  - $A$ and $B^c$ are independent
  - $A^c$ and $B$ are independent
  - $A^c$ and $B^c$ are independent
- Independence is not the same as mutual exclusivity
  - Mutually exclusive events ($P(A \cap B) = 0$) with non-zero probabilities are dependent, not independent
- Independence is symmetric: If $A$ is independent of $B$, then $B$ is independent of $A$

### Pairwise vs. Mutual Independence
- **Pairwise Independence**: Each pair of events is independent
  $$P(A_i \cap A_j) = P(A_i)P(A_j) \text{ for all } i \neq j$$
- **Mutual Independence**: For $n$ events $A_1, A_2, \ldots, A_n$, we require:
  $$P(A_1 \cap A_2 \cap \ldots \cap A_k) = P(A_1) \times P(A_2) \times \ldots \times P(A_k)$$
  for all subsets $\{A_1, A_2, \ldots, A_k\}$ where $k \leq n$
- Pairwise independence does not imply mutual independence
- Mutual independence implies pairwise independence

## Independent Random Variables

### Definition
- Random variables $X$ and $Y$ are independent if any events defined in terms of $X$ are independent of any events defined in terms of $Y$
- Formally: For any sets $A$ and $B$, $P(X \in A, Y \in B) = P(X \in A) \times P(Y \in B)$

### For Discrete Random Variables
- $X$ and $Y$ are independent if and only if:
  $$P(X = x, Y = y) = P(X = x) \times P(Y = y) \text{ for all } x, y$$
- The joint PMF equals the product of the marginal PMFs
- For multiple discrete random variables:
  $$P(X_1 = x_1, \ldots, X_n = x_n) = \prod_{i=1}^n P(X_i = x_i)$$

### For Continuous Random Variables
- $X$ and $Y$ are independent if and only if:
  $$f_{X,Y}(x, y) = f_X(x) \times f_Y(y) \text{ for all } x, y$$
- The joint PDF equals the product of the marginal PDFs
- For multiple continuous random variables:
  $$f_{X_1,\ldots,X_n}(x_1,\ldots,x_n) = \prod_{i=1}^n f_{X_i}(x_i)$$

### For Mixed Random Variables
- For a discrete random variable $X$ and a continuous random variable $Y$:
  $$P(X = x, Y \in A) = P(X = x) \times P(Y \in A) \text{ for all } x, A$$

## Properties of Independent Random Variables

### Expectation
- $E[XY] = E[X]E[Y]$
- $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ for any functions $g$ and $h$
- For multiple independent random variables:
  $$E\left[\prod_{i=1}^n X_i\right] = \prod_{i=1}^n E[X_i]$$

### Variance
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
- For $n$ independent random variables:
  $$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i)$$
- For linear combinations:
  $$\text{Var}\left(\sum_{i=1}^n a_iX_i\right) = \sum_{i=1}^n a_i^2 \text{Var}(X_i)$$

### Covariance and Correlation
- $\text{Cov}(X, Y) = 0$
- $\text{Corr}(X, Y) = 0$
- Note: Zero covariance/correlation implies independence only for jointly normally distributed random variables
- For multiple independent random variables:
  $$\text{Cov}(X_i, X_j) = 0 \text{ for all } i \neq j$$

### Moment Generating Functions
- For independent $X$ and $Y$:
  $$M_{X+Y}(t) = M_X(t) \times M_Y(t)$$
- For multiple independent random variables:
  $$M_{\sum_{i=1}^n X_i}(t) = \prod_{i=1}^n M_{X_i}(t)$$

## Testing for Independence

### Statistical Tests
- **Chi-Square Test of Independence**:
  $$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
  where $O_{ij}$ are observed frequencies and $E_{ij}$ are expected frequencies
- **Correlation Tests**:
  - Pearson correlation: $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}$
  - Spearman's rank correlation
  - Kendall's tau
- **Mutual Information**:
  $$I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

### Graphical Methods
- Scatter plots
- Contingency tables
- Heatmaps of joint distributions
- Q-Q plots for comparing distributions

## Independence in Machine Learning

### Feature Independence
- **Naive Bayes Classifiers**:
  $$P(Y|X_1,\ldots,X_n) \propto P(Y) \prod_{i=1}^n P(X_i|Y)$$
- **Independent Component Analysis (ICA)**:
  $$\mathbf{X} = \mathbf{A}\mathbf{S}$$
  where $\mathbf{S}$ contains independent sources
- **Orthogonality in Feature Engineering**:
  $$\mathbf{X}^T\mathbf{X} = \mathbf{I}$$

### Model Assumptions
- **I.I.D. (Independent and Identically Distributed) samples**:
  $$P(\mathbf{X}_1,\ldots,\mathbf{X}_n) = \prod_{i=1}^n P(\mathbf{X}_i)$$
- Cross-validation assumes independence between training and testing sets
- Time series models deal with dependent observations

### Sampling Methods
- Random sampling relies on independence
- Bootstrap methods
- Markov Chain Monte Carlo (MCMC) for dependent samples

## Independence vs. Conditional Independence

### Conditional Independence
- $X$ and $Y$ are conditionally independent given $Z$ if:
  $$P(X = x, Y = y | Z = z) = P(X = x | Z = z) \times P(Y = y | Z = z)$$
  for all $x, y, z$
- Variables that are dependent can become independent when conditioned on another variable
- For continuous random variables:
  $$f_{X,Y|Z}(x,y|z) = f_{X|Z}(x|z) \times f_{Y|Z}(y|z)$$

### Graphical Models
- **Bayesian Networks**: Represent conditional independence relationships
- **Markov Random Fields**: Encode pairwise independence
- **d-separation**: Graphical criterion for conditional independence

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation concepts
- [[L2_1_Joint_Probability|Joint Probability]]: Probability of multiple events
- [[L2_1_Conditional_Probability|Conditional Probability]]: Dependence between events
- [[L2_1_Multivariate_Analysis|Multivariate Analysis]]: Multiple random variables
- [[L2_1_Graphical_Models|Graphical Models]]: Representing independence structures 