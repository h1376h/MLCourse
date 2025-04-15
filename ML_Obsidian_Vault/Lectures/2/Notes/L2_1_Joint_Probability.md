# Joint Probability

Joint probability deals with the probability of multiple events occurring simultaneously or multiple random variables taking specific values together.

## Joint Probability of Events

### Definition
- The joint probability of events $A$ and $B$, denoted $P(A \cap B)$ or $P(A, B)$, is the probability that both events occur simultaneously
- For $n$ events: $P(A_1 \cap A_2 \cap \ldots \cap A_n)$ is the probability that all $n$ events occur simultaneously

### Properties
- $0 \leq P(A \cap B) \leq \min(P(A), P(B))$
- $P(A \cap B) \leq P(A)$ and $P(A \cap B) \leq P(B)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Chain Rule**: $P(A_1, \ldots, A_n) = P(A_1)P(A_2|A_1)\cdots P(A_n|A_1,\ldots,A_{n-1})$

## Joint Probability Distributions

### Joint Probability Mass Function (Joint PMF)
- For discrete random variables $X$ and $Y$
- $P(X = x, Y = y)$ gives the probability that $X = x$ and $Y = y$ occur simultaneously
- Properties:
  - $0 \leq P(X = x, Y = y) \leq 1$
  - $\sum_x\sum_y P(X = x, Y = y) = 1$ (sum over all possible combinations)
  - $P((X,Y) \in A) = \sum_{(x,y) \in A} P(X = x, Y = y)$

### Joint Probability Density Function (Joint PDF)
- For continuous random variables $X$ and $Y$
- $f(x, y)$ represents the probability density at the point $(x, y)$
- Properties:
  - $f(x, y) \geq 0$
  - $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(x, y) dx dy = 1$ (integral over the entire range)
  - $P((X, Y) \in R) = \iint_R f(x, y) dx dy$ (probability that $(X, Y)$ falls in region $R$)
  - $f(x, y) = \frac{\partial^2 F(x, y)}{\partial x \partial y}$ where $F$ is the joint CDF

### Joint Cumulative Distribution Function (Joint CDF)
- $F(x, y) = P(X \leq x, Y \leq y)$
- Properties:
  - Monotonic in each argument
  - Right-continuous in each argument
  - $\lim_{x \to -\infty} F(x, y) = 0$ and $\lim_{y \to -\infty} F(x, y) = 0$
  - $\lim_{x \to \infty, y \to \infty} F(x, y) = 1$

### Visualizing Joint Distributions
- **Discrete**: Tables, heatmaps, 3D histograms
- **Continuous**: Contour plots, 3D surface plots
- **Marginal Plots**: Histograms along each axis
- **Conditional Plots**: Slices of the joint distribution

## Marginal Distributions

### Marginal Probability Mass Function
- Obtained by summing the joint PMF over all values of one variable
- For discrete variables:
  - $P(X = x) = \sum_y P(X = x, Y = y)$
  - $P(Y = y) = \sum_x P(X = x, Y = y)$

### Marginal Probability Density Function
- Obtained by integrating the joint PDF over all values of one variable
- For continuous variables:
  - $f_X(x) = \int_{-\infty}^{\infty} f(x, y) dy$
  - $f_Y(y) = \int_{-\infty}^{\infty} f(x, y) dx$

## Independence

### Definition
- Random variables $X$ and $Y$ are independent if and only if:
  - $P(X = x, Y = y) = P(X = x) \times P(Y = y)$ for all $x, y$ (discrete case)
  - $f(x, y) = f_X(x) \times f_Y(y)$ for all $x, y$ (continuous case)
  - $F(x, y) = F_X(x) \times F_Y(y)$ for all $x, y$ (CDF definition)

### Properties of Independent Random Variables
- $\text{Cov}(X, Y) = 0$ (zero covariance)
- $E[XY] = E[X]E[Y]$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
- $M_{X+Y}(t) = M_X(t)M_Y(t)$ (product of moment generating functions)
- $f_{X+Y}(z) = \int f_X(x)f_Y(z-x)dx$ (convolution of PDFs)

## Covariance and Correlation

### Covariance
- Measures the joint variability of two random variables
- $\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$
- Properties:
  - $\text{Cov}(X, X) = \text{Var}(X)$
  - $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
  - $\text{Cov}(aX + b, cY + d) = ac\text{Cov}(X, Y)$
  - $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$

### Correlation
- Normalized measure of the linear relationship between two random variables
- $\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X\sigma_Y}$ (where $\sigma_X$ and $\sigma_Y$ are standard deviations)
- Properties:
  - $-1 \leq \text{Corr}(X, Y) \leq 1$
  - $\text{Corr}(X, Y) = 1$ if $Y = aX + b$ with $a > 0$
  - $\text{Corr}(X, Y) = -1$ if $Y = aX + b$ with $a < 0$
  - $\text{Corr}(X, Y) = 0$ indicates no linear relationship

## Applications in Machine Learning

### Multivariate Modeling
- Predicting multiple target variables simultaneously
- Capturing dependencies between features
- **Multivariate Regression**: $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$
- **Multivariate Time Series**: Modeling dependencies across time

### Latent Variable Models
- **Hidden Markov Models**: $P(\mathbf{X}, \mathbf{Z}) = P(\mathbf{Z}_1)\prod_{t=2}^T P(\mathbf{Z}_t|\mathbf{Z}_{t-1})\prod_{t=1}^T P(\mathbf{X}_t|\mathbf{Z}_t)$
- **Gaussian Mixture Models**: $f(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\mathbf{\mu}_k, \mathbf{\Sigma}_k)$
- **Factor Analysis**: $\mathbf{X} = \mathbf{W}\mathbf{Z} + \mathbf{\mu} + \mathbf{\epsilon}$

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: $\mathbf{Y} = \mathbf{U}^T\mathbf{X}$
- **t-SNE**: Preserves local structure in high dimensions
- **Canonical Correlation Analysis**: Finds linear combinations with maximum correlation

## Related Topics
- [[L2_1_Conditional_Probability|Conditional Probability]]: Probability of events given other events
- [[L2_1_Independence|Independence]]: When events don't influence each other
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation concepts
- [[L2_1_Joint_Distributions|Joint Distributions]]: More detailed discussion of joint distributions
- [[L2_1_Joint_Distributions_Examples|Joint Distributions Examples]]: Practical examples
- [[L2_1_Covariance_Correlation|Covariance and Correlation]]: Further exploration of relationships between variables 