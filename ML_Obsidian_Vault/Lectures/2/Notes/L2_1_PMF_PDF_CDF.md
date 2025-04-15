# Probability Mass Function (PMF), Probability Density Function (PDF), and Cumulative Distribution Function (CDF)

## Probability Mass Function (PMF)
- Definition: A function that gives the probability that a discrete random variable is exactly equal to some value
- Properties:
  - $P(X = x) \geq 0$ for all $x$
  - $\sum_{x} P(X = x) = 1$
- Example: For a fair six-sided die, PMF is $P(X = x) = \frac{1}{6}$ for $x = 1,2,3,4,5,6$

## Probability Density Function (PDF)
- Definition: A function that describes the relative likelihood for a continuous random variable to take on a given value
- Properties:
  - $f(x) \geq 0$ for all $x$
  - $\int_{-\infty}^{\infty} f(x) dx = 1$
  - $P(a \leq X \leq b) = \int_{a}^{b} f(x) dx$
- Example: Normal distribution PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

## Cumulative Distribution Function (CDF)
- Definition: A function that gives the probability that a random variable is less than or equal to a certain value
- Properties:
  - $F(x) = P(X \leq x)$
  - $F(x)$ is non-decreasing
  - $\lim_{x \to -\infty} F(x) = 0$
  - $\lim_{x \to \infty} F(x) = 1$
- For discrete variables: $F(x) = \sum_{k \leq x} P(X = k)$
- For continuous variables: $F(x) = \int_{-\infty}^{x} f(t) dt$

## Relationships
- PDF is the derivative of CDF for continuous variables: $f(x) = \frac{d}{dx} F(x)$
- CDF is the integral of PDF: $F(x) = \int_{-\infty}^{x} f(t) dt$
- For discrete variables, CDF is the sum of PMF values up to x

## Applications in Machine Learning
- PMF: Used in discrete probability models (e.g., Naive Bayes, Multinomial distributions)
- PDF: Essential for continuous probability models (e.g., Gaussian Mixture Models, Kernel Density Estimation)
- CDF: Used in statistical tests, quantile functions, and generating random variables

## Visualization
- PMF: Bar plots
- PDF: Smooth curves
- CDF: Monotonically increasing curves from 0 to 1

## Examples
See [[L2_1_PMF_PDF_CDF_Examples|PMF, PDF, and CDF Examples]] for detailed examples and applications. 