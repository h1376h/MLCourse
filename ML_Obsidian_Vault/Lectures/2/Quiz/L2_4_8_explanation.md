# Question 8: MLE for Mixture of Normal Distributions

## Problem Statement
You are given a dataset of n independent observations from a mixture of two normal distributions with equal mixing proportions. The probability density function is:

$$f(x|\mu_1, \mu_2, \sigma) = \frac{1}{2} \left( \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu_1)^2}{2\sigma^2}} + \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu_2)^2}{2\sigma^2}} \right)$$

where $\mu_1$ and $\mu_2$ are unknown means and $\sigma$ is known.

### Task
1. Write down the log-likelihood function for this model
2. Derive the score equations for $\mu_1$ and $\mu_2$
3. Discuss the challenges in finding the MLE for this model and propose a solution

## Understanding the Probability Model

A mixture model is a probabilistic model that assumes the observed data comes from multiple underlying distributions. In this case, we have a mixture of two normal distributions with equal mixing proportions (0.5 each). The parameters we need to estimate are the means of these two normal distributions ($\mu_1$ and $\mu_2$), while the standard deviation $\sigma$ is assumed to be known.

Key characteristics of this model include:
- Each observation has a 50% chance of coming from either normal distribution
- The overall distribution can be bimodal when the means are sufficiently separated
- The model has more complex parameter estimation challenges than a single distribution
- Applications include modeling heterogeneous populations and clustering

## Solution

The mixture of normal distributions is a probabilistic model that combines multiple normal distributions with different parameters. It's used to model data that comes from heterogeneous populations and can represent multimodal data distributions.

### Step 1: Formulate the likelihood function
For n independent observations $x_1, x_2, \ldots, x_n$, the likelihood function is:

$$L(\mu_1, \mu_2) = \prod_{i=1}^n f(x_i|\mu_1, \mu_2, \sigma) = \prod_{i=1}^n \frac{1}{2} \left( \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}} \right)$$

### Step 2: Take the logarithm to get the log-likelihood
Taking the natural logarithm, we get the log-likelihood function:

$$\ell(\mu_1, \mu_2) = \sum_{i=1}^n \log \left[ \frac{1}{2} \left( \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}} \right) \right]$$

Simplifying:

$$\ell(\mu_1, \mu_2) = \sum_{i=1}^n \log \left[ \frac{1}{2\sqrt{2\pi}\sigma} \left( e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}} \right) \right]$$

$$\ell(\mu_1, \mu_2) = n \log\left(\frac{1}{2\sqrt{2\pi}\sigma}\right) + \sum_{i=1}^n \log \left( e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}} \right)$$

Since the first term is constant with respect to $\mu_1$ and $\mu_2$, it doesn't affect the maximization, so we can focus on:

$$\ell(\mu_1, \mu_2) \propto \sum_{i=1}^n \log \left( e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}} \right)$$

### Step 3: Find the critical points by taking the derivative
The score equations are obtained by taking the partial derivatives of the log-likelihood with respect to each parameter and setting them to zero.

For $\mu_1$:

$$\frac{\partial \ell}{\partial \mu_1} = \sum_{i=1}^n \frac{\frac{(x_i-\mu_1)}{\sigma^2} e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}}}{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}} = 0$$

For $\mu_2$:

$$\frac{\partial \ell}{\partial \mu_2} = \sum_{i=1}^n \frac{\frac{(x_i-\mu_2)}{\sigma^2} e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}}{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}} = 0$$

### Step 4: Introducing responsibilities for component analysis
These equations can be rewritten in terms of responsibilities or posterior probabilities. If we define:

$$\gamma_{i1} = \frac{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}}}{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}}$$

$$\gamma_{i2} = \frac{e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}}{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}}$$

Then the score equations become:

$$\frac{\partial \ell}{\partial \mu_1} = \frac{1}{\sigma^2} \sum_{i=1}^n \gamma_{i1} (x_i - \mu_1) = 0$$

$$\frac{\partial \ell}{\partial \mu_2} = \frac{1}{\sigma^2} \sum_{i=1}^n \gamma_{i2} (x_i - \mu_2) = 0$$

### Step 5: Solve for the MLE estimate
Solving for $\mu_1$ and $\mu_2$:

$$\mu_1 = \frac{\sum_{i=1}^n \gamma_{i1} x_i}{\sum_{i=1}^n \gamma_{i1}}$$

$$\mu_2 = \frac{\sum_{i=1}^n \gamma_{i2} x_i}{\sum_{i=1}^n \gamma_{i2}}$$

### Step 6: Challenges in direct optimization
Finding the MLE for a mixture model presents several challenges:

1. **No Closed-Form Solution**: The score equations don't have a direct closed-form solution because the responsibilities $\gamma_{i1}$ and $\gamma_{i2}$ depend on the values of $\mu_1$ and $\mu_2$ themselves, creating a circular dependency.

2. **Multiple Modes in the Likelihood Surface**: The likelihood function for mixture models typically has multiple modes. In particular, there's symmetry in the parameter space: swapping $\mu_1$ and $\mu_2$ gives the same likelihood value (known as the "label switching" problem).

3. **Likelihood Degeneracy**: If $\mu_1 = \mu_2$, the model reduces to a single normal distribution, which can be a local maximum of the likelihood but not the global maximum for truly bimodal data.

4. **Convergence Issues**: Direct optimization of the likelihood can be challenging due to these properties, leading to convergence issues or suboptimal solutions.

### Step 7: Propose a solution using the Expectation-Maximization (EM) algorithm
The standard approach for mixture model estimation is the Expectation-Maximization (EM) algorithm, which iteratively computes the responsibilities (E-step) and updates the parameters (M-step):

1. **Initialize**: Start with initial values for $\mu_1$ and $\mu_2$.
   
2. **E-step**: Calculate the responsibilities for each data point:
   $$\gamma_{i1} = \frac{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}}}{e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} + e^{-\frac{(x_i-\mu_2)^2}{2\sigma^2}}}$$
   $$\gamma_{i2} = 1 - \gamma_{i1}$$

3. **M-step**: Update the parameters:
   $$\mu_1 = \frac{\sum_{i=1}^n \gamma_{i1} x_i}{\sum_{i=1}^n \gamma_{i1}}$$
   $$\mu_2 = \frac{\sum_{i=1}^n \gamma_{i2} x_i}{\sum_{i=1}^n \gamma_{i2}}$$

4. **Iterate**: Repeat steps 2 and 3 until convergence.

This approach guarantees monotonic increase in the likelihood and typically converges to a local maximum. To address the issue of multiple local maxima, multiple random initializations can be used, and the solution with the highest likelihood is selected.

## Visual Explanations

### Mixture Distribution for Different Parameter Values
![Mixture PDFs](Images/L2_4_Quiz_8/mixture_pdfs.png)

This figure shows how the shape of the mixture distribution changes with different mean parameters and illustrates the bimodal nature of the distribution when means are well-separated.

### Likelihood Surface
![Likelihood Surface](Images/L2_4_Quiz_8/likelihood_surface.png)

This visualization of the log-likelihood function in the parameter space shows the multiple modes and symmetry in the likelihood surface and highlights the MLE estimates found by optimization.

### MLE Fit to Data
![MLE Fit](Images/L2_4_Quiz_8/mle_fit.png)

This figure shows how well the MLE estimates fit the observed data, displays both the mixture distribution and its component distributions, and demonstrates how the estimated means align with the modes in the data.

### Label Switching Problem
![Label Switching](Images/L2_4_Quiz_8/label_switching.png)

This visualization illustrates the symmetry in the parameter space, shows how swapping $\mu_1$ and $\mu_2$ yields the same likelihood, and demonstrates why initialization matters in finding the correct solution.

### EM Algorithm Convergence
![EM Algorithm](Images/L2_4_Quiz_8/em_algorithm.png)

This figure shows how the EM algorithm iteratively improves the parameter estimates, demonstrates convergence to the correct solution even from suboptimal initial values, and illustrates the monotonic improvement in fit quality.

## Key Insights

### Likelihood Structure
- The log-likelihood for mixture models is complex with multiple modes
- Direct optimization can be challenging and sensitive to initialization
- Symmetry in the parameter space creates identifiability issues

### Score Equations
- The score equations have an intuitive interpretation in terms of weighted averages
- Each mean parameter is a weighted average of the data, with weights proportional to the component responsibilities
- These equations lead naturally to the EM algorithm

### EM Algorithm Benefits
- Provides a stable iterative approach to finding the MLE
- Guarantees monotonic improvement in the likelihood
- Can be easily extended to more complex mixture models

## Conclusion

Maximum likelihood estimation for mixture models presents several challenges:
- The score equations don't have a direct closed-form solution
- The likelihood surface has multiple modes due to label switching
- Direct optimization can be unstable or converge to suboptimal solutions

The Expectation-Maximization (EM) algorithm provides an elegant solution to these challenges by iteratively:
1. Computing the responsibilities of each component for each data point (E-step)
2. Updating the parameters based on these responsibilities (M-step)

This approach is not only theoretically sound but also practical and widely used for mixture model estimation in various applications, including clustering, density estimation, and classification problems. 