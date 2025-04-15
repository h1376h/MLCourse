# Question 9: MLE for Geometric Distribution

## Problem Statement
A data scientist is studying the number of attempts needed before the first success in a series of independent trials. The data follows a geometric distribution with parameter $p$ (probability of success on each trial). The probability mass function is:

$$f(x|p) = p(1-p)^{x-1} \text{ for } x = 1, 2, 3, \ldots$$

The scientist observes the following number of attempts until first success in 8 independent experiments: 3, 1, 4, 2, 5, 2, 1, 3.

### Task
1. Derive the maximum likelihood estimator $\hat{p}_{MLE}$ for the parameter $p$
2. Calculate the numerical value of $\hat{p}_{MLE}$ for the given data
3. Calculate the expected number of trials until first success using your MLE estimate
4. Find the variance of the number of trials based on your MLE estimate

## Understanding the Probability Model

The geometric distribution models the number of trials needed to achieve the first success in a sequence of Bernoulli trials. Each Bernoulli trial has probability $p$ of success and probability $1-p$ of failure. The geometric distribution has a single parameter $p$ that determines its shape.

## Solution

The geometric distribution is a discrete probability distribution that models the number of trials needed until the first success occurs in a sequence of independent Bernoulli trials. It has a single parameter $p$ that represents the probability of success on each trial.

### Step 1: Formulate the likelihood function
For $n$ independent observations $x_1, x_2, \ldots, x_n$, the likelihood function is:

$$L(p) = \prod_{i=1}^n f(x_i|p) = \prod_{i=1}^n p(1-p)^{x_i-1} = p^n \prod_{i=1}^n (1-p)^{x_i-1} = p^n (1-p)^{\sum_{i=1}^n (x_i-1)}$$

### Step 2: Take the logarithm to get the log-likelihood
Taking the natural logarithm, we get the log-likelihood function:

$$\ell(p) = \log L(p) = n \log p + \sum_{i=1}^n (x_i-1) \log(1-p)$$

### Step 3: Find the critical points by taking the derivative
To find the maximum, we take the derivative with respect to $p$ and set it to zero:

$$\frac{d\ell}{dp} = \frac{n}{p} - \frac{\sum_{i=1}^n (x_i-1)}{1-p} = 0$$

### Step 4: Solve for the MLE estimate
Solving for $p$:

$$\frac{n}{p} = \frac{\sum_{i=1}^n (x_i-1)}{1-p}$$

$$n(1-p) = p\sum_{i=1}^n (x_i-1)$$

$$n - np = p\sum_{i=1}^n (x_i-1)$$

$$n = np + p\sum_{i=1}^n (x_i-1)$$

$$n = p(n + \sum_{i=1}^n (x_i-1))$$

$$n = p(n + \sum_{i=1}^n x_i - n)$$

$$n = p\sum_{i=1}^n x_i$$

$$p = \frac{n}{\sum_{i=1}^n x_i}$$

Therefore, the MLE is:

$$\hat{p}_{MLE} = \frac{n}{\sum_{i=1}^n X_i} = \frac{1}{\bar{X}}$$

where $\bar{X}$ is the sample mean.

### Step 5: Verify it's a maximum
The second derivative of the log-likelihood is:

$$\frac{d^2\ell}{dp^2} = -\frac{n}{p^2} - \frac{\sum_{i=1}^n (x_i-1)}{(1-p)^2} < 0$$

Since the second derivative is negative for all $p \in (0, 1)$, we confirm that our critical point is indeed a maximum.

### Step 6: Calculate the numerical value of the MLE
For the given data [3, 1, 4, 2, 5, 2, 1, 3]:
- $n = 8$
- $\sum_{i=1}^n x_i = 3 + 1 + 4 + 2 + 5 + 2 + 1 + 3 = 21$
- $\bar{X} = 21/8 = 2.625$

Therefore:
$$\hat{p}_{MLE} = \frac{1}{\bar{X}} = \frac{1}{2.625} = 0.3810$$

### Step 7: Calculate the expected number of trials
For a geometric distribution with parameter $p$, the expected value (mean) is:

$$E[X] = \frac{1}{p}$$

Using our MLE estimate:

$$E[X] = \frac{1}{\hat{p}_{MLE}} = \frac{1}{0.3810} = 2.625$$

This is exactly equal to the sample mean (as expected from our MLE derivation).

### Step 8: Calculate the variance of the number of trials
For a geometric distribution with parameter $p$, the variance is:

$$Var[X] = \frac{1-p}{p^2}$$

Using our MLE estimate:

$$Var[X] = \frac{1-\hat{p}_{MLE}}{\hat{p}_{MLE}^2} = \frac{1-0.3810}{0.3810^2} = \frac{0.6190}{0.1452} = 4.264$$

## Visual Explanations

### Geometric Distribution for Different p Values
![Geometric PMFs](Images/L2_4_Quiz_9/geometric_pmfs.png)

This figure shows how the distribution shape changes with different p values and helps understand the role of the success probability parameter.

### Likelihood Surface
![Likelihood Surface](Images/L2_4_Quiz_9/likelihood_surface.png)

This visualization of the log-likelihood function shows the maximum point corresponding to the MLE and demonstrates the concavity of the likelihood function.

### MLE Fit to Data
![MLE Fit](Images/L2_4_Quiz_9/mle_fit.png)

This figure shows how well the MLE estimate fits the observed data by comparing the estimated PMF with the data histogram.

### Expected Value and Variance
![Expected Value and Variance](Images/L2_4_Quiz_9/expected_variance.png)

This visualization shows how the expected value and variance change with different p values and illustrates the relationship between these measures and the success probability.

## Key Insights

### MLE Properties
- The MLE for the geometric distribution is the reciprocal of the sample mean
- It provides a simple and intuitive estimator for the success probability
- The MLE estimate directly relates to the expected number of trials
- The estimator is consistent and asymptotically normal

### Practical Considerations
- The geometric distribution is useful for modeling "time until first success" scenarios
- Common applications include modeling the number of attempts until a sale is made, the number of tosses until a coin lands heads, etc.
- The MLE estimator is easy to compute and interpret

### Limitations
- The MLE can be sensitive to outliers, especially for small sample sizes
- The geometric distribution assumes that the success probability remains constant across all trials
- For small samples, the estimator might have significant bias

## Conclusion

The MLE for the geometric distribution is a powerful and practical estimator that:
- Has a simple formula: $\hat{p}_{MLE} = \frac{1}{\bar{X}}$
- Provides a direct relationship to the expected number of trials
- Enables straightforward calculation of variance and other distributional properties
- Is widely applicable to many real-world scenarios involving "time until first success"
