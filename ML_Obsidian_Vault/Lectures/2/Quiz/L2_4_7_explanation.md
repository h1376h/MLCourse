# Question 7: MLE for Gamma Distribution

## Problem Statement
Consider a dataset of n independent observations from a gamma distribution with known shape parameter $\alpha = 2$ and unknown rate parameter $\beta$. The probability density function is:

$$f(x|\beta) = \frac{\beta^2 x e^{-\beta x}}{\Gamma(2)} \text{ for } x > 0$$

where $\Gamma(2) = 1$ is the gamma function evaluated at 2.

### Task
1. Derive the maximum likelihood estimator $\hat{\beta}_{MLE}$ for $\beta$
2. Calculate the Fisher information $I(\beta)$
3. Construct an approximate 95% confidence interval for $\beta$

## Understanding the Probability Model

The gamma distribution is a flexible continuous probability distribution that is often used to model positive-valued random variables. It has two parameters: a shape parameter $\alpha$ and a rate parameter $\beta$. In this problem, the shape parameter is known to be $\alpha = 2$, and we need to estimate the rate parameter $\beta$.

When $\alpha = 2$, the gamma distribution has the PDF:

$$f(x|\beta) = \beta^2 x e^{-\beta x}$$

This special case is also known as the Erlang distribution with shape parameter 2.

## Solution

The gamma distribution is a continuous probability distribution that models positive-valued random variables. It's commonly used in survival analysis, reliability engineering, and modeling waiting times. It has two parameters: a shape parameter α and a rate parameter β.

### Step 1: Formulate the likelihood function
For n independent observations $x_1, x_2, \ldots, x_n$, the likelihood function is:

$$L(\beta) = \prod_{i=1}^n f(x_i|\beta) = \prod_{i=1}^n \beta^2 x_i e^{-\beta x_i} = \beta^{2n} \left(\prod_{i=1}^n x_i\right) e^{-\beta \sum_{i=1}^n x_i}$$

### Step 2: Take the logarithm to get the log-likelihood
Taking the natural logarithm, we get the log-likelihood function:

$$\ell(\beta) = \log L(\beta) = 2n \log \beta + \sum_{i=1}^n \log x_i - \beta \sum_{i=1}^n x_i$$

Note that the term $\sum_{i=1}^n \log x_i$ doesn't depend on $\beta$ and won't affect the maximization.

### Step 3: Find the critical points by taking the derivative
To find the maximum, we take the derivative with respect to $\beta$ and set it to zero:

$$\frac{d\ell}{d\beta} = \frac{2n}{\beta} - \sum_{i=1}^n x_i = 0$$

### Step 4: Solve for the MLE estimate
Solving for $\beta$:

$$\frac{2n}{\beta} = \sum_{i=1}^n x_i$$

$$\beta = \frac{2n}{\sum_{i=1}^n x_i} = \frac{2}{\bar{x}}$$

where $\bar{x}$ is the sample mean. Therefore, the MLE is:

$$\hat{\beta}_{MLE} = \frac{\alpha}{\bar{x}} = \frac{2}{\bar{x}}$$

### Step 5: Verify it's a maximum
The second derivative of the log-likelihood is:

$$\frac{d^2\ell}{d\beta^2} = -\frac{2n}{\beta^2}$$

Since the second derivative is negative for all $\beta > 0$, we confirm that our critical point is indeed a maximum.

### Step 6: Calculate the Fisher information
The Fisher information is the negative expected value of the second derivative of the log-likelihood:

$$I_1(\beta) = E\left[-\frac{d^2\ell}{d\beta^2}\right] = \frac{2}{\beta^2}$$

For n observations, the Fisher information is:

$$I_n(\beta) = n \cdot I_1(\beta) = \frac{2n}{\beta^2}$$

### Step 7: Construct a 95% confidence interval
For large samples, the MLE $\hat{\beta}_{MLE}$ is approximately normally distributed with mean $\beta$ and variance $\frac{1}{I_n(\beta)}$. 

The standard error of the estimator is:

$$SE(\hat{\beta}_{MLE}) = \sqrt{\frac{1}{I_n(\hat{\beta}_{MLE})}} = \sqrt{\frac{\hat{\beta}_{MLE}^2}{2n}} = \frac{\hat{\beta}_{MLE}}{\sqrt{2n}}$$

A 95% confidence interval for $\beta$ is:

$$\hat{\beta}_{MLE} \pm 1.96 \cdot SE(\hat{\beta}_{MLE}) = \hat{\beta}_{MLE} \pm 1.96 \cdot \frac{\hat{\beta}_{MLE}}{\sqrt{2n}}$$

This can be rewritten as:

$$\hat{\beta}_{MLE} \cdot \left[1 - 1.96 \cdot \frac{1}{\sqrt{2n}}, 1 + 1.96 \cdot \frac{1}{\sqrt{2n}}\right]$$

## Visual Explanations

### Gamma Distribution for Different β Values
![Gamma PDFs](Images/L2_4_Quiz_7/gamma_pdfs.png)

This figure shows how the shape of the gamma distribution changes with different β values and illustrates the effect of the rate parameter on the distribution.

### Likelihood Surface
![Likelihood Surface](Images/L2_4_Quiz_7/likelihood_surface.png)

This visualization of the log-likelihood function shows the maximum point corresponding to the MLE and demonstrates the concavity of the likelihood function.

### MLE Fit to Data
![MLE Fit](Images/L2_4_Quiz_7/mle_fit.png)

This figure shows how well the MLE estimate fits the observed data by comparing the estimated PDF with the data histogram.

### Fisher Information
![Fisher Information](Images/L2_4_Quiz_7/fisher_information.png)

This visualization shows how the Fisher information changes with the parameter value and illustrates that the Fisher information is higher for smaller values of β.

### Confidence Intervals
![Confidence Intervals](Images/L2_4_Quiz_7/confidence_intervals.png)

This figure illustrates confidence intervals for different sample sizes, demonstrates the coverage properties of the constructed intervals, and shows how the precision improves with larger sample sizes.

## Key Insights

### MLE Properties
- The MLE for the gamma distribution rate parameter is a simple function of the sample mean: $\hat{\beta}_{MLE} = \frac{\alpha}{\bar{x}}$
- This estimator is consistent and asymptotically efficient
- For the special case of $\alpha = 2$, the estimator is $\hat{\beta}_{MLE} = \frac{2}{\bar{x}}$

### Fisher Information Analysis
- The Fisher information is inversely proportional to the square of the parameter value
- This implies that smaller values of $\beta$ can be estimated more precisely
- For $\alpha = 2$, the Fisher information for a single observation is $I(\beta) = \frac{2}{\beta^2}$

### Confidence Interval Properties
- The confidence interval width is proportional to the parameter value
- The relative precision (width divided by the parameter value) depends only on the sample size
- As the sample size increases, the confidence interval becomes narrower

## Conclusion

The maximum likelihood estimator for the gamma distribution rate parameter $\beta$ (with known shape parameter $\alpha = 2$) is:
- Simple to compute: $\hat{\beta}_{MLE} = \frac{2}{\bar{x}}$
- Consistent: it converges to the true parameter value as sample size increases
- Asymptotically efficient: it achieves the Cramér-Rao lower bound for large samples

The Fisher information provides a measure of the amount of information the data contains about the parameter, and it allows us to construct confidence intervals for the parameter. For a 95% confidence interval:

$$\hat{\beta}_{MLE} \cdot \left[1 - 1.96 \cdot \frac{1}{\sqrt{2n}}, 1 + 1.96 \cdot \frac{1}{\sqrt{2n}}\right]$$

This approach illustrates the power of maximum likelihood estimation combined with asymptotic theory for statistical inference. 