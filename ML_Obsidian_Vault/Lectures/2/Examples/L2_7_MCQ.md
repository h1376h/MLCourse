# Question 2.7: MAP and Full Bayesian Inference Multiple Choice

This document provides practice multiple-choice questions on Maximum A Posteriori (MAP) estimation and Full Bayesian Inference to test your understanding of key concepts.

## Questions

### Question 1
For a Beta($\alpha$, $\beta$) prior and a binomial likelihood with $s$ successes in $n$ trials, the MAP estimate for the probability parameter $\theta$ is:

**Options:**
A) $\frac{\alpha + s - 1}{\alpha + \beta + n - 2}$
B) $\frac{\alpha + s}{\alpha + \beta + n}$
C) $\frac{s}{n}$
D) $\frac{\alpha}{\alpha + \beta}$

**Answer:** A

**Explanation:** The MAP estimate is the mode of the posterior distribution. For a Beta($\alpha$, $\beta$) prior and binomial likelihood, the posterior is Beta($\alpha + s$, $\beta + n - s$). The mode of a Beta($a$, $b$) distribution is $\frac{a - 1}{a + b - 2}$ when $a, b > 1$, which gives $\frac{\alpha + s - 1}{\alpha + \beta + n - 2}$.

### Question 2
A coin has been flipped 5 times, resulting in 4 heads. If we use a Beta(2, 2) prior distribution for the probability of heads, what is the posterior mean probability of getting heads?

**Options:**
A) 0.5
B) 0.6
C) 0.67
D) 0.8

**Answer:** C

**Explanation:** With a Beta(2, 2) prior and data of 4 heads in 5 flips, the posterior is Beta(2+4, 2+1) = Beta(6, 3). The mean of this distribution is $\frac{6}{6+3} = \frac{6}{9} = 0.67$.

### Question 3
Consider a linear regression problem where you want to prevent overfitting by adding L2 regularization. From a Bayesian perspective, this regularization corresponds to which prior distribution on the regression coefficients?

**Options:**
A) Uniform prior
B) Gaussian prior
C) Laplace prior
D) Cauchy prior

**Answer:** B

**Explanation:** L2 regularization (ridge regression) corresponds to assuming a Gaussian (normal) prior on the regression coefficients in Bayesian inference. The penalty term $\lambda\sum_{i=1}^{p} \beta_i^2$ in ridge regression is equivalent to assuming that the coefficients follow a normal distribution with mean 0.

### Question 4
The Bayesian Information Criterion (BIC) is an approximation to which quantity in Bayesian model selection?

**Options:**
A) Posterior probability
B) Likelihood function
C) Prior probability
D) Negative log marginal likelihood (evidence)

**Answer:** D

**Explanation:** The BIC approximates the negative log of the marginal likelihood (evidence), which is used for Bayesian model selection. The formula is typically written as:

$$\text{BIC} = -2\ln(L) + k\ln(n)$$

where $L$ is the likelihood, $k$ is the number of parameters, and $n$ is the number of data points.

### Question 5
In Bayesian Model Averaging, predictions are made by:

**Options:**
A) Selecting the model with the highest posterior probability
B) Averaging predictions across models, weighted by their posterior probabilities
C) Averaging predictions across models, weighted by their likelihoods
D) Using only the model with the highest evidence

**Answer:** B

**Explanation:** Bayesian Model Averaging makes predictions by averaging the predictions of multiple models, with each model's contribution weighted by its posterior probability. The formula is:

$$p(y|x,D) = \sum_{i=1}^{M} p(y|x,M_i,D)p(M_i|D)$$

where $p(M_i|D)$ is the posterior probability of model $M_i$.

### Question 6
In a study comparing two treatments, the Bayes factor $\text{BF}_{12} = 8$ means:

**Options:**
A) Treatment 1 is 8 times more effective than Treatment 2
B) The data are 8 times more likely under Model 1 than Model 2
C) The posterior odds are always 8 regardless of the prior odds
D) The probability of Model 1 being correct is 8 times that of Model 2

**Answer:** B

**Explanation:** A Bayes factor $\text{BF}_{12} = 8$ means that the observed data are 8 times more likely under Model 1 than under Model 2. Mathematically:

$$\text{BF}_{12} = \frac{p(D|M_1)}{p(D|M_2)} = 8$$

where $p(D|M_i)$ is the marginal likelihood of the data under model $M_i$. Option A is incorrect as the Bayes factor concerns the likelihood of data, not treatment effectiveness. Option C is incorrect because the posterior odds equal the Bayes factor multiplied by the prior odds, not just the Bayes factor alone. Option D is incorrect because posterior probabilities depend on both the Bayes factor and the prior probabilities.

### Question 7
Variational Inference is a technique for approximating complex posterior distributions in Bayesian inference. Which of the following statements about Variational Inference is FALSE?

**Options:**
A) It approximates the posterior by finding a simpler distribution that minimizes the KL divergence
B) It is typically faster than Markov Chain Monte Carlo (MCMC) methods
C) It always provides an exact representation of the true posterior distribution
D) It can be used when the posterior distribution is analytically intractable

**Answer:** C

**Explanation:** Variational Inference provides an approximation, not an exact representation of the true posterior distribution. This is its fundamental limitation but also what makes it computationally feasible for complex models. It minimizes the KL divergence between the true posterior $p(\theta|D)$ and a simpler approximating distribution $q(\theta)$:

$$\text{KL}(q(\theta)||p(\theta|D)) = \int q(\theta) \log \frac{q(\theta)}{p(\theta|D)} d\theta$$

### Question 8
Which of the following optimization methods is commonly used for finding the MAP estimate when the posterior distribution is complex?

**Options:**
A) Grid search
B) Gradient descent
C) Random sampling
D) Exhaustive enumeration

**Answer:** B

**Explanation:** Gradient descent (and its variants) is commonly used for finding MAP estimates when dealing with complex posterior distributions that don't have analytical solutions. The algorithm uses the gradient of the log posterior $\nabla \log p(\theta|D)$ to iteratively update the parameter estimates:

$$\theta^{(t+1)} = \theta^{(t)} + \eta \nabla \log p(\theta^{(t)}|D)$$

where $\eta$ is the learning rate.

### Question 9
Consider L1 regularization in linear regression (Lasso regression). From a Bayesian perspective, this corresponds to which prior distribution on the regression coefficients?

**Options:**
A) Gaussian prior
B) Laplace prior
C) Uniform prior
D) Beta prior

**Answer:** B

**Explanation:** L1 regularization (Lasso regression) corresponds to assuming a Laplace (double exponential) prior on the regression coefficients in Bayesian inference. The penalty term $\lambda\sum_{i=1}^{p} |\beta_i|$ in Lasso regression is equivalent to placing a Laplace prior with density:

$$p(\beta_i) \propto \exp(-\lambda|\beta_i|)$$

### Question 10
When using posterior sampling in full Bayesian inference, which of the following statements is TRUE?

**Options:**
A) Only a single sample is needed to represent the posterior distribution
B) Samples allow us to estimate expectations under the posterior distribution
C) The mean of the samples always equals the MAP estimate
D) Posterior sampling is only necessary for continuous parameters

**Answer:** B

**Explanation:** Samples from the posterior distribution allow us to estimate expectations (like means, variances, and other statistics) under the posterior distribution, which is a key advantage of full Bayesian inference. For any function $f(\theta)$, the expectation can be approximated as:

$$E[f(\theta)|D] = \int f(\theta)p(\theta|D)d\theta \approx \frac{1}{N}\sum_{i=1}^{N}f(\theta^{(i)})$$

where $\theta^{(i)}$ are samples from the posterior $p(\theta|D)$.

### Question 11
For computing the marginal likelihood (model evidence) in Bayesian model selection, which of the following methods is NOT commonly used?

**Options:**
A) Laplace approximation
B) Bridge sampling
C) Maximum likelihood estimation
D) Importance sampling

**Answer:** C

**Explanation:** Maximum likelihood estimation (MLE) is not used for computing the marginal likelihood. MLE finds parameter values that maximize the likelihood, while the marginal likelihood requires integrating over all possible parameter values:

$$p(D|M) = \int p(D|\theta,M)p(\theta|M)d\theta$$

where $p(D|\theta,M)$ is the likelihood and $p(\theta|M)$ is the prior under model $M$.

### Question 12
What is the most significant computational challenge in implementing full Bayesian inference for complex models?

**Options:**
A) Storing the full posterior distribution in memory
B) Computing high-dimensional integrals for posterior expectations
C) Finding a closed-form expression for the likelihood function
D) Determining appropriate point estimates

**Answer:** B

**Explanation:** Computing high-dimensional integrals required for posterior expectations is the most significant computational challenge in implementing full Bayesian inference for complex models. The posterior expectation of a function $f(\theta)$ is:

$$E[f(\theta)|D] = \int f(\theta)p(\theta|D)d\theta$$

This integral becomes increasingly difficult to compute as the dimension of $\theta$ increases, often requiring sophisticated sampling or approximation methods like MCMC. While memory limitations (Option A) can be an issue, they are generally addressed through sampling approaches. Options C and D are not unique challenges to Bayesian inference compared to the fundamental challenge of computing these integrals.

### Question 13
Instead of the Bayes estimator, one can also use the MAP (maximum a posteriori estimator) in order to get a point estimate of the parameter of interest $\theta$ based on data. Which of the following expressions define the MAP?

**Options:**
A) $\underset{\theta}{\operatorname{argmax}} \log L(\theta)p(\theta)$
B) $\underset{\theta}{\operatorname{argmax}} \log p(\theta)$
C) $\underset{\theta}{\operatorname{argmax}} p(\text{data})$
D) $\underset{\theta}{\operatorname{argmax}} \log L(\theta)$

**Answer:** A

**Explanation:** The MAP estimator maximizes the posterior probability $p(\theta|\text{data})$, which is proportional to the product of the likelihood $L(\theta)$ and the prior $p(\theta)$. Taking the logarithm (which preserves the maximum), we get:

$$\log(L(\theta)p(\theta)) = \log L(\theta) + \log p(\theta)$$

Therefore, $\underset{\theta}{\operatorname{argmax}} \log L(\theta)p(\theta)$ correctly defines the MAP estimator.

### Question 14
Maximum a posteriori (MAP) classifies labels with the highest posterior probability value. Using Bayes Rule, we can find the solution as

$$\underset{Y}{\operatorname{argmax}} \frac{p(X|Y)p(Y)}{p(X)}$$

Now, pick the correct statements about the maximum likelihood (ML) solution.

**Options:**
A) ML ignores the prior and evidence to reach the simple solution.
B) ML ignores the evidence for the simple solution.
C) As we cannot ignore the prior and evidence, ML solution is equivalent to the MAP.
D) ML is equivalent to MAP when the prior distribution is uniform.

**Answer:** A, B, D

**Explanation:** Maximum likelihood estimation focuses solely on maximizing the likelihood function $p(X|Y)$. Option A is correct because ML ignores both the prior $p(Y)$ and the evidence $p(X)$ from Bayes' rule. Option B is also correct since ML specifically ignores the evidence term $p(X)$ in the denominator. Option C is incorrect because ML and MAP are generally different unless the prior is uniform. Option D is correct because when the prior $p(Y)$ is uniform (constant for all values of $Y$), the MAP estimate becomes equivalent to the ML estimate since the uniform prior doesn't influence which value of $Y$ maximizes the posterior probability.

### Question 15
When calculating the MAP estimate for a normal distribution with known variance, which of the following components is NOT needed?

**Options:**
A) The prior mean $\mu_0$
B) The sample median
C) The prior variance $\sigma_0^2$
D) The data variance $\sigma^2$

**Answer:** B

**Explanation:** The MAP formula for normal distributions with known variance uses the prior mean $\mu_0$, prior variance $\sigma_0^2$, sum of observations (or sample mean and count), and data variance $\sigma^2$. The formula is:

$$\hat{\mu}_{MAP} = \frac{\mu_0 + \frac{\sigma_0^2}{\sigma^2}\sum_{i=1}^N x_i}{1 + \frac{\sigma_0^2}{\sigma^2}N}$$

The sample median is not part of this calculation.

### Question 16
Which of the following best describes the relationship between MAP and regularization in machine learning?

**Options:**
A) They are unrelated concepts
B) MAP is a special case of regularization
C) Regularization is a special case of MAP
D) Certain types of regularization can be interpreted as performing MAP estimation

**Answer:** D

**Explanation:** L2 regularization corresponds to MAP estimation with a Gaussian prior, while L1 regularization corresponds to MAP with a Laplace prior. For example, ridge regression with penalty $\lambda\sum_{i=1}^{p} \beta_i^2$ is equivalent to MAP estimation with a prior $p(\beta_i) \propto \exp(-\lambda\beta_i^2/2)$. This connection provides a Bayesian interpretation for common regularization techniques.

### Question 17
In a sensor fusion scenario with two sensors measuring the same quantity, how could MAP estimation be useful?

**Options:**
A) It cannot be applied to sensor fusion problems
B) It can combine measurements from both sensors optimally considering their different error characteristics
C) It always selects the reading from the more accurate sensor
D) It simply averages the readings from both sensors

**Answer:** B

**Explanation:** MAP estimation provides a principled way to combine measurements from multiple sensors by treating one sensor's reading as the prior and the other as new data, or by combining both as data with a separate prior belief. If we have two sensors with measurements $x_1$ and $x_2$ with variances $\sigma_1^2$ and $\sigma_2^2$, the MAP estimate of the true value $\mu$ would be:

$$\hat{\mu}_{MAP} = \frac{\frac{x_1}{\sigma_1^2} + \frac{x_2}{\sigma_2^2}}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}}$$

This gives optimal weights based on each sensor's reliability.

### Question 18
If the variance ratio ($r = \frac{\sigma_0^2}{\sigma^2}$) in MAP estimation equals 9, what does this indicate about our trust in the prior versus the data?

**Options:**
A) We trust the prior 9 times more than the data
B) We trust the data 9 times more than the prior
C) We trust the prior and data equally
D) The ratio doesn't relate to trust levels

**Answer:** B

**Explanation:** A variance ratio greater than 1 indicates more trust in the data than the prior. With $r = 9$, each data point has 9 times more influence on the MAP estimate than we would expect if we trusted the prior and data equally. This can be seen in the simplified MAP formula:

$$\hat{\mu}_{MAP} = \frac{\mu_0 + r \sum_{i=1}^N x_i}{1 + r \times N}$$

The larger $r$ is, the more the estimate is pulled toward the data.

## Related Topics

- [[L2_4_MCQ|Maximum Likelihood Estimation Multiple Choice Questions]]
- [[Bayesian Inference]]
- [[Model Selection]] 