# Multinomial Distribution MLE Examples

This document provides practical examples of Maximum Likelihood Estimation (MLE) for Multinomial distributions, illustrating the concept of estimating categorical probability parameters and its significance in analyzing data with multiple discrete outcomes.

## Key Concepts and Formulas

The Multinomial distribution is an extension of the binomial distribution to multiple categories. It models the probability of counts for each category in a fixed number of independent trials, where each trial results in exactly one of k possible outcomes.

### The Multinomial MLE Formula

For a multinomial distribution with $k$ categories, the likelihood function is:

$$L(\theta) = \prod_{i=1}^{k} \theta_i^{x_i}$$

The log-likelihood function:

$$\ell(\theta) = \sum_{i=1}^{k} x_i \ln(\theta_i)$$

Subject to the constraint:

$$\sum_{i=1}^{k} \theta_i = 1$$

Using Lagrange multipliers and solving the optimization problem, the maximum likelihood estimator for each probability parameter is:

$$\hat{\theta}_i = \frac{x_i}{\sum_{j=1}^{k} x_j}$$

Where:
- $\theta_i$ = probability of category $i$
- $x_i$ = observed count for category $i$
- $\sum_{j=1}^{k} x_j$ = total number of observations

## Categorical Data Examples

The following examples demonstrate MLE for categorical variables (Multinomial distribution):

- **Six-Sided Die**: Testing fairness of a die

### Example 1: Six-Sided Die

#### Problem Statement
You want to test if a six-sided die is fair by analyzing the frequency of each face. After rolling the die 50 times, you observe the following counts: Face 1: 7 times, Face 2: 5 times, Face 3: 9 times, Face 4: 12 times, Face 5: 8 times, and Face 6: 9 times. Assuming the rolls follow a multinomial distribution, MLE can help estimate the probability parameter for each face.

In this example:
- The data consists of 50 die rolls with counts for each of the six faces
- We assume the rolls follow a multinomial distribution
- MLE estimates the probability parameters for each face
- The analysis relies solely on the observed data without prior assumptions

#### Solution

##### Step 1: Gather the data
- Observed counts:
  - Face 1: 7 times
  - Face 2: 5 times
  - Face 3: 9 times
  - Face 4: 12 times
  - Face 5: 8 times
  - Face 6: 9 times
- Total number of rolls (N) = 50

##### Step 2: Define the likelihood function
For multinomially distributed data, the likelihood function is:

$$L(p_1, p_2, \dots, p_6 | \text{data}) = \frac{N!}{x_1! \times x_2! \times \dots \times x_6!} \times p_1^{x_1} \times p_2^{x_2} \times \dots \times p_6^{x_6}$$

Where:
- $p_1, p_2, \dots, p_6$ are the probability parameters we're trying to estimate
- $x_1, x_2, \dots, x_6$ are the observed counts for each category
- $N$ is the total number of observations ($\sum x_i$)

##### Step 3: Calculate MLE
For a multinomial distribution, the MLE for each probability parameter is simply the proportion of occurrences in the category:

$$p_i^{\text{MLE}} = \frac{x_i}{N}$$

Therefore:
- $p_1^{\text{MLE}} = \frac{7}{50} = 0.14$ (14%)
- $p_2^{\text{MLE}} = \frac{5}{50} = 0.10$ (10%)
- $p_3^{\text{MLE}} = \frac{9}{50} = 0.18$ (18%)
- $p_4^{\text{MLE}} = \frac{12}{50} = 0.24$ (24%)
- $p_5^{\text{MLE}} = \frac{8}{50} = 0.16$ (16%)
- $p_6^{\text{MLE}} = \frac{9}{50} = 0.18$ (18%)

##### Step 4: Confidence intervals
We can calculate approximate 95% confidence intervals for each parameter using:

$$\text{CI} = p_i^{\text{MLE}} \pm 1.96 \times \sqrt{\frac{p_i^{\text{MLE}} \times (1-p_i^{\text{MLE}})}{N}}$$

For example, for Face 1:

$$\text{CI} = 0.14 \pm 1.96 \times \sqrt{\frac{0.14 \times 0.86}{50}}$$

$$\text{CI} = 0.14 \pm 1.96 \times \sqrt{0.0024}$$

$$\text{CI} = 0.14 \pm 1.96 \times 0.049$$

$$\text{CI} = 0.14 \pm 0.096$$

$$\text{CI} = [0.044, 0.236]$$

##### Step 5: Interpret the results
Based on the MLE analysis, the estimated probabilities differ from what we'd expect from a fair die ($\frac{1}{6} \approx 0.167$ for each face). Face 4 appears to have the highest probability (0.24), while Face 2 has the lowest (0.10). However, the confidence intervals for all faces include the theoretical probability of 0.167, so we don't have strong evidence that the die is unfair based on this sample.

![Dice Rolls MLE Example](../Images/multinomial_mle_dice_rolls.png)

### Example 2: Survey Responses

#### Problem Statement
A marketing team conducted a survey asking customers to rate their product on a scale from 1 to 5 stars. Out of 200 responses, they received: 1 star: 15 responses, 2 stars: 25 responses, 3 stars: 60 responses, 4 stars: 70 responses, and 5 stars: 30 responses. The team wants to estimate the true probability distribution of customer ratings.

In this example:
- The data consists of 200 survey responses across 5 rating categories
- We assume the ratings follow a multinomial distribution
- MLE estimates the probability parameters for each rating category
- The analysis relies solely on the observed data without prior assumptions

#### Solution

##### Step 1: Gather the data
- Observed counts:
  - 1 star: 15 responses
  - 2 stars: 25 responses
  - 3 stars: 60 responses
  - 4 stars: 70 responses
  - 5 stars: 30 responses
- Total number of responses (N) = 200

##### Step 2: Apply the MLE formula
For a multinomial distribution, the MLE for each probability parameter is:

$$p_i^{\text{MLE}} = \frac{x_i}{N}$$

Therefore:
- $p_1^{\text{MLE}} = \frac{15}{200} = 0.075$ (7.5%)
- $p_2^{\text{MLE}} = \frac{25}{200} = 0.125$ (12.5%)
- $p_3^{\text{MLE}} = \frac{60}{200} = 0.30$ (30%)
- $p_4^{\text{MLE}} = \frac{70}{200} = 0.35$ (35%)
- $p_5^{\text{MLE}} = \frac{30}{200} = 0.15$ (15%)

##### Step 3: Interpret the results
Based on the MLE analysis, the estimated distribution shows that 4-star ratings are most common (35%), followed by 3-star ratings (30%). The majority of ratings (80%) are in the 3-5 star range, indicating generally positive customer sentiment. The marketing team can use these estimates to understand the true distribution of customer opinions about their product.

## Key Insights and Takeaways

### Theoretical Insights
- The MLE for a multinomial distribution parameter is simply the proportion of observations in that category
- This estimate represents the value of θᵢ that makes the observed data most likely
- The constraint that all probabilities must sum to 1 is handled implicitly by using proportions
- As sample size increases, the MLE approaches the true parameter values

### Practical Applications
- Testing the fairness of dice, cards, or other gaming equipment
- Analyzing survey data with multiple response categories
- Market share analysis across multiple competitors
- Political polling and vote distribution estimation
- DNA sequence analysis and nucleotide frequency estimation

### Common Pitfalls
- Small sample sizes can lead to unreliable estimates
- Zero counts in categories require special handling in some contexts
- The assumption of independence between trials may not hold in all applications
- Confidence intervals become less reliable with small expected counts in any category

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Code/multinomial_mle_examples.py
```

## Related Topics

- [[L2_3_Likelihood_Examples|Likelihood Examples]]: General concepts of likelihood
- [[L2_4_MLE_Examples|MLE Examples]]: Other distributions' MLE calculations
