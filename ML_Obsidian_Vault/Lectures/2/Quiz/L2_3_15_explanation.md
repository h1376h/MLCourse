# Question 15: Likelihood Functions and Distribution Fitting

## Problem Statement
A researcher has collected 50 data points on a continuous random variable X, which takes values between 0 and 1. The researcher wants to determine the best probabilistic model for this data by comparing three different distribution families: Beta, Normal, and Exponential. The following figures show different aspects of the likelihood analysis.

![Data Histogram](../Images/L2_3_Quiz_15/graph1_data_histogram.png)
![Beta Log-Likelihood Contour](../Images/L2_3_Quiz_15/graph2_beta_loglik_contour.png)
![Log-Likelihood Comparisons](../Images/L2_3_Quiz_15/graph3_loglik_comparisons.png)
![Fitted PDFs](../Images/L2_3_Quiz_15/graph4_fitted_pdfs.png)
![Probability vs Likelihood](../Images/L2_3_Quiz_15/graph5_probability_vs_likelihood.png)
![Likelihood Ratio Comparison](../Images/L2_3_Quiz_15/graph6_likelihood_ratio.png)

## Task
Using only the information provided in these graphs, answer the following questions:

1. Based on Figure 3, explain how the likelihood function responds to changes in parameter values for each distribution model. Why does the likelihood function peak at certain parameter values?
2. According to Figure 6, which distribution family best fits the observed data? Explain your reasoning.
3. Looking at Figure 4, visually assess how well each fitted distribution matches the observed data histogram.
4. Using Figure 5, explain the key difference between probability and likelihood in your own words.
5. Based on all the information provided, which distribution would you recommend using to model this data? Justify your answer.

## Solution

### Question 1: Likelihood Function Behavior

From Figure 3 (Log-Likelihood Comparisons), we can observe how the likelihood function changes as parameter values vary for each distribution:

- **Normal Distribution**: The likelihood function forms a smooth curve that peaks when the mean parameter is around 0.59. This peak indicates that this parameter value makes the observed data most probable under a Normal distribution model with fixed standard deviation of 0.2. As we move away from this optimal value, the likelihood decreases, showing that these alternative parameter values make the observed data less likely.

- **Exponential Distribution**: The likelihood function for the rate parameter shows a clear peak around 1.69. The steeper drop-off to the left indicates that very small rate values (which would create a more spread-out distribution) are particularly unlikely to have generated the observed data. The more gradual decrease to the right suggests that larger rate values (creating more concentrated distributions) are somewhat more plausible but still less likely than the optimal value.

- **Beta Distribution**: When we fix the beta parameter at approximately 2.03, the likelihood function for the alpha parameter peaks near 3.06. The shape of this curve demonstrates how sensitive the model fit is to changes in the alpha parameter, with the likelihood falling off more rapidly for smaller alpha values than for larger ones.

The likelihood function peaks at certain parameter values because these are the values that make the observed data most probable under each distribution model. For continuous distributions, the likelihood function is calculated by evaluating the probability density function at each observed data point and taking the product (or sum of logarithms for computational stability). The parameter values that maximize this product are those that best align the distribution's shape with the pattern of observed data.

### Question 2: Best-Fitting Distribution

From Figure 6 (Likelihood Ratio Comparison), we can see that the Beta distribution has the highest log-likelihood value, followed by the Normal distribution, and then the Exponential distribution. 

The log-likelihood values are approximately:
- Beta: -21.96
- Normal: -36.44
- Exponential: -44.17

The relative likelihood values (shown by the bar heights) indicate that the Beta distribution is exponentially more likely to have generated the data compared to the other distributions. The likelihood ratio between Beta and Normal is so large that the Normal's relative likelihood is nearly zero on this scale.

Therefore, the Beta distribution provides the best fit to the observed data according to the likelihood criterion.

### Question 3: Visual Assessment of Fit

Looking at Figure 4 (Fitted PDFs), we can visually assess how well each fitted distribution matches the observed data histogram:

- **Beta Distribution (purple line)**: This distribution appears to match the shape of the histogram very well. It captures the slight right skew in the data and aligns with the central tendency of the observations.

- **Normal Distribution (green line)**: While it captures the central tendency of the data, it extends significantly beyond the [0,1] range (which is impossible for the data) and doesn't match the slight skewness in the histogram.

- **Exponential Distribution (blue line)**: This distribution shows a poor fit, as it suggests a monotonically decreasing density, whereas the data appears to have a peak around 0.6.

The visual assessment confirms that the Beta distribution provides the best fit to the data, which aligns with the likelihood-based conclusion.

### Question 4: Probability vs. Likelihood

Figure 5 (Probability vs. Likelihood) illustrates the fundamental difference between probability and likelihood:

- **Probability** (top panel): Here, we fix the distribution (with set parameters) and vary the data point. The y-axis shows the probability density for different possible data values. Probability answers the question: "Given a specific distribution, what is the chance of observing a particular data point?"

- **Likelihood** (bottom panel): Here, we fix the data point (x = 0.5) and vary the distribution parameter (alpha). The y-axis shows how likely that fixed data point would be under different parameter values. Likelihood answers the question: "Given the observed data, which parameter values would make this data most probable?"

The key difference is in what's fixed and what varies:
- In probability, the distribution is fixed, and we assess different possible data values.
- In likelihood, the data is fixed, and we assess different possible distribution parameters.

Probability flows from parameters to data, while likelihood flows from data to parameters. This is why likelihood is fundamental to parameter estimation in statistics.

### Question 5: Recommended Distribution

Based on all the information provided, I would recommend using the Beta distribution to model this data for the following reasons:

1. **Highest Likelihood**: The Beta distribution has the highest log-likelihood value, indicating it best explains the observed data among the three candidates.

2. **Appropriate Domain**: The Beta distribution is defined on the interval [0,1], which matches the range of the observed data. Unlike the Normal distribution, it doesn't assign probability to impossible values outside this range.

3. **Flexible Shape**: The Beta distribution with parameters alpha ≈ 3.06 and beta ≈ 2.03 captures the slight right skew in the data that the symmetric Normal distribution cannot model accurately.

4. **Visual Fit**: As seen in Figure 4, the Beta distribution provides the best visual fit to the histogram of the observed data.

The evidence from both likelihood analysis and visual inspection strongly supports choosing the Beta distribution as the most appropriate model for this dataset. 