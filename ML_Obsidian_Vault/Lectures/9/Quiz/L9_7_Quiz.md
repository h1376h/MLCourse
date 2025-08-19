# Lecture 9.7: Bootstrap and Resampling Methods Quiz

## Overview
This quiz contains 25 comprehensive questions covering bootstrap methods, including bootstrap estimation, confidence intervals, different bootstrap types, advantages, limitations, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
You have a dataset with 5 values: [2, 4, 6, 8, 10] and want to use bootstrap resampling.

#### Task
1. What is the mean of the original dataset?
2. If you create a bootstrap sample by sampling with replacement, how many possible different samples could you get?
3. What's the probability of getting the original sample [2, 4, 6, 8, 10] in a bootstrap sample?
4. What's the probability of getting a sample with all values equal to 2?
5. What's the main advantage of bootstrap resampling?

For a detailed explanation of this question, see [Question 1: Basic Bootstrap Concepts](L9_7_1_explanation.md).

## Question 2

### Problem Statement
Consider a dataset with 4 values: [3, 7, 11, 15] and you want to estimate the standard deviation using bootstrap.

#### Task
1. Calculate the standard deviation of the original dataset
2. If you create 3 bootstrap samples: [3, 3, 7, 11], [7, 11, 11, 15], [3, 7, 15, 15], calculate the standard deviation of each
3. What's the mean of these bootstrap standard deviations?
4. What's the difference between the original and bootstrap mean?
5. Why might bootstrap estimates differ from the original statistic?

For a detailed explanation of this question, see [Question 2: Bootstrap Standard Deviation](L9_7_2_explanation.md).

## Question 3

### Problem Statement
You have a dataset with 6 values: [1, 3, 5, 7, 9, 11] and want to create bootstrap confidence intervals.

#### Task
1. Calculate the mean of the original dataset
2. If you create 4 bootstrap samples and get means [4.5, 5.2, 4.8, 5.5], what's the bootstrap mean?
3. What's the standard deviation of these bootstrap means?
4. If you want a 95% confidence interval, what percentile values would you use?
5. What does the width of the confidence interval tell you about the estimate's precision?

For a detailed explanation of this question, see [Question 3: Bootstrap Confidence Intervals](L9_7_3_explanation.md).

## Question 4

### Problem Statement
Consider a dataset with 5 values: [10, 20, 30, 40, 50] and you want to estimate the median using bootstrap.

#### Task
1. What is the median of the original dataset?
2. If you create 3 bootstrap samples: [10, 20, 30, 30, 50], [20, 20, 30, 40, 40], [10, 30, 40, 50, 50], calculate the median of each
3. What's the mean of these bootstrap medians?
4. What's the difference between the original and bootstrap median?
5. Why might the median be more stable than the mean in bootstrap resampling?

For a detailed explanation of this question, see [Question 4: Bootstrap Median Estimation](L9_7_4_explanation.md).

## Question 5

### Problem Statement
You have a dataset with 4 values: [2, 4, 6, 8] and want to understand bootstrap sampling probabilities.

#### Task
1. What's the probability of selecting value 2 in a single bootstrap draw?
2. What's the probability of NOT selecting value 2 in a single bootstrap draw?
3. What's the probability of getting a bootstrap sample with all values equal to 2?
4. What's the probability of getting a bootstrap sample with no value 2?
5. What's the expected number of 2s in a bootstrap sample of size 4?

For a detailed explanation of this question, see [Question 5: Bootstrap Sampling Probabilities](L9_7_5_explanation.md).

## Question 6

### Problem Statement
Consider a dataset with 6 values: [5, 10, 15, 20, 25, 30] and you want to estimate the variance using bootstrap.

#### Task
1. Calculate the variance of the original dataset
2. If you create 3 bootstrap samples: [5, 10, 15, 20, 25, 30], [10, 15, 20, 25, 25, 30], [5, 15, 20, 20, 25, 30], calculate the variance of each
3. What's the mean of these bootstrap variances?
4. What's the difference between the original and bootstrap variance?
5. Why might bootstrap variance estimates be biased?

For a detailed explanation of this question, see [Question 6: Bootstrap Variance Estimation](L9_7_6_explanation.md).

## Question 7

### Problem Statement
You have a dataset with 5 values: [1, 2, 3, 4, 5] and want to create bootstrap samples.

#### Task
1. What's the mean of the original dataset?
2. If you create 3 bootstrap samples: [1, 2, 3, 4, 5], [2, 3, 4, 4, 5], [1, 3, 3, 4, 5], calculate the mean of each
3. What's the standard deviation of these bootstrap means?
4. What's the 95% confidence interval for the mean?
5. What does this confidence interval tell you about the population mean?

For a detailed explanation of this question, see [Question 7: Bootstrap Mean Confidence](L9_7_7_explanation.md).

## Question 8

### Problem Statement
Consider a dataset with 4 values: [6, 12, 18, 24] and you want to estimate the coefficient of variation using bootstrap.

#### Task
1. Calculate the coefficient of variation (CV = standard deviation/mean) of the original dataset
2. If you create 3 bootstrap samples: [6, 12, 18, 24], [12, 18, 18, 24], [6, 12, 18, 18], calculate the CV of each
3. What's the mean of these bootstrap CVs?
4. What's the difference between the original and bootstrap CV?
5. Why might the coefficient of variation be useful in bootstrap analysis?

For a detailed explanation of this question, see [Question 8: Bootstrap Coefficient of Variation](L9_7_8_explanation.md).

## Question 9

### Problem Statement
You have a dataset with 6 values: [2, 4, 6, 8, 10, 12] and want to understand bootstrap bias.

#### Task
1. What's the mean of the original dataset?
2. If you create 4 bootstrap samples and get means [6.5, 6.8, 6.2, 6.9], what's the bootstrap mean?
3. What's the bias (bootstrap mean - original mean)?
4. Is this bias positive or negative?
5. What does this bias tell you about the bootstrap estimator?

For a detailed explanation of this question, see [Question 9: Bootstrap Bias Analysis](L9_7_9_explanation.md).

## Question 10

### Problem Statement
Consider a dataset with 5 values: [3, 6, 9, 12, 15] and you want to estimate the range using bootstrap.

#### Task
1. What's the range of the original dataset?
2. If you create 3 bootstrap samples: [3, 6, 9, 12, 15], [6, 9, 12, 12, 15], [3, 6, 9, 9, 15], calculate the range of each
3. What's the mean of these bootstrap ranges?
4. What's the difference between the original and bootstrap range?
5. Why might the range be less stable than other statistics in bootstrap resampling?

For a detailed explanation of this question, see [Question 10: Bootstrap Range Estimation](L9_7_10_explanation.md).

## Question 11

### Problem Statement
You have a dataset with 4 values: [8, 16, 24, 32] and want to create percentile bootstrap confidence intervals.

#### Task
1. What's the mean of the original dataset?
2. If you create 5 bootstrap samples and get means [20.5, 21.2, 19.8, 20.9, 21.5], what's the bootstrap mean?
3. What's the standard deviation of these bootstrap means?
4. If you want a 90% confidence interval, what percentile values would you use?
5. What's the advantage of percentile bootstrap over normal approximation?

For a detailed explanation of this question, see [Question 11: Percentile Bootstrap Confidence](L9_7_11_explanation.md).

## Question 12

### Problem Statement
Consider a dataset with 6 values: [4, 8, 12, 16, 20, 24] and you want to estimate the interquartile range using bootstrap.

#### Task
1. What's the interquartile range (IQR) of the original dataset?
2. If you create 3 bootstrap samples: [4, 8, 12, 16, 20, 24], [8, 12, 16, 16, 20, 24], [4, 8, 12, 12, 20, 24], calculate the IQR of each
3. What's the mean of these bootstrap IQRs?
4. What's the difference between the original and bootstrap IQR?
5. Why might the IQR be more robust than the range in bootstrap resampling?

For a detailed explanation of this question, see [Question 12: Bootstrap Interquartile Range](L9_7_12_explanation.md).

## Question 13

### Problem Statement
You have a dataset with 5 values: [5, 10, 15, 20, 25] and want to understand bootstrap efficiency.

#### Task
1. What's the mean of the original dataset?
2. If you create 3 bootstrap samples: [5, 10, 15, 20, 25], [10, 15, 20, 20, 25], [5, 15, 15, 20, 25], calculate the mean of each
3. What's the standard deviation of these bootstrap means?
4. What's the standard error of the mean?
5. How does the bootstrap standard error compare to the theoretical standard error?

For a detailed explanation of this question, see [Question 13: Bootstrap Efficiency](L9_7_13_explanation.md).

## Question 14

### Problem Statement
Consider a dataset with 4 values: [10, 20, 30, 40] and you want to estimate the geometric mean using bootstrap.

#### Task
1. Calculate the geometric mean of the original dataset
2. If you create 3 bootstrap samples: [10, 20, 30, 40], [20, 30, 30, 40], [10, 20, 30, 30], calculate the geometric mean of each
3. What's the mean of these bootstrap geometric means?
4. What's the difference between the original and bootstrap geometric mean?
5. Why might the geometric mean be useful in certain applications?

For a detailed explanation of this question, see [Question 14: Bootstrap Geometric Mean](L9_7_14_explanation.md).

## Question 15

### Problem Statement
You have a dataset with 6 values: [1, 3, 5, 7, 9, 11] and want to create bias-corrected bootstrap confidence intervals.

#### Task
1. What's the mean of the original dataset?
2. If you create 4 bootstrap samples and get means [5.2, 5.5, 4.8, 5.9], what's the bootstrap mean?
3. What's the bias correction factor?
4. If you want a 95% confidence interval, what percentile values would you use after bias correction?
5. What's the advantage of bias-corrected bootstrap over standard percentile bootstrap?

For a detailed explanation of this question, see [Question 15: Bias-Corrected Bootstrap](L9_7_15_explanation.md).

## Question 16

### Problem Statement
Consider a dataset with 5 values: [2, 4, 6, 8, 10] and you want to estimate the harmonic mean using bootstrap.

#### Task
1. Calculate the harmonic mean of the original dataset
2. If you create 3 bootstrap samples: [2, 4, 6, 8, 10], [4, 6, 8, 8, 10], [2, 4, 6, 6, 10], calculate the harmonic mean of each
3. What's the mean of these bootstrap harmonic means?
4. What's the difference between the original and bootstrap harmonic mean?
5. Why might the harmonic mean be useful in certain applications?

For a detailed explanation of this question, see [Question 16: Bootstrap Harmonic Mean](L9_7_16_explanation.md).

## Question 17

### Problem Statement
You have a dataset with 4 values: [6, 12, 18, 24] and want to understand bootstrap consistency.

#### Task
1. What's the mean of the original dataset?
2. If you create 4 bootstrap samples and get means [15.2, 15.8, 14.9, 15.5], what's the bootstrap mean?
3. What's the standard deviation of these bootstrap means?
4. What's the coefficient of variation of the bootstrap means?
5. What does this coefficient of variation tell you about bootstrap consistency?

For a detailed explanation of this question, see [Question 17: Bootstrap Consistency](L9_7_17_explanation.md).

## Question 18

### Problem Statement
Consider a dataset with 6 values: [3, 6, 9, 12, 15, 18] and you want to estimate the mode using bootstrap.

#### Task
1. What's the mode of the original dataset?
2. If you create 3 bootstrap samples: [3, 6, 9, 12, 15, 18], [6, 9, 12, 12, 15, 18], [3, 6, 9, 9, 15, 18], calculate the mode of each
3. What's the most common mode across these bootstrap samples?
4. What's the difference between the original and most common bootstrap mode?
5. Why might the mode be less reliable than other statistics in bootstrap resampling?

For a detailed explanation of this question, see [Question 18: Bootstrap Mode Estimation](L9_7_18_explanation.md).

## Question 19

### Problem Statement
You have a dataset with 5 values: [4, 8, 12, 16, 20] and want to create accelerated bootstrap confidence intervals.

#### Task
1. What's the mean of the original dataset?
2. If you create 5 bootstrap samples and get means [12.2, 12.8, 11.9, 12.5, 12.9], what's the bootstrap mean?
3. What's the acceleration factor?
4. If you want a 90% confidence interval, what percentile values would you use after acceleration?
5. What's the advantage of accelerated bootstrap over standard percentile bootstrap?

For a detailed explanation of this question, see [Question 19: Accelerated Bootstrap](L9_7_19_explanation.md).

## Question 20

### Problem Statement
Consider a dataset with 4 values: [5, 10, 15, 20] and you want to estimate the skewness using bootstrap.

#### Task
1. Calculate the skewness of the original dataset
2. If you create 3 bootstrap samples: [5, 10, 15, 20], [10, 15, 15, 20], [5, 10, 15, 15], calculate the skewness of each
3. What's the mean of these bootstrap skewness values?
4. What's the difference between the original and bootstrap skewness?
5. Why might skewness be more sensitive to bootstrap resampling than other statistics?

For a detailed explanation of this question, see [Question 20: Bootstrap Skewness](L9_7_20_explanation.md).

## Question 21

### Problem Statement
You have a dataset with 6 values: [2, 4, 6, 8, 10, 12] and want to understand bootstrap coverage.

#### Task
1. What's the mean of the original dataset?
2. If you create 6 bootstrap samples and get means [6.8, 7.2, 6.5, 7.0, 6.9, 7.1], what's the bootstrap mean?
3. What's the standard deviation of these bootstrap means?
4. What's the 95% confidence interval for the mean?
5. What does "coverage" mean in the context of bootstrap confidence intervals?

For a detailed explanation of this question, see [Question 21: Bootstrap Coverage](L9_7_21_explanation.md).

## Question 22

### Problem Statement
Consider a dataset with 5 values: [7, 14, 21, 28, 35] and you want to estimate the kurtosis using bootstrap.

#### Task
1. Calculate the kurtosis of the original dataset
2. If you create 3 bootstrap samples: [7, 14, 21, 28, 35], [14, 21, 28, 28, 35], [7, 14, 21, 21, 35], calculate the kurtosis of each
3. What's the mean of these bootstrap kurtosis values?
4. What's the difference between the original and bootstrap kurtosis?
5. Why might kurtosis be particularly sensitive to bootstrap resampling?

For a detailed explanation of this question, see [Question 22: Bootstrap Kurtosis](L9_7_22_explanation.md).

## Question 23

### Problem Statement
You have a dataset with 4 values: [9, 18, 27, 36] and want to create studentized bootstrap confidence intervals.

#### Task
1. What's the mean of the original dataset?
2. If you create 4 bootstrap samples and get means [22.8, 23.5, 22.2, 23.1], what's the bootstrap mean?
3. What's the standard deviation of these bootstrap means?
4. What's the studentized statistic?
5. What's the advantage of studentized bootstrap over standard percentile bootstrap?

For a detailed explanation of this question, see [Question 23: Studentized Bootstrap](L9_7_23_explanation.md).

## Question 24

### Problem Statement
Consider a dataset with 6 values: [1, 2, 3, 4, 5, 6] and you want to estimate the coefficient of skewness using bootstrap.

#### Task
1. Calculate the coefficient of skewness of the original dataset
2. If you create 3 bootstrap samples: [1, 2, 3, 4, 5, 6], [2, 3, 4, 4, 5, 6], [1, 2, 3, 3, 5, 6], calculate the coefficient of skewness of each
3. What's the mean of these bootstrap coefficients of skewness?
4. What's the difference between the original and bootstrap coefficient of skewness?
5. Why might the coefficient of skewness be more interpretable than raw skewness?

For a detailed explanation of this question, see [Question 24: Bootstrap Coefficient of Skewness](L9_7_24_explanation.md).

## Question 25

### Problem Statement
You have a dataset with 5 values: [8, 16, 24, 32, 40] and want to understand bootstrap robustness.

#### Task
1. What's the mean of the original dataset?
2. If you create 5 bootstrap samples and get means [24.2, 24.8, 23.9, 24.5, 24.9], what's the bootstrap mean?
3. What's the median of these bootstrap means?
4. What's the difference between the mean and median of the bootstrap means?
5. What does this difference tell you about the robustness of the bootstrap estimator?

For a detailed explanation of this question, see [Question 25: Bootstrap Robustness](L9_7_25_explanation.md).
