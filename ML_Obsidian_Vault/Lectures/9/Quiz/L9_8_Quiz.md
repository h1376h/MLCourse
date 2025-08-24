# Lecture 9.8: Advanced Evaluation Topics Quiz

## Overview
This quiz contains 25 comprehensive questions covering advanced evaluation topics, including statistical significance, model comparison, statistical tests, multiple comparison problems, evaluation bias, and best practices. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
You're comparing two models and want to determine if their performance difference is statistically significant.

**Model Performance:**
- Model A: $85\%$ accuracy on $1000$ test samples
- Model B: $87\%$ accuracy on $1000$ test samples

#### Task
1. Calculate the standard error for each model's accuracy
2. Calculate the standard error of the difference
3. Calculate the $z$-score for the difference
4. Is this difference statistically significant at the $5\%$ level?
5. What does statistical significance mean in this context?

For a detailed explanation of this question, see [Question 1: Statistical Significance Testing](L9_8_1_explanation.md).

## Question 2

### Problem Statement
Consider three models with the following performance on the same test set:

**Model Performance:**
- Model X: $82\%$ accuracy, $1000$ samples
- Model Y: $85\%$ accuracy, $1000$ samples
- Model Z: $88\%$ accuracy, $1000$ samples

#### Task
1. Calculate the standard error for each model
2. Which pair of models has the largest performance difference?
3. Calculate the $z$-score for the difference between Models X and Z
4. Is this difference statistically significant at the $1\%$ level?
5. What statistical test would you use to compare all three models simultaneously?

For a detailed explanation of this question, see [Question 2: Multiple Model Comparison](L9_8_2_explanation.md).

## Question 3

### Problem Statement
You're evaluating a model using 5-fold cross-validation and want to test if the performance is significantly different from a baseline.

**Cross-Validation Results:**
- Fold 1: $83\%$ accuracy
- Fold 2: $85\%$ accuracy
- Fold 3: $82\%$ accuracy
- Fold 4: $86\%$ accuracy
- Fold 5: $84\%$ accuracy

**Baseline Performance:** $80\%$ accuracy

#### Task
1. Calculate the mean and standard deviation of the cross-validation results
2. Calculate the standard error of the mean
3. Calculate the $t$-statistic for comparing with the baseline
4. Is the model significantly better than the baseline at the $5\%$ level?
5. What are the degrees of freedom for this $t$-test?

For a detailed explanation of this question, see [Question 3: Cross-Validation Significance](L9_8_3_explanation.md).

## Question 4

### Problem Statement
Consider a dataset with 500 samples where you're comparing two classification algorithms.

**Algorithm Performance:**
- Algorithm A: $78\%$ accuracy, $500$ samples
- Algorithm B: $82\%$ accuracy, $500$ samples

#### Task
1. Calculate the standard error for each algorithm
2. Calculate the standard error of the difference
3. Calculate the $95\%$ confidence interval for the difference
4. Does this confidence interval contain zero?
5. What does this tell you about statistical significance?

For a detailed explanation of this question, see [Question 4: Confidence Intervals](L9_8_4_explanation.md).

## Question 5

### Problem Statement
You're comparing three models and need to control for multiple comparisons.

**Model Performance:**
- Model 1 vs Model 2: $p$-value = $0.03$
- Model 1 vs Model 3: $p$-value = $0.01$
- Model 2 vs Model 3: $p$-value = $0.04$

#### Task
1. If you use Bonferroni correction with $\alpha = 0.05$, what's the adjusted significance level?
2. Which comparisons are significant after Bonferroni correction?
3. What's the advantage of Bonferroni correction?
4. What's the disadvantage of Bonferroni correction?
5. What alternative method could you use for multiple comparisons?

For a detailed explanation of this question, see [Question 5: Multiple Comparison Problem](L9_8_5_explanation.md).

## Question 6

### Problem Statement
Consider a model that achieves different performance on different datasets.

**Dataset Performance:**
- Dataset A: $88\%$ accuracy, $800$ samples
- Dataset B: $85\%$ accuracy, $800$ samples
- Dataset C: $90\%$ accuracy, $800$ samples

#### Task
1. Calculate the mean and standard deviation across datasets
2. Calculate the coefficient of variation
3. What does the coefficient of variation tell you about model stability?
4. How would you test if the performance differences are statistically significant?
5. What statistical test would be most appropriate for this comparison?

For a detailed explanation of this question, see [Question 6: Model Stability Analysis](L9_8_6_explanation.md).

## Question 7

### Problem Statement
You're evaluating a model using different evaluation metrics and want to understand their relationships.

**Model Results:**
- Accuracy: $85\%$
- Precision: $82\%$
- Recall: $88\%$
- F1-Score: $85\%$

#### Task
1. Calculate the harmonic mean of precision and recall
2. Does this match the F1-score? Explain
3. If you had to choose one metric to report, which would you pick and why?
4. What additional information would you need to make a complete assessment?
5. How would you handle conflicting metric results?

For a detailed explanation of this question, see [Question 7: Metric Relationships](L9_8_7_explanation.md).

## Question 8

### Problem Statement
Consider a model that shows different performance across different demographic groups.

**Group Performance:**
- Group A: $88\%$ accuracy, $400$ samples
- Group B: $82\%$ accuracy, $400$ samples
- Group C: $85\%$ accuracy, $400$ samples

#### Task
1. Calculate the overall accuracy across all groups
2. Calculate the standard deviation of group performances
3. What does this variation suggest about model fairness?
4. How would you test if the group differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 8: Fairness Evaluation](L9_8_8_explanation.md).

## Question 9

### Problem Statement
You're comparing two models using paired t-test on the same test samples.

**Paired Results (Model A vs Model B):**
- Sample 1: A=$0.85$, B=$0.87$
- Sample 2: A=$0.82$, B=$0.84$
- Sample 3: A=$0.88$, B=$0.86$
- Sample 4: A=$0.83$, B=$0.85$
- Sample 5: A=$0.86$, B=$0.88$

#### Task
1. Calculate the differences (B - A) for each sample
2. Calculate the mean and standard deviation of the differences
3. Calculate the $t$-statistic
4. Is the difference statistically significant at the $5\%$ level?
5. What are the degrees of freedom for this test?

For a detailed explanation of this question, see [Question 9: Paired T-Test](L9_8_9_explanation.md).

## Question 10

### Problem Statement
Consider a model that shows performance degradation over time.

**Temporal Performance:**
- Week 1: $88\%$ accuracy, $200$ samples
- Week 2: $86\%$ accuracy, $200$ samples
- Week 3: $84\%$ accuracy, $200$ samples
- Week 4: $82\%$ accuracy, $200$ samples

#### Task
1. Calculate the mean and standard deviation across weeks
2. Calculate the trend (slope) of performance over time
3. What does this trend suggest about model stability?
4. How would you test if the performance change is statistically significant?
5. What would you do if the degradation is significant?

For a detailed explanation of this question, see [Question 10: Temporal Performance Analysis](L9_8_10_explanation.md).

## Question 11

### Problem Statement
You're evaluating a model using different feature subsets and want to understand the impact.

**Feature Subset Performance:**
- All features: $87\%$ accuracy, $1000$ samples
- Core features only: $84\%$ accuracy, $1000$ samples
- Extended features: $88\%$ accuracy, $1000$ samples

#### Task
1. Calculate the standard error for each configuration
2. Which feature subset performs best?
3. Calculate the $z$-score for comparing core vs extended features
4. Is this difference statistically significant at the $5\%$ level?
5. What would you recommend based on these results?

For a detailed explanation of this question, see [Question 11: Feature Subset Comparison](L9_8_11_explanation.md).

## Question 12

### Problem Statement
Consider a model that shows different performance across different data sources.

**Source Performance:**
- Source 1: $85\%$ accuracy, $300$ samples
- Source 2: $88\%$ accuracy, $300$ samples
- Source 3: $82\%$ accuracy, $300$ samples

#### Task
1. Calculate the mean and standard deviation across sources
2. Calculate the coefficient of variation
3. What does this variation suggest about data quality?
4. How would you test if the source differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 12: Data Source Analysis](L9_8_12_explanation.md).

## Question 13

### Problem Statement
You're comparing two models using McNemar's test for paired nominal data.

**Contingency Table:**
|                | Model B Correct | Model B Incorrect |
|----------------|-----------------|-------------------|
| Model A Correct| $150$           | $25$              |
| Model A Incorrect| $20$           | $55$              |

#### Task
1. Calculate the McNemar test statistic
2. What are the degrees of freedom for this test?
3. Is the difference statistically significant at the $5\%$ level?
4. Which model performs better?
5. What's the advantage of McNemar's test over other comparison methods?

For a detailed explanation of this question, see [Question 13: McNemar's Test](L9_8_13_explanation.md).

## Question 14

### Problem Statement
Consider a model that shows performance variation across different time periods.

**Period Performance:**
- Morning: $86\%$ accuracy, $400$ samples
- Afternoon: $88\%$ accuracy, $400$ samples
- Evening: $84\%$ accuracy, $400$ samples
- Night: $85\%$ accuracy, $400$ samples

#### Task
1. Calculate the mean and standard deviation across periods
2. Calculate the range of performance
3. What does this variation suggest about model robustness?
4. How would you test if the period differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 14: Temporal Robustness](L9_8_14_explanation.md).

## Question 15

### Problem Statement
You're evaluating a model using different evaluation protocols and want to understand the impact.

**Protocol Performance:**
- Protocol 1: $85\%$ accuracy, $800$ samples
- Protocol 2: $87\%$ accuracy, $800$ samples
- Protocol 3: $83\%$ accuracy, $800$ samples

#### Task
1. Calculate the mean and standard deviation across protocols
2. Calculate the coefficient of variation
3. What does this variation suggest about evaluation consistency?
4. How would you test if the protocol differences are statistically significant?
5. What would you recommend based on these results?

For a detailed explanation of this question, see [Question 15: Evaluation Protocol Analysis](L9_8_15_explanation.md).

## Question 16

### Problem Statement
Consider a model that shows performance variation across different user segments.

**Segment Performance:**
- Segment A: $89\%$ accuracy, $500$ samples
- Segment B: $85\%$ accuracy, $500$ samples
- Segment C: $87\%$ accuracy, $500$ samples

#### Task
1. Calculate the mean and standard deviation across segments
2. Calculate the range of performance
3. What does this variation suggest about model generalization?
4. How would you test if the segment differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 16: User Segment Analysis](L9_8_16_explanation.md).

## Question 17

### Problem Statement
You're comparing two models using Wilcoxon signed-rank test for non-parametric comparison.

**Paired Results (Model A vs Model B):**
- Sample 1: A=$0.82$, B=$0.85$
- Sample 2: A=$0.88$, B=$0.86$
- Sample 3: A=$0.84$, B=$0.87$
- Sample 4: A=$0.86$, B=$0.88$
- Sample 5: A=$0.83$, B=$0.84$

#### Task
1. Calculate the differences (B - A) for each sample
2. Rank the absolute differences
3. Calculate the sum of ranks for positive differences
4. Calculate the sum of ranks for negative differences
5. What does this tell you about which model performs better?

For a detailed explanation of this question, see [Question 17: Wilcoxon Signed-Rank Test](L9_8_17_explanation.md).

## Question 18

### Problem Statement
Consider a model that shows performance variation across different geographic regions.

**Regional Performance:**
- Region 1: $87\%$ accuracy, $600$ samples
- Region 2: $84\%$ accuracy, $600$ samples
- Region 3: $89\%$ accuracy, $600$ samples

#### Task
1. Calculate the mean and standard deviation across regions
2. Calculate the coefficient of variation
3. What does this variation suggest about geographic generalization?
4. How would you test if the regional differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 18: Geographic Analysis](L9_8_18_explanation.md).

## Question 19

### Problem Statement
You're evaluating a model using different evaluation metrics and want to understand trade-offs.

**Metric Results:**
- Accuracy: $86\%$
- Precision: $84\%$
- Recall: $88\%$
- Specificity: $85\%$

#### Task
1. Calculate the F1-score
2. Calculate the balanced accuracy
3. Which metric would you prioritize for this application?
4. What additional information would you need to make a complete assessment?
5. How would you handle conflicting metric results?

For a detailed explanation of this question, see [Question 19: Metric Trade-offs](L9_8_19_explanation.md).

## Question 20

### Problem Statement
Consider a model that shows performance variation across different data collection methods.

**Collection Method Performance:**
- Method 1: $88\%$ accuracy, $400$ samples
- Method 2: $85\%$ accuracy, $400$ samples
- Method 3: $87\%$ accuracy, $400$ samples

#### Task
1. Calculate the mean and standard deviation across methods
2. Calculate the range of performance
3. What does this variation suggest about data quality?
4. How would you test if the method differences are statistically significant?
5. What would you recommend based on these results?

For a detailed explanation of this question, see [Question 20: Data Collection Analysis](L9_8_20_explanation.md).

## Question 21

### Problem Statement
You're comparing multiple models and need to control for family-wise error rate.

**Model Comparison Results:**
- Model A vs B: $p$-value = $0.02$
- Model A vs C: $p$-value = $0.01$
- Model A vs D: $p$-value = $0.03$
- Model B vs C: $p$-value = $0.04$
- Model B vs D: $p$-value = $0.02$
- Model C vs D: $p$-value = $0.01$

#### Task
1. How many comparisons are being made?
2. If you use Bonferroni correction with $\alpha = 0.05$, what's the adjusted significance level?
3. Which comparisons are significant after Bonferroni correction?
4. What's the advantage of controlling family-wise error rate?
5. What alternative method could you use?

For a detailed explanation of this question, see [Question 21: Family-Wise Error Rate](L9_8_21_explanation.md).

## Question 22

### Problem Statement
Consider a model that shows performance variation across different experimental conditions.

**Condition Performance:**
- Condition 1: $86\%$ accuracy, $500$ samples
- Condition 2: $89\%$ accuracy, $500$ samples
- Condition 3: $83\%$ accuracy, $500$ samples

#### Task
1. Calculate the mean and standard deviation across conditions
2. Calculate the coefficient of variation
3. What does this variation suggest about experimental control?
4. How would you test if the condition differences are statistically significant?
5. What would you do if the differences are significant?

For a detailed explanation of this question, see [Question 22: Experimental Condition Analysis](L9_8_22_explanation.md).

## Question 23

### Problem Statement
You're evaluating a model using different evaluation criteria and want to understand their relationships.

**Criteria Results:**
- Criterion 1: $85\%$ score, weight = $0.4$
- Criterion 2: $88\%$ score, weight = $0.3$
- Criterion 3: $82\%$ score, weight = $0.3$

#### Task
1. Calculate the weighted average score
2. If you had to choose one criterion, which would you pick and why?
3. What additional information would you need to make a complete assessment?
4. How would you handle conflicting criteria results?
5. What would you do if the weighted score is below your threshold?

For a detailed explanation of this question, see [Question 23: Multi-Criteria Evaluation](L9_8_23_explanation.md).

## Question 24

### Problem Statement
Consider a model that shows performance variation across different validation strategies.

**Validation Strategy Performance:**
- Strategy 1: $87\%$ accuracy, $600$ samples
- Strategy 2: $84\%$ accuracy, $600$ samples
- Strategy 3: $86\%$ accuracy, $600$ samples

#### Task
1. Calculate the mean and standard deviation across strategies
2. Calculate the range of performance
3. What does this variation suggest about validation robustness?
4. How would you test if the strategy differences are statistically significant?
5. What would you recommend based on these results?

For a detailed explanation of this question, see [Question 24: Validation Strategy Analysis](L9_8_24_explanation.md).

## Question 25

### Problem Statement
You're comparing two models using different statistical tests and want to understand their differences.

**Model Performance:**
- Model X: $85\%$ accuracy, $1000$ samples
- Model Y: $88\%$ accuracy, $1000$ samples

#### Task
1. Calculate the standard error for each model
2. Calculate the standard error of the difference
3. Calculate the $z$-score for the difference
4. Calculate the $95\%$ confidence interval for the difference
5. What do these results tell you about the practical significance of the difference?

For a detailed explanation of this question, see [Question 25: Practical vs Statistical Significance](L9_8_25_explanation.md).
