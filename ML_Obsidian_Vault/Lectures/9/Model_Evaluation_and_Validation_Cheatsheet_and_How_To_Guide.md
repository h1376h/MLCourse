# Model Evaluation and Validation Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core Evaluation Formulas

**Classification Metrics:**
- **Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$
- **F1 Score**: $\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Regression Metrics:**
- **MSE**: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE**: $\text{RMSE} = \sqrt{\text{MSE}}$
- **MAE**: $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **R²**: $\text{R²} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

**Statistical Tests:**
- **Standard Error**: $\text{SE} = \sqrt{\frac{p(1-p)}{n}}$
- **Z-score**: $z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\text{SE}_1^2 + \text{SE}_2^2}}$

**Validation Methods:**
- **K-fold CV**: Each sample used $(k-1)$ times for training, $1$ time for testing
- **Bootstrap**: $P(\text{sample not selected}) = (1 - \frac{1}{n})^n$

---

## 🎯 Question Type 1: Overfitting and Underfitting Detection

### How to Approach:

**Step 1: Identify Performance Patterns**
- **Overfitting**: Training accuracy >> Test accuracy
- **Underfitting**: Training accuracy ≈ Test accuracy (both low)
- **Good fit**: Training accuracy ≈ Test accuracy (both high)

**Step 2: Calculate Generalization Gap**
- **Gap**: |Training error - Test error|
- **Large gap**: Overfitting
- **Small gap**: Good generalization

**Step 3: Analyze Learning Curves**
- **Overfitting**: Training error decreases, validation error increases
- **Underfitting**: Both errors remain high
- **Good fit**: Both errors decrease and converge

### Example Template:
```
Given: Model with [training]% training and [test]% test performance
1. Performance analysis:
   - Training accuracy: [value]%
   - Test accuracy: [value]%
   - Generalization gap: |[training] - [test]| = [gap]%
2. Pattern identification:
   - [Overfitting/Underfitting/Good fit] because [reasoning]
3. Recommendations:
   - For [overfitting/underfitting]: [specific solution]
```

---

## 🎯 Question Type 2: Classification Metrics Calculation

### How to Approach:

**Step 1: Build Confusion Matrix**
- **TP**: True Positives (correctly predicted positive)
- **TN**: True Negatives (correctly predicted negative)
- **FP**: False Positives (incorrectly predicted positive)
- **FN**: False Negatives (incorrectly predicted negative)

**Step 2: Calculate Metrics**
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity, true positive rate
- **F1 Score**: Harmonic mean of precision and recall

**Step 3: Interpret Results**
- **High precision**: Few false positives
- **High recall**: Few false negatives
- **Balanced F1**: Good overall performance

### Example Template:
```
Given: Confusion matrix with [TP], [TN], [FP], [FN]
1. Metrics calculation:
   - Accuracy = ([TP] + [TN]) / ([TP] + [TN] + [FP] + [FN]) = [value]
   - Precision = [TP] / ([TP] + [FP]) = [value]
   - Recall = [TP] / ([TP] + [FN]) = [value]
   - F1 Score = 2 × [precision] × [recall] / ([precision] + [recall]) = [value]
2. Interpretation:
   - [High/Medium/Low] precision indicates [few/many] false positives
   - [High/Medium/Low] recall indicates [few/many] false negatives
```

---

## 🎯 Question Type 3: Regression Metrics Calculation

### How to Approach:

**Step 1: Calculate Error Metrics**
- **MSE**: Mean squared error (sensitive to outliers)
- **RMSE**: Root mean squared error (same units as target)
- **MAE**: Mean absolute error (robust to outliers)

**Step 2: Calculate Goodness of Fit**
- **R²**: Coefficient of determination
- **Adjusted R²**: Penalized for number of features

**Step 3: Compare Models**
- **Lower error metrics**: Better prediction accuracy
- **Higher R²**: Better fit to data

### Example Template:
```
Given: Actual values [y] and predicted values [ŷ]
1. Error calculations:
   - MSE = Σ(yᵢ - ŷᵢ)² / n = [value]
   - RMSE = √MSE = [value]
   - MAE = Σ|yᵢ - ŷᵢ| / n = [value]
2. Goodness of fit:
   - R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)² = [value]
   - Variance explained: [R² × 100]%
```

---

## 🎯 Question Type 4: ROC Curve and AUC Analysis

### How to Approach:

**Step 1: Calculate TPR and FPR**
- **TPR (Sensitivity)**: $\frac{TP}{TP + FN}$
- **FPR (1-Specificity)**: $\frac{FP}{FP + TN}$

**Step 2: Plot ROC Curve**
- **X-axis**: False Positive Rate (FPR)
- **Y-axis**: True Positive Rate (TPR)
- **Points**: (FPR, TPR) for each threshold

**Step 3: Calculate AUC**
- **Trapezoidal rule**: Sum of trapezoid areas
- **Perfect classifier**: AUC = 1.0
- **Random classifier**: AUC = 0.5

### Example Template:
```
Given: Model predictions with [thresholds] and [performance]
1. TPR/FPR calculation:
   - Threshold [value]: TPR = [value], FPR = [value]
2. AUC interpretation:
   - AUC = [value]
   - [Excellent/Good/Poor] discrimination
```

---

## 🎯 Question Type 5: Cross-Validation Analysis

### How to Approach:

**Step 1: Calculate CV Parameters**
- **Fold size**: n/K for k-fold CV
- **Training samples**: (K-1) × fold_size
- **Test samples**: fold_size
- **Total models**: K models trained

**Step 2: Analyze CV Results**
- **Mean performance**: Average across folds
- **Standard deviation**: Measure of stability
- **Overfitting detection**: High variance across folds

### Example Template:
```
Given: [n] samples with [k]-fold cross-validation
1. CV parameters:
   - Fold size: [n]/[k] = [fold_size] samples per fold
   - Total models: [k] models trained
2. Performance analysis:
   - Mean: [mean_value]
   - Standard deviation: [std_value]
   - Stability: [high/medium/low] based on [cv_value]
```

---

## 🎯 Question Type 6: Sampling Techniques Analysis

### How to Approach:

**Step 1: Identify Sampling Method**
- **Simple random**: Equal probability for all samples
- **Stratified**: Maintains subgroup proportions
- **Systematic**: Regular interval sampling
- **Cluster**: Group-based sampling

**Step 2: Calculate Sampling Parameters**
- **Sample size**: Number of samples to select
- **Sampling fraction**: n/N ratio
- **Selection probability**: P(sample selected)

### Example Template:
```
Given: Population of [N] with [groups] and sample size [n]
1. Sampling parameters:
   - Sample size: [n] samples
   - Sampling fraction: [n]/[N] = [fraction]
   - Selection probability: [n]/[N] = [prob]
2. Method comparison:
   - [Method]: [bias] bias, [variance] variance
```

---

## 🎯 Question Type 7: Bootstrap Analysis

### How to Approach:

**Step 1: Calculate Bootstrap Probabilities**
- **Selection probability**: P(value selected) = 1/n
- **Non-selection probability**: P(value not selected) = (1-1/n)^n
- **Bootstrap sample count**: n^n possible samples

**Step 2: Estimate Bootstrap Statistics**
- **Bootstrap mean**: Average of bootstrap estimates
- **Bootstrap bias**: Difference from original statistic
- **Bootstrap standard error**: Standard deviation of estimates

### Example Template:
```
Given: Dataset [X] with [n] values and [B] bootstrap samples
1. Probability calculations:
   - Selection probability: 1/[n] = [prob]
   - Non-selection probability: (1 - 1/[n])^[n] = [non_prob]
2. Bootstrap statistics:
   - Bootstrap mean: [mean of estimates]
   - Bootstrap bias: [bootstrap mean] - [original] = [bias]
   - Bootstrap SE: [standard deviation of estimates]
```

---

## 🎯 Question Type 8: Statistical Significance Testing

### How to Approach:

**Step 1: Formulate Hypotheses**
- **Null hypothesis (H₀)**: No difference/effect
- **Alternative hypothesis (H₁)**: There is a difference/effect
- **Significance level (α)**: Type I error rate

**Step 2: Calculate Test Statistics**
- **Z-test**: For large samples, known variance
- **T-test**: For small samples, unknown variance
- **Standard error**: SE = √(SE₁² + SE₂²)

**Step 3: Make Decision**
- **Reject H₀**: Test statistic > critical value
- **Fail to reject H₀**: Test statistic ≤ critical value

### Example Template:
```
Given: Model A ([performance1]) vs Model B ([performance2])
1. Hypothesis formulation:
   - H₀: μA = μB (no difference in performance)
   - H₁: μA ≠ μB (different performance)
   - α = [0.05] (significance level)
2. Test statistic calculation:
   - Standard error: SE = √([SE₁²] + [SE₂²]) = [SE_value]
   - Z-score: z = ([performance1] - [performance2]) / [SE_value] = [z_value]
3. Decision:
   - [Reject/Fail to reject] H₀ because [reasoning]
```

---

## 🎯 Question Type 9: Model Comparison and Selection

### How to Approach:

**Step 1: Calculate Performance Differences**
- **Absolute difference**: |Metric_A - Metric_B|
- **Relative difference**: (Metric_A - Metric_B) / Metric_B
- **Statistical significance**: Z-test or t-test

**Step 2: Consider Trade-offs**
- **Performance vs complexity**: Better performance vs simpler model
- **Speed vs accuracy**: Faster vs more accurate
- **Interpretability vs performance**: Explainable vs black box

**Step 3: Make Recommendation**
- **Best model**: Based on primary criteria
- **Justification**: Clear reasoning for selection

### Example Template:
```
Given: Model A vs Model B with [metrics]
1. Performance comparison:
   - Model A: [metric1] = [value1]
   - Model B: [metric1] = [value1]
   - Absolute difference: |[value1A] - [value1B]| = [diff1]
2. Trade-off analysis:
   - Performance: [Model A/B] is [better/worse]
   - Complexity: [Model A/B] is [simpler/more complex]
3. Recommendation: [Model A/B] because [justification]
```

---

## 🎯 Question Type 10: Evaluation Best Practices and Pitfalls

### How to Approach:

**Step 1: Identify Common Pitfalls**
- **Data leakage**: Using test data in training
- **Overfitting**: Model memorizes training data
- **Selection bias**: Non-representative data

**Step 2: Apply Best Practices**
- **Proper data splitting**: Train/validation/test sets
- **Cross-validation**: Robust performance estimation
- **Multiple metrics**: Comprehensive evaluation

**Step 3: Validate Results**
- **Reproducibility**: Same results on different runs
- **Stability**: Consistent performance across folds
- **Generalization**: Performance on unseen data

### Example Template:
```
Given: [evaluation scenario] with [potential issues]
1. Pitfall identification:
   - Data leakage: [present/absent] - [description]
   - Overfitting: [present/absent] - [indication]
2. Best practice application:
   - Data splitting: [train/validation/test] = [60/20/20]%
   - Cross-validation: [k]-fold with [stratification]
3. Validation results:
   - Reproducibility: [consistent/variable] across [runs]
   - Stability: [low/high] variance across [folds]
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify evaluation type** - classification, regression, or comparison
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### Common Mistakes to Avoid:
- **Confusing precision and recall** - precision is TP/(TP+FP), recall is TP/(TP+FN)
- **Forgetting to square root RMSE** - RMSE = √MSE
- **Ignoring class imbalance** - accuracy can be misleading
- **Not considering statistical significance** - differences may not be meaningful
- **Overlooking data leakage** - using test data in training

### Quick Reference Decision Trees:

**Which Evaluation Metric?**
```
Classification → Accuracy, Precision, Recall, F1
Regression → MSE, RMSE, MAE, R²
Imbalanced → Precision, Recall, F1, AUC
```

**Which Validation Method?**
```
Small (<100) → Leave-one-out CV
Medium (100-1000) → 5-10 fold CV
Large (>1000) → Holdout or 3-5 fold CV
```

**How to Detect Overfitting?**
```
High train, low test → Overfitting
Low train, low test → Underfitting
High train, high test → Good fit
```

---

*This guide covers the most common Model Evaluation and Validation question types. Practice with each approach and adapt based on specific problem requirements. Remember: proper evaluation is crucial for reliable model deployment!*
