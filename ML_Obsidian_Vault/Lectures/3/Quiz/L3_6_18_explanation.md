# Question 18: Linear Regression Model Diagnostics and Improvement

## Problem Statement
Analyze the following scenario and answer the questions.

You are building a linear regression model to predict house prices. After training your model, you notice that:
- The training error is very low
- The test error is much higher than the training error
- The residuals exhibit a clear pattern when plotted against the predicted values
- The learning curve shows that validation error initially decreases with more training data but then plateaus

### Task
1. What problem(s) is your model likely facing? Be specific.
2. Describe TWO specific strategies you could use to address the identified problems.
3. Which evaluation technique would be most appropriate to assess if your strategies have improved the model? Explain your choice.
4. If you had to choose between collecting more training data or adding more features to your model in this scenario, which would you recommend? Justify your answer.

## Understanding the Problem
This problem focuses on diagnosing and addressing issues in a linear regression model for house price prediction. The symptoms described are common indicators of model problems that impact predictive performance. 

In a well-performing model, we expect:
- Similar errors on training and test sets, i.e., $|\text{MSE}_{\text{train}} - \text{MSE}_{\text{test}}| \approx 0$
- Randomly distributed residuals with no pattern, i.e., $\text{Cov}(\hat{\epsilon}, \hat{y}) \approx 0$
- Continuous improvement with additional data, i.e., $\frac{d}{dn}\text{MSE}_{\text{validation}}(n) < 0$ for all $n$

The symptoms described suggest systematic issues with the model that need to be identified and resolved for better performance.

## Solution

### Task 1: Identifying the Problem(s)

#### Step 1: Diagnosing overfitting and model misspecification
Based on the symptoms described, we can identify two primary issues:

1. **Overfitting**: The very low training error compared to much higher test error is a classic symptom of overfitting. Mathematically, if we denote the training error as $\text{MSE}_{\text{train}}$ and the test error as $\text{MSE}_{\text{test}}$, we observe:

   $$\text{MSE}_{\text{train}} \ll \text{MSE}_{\text{test}}$$

   This indicates the model is fitting too closely to the training data, including its noise, rather than learning the underlying pattern.

2. **Model Misspecification**: The pattern in residuals suggests that the model is not capturing some important relationships in the data. If we denote the residuals as $\hat{\epsilon}_i = y_i - \hat{y}_i$, we observe a systematic pattern when plotting $\hat{\epsilon}_i$ against $\hat{y}_i$. This violates the assumption of independence between residuals and predicted values:

   $$\text{Cov}(\hat{\epsilon}, \hat{y}) \neq 0$$

To demonstrate these issues, let's examine a synthetic house price prediction scenario with similar symptoms:

![Actual vs Predicted](../Images/L3_6_Quiz_18/actual_vs_predicted_overfit.png)

In this visualization, we can see how the training predictions fit almost perfectly to the actual values, while test predictions show significant deviations. The quantitative results confirm this extreme overfitting:
- Training MSE $\approx 0.00$
- Test MSE $\approx 1.46 \times 10^{12}$
- Training $R^2 \approx 1.00$
- Test $R^2 \approx -8.78 \times 10^{10}$
- Error ratio (Test/Train) $\approx 9.57 \times 10^{30}$

#### Step 2: Mathematical analysis through bias-variance decomposition

The expected test error for any model can be decomposed as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2_{\epsilon}$$

Where:
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$ is the expected difference between our model's prediction and the true function
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)^2] - \mathbb{E}[\hat{f}(x)]^2$ is the variance of our model's predictions
- $\sigma^2_{\epsilon}$ is the irreducible error due to noise

In our overfit model, the variance term dominates, causing poor generalization performance.

#### Step 3: Analysis of residual patterns

In a well-specified model, the residuals $\hat{\epsilon}_i = y_i - \hat{y}_i$ should be:

1. Independent of predicted values: $\text{Cov}(\hat{\epsilon}, \hat{y}) = 0$
2. Homoscedastic: $\text{Var}(\hat{\epsilon}_i | \hat{y}_i) = \sigma^2$ (constant)
3. Normally distributed: $\hat{\epsilon}_i \sim \mathcal{N}(0, \sigma^2)$

![Residual Pattern](../Images/L3_6_Quiz_18/residuals_vs_predicted.png)

The clear curved pattern in this residuals plot suggests a systematic relationship between residuals and predictions:

$$\hat{\epsilon}_i = g(\hat{y}_i) + \nu_i$$

Where $g$ is some non-zero function and $\nu_i$ is random noise. This indicates that important non-linear relationships are missing from our model.

#### Step 4: Learning curve analysis

![Learning Curve](../Images/L3_6_Quiz_18/learning_curve_overfit.png)

In a properly specified model with $n$ training samples, we expect:

$$\mathbb{E}[\text{MSE}_{\text{train}}(n)] = \sigma^2 \left(1 - \frac{p}{n}\right)$$
$$\mathbb{E}[\text{MSE}_{\text{validation}}(n)] = \sigma^2 \left(1 + \frac{p}{n}\right)$$

Our learning curve shows:
1. A plateauing validation error: $\frac{d}{dn}\text{MSE}_{\text{validation}}(n) \approx 0$ for large $n$
2. A persistent gap: $\text{MSE}_{\text{validation}}(n) - \text{MSE}_{\text{train}}(n) \gg 0$ 

This confirms that our model suffers from both overfitting and misspecification.

### Task 2: Strategies to Address the Problems

#### Strategy 1: Regularization to address overfitting

Ridge regression adds a penalty term to the loss function:

$$\hat{\beta}_{\text{ridge}} = \arg\min_{\beta} \left\{ \|y - X\beta\|^2 + \alpha \|\beta\|^2_2 \right\}$$

Where $\alpha > 0$ is the regularization parameter. The closed-form solution is:

$$\hat{\beta}_{\text{ridge}} = (X^TX + \alpha I)^{-1}X^Ty$$

Ridge regression improves the conditioning of the $X^TX$ matrix:

$$\kappa_{\text{ridge}} = \frac{\lambda_{\max}(X^TX) + \alpha}{\lambda_{\min}(X^TX) + \alpha}$$

As $\alpha$ increases, variance decreases at the cost of increased bias:

$$\text{Bias}[\hat{\beta}_{\text{ridge}}] = \alpha (X^TX + \alpha I)^{-1} \beta$$
$$\text{Var}[\hat{\beta}_{\text{ridge}}] = \sigma^2 (X^TX + \alpha I)^{-1} X^TX (X^TX + \alpha I)^{-1}$$

Applied to our model with $\alpha = 10.0$, ridge regression dramatically improves performance:
- Training MSE increases from $\approx 0$ to $3.78$ (indicating less overfitting)
- Test MSE decreases from $\approx 1.46 \times 10^{12}$ to $4.47$
- Error ratio decreases from $\approx 9.57 \times 10^{30}$ to $1.18$

The residuals also show significant improvement:

![Residuals Comparison](../Images/L3_6_Quiz_18/residuals_comparison.png)

#### Strategy 2: Feature engineering to address model misspecification

To capture non-linear relationships, we add transformed features. If the true relationship is:

$$y = f(X) + \epsilon$$

We can approximate it using basis function expansion:

$$f(X) \approx \sum_{j=1}^{m} \beta_j \phi_j(X)$$

Where $\phi_j$ are basis functions. For polynomial features:

$$\phi_j(X) = \prod_{k=1}^{p} X_k^{a_{jk}}$$

With $\sum_{k=1}^{p} a_{jk} \leq d$, where $d$ is the polynomial degree.

Based on our analysis of the residual patterns, we added specific non-linear terms:
- $X_1^2$ (square footage squared)
- $X_1X_2$ (size × bedrooms interaction)
- $X_1X_3^2$ (size × age² interaction)

Our new feature matrix becomes:

$$X_{\text{new}} = [X_1, X_2, X_3, X_1^2, X_1X_2, X_1X_3^2]$$

After feature engineering, we observe:
- Training MSE: $4.07$
- Test MSE: $4.24$
- Error ratio: $1.04$

This indicates our model is now well-specified and not overfitting.

### Task 3: Evaluation Technique

#### Cross-validation as the optimal evaluation strategy

K-fold cross-validation divides the training data into $K$ subsets (folds). For each $k \in \{1,2,...,K\}$, we train on all folds except the $k$-th fold, and validate on the $k$-th fold.

Formally, let $\kappa: \{1,2,...,n\} \mapsto \{1,2,...,K\}$ be a function that indicates the fold assignment for each observation. The K-fold cross-validation error is:

$$\text{CV}(K) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{f}^{-\kappa(i)}(x_i)\right)^2$$

Where $\hat{f}^{-\kappa(i)}$ is the model trained on all folds except $\kappa(i)$.

Cross-validation is the most appropriate evaluation technique because:

1. **Unbiasedness**: $\mathbb{E}[\text{CV}(K)]$ is approximately unbiased for expected test error
2. **Consistency**: As $n \to \infty$, CV selects the optimal model with probability approaching 1
3. **Efficiency**: CV makes efficient use of limited data
4. **Robustness**: It provides reliable estimates for both simple and complex models

For our models, we used $K=5$ cross-validation to obtain reliable estimates of generalization performance.

Additionally, the combination of:
- Residual analysis (to verify homoscedasticity and normality)
- Learning curve analysis (to verify convergence properties)
- Model comparison metrics (MSE, $R^2$, error ratio)

Provides a comprehensive evaluation of model improvements.

### Task 4: More Data vs. More Features

#### Theoretical analysis of sample size vs. model complexity

For a model with complexity (number of parameters) $p$ and training size $n$, the expected test error can be approximated as:

$$\mathbb{E}[\text{MSE}_{\text{test}}(n,p)] \approx \sigma^2 + \text{Bias}^2(p) + \frac{p\sigma^2}{n}$$

Where:
- $\sigma^2$ is the irreducible error
- $\text{Bias}^2(p)$ is the squared bias, which decreases as $p$ increases
- $\frac{p\sigma^2}{n}$ is the variance term, which increases with $p$ and decreases with $n$

When the model is misspecified (inadequate complexity), $\text{Bias}^2(p)$ dominates and:

$$\lim_{n \to \infty} \mathbb{E}[\text{MSE}_{\text{test}}(n,p)] = \sigma^2 + \text{Bias}^2(p) > \sigma^2$$

Thus, increasing $n$ cannot reduce error below $\sigma^2 + \text{Bias}^2(p)$.

The comparison of our learning curves shows:

![Training Size Comparison](../Images/L3_6_Quiz_18/training_size_comparison.png)

From the analysis and empirical results, **adding more features** would be more beneficial than collecting more data because:

1. The plateau in the learning curve indicates that additional data will not significantly improve model performance
2. The bias term from model misspecification dominates: $\text{Bias}^2(p) \gg \frac{p\sigma^2}{n}$
3. Empirical results show better performance with feature engineering: $4.24$ vs. $4.47$ MSE

Empirical results support this conclusion:
- Feature-engineered model: Test MSE = $4.24$
- Regularized model with original features: Test MSE = $4.47$

The properly specified model with appropriate features outperforms the regularized model on the same dataset, confirming that addressing model specification is more important than increasing sample size in this scenario.

## Visual Explanations

### Model performance comparison

![Model Comparison](../Images/L3_6_Quiz_18/model_comparison.png)

This visualization shows the performance of all models according to the optimality criteria:

$$\mathcal{M}^* = \arg\min_{\mathcal{M} \in \mathfrak{M}} \mathbb{E}_{(X,y) \sim \mathcal{D}}[(y - \mathcal{M}(X))^2]$$

Both improved models significantly outperform the original overfit model, with feature engineering providing slightly better results than regularization alone.

## Conclusion

### Task 1: Problem diagnosis
- **Overfitting**: Formalized by $\text{MSE}_{\text{train}} \ll \text{MSE}_{\text{test}}$ ($0.00$ vs. $1.46 \times 10^{12}$)
- **Model misspecification**: Characterized by systematic patterns in residuals with $\text{Cov}(\hat{\epsilon}, \hat{y}) \neq 0$

### Task 2: Solution strategies
- **Regularization**: Implemented ridge regression with $\alpha = 10.0$, reducing test MSE to $4.47$
- **Feature engineering**: Added non-linear terms ($X_1^2, X_1X_2, X_1X_3^2$), reducing test MSE to $4.24$

### Task 3: Evaluation technique
- **Cross-validation**: Provides unbiased, consistent, and efficient estimates of generalization performance
- **Comprehensive evaluation framework**: Combines residual analysis, learning curves, and comparative metrics

### Task 4: Data vs. features recommendation
- **Adding more features** is preferable because:
  - The learning curve plateaus, indicating limited benefit from more data
  - The bias term from model misspecification dominates: $\text{Bias}^2(p) \gg \frac{p\sigma^2}{n}$
  - Empirical results show better performance with feature engineering: $4.24$ vs. $4.47$ MSE 