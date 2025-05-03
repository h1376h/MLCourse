# Question 19: LDA for Credit Approval

## Problem Statement
A bank is using LDA to classify credit applications as approved (1) or denied (0) based on annual income (thousands of dollars) and debt-to-income ratio (percentage). The following table shows training data from previous applications:

| Income ($K) | Debt-to-Income (%) | Credit Approved (y) |
|-------------|---------------------|---------------------|
| 65          | 28                  | 1                   |
| 50          | 32                  | 0                   |
| 79          | 22                  | 1                   |
| 48          | 40                  | 0                   |
| 95          | 18                  | 1                   |
| 36          | 36                  | 0                   |
| 72          | 30                  | 1                   |
| 60          | 34                  | 0                   |
| 85          | 24                  | 1                   |
| 42          | 38                  | 0                   |

### Task
1. Calculate the class means for approved and denied applications
2. Calculate the pooled within-class covariance matrix
3. Find the between-class covariance matrix $S_B$
4. Determine the optimal projection direction for the LDA by finding the eigenvector corresponding to the largest eigenvalue of $S_W^{-1}S_B$
5. Calculate the threshold for classification assuming the prior probabilities are $P(y=1) = 0.3$ and $P(y=0) = 0.7$
6. For a new applicant with income $55K and debt-to-income ratio 25%, which class would LDA predict? Will their credit application be approved or denied?

## Understanding the Problem
Linear Discriminant Analysis (LDA) is a statistical method for finding a linear combination of features that best separates two or more classes. For binary classification, LDA projects the data onto a single dimension that maximizes between-class separation while minimizing within-class variance. This problem involves using LDA to develop a credit approval model based on two features: income and debt-to-income ratio.

## Solution

### Step 1: Calculate the class means for approved and denied applications
First, we separate the data points by class:

**Approved applications (Class 1):**
- Sample 1: Income = $65K, Debt-to-Income = 28%
- Sample 2: Income = $79K, Debt-to-Income = 22%
- Sample 3: Income = $95K, Debt-to-Income = 18%
- Sample 4: Income = $72K, Debt-to-Income = 30%
- Sample 5: Income = $85K, Debt-to-Income = 24%

**Denied applications (Class 0):**
- Sample 1: Income = $50K, Debt-to-Income = 32%
- Sample 2: Income = $48K, Debt-to-Income = 40%
- Sample 3: Income = $36K, Debt-to-Income = 36%
- Sample 4: Income = $60K, Debt-to-Income = 34%
- Sample 5: Income = $42K, Debt-to-Income = 38%

Now, we calculate the mean vector for each class:

For the approved class (y=1):
$$\begin{align}
\text{Mean Income} &= \frac{65 + 79 + 95 + 72 + 85}{5} = \frac{396}{5} = 79.2 \\
\text{Mean Debt-to-Income} &= \frac{28 + 22 + 18 + 30 + 24}{5} = \frac{122}{5} = 24.4
\end{align}$$

Therefore, $\boldsymbol{\mu}_1 = \begin{pmatrix} 79.2 \\ 24.4 \end{pmatrix}$

For the denied class (y=0):
$$\begin{align}
\text{Mean Income} &= \frac{50 + 48 + 36 + 60 + 42}{5} = \frac{236}{5} = 47.2 \\
\text{Mean Debt-to-Income} &= \frac{32 + 40 + 36 + 34 + 38}{5} = \frac{180}{5} = 36.0
\end{align}$$

Therefore, $\boldsymbol{\mu}_0 = \begin{pmatrix} 47.2 \\ 36.0 \end{pmatrix}$

We can observe that approved applications tend to have higher income and lower debt-to-income ratio compared to denied applications, which aligns with intuitive expectations.

### Step 2: Calculate the pooled within-class covariance matrix
To calculate the covariance matrices for each class, we first need to center the data by subtracting the respective class means:

**Approved class (centered data):**
$$\mathbf{X}_1 - \boldsymbol{\mu}_1 = 
\begin{pmatrix} 
65 - 79.2 & 28 - 24.4 \\
79 - 79.2 & 22 - 24.4 \\
95 - 79.2 & 18 - 24.4 \\
72 - 79.2 & 30 - 24.4 \\
85 - 79.2 & 24 - 24.4
\end{pmatrix} = 
\begin{pmatrix} 
-14.2 & 3.6 \\
-0.2 & -2.4 \\
15.8 & -6.4 \\
-7.2 & 5.6 \\
5.8 & -0.4
\end{pmatrix}$$

Now compute the covariance matrix for the approved class:
$$\mathbf{S}_1 = \frac{1}{n_1 - 1}\sum_{i=1}^{n_1} (\mathbf{x}_i - \boldsymbol{\mu}_1)(\mathbf{x}_i - \boldsymbol{\mu}_1)^T$$

This gives us:
$$\mathbf{S}_1 = \begin{pmatrix} 134.2 & -48.6 \\ -48.6 & 22.8 \end{pmatrix}$$

**Denied class (centered data):**
$$\mathbf{X}_0 - \boldsymbol{\mu}_0 = 
\begin{pmatrix} 
50 - 47.2 & 32 - 36.0 \\
48 - 47.2 & 40 - 36.0 \\
36 - 47.2 & 36 - 36.0 \\
60 - 47.2 & 34 - 36.0 \\
42 - 47.2 & 38 - 36.0
\end{pmatrix} = 
\begin{pmatrix} 
2.8 & -4.0 \\
0.8 & 4.0 \\
-11.2 & 0.0 \\
12.8 & -2.0 \\
-5.2 & 2.0
\end{pmatrix}$$

Computing the covariance matrix for the denied class:
$$\mathbf{S}_0 = \begin{pmatrix} 81.2 & -11.0 \\ -11.0 & 10.0 \end{pmatrix}$$

The pooled within-class covariance matrix $\mathbf{S}_W$ is calculated as a weighted average of individual class covariances:

$$\mathbf{S}_W = \frac{(n_1 - 1)\mathbf{S}_1 + (n_0 - 1)\mathbf{S}_0}{n_1 + n_0 - 2}$$

Substituting our values:
$$\begin{align}
\mathbf{S}_W &= \frac{(5 - 1)\mathbf{S}_1 + (5 - 1)\mathbf{S}_0}{5 + 5 - 2} \\
&= \frac{4\mathbf{S}_1 + 4\mathbf{S}_0}{8} \\
&= \frac{4\begin{pmatrix} 134.2 & -48.6 \\ -48.6 & 22.8 \end{pmatrix} + 4\begin{pmatrix} 81.2 & -11.0 \\ -11.0 & 10.0 \end{pmatrix}}{8} \\
&= \begin{pmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{pmatrix}
\end{align}$$

The negative off-diagonal elements indicate a negative correlation between income and debt-to-income ratio within each class, which makes sense as higher-income individuals generally have lower debt-to-income ratios.

### Step 3: Calculate the between-class scatter matrix
The between-class scatter matrix $\mathbf{S}_B$ represents the separation between classes and is calculated as:

$$\mathbf{S}_B = \frac{n_1 n_0}{n} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T$$

First, let's calculate the difference between class means:
$$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 = \begin{pmatrix} 79.2 \\ 24.4 \end{pmatrix} - \begin{pmatrix} 47.2 \\ 36.0 \end{pmatrix} = \begin{pmatrix} 32.0 \\ -11.6 \end{pmatrix}$$

Now, we compute the outer product:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T = \begin{pmatrix} 32.0 \\ -11.6 \end{pmatrix} \begin{pmatrix} 32.0 & -11.6 \end{pmatrix} = \begin{pmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{pmatrix}$$

Finally, the between-class scatter matrix:
$$\begin{align}
\mathbf{S}_B &= \frac{n_1 n_0}{n} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T \\
&= \frac{5 \cdot 5}{10} \begin{pmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{pmatrix} \\
&= 2.5 \begin{pmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{pmatrix} \\
&= \begin{pmatrix} 2560.0 & -928.0 \\ -928.0 & 336.4 \end{pmatrix}
\end{align}$$

For binary classification, $\mathbf{S}_B$ always has rank 1 because it's the outer product of a single vector, which is confirmed in our calculation.

### Step 4: Determine the optimal projection direction
For binary classification, the optimal projection direction $\mathbf{w}$ can be calculated directly using:

$$\mathbf{w} = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$$

First, we calculate the inverse of the pooled within-class covariance matrix:
$$\mathbf{S}_W^{-1} = \begin{pmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{pmatrix}^{-1} = \begin{pmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{pmatrix}$$

Then, we compute the optimal projection direction:
$$\begin{align}
\mathbf{w} &= \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \\
&= \begin{pmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{pmatrix} \begin{pmatrix} 32.0 \\ -11.6 \end{pmatrix} \\
&= \begin{pmatrix} 0.2040 \\ -0.3367 \end{pmatrix}
\end{align}$$

We can normalize $\mathbf{w}$ to unit length for easier interpretation:
$$\begin{align}
\|\mathbf{w}\| &= \sqrt{0.2040^2 + (-0.3367)^2} = 0.3937 \\
\mathbf{w}_{\text{norm}} &= \frac{\mathbf{w}}{\|\mathbf{w}\|} = \frac{1}{0.3937}\begin{pmatrix} 0.2040 \\ -0.3367 \end{pmatrix} = \begin{pmatrix} 0.5181 \\ -0.8553 \end{pmatrix}
\end{align}$$

Alternatively, we can find this direction as the eigenvector corresponding to the largest eigenvalue of $\mathbf{S}_W^{-1}\mathbf{S}_B$:

$$\mathbf{S}_W^{-1}\mathbf{S}_B = \begin{pmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{pmatrix} \begin{pmatrix} 2560.0 & -928.0 \\ -928.0 & 336.4 \end{pmatrix} = \begin{pmatrix} 16.3163 & -5.9146 \\ -26.9375 & 9.7648 \end{pmatrix}$$

Computing the eigenvalues and eigenvectors:
- Eigenvalues: $\lambda_1 = 26.0811$, $\lambda_2 \approx 0$
- First eigenvector: $\mathbf{v}_1 = \begin{pmatrix} 0.5181 \\ -0.8553 \end{pmatrix}$

The projection direction indicates that LDA assigns a positive weight to income and a negative weight to debt-to-income ratio, which aligns with the expectation that higher income is favorable for approval, while higher debt-to-income ratio is unfavorable.

### Step 5: Calculate the threshold for classification
The threshold for classification depends on the projected class means and prior probabilities.

First, we project the class means onto the direction $\mathbf{w}$:

$$\begin{align}
\mathbf{w}^T\boldsymbol{\mu}_1 &= \begin{pmatrix} 0.2040 & -0.3367 \end{pmatrix} \begin{pmatrix} 79.2 \\ 24.4 \end{pmatrix} \\
&= 0.2040 \cdot 79.2 + (-0.3367) \cdot 24.4 \\
&= 16.1531 + (-8.2159) \\
&= 7.9372
\end{align}$$

$$\begin{align}
\mathbf{w}^T\boldsymbol{\mu}_0 &= \begin{pmatrix} 0.2040 & -0.3367 \end{pmatrix} \begin{pmatrix} 47.2 \\ 36.0 \end{pmatrix} \\
&= 0.2040 \cdot 47.2 + (-0.3367) \cdot 36.0 \\
&= 9.6266 + (-12.1219) \\
&= -2.4953
\end{align}$$

With equal prior probabilities, the threshold would be the midpoint of the projected means:
$$\text{threshold}_{\text{equal}} = \frac{\mathbf{w}^T\boldsymbol{\mu}_1 + \mathbf{w}^T\boldsymbol{\mu}_0}{2} = \frac{7.9372 + (-2.4953)}{2} = 2.7209$$

However, given the unequal prior probabilities $P(y=1) = 0.3$ and $P(y=0) = 0.7$, we adjust the threshold according to:

$$\text{threshold} = \frac{\mathbf{w}^T\boldsymbol{\mu}_1 + \mathbf{w}^T\boldsymbol{\mu}_0}{2} + \frac{1}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}} \ln\left(\frac{P(y=0)}{P(y=1)}\right)$$

First, we calculate $\mathbf{w}^T\mathbf{S}_W\mathbf{w}$:
$$\begin{align}
\mathbf{w}^T\mathbf{S}_W\mathbf{w} &= \begin{pmatrix} 0.2040 & -0.3367 \end{pmatrix} \begin{pmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{pmatrix} \begin{pmatrix} 0.2040 \\ -0.3367 \end{pmatrix} \\
&= 10.4324
\end{align}$$

Next, the logarithm of the prior ratio:
$$\ln\left(\frac{P(y=0)}{P(y=1)}\right) = \ln\left(\frac{0.7}{0.3}\right) = \ln(2.3333) = 0.8473$$

Finally, the threshold with priors:
$$\begin{align}
\text{threshold} &= \frac{7.9372 + (-2.4953)}{2} + \frac{0.8473}{10.4324} \\
&= 2.7209 + 0.0812 \\
&= 2.8022
\end{align}$$

The threshold shifts slightly toward the approved class mean due to the lower prior probability assigned to approvals, making approvals more selective.

### Step 6: Predict class for a new applicant
For a new applicant with income $55K and debt-to-income ratio 25%, we project their data onto the LDA direction:

$$\begin{align}
\mathbf{w}^T\mathbf{x}_{\text{new}} &= \begin{pmatrix} 0.2040 & -0.3367 \end{pmatrix} \begin{pmatrix} 55 \\ 25 \end{pmatrix} \\
&= 0.2040 \cdot 55 + (-0.3367) \cdot 25 \\
&= 11.2174 + (-8.4180) \\
&= 2.7995
\end{align}$$

Comparing with the threshold:
- Projected value: $2.7995$
- Threshold with priors: $2.8022$

Since the projected value $(2.7995)$ is less than the threshold $(2.8022)$, the LDA model predicts that this application will be denied. This is a very close decision, as the applicant's projected value is only slightly below the threshold.

## Visual Explanations

### LDA for Credit Approval
![LDA for Credit Approval Decision](../Images/L4_4_Quiz_19/lda_credit_approval.png)

This visualization shows the data points in the original feature space, with green circles representing approved applications and red crosses representing denied applications. The decision boundary is shown as a black line, which is perpendicular to the LDA direction (blue arrow). The new applicant is shown as a purple diamond, falling just on the "denied" side of the boundary.

### LDA Projection
![LDA Projection](../Images/L4_4_Quiz_19/lda_projection.png)

This plot shows all data points projected onto the LDA direction. Approved applications (green) are well-separated from denied applications (red). The threshold with priors (solid vertical line) defines the decision boundary. The new applicant (purple diamond) falls just to the left of the threshold, resulting in a denial prediction.

### Feature Importance
![Feature Importance](../Images/L4_4_Quiz_19/feature_importance.png)

This bar chart shows the relative importance of each feature in the LDA model. Debt-to-income ratio has a higher weight (62%) compared to income (38%), indicating that it has more influence on the classification decision. The negative weight for debt-to-income ratio means that higher values push toward denial.

## Key Insights

### Statistical Foundations
- LDA works by finding a projection direction that maximizes the ratio of between-class variance to within-class variance
- For binary classification, there is only one discriminant direction (the rank of $\mathbf{S}_B$ is 1)
- The covariance structure within classes influences the direction of optimal separation
- The negative correlation between income and debt-to-income ratio is accounted for in the LDA model through the pooled covariance matrix

### Practical Implications
- The model exhibits expected behavior: higher income increases approval likelihood, while higher debt-to-income ratio decreases it
- The debt-to-income ratio has a greater impact on the decision than income (62% vs. 38% importance)
- Prior probabilities affect the threshold placement and therefore the decision boundary
- Using priors of $P(y=1) = 0.3$ and $P(y=0) = 0.7$ makes approval more selective by shifting the threshold toward the approved class mean

### Decision Making
- The LDA model provides a principled way to make credit decisions by projecting applicant data onto the optimal direction
- The projection reduces the original 2D problem to a 1D comparison against a threshold
- The model can be adjusted by changing priors to reflect different approval policies (more or less selective)
- The closeness of the new applicant to the threshold suggests uncertainty in the decision

## Conclusion
The LDA analysis of credit approval data has yielded several important results:

1. There are clear statistical differences between approved and denied applications, with approved applications having higher income (mean $79.2K vs $47.2K) and lower debt-to-income ratios (mean 24.4% vs 36.0%).

2. The optimal LDA projection direction $\mathbf{w} = \begin{pmatrix} 0.2040 \\ -0.3367 \end{pmatrix}$ assigns a positive weight to income and a negative weight to debt-to-income ratio, confirming that higher income contributes positively to approval while higher debt-to-income ratio contributes negatively.

3. With the given prior probabilities $P(y=1) = 0.3$ and $P(y=0) = 0.7$, the classification threshold is 2.8022.

4. For a new applicant with income $55K and debt-to-income ratio 25%, the model predicts a denial of credit, though the decision is very close (projected value 2.7995 vs. threshold 2.8022).

This LDA model provides a principled statistical approach to credit approval decisions based on the patterns observed in historical data while incorporating prior beliefs about approval rates. 