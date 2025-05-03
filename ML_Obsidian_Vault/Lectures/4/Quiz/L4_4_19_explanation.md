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

Therefore, $\boldsymbol{\mu}_1 = \begin{bmatrix} 79.2 \\ 24.4 \end{bmatrix}$

For the denied class (y=0):
$$\begin{align}
\text{Mean Income} &= \frac{50 + 48 + 36 + 60 + 42}{5} = \frac{236}{5} = 47.2 \\
\text{Mean Debt-to-Income} &= \frac{32 + 40 + 36 + 34 + 38}{5} = \frac{180}{5} = 36.0
\end{align}$$

Therefore, $\boldsymbol{\mu}_0 = \begin{bmatrix} 47.2 \\ 36.0 \end{bmatrix}$

We can observe that approved applications tend to have higher income and lower debt-to-income ratio compared to denied applications, which aligns with intuitive expectations.

### Step 2: Calculate the pooled within-class covariance matrix
To calculate the covariance matrices for each class, we first need to center the data by subtracting the respective class means:

**Approved class (centered data):**
$$\mathbf{X}_1 - \boldsymbol{\mu}_1 = 
\begin{bmatrix} 
65 - 79.2 & 28 - 24.4 \\
79 - 79.2 & 22 - 24.4 \\
95 - 79.2 & 18 - 24.4 \\
72 - 79.2 & 30 - 24.4 \\
85 - 79.2 & 24 - 24.4
\end{bmatrix} = 
\begin{bmatrix} 
-14.2 & 3.6 \\
-0.2 & -2.4 \\
15.8 & -6.4 \\
-7.2 & 5.6 \\
5.8 & -0.4
\end{bmatrix}$$

Now compute the covariance matrix for the approved class:
$$\mathbf{S}_1 = \frac{1}{n_1 - 1}\sum_{i=1}^{n_1} (\mathbf{x}_i - \boldsymbol{\mu}_1)(\mathbf{x}_i - \boldsymbol{\mu}_1)^T$$

Computing the elements of the covariance matrix:
$$\begin{align}
S_1[0,0] &= \text{Variance(Income)} = \frac{(-14.2)^2 + (-0.2)^2 + (15.8)^2 + (-7.2)^2 + (5.8)^2}{4} = 134.2 \\
S_1[0,1] = S_1[1,0] &= \text{Covariance(Income, Debt-to-Income)} \\
&= \frac{(-14.2 \times 3.6) + (-0.2 \times -2.4) + (15.8 \times -6.4) + (-7.2 \times 5.6) + (5.8 \times -0.4)}{4} \\
&= \frac{-51.12 + 0.48 - 101.12 - 40.32 - 2.32}{4} = -48.6 \\
S_1[1,1] &= \text{Variance(Debt-to-Income)} = \frac{(3.6)^2 + (-2.4)^2 + (-6.4)^2 + (5.6)^2 + (-0.4)^2}{4} = 22.8
\end{align}$$

This gives us:
$$\mathbf{S}_1 = \begin{bmatrix} 134.2 & -48.6 \\ -48.6 & 22.8 \end{bmatrix}$$

**Denied class (centered data):**
$$\mathbf{X}_0 - \boldsymbol{\mu}_0 = 
\begin{bmatrix} 
50 - 47.2 & 32 - 36.0 \\
48 - 47.2 & 40 - 36.0 \\
36 - 47.2 & 36 - 36.0 \\
60 - 47.2 & 34 - 36.0 \\
42 - 47.2 & 38 - 36.0
\end{bmatrix} = 
\begin{bmatrix} 
2.8 & -4.0 \\
0.8 & 4.0 \\
-11.2 & 0.0 \\
12.8 & -2.0 \\
-5.2 & 2.0
\end{bmatrix}$$

Computing the elements of the covariance matrix:
$$\begin{align}
S_0[0,0] &= \text{Variance(Income)} = \frac{(2.8)^2 + (0.8)^2 + (-11.2)^2 + (12.8)^2 + (-5.2)^2}{4} = 81.2 \\
S_0[0,1] = S_0[1,0] &= \text{Covariance(Income, Debt-to-Income)} \\
&= \frac{(2.8 \times -4.0) + (0.8 \times 4.0) + (-11.2 \times 0.0) + (12.8 \times -2.0) + (-5.2 \times 2.0)}{4} \\
&= \frac{-11.2 + 3.2 + 0.0 - 25.6 - 10.4}{4} = -11.0 \\
S_0[1,1] &= \text{Variance(Debt-to-Income)} = \frac{(-4.0)^2 + (4.0)^2 + (0.0)^2 + (-2.0)^2 + (2.0)^2}{4} = 10.0
\end{align}$$

This gives us:
$$\mathbf{S}_0 = \begin{bmatrix} 81.2 & -11.0 \\ -11.0 & 10.0 \end{bmatrix}$$

The pooled within-class covariance matrix $\mathbf{S}_W$ is calculated as a weighted average of individual class covariances:

$$\mathbf{S}_W = \frac{(n_1 - 1)\mathbf{S}_1 + (n_0 - 1)\mathbf{S}_0}{n_1 + n_0 - 2}$$

Substituting our values:
$$\begin{align}
\mathbf{S}_W &= \frac{(5 - 1)\mathbf{S}_1 + (5 - 1)\mathbf{S}_0}{5 + 5 - 2} \\
&= \frac{4\mathbf{S}_1 + 4\mathbf{S}_0}{8} \\
&= \frac{4\begin{bmatrix} 134.2 & -48.6 \\ -48.6 & 22.8 \end{bmatrix} + 4\begin{bmatrix} 81.2 & -11.0 \\ -11.0 & 10.0 \end{bmatrix}}{8} \\
&= \frac{\begin{bmatrix} 536.8 & -194.4 \\ -194.4 & 91.2 \end{bmatrix} + \begin{bmatrix} 324.8 & -44.0 \\ -44.0 & 40.0 \end{bmatrix}}{8} \\
&= \frac{\begin{bmatrix} 861.6 & -238.4 \\ -238.4 & 131.2 \end{bmatrix}}{8} \\
&= \begin{bmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{bmatrix}
\end{align}$$

The negative off-diagonal elements indicate a negative correlation between income and debt-to-income ratio within each class, which makes sense as higher-income individuals generally have lower debt-to-income ratios.

### Step 3: Calculate the between-class scatter matrix
The between-class scatter matrix $\mathbf{S}_B$ represents the separation between classes and is calculated as:

$$\mathbf{S}_B = \frac{n_1 n_0}{n} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T$$

First, let's calculate the difference between class means:
$$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 = \begin{bmatrix} 79.2 \\ 24.4 \end{bmatrix} - \begin{bmatrix} 47.2 \\ 36.0 \end{bmatrix} = \begin{bmatrix} 32.0 \\ -11.6 \end{bmatrix}$$

Now, we compute the outer product:
$$\begin{align}
(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T &= \begin{bmatrix} 32.0 \\ -11.6 \end{bmatrix} \begin{bmatrix} 32.0 & -11.6 \end{bmatrix} \\
&= \begin{bmatrix} 
32.0 \times 32.0 & 32.0 \times (-11.6) \\
(-11.6) \times 32.0 & (-11.6) \times (-11.6)
\end{bmatrix} \\
&= \begin{bmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{bmatrix}
\end{align}$$

Finally, the between-class scatter matrix:
$$\begin{align}
\mathbf{S}_B &= \frac{n_1 n_0}{n} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T \\
&= \frac{5 \cdot 5}{10} \begin{bmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{bmatrix} \\
&= 2.5 \begin{bmatrix} 1024.0 & -371.2 \\ -371.2 & 134.6 \end{bmatrix} \\
&= \begin{bmatrix} 2560.0 & -928.0 \\ -928.0 & 336.4 \end{bmatrix}
\end{align}$$

For binary classification, $\mathbf{S}_B$ always has rank 1 because it's the outer product of a single vector, which is confirmed in our calculation.

### Step 4: Determine the optimal projection direction
For binary classification, the optimal projection direction $\mathbf{w}$ can be calculated directly using:

$$\mathbf{w} = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$$

First, we calculate the inverse of the pooled within-class covariance matrix:
$$\mathbf{S}_W^{-1} = \begin{bmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{bmatrix}^{-1} = \begin{bmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{bmatrix}$$

The inverse is calculated as follows:
$$\begin{align}
\det(\mathbf{S}_W) &= 107.7 \times 16.4 - (-29.8) \times (-29.8) \\
&= 1766.28 - 888.04 \\
&= 878.24 \\
\mathbf{S}_W^{-1} &= \frac{1}{\det(\mathbf{S}_W)} \begin{bmatrix} 16.4 & 29.8 \\ 29.8 & 107.7 \end{bmatrix} \\
&= \frac{1}{878.24} \begin{bmatrix} 16.4 & 29.8 \\ 29.8 & 107.7 \end{bmatrix} \\
&= \begin{bmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{bmatrix}
\end{align}$$

Then, we compute the optimal projection direction:
$$\begin{align}
\mathbf{w} &= \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \\
&= \begin{bmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{bmatrix} \begin{bmatrix} 32.0 \\ -11.6 \end{bmatrix} \\
&= \begin{bmatrix} 
0.0187 \times 32.0 + 0.0339 \times (-11.6) \\
0.0339 \times 32.0 + 0.1226 \times (-11.6)
\end{bmatrix} \\
&= \begin{bmatrix}
0.5984 - 0.3944 \\
1.0848 - 1.4215
\end{bmatrix} \\
&= \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix}
\end{align}$$

We can normalize $\mathbf{w}$ to unit length for easier interpretation:
$$\begin{align}
\|\mathbf{w}\| &= \sqrt{0.2040^2 + (-0.3367)^2} = \sqrt{0.0416 + 0.1134} = \sqrt{0.155} = 0.3937 \\
\mathbf{w}_{\text{norm}} &= \frac{\mathbf{w}}{\|\mathbf{w}\|} = \frac{1}{0.3937}\begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix} = \begin{bmatrix} 0.5181 \\ -0.8553 \end{bmatrix}
\end{align}$$

Alternatively, we can find this direction as the eigenvector corresponding to the largest eigenvalue of $\mathbf{S}_W^{-1}\mathbf{S}_B$:

$$\begin{align}
\mathbf{S}_W^{-1}\mathbf{S}_B &= \begin{bmatrix} 0.0187 & 0.0339 \\ 0.0339 & 0.1226 \end{bmatrix} \begin{bmatrix} 2560.0 & -928.0 \\ -928.0 & 336.4 \end{bmatrix}
\end{align}$$

Calculating the elements of $\mathbf{S}_W^{-1}\mathbf{S}_B$ in detail:

$$\begin{align}
[\mathbf{S}_W^{-1}\mathbf{S}_B]_{00} &= 0.0187 \times 2560.0 + 0.0339 \times (-928.0) \\
&= 47.8047 - 31.4884 = 16.3163 \\
[\mathbf{S}_W^{-1}\mathbf{S}_B]_{01} &= 0.0187 \times (-928.0) + 0.0339 \times 336.4 \\
&= -17.3292 + 11.4146 = -5.9146 \\
[\mathbf{S}_W^{-1}\mathbf{S}_B]_{10} &= 0.0339 \times 2560.0 + 0.1226 \times (-928.0) \\
&= 86.8646 - 113.8021 = -26.9375 \\
[\mathbf{S}_W^{-1}\mathbf{S}_B]_{11} &= 0.0339 \times (-928.0) + 0.1226 \times 336.4 \\
&= -31.4884 + 41.2533 = 9.7648
\end{align}$$

Therefore:
$$\mathbf{S}_W^{-1}\mathbf{S}_B = \begin{bmatrix} 16.3163 & -5.9146 \\ -26.9375 & 9.7648 \end{bmatrix}$$

Computing the eigenvalues and eigenvectors:
- Eigenvalues: $\lambda_1 = 26.0811$, $\lambda_2 \approx 0$
- First eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 0.5181 \\ -0.8553 \end{bmatrix}$

The eigenvalue interpretation:
- $\lambda_1 = 26.0811$ - This large value indicates strong class separation along the first eigenvector direction.
- $\lambda_2 \approx 0$ - This zero value is expected in binary classification since $\mathbf{S}_B$ has rank 1.

The projection direction indicates that LDA assigns a positive weight to income and a negative weight to debt-to-income ratio, which aligns with the expectation that higher income is favorable for approval, while higher debt-to-income ratio is unfavorable.

### Step 5: Calculate the threshold for classification
The threshold for classification depends on the projected class means and prior probabilities.

First, we project the class means onto the direction $\mathbf{w}$:

$$\begin{align}
\mathbf{w}^T\boldsymbol{\mu}_1 &= \begin{bmatrix} 0.2040 & -0.3367 \end{bmatrix} \begin{bmatrix} 79.2 \\ 24.4 \end{bmatrix} \\
&= 0.2040 \times 79.2 + (-0.3367) \times 24.4 \\
&= 16.1531 + (-8.2159) \\
&= 7.9372
\end{align}$$

$$\begin{align}
\mathbf{w}^T\boldsymbol{\mu}_0 &= \begin{bmatrix} 0.2040 & -0.3367 \end{bmatrix} \begin{bmatrix} 47.2 \\ 36.0 \end{bmatrix} \\
&= 0.2040 \times 47.2 + (-0.3367) \times 36.0 \\
&= 9.6266 + (-12.1219) \\
&= -2.4953
\end{align}$$

With equal prior probabilities, the threshold would be the midpoint of the projected means:
$$\text{threshold}_{\text{equal}} = \frac{\mathbf{w}^T\boldsymbol{\mu}_1 + \mathbf{w}^T\boldsymbol{\mu}_0}{2} = \frac{7.9372 + (-2.4953)}{2} = \frac{5.4419}{2} = 2.7209$$

However, given the unequal prior probabilities $P(y=1) = 0.3$ and $P(y=0) = 0.7$, we adjust the threshold according to:

$$\text{threshold} = \frac{\mathbf{w}^T\boldsymbol{\mu}_1 + \mathbf{w}^T\boldsymbol{\mu}_0}{2} + \frac{1}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}} \ln\left(\frac{P(y=0)}{P(y=1)}\right)$$

This formula comes from the decision rule in discriminant analysis with Gaussian class-conditional densities and shared covariance. The detailed derivation involves:

1. Starting with the log-likelihood ratio: $\log\frac{P(y=1|x)}{P(y=0|x)} > 0$ for classification as Class 1
2. Applying Bayes' rule: $\log\frac{p(x|y=1)P(y=1)}{p(x|y=0)P(y=0)} > 0$
3. For multivariate Gaussians with shared covariance $\Sigma$, expanding and simplifying:
   $$-0.5(x-\mu_1)^T\Sigma^{-1}(x-\mu_1) + \log P(y=1) > -0.5(x-\mu_0)^T\Sigma^{-1}(x-\mu_0) + \log P(y=0)$$
4. After algebraic manipulations, this leads to:
   $$w^Tx > \frac{w^T\mu_1 + w^T\mu_0}{2} + \frac{1}{w^T\Sigma w}\ln\left(\frac{P(y=0)}{P(y=1)}\right)$$

First, we calculate $\mathbf{w}^T\mathbf{S}_W\mathbf{w}$:
$$\begin{align}
\mathbf{w}^T\mathbf{S}_W\mathbf{w} &= \begin{bmatrix} 0.2040 & -0.3367 \end{bmatrix} \begin{bmatrix} 107.7 & -29.8 \\ -29.8 & 16.4 \end{bmatrix} \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix} \\
&= \begin{bmatrix} 0.2040 \times 107.7 + (-0.3367) \times (-29.8) & 0.2040 \times (-29.8) + (-0.3367) \times 16.4 \end{bmatrix} \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix} \\
&= \begin{bmatrix} 21.9708 + 10.0332 & -6.0792 - 5.5219 \end{bmatrix} \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix} \\
&= \begin{bmatrix} 32.0040 & -11.6011 \end{bmatrix} \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix} \\
&= 32.0040 \times 0.2040 + (-11.6011) \times (-0.3367) \\
&= 6.5288 + 3.9036 \\
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
\mathbf{w}^T\mathbf{x}_{\text{new}} &= \begin{bmatrix} 0.2040 & -0.3367 \end{bmatrix} \begin{bmatrix} 55 \\ 25 \end{bmatrix} \\
&= 0.2040 \times 55 + (-0.3367) \times 25 \\
&= 11.2174 + (-8.4180) \\
&= 2.7995
\end{align}$$

Comparing with the threshold:
- Projected value: $2.7995$
- Threshold with priors: $2.8022$

Since the projected value $(2.7995)$ is less than the threshold $(2.8022)$, the LDA model predicts that this application will be denied. This is a very close decision, as the applicant's projected value is only slightly below the threshold.

#### Decision Boundary Sensitivity Analysis
For this new applicant, the decision is extremely close. We can analyze how much each feature would need to change to flip the decision:

- Income increase needed to change decision:
  $$\begin{align}
  2.8022 &= 0.2040 \times (55 + \Delta \text{Income}) + (-0.3367) \times 25 \\
  2.8022 &= 11.2174 + 0.2040 \times \Delta \text{Income} - 8.4180 \\
  2.8022 &= 2.7995 + 0.2040 \times \Delta \text{Income} \\
  0.0027 &= 0.2040 \times \Delta \text{Income} \\
  \Delta \text{Income} &= \frac{0.0027}{0.2040} = 0.01324 \approx \$0.01K
  \end{align}$$

- Debt-to-Income decrease needed:
  $$\begin{align}
  2.8022 &= 0.2040 \times 55 + (-0.3367) \times (25 - \Delta \text{DTI}) \\
  2.8022 &= 11.2174 - 8.4180 + 0.3367 \times \Delta \text{DTI} \\
  2.8022 &= 2.7995 + 0.3367 \times \Delta \text{DTI} \\
  0.0027 &= 0.3367 \times \Delta \text{DTI} \\
  \Delta \text{DTI} &= \frac{0.0027}{0.3367} = 0.00802 \approx 0.01\%
  \end{align}$$

This sensitivity analysis shows that even a very small change in either feature could change the decision from denied to approved, highlighting how close this decision is.

### Feature Importance Analysis
We can analyze the relative importance of each feature in the LDA model:

$$\begin{align}
\text{Income importance} &= \frac{|w_1|}{\sum_i |w_i|} = \frac{|0.2040|}{|0.2040| + |-0.3367|} = \frac{0.2040}{0.5407} = 0.3774 \approx 37.7\% \\
\text{DTI importance} &= \frac{|w_2|}{\sum_i |w_i|} = \frac{|-0.3367|}{|0.2040| + |-0.3367|} = \frac{0.3367}{0.5407} = 0.6226 \approx 62.3\%
\end{align}$$

But we can also measure feature importance by the effect size - how much each feature contributes to the separation between classes:

$$\begin{align}
\text{Income effect} &= |w_1| \times |\mu_{1,1} - \mu_{0,1}| = 0.2040 \times |79.2 - 47.2| = 0.2040 \times 32.0 = 6.5265 \\
\text{DTI effect} &= |w_2| \times |\mu_{1,2} - \mu_{0,2}| = 0.3367 \times |24.4 - 36.0| = 0.3367 \times 11.6 = 3.9059
\end{align}$$

Total effect: $6.5265 + 3.9059 = 10.4324$

$$\begin{align}
\text{Income contribution} &= \frac{6.5265}{10.4324} = 0.6256 \approx 62.6\% \\
\text{DTI contribution} &= \frac{3.9059}{10.4324} = 0.3744 \approx 37.4\%
\end{align}$$

This shows that while Debt-to-Income ratio has a higher weight in the LDA function (62.3%), Income actually contributes more to the overall class separation (62.6%) due to the larger difference between classes on this feature.

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
- The debt-to-income ratio has a greater impact on the decision than income when considering raw weights (62.3% vs. 37.7% importance)
- However, income contributes more to the total class separation due to the larger difference between classes (62.6% vs. 37.4%)
- Prior probabilities affect the threshold placement and therefore the decision boundary
- Using priors of $P(y=1) = 0.3$ and $P(y=0) = 0.7$ makes approval more selective by shifting the threshold toward the approved class mean

### Decision Making
- The LDA model provides a principled way to make credit decisions by projecting applicant data onto the optimal direction
- The projection reduces the original 2D problem to a 1D comparison against a threshold
- The model can be adjusted by changing priors to reflect different approval policies (more or less selective)
- The closeness of the new applicant to the threshold suggests uncertainty in the decision
- For the new applicant, even a tiny change in income ($0.01K increase) or debt-to-income ratio (0.01% decrease) would change the decision from denied to approved

## Conclusion
The LDA analysis of credit approval data has yielded several important results:

1. There are clear statistical differences between approved and denied applications, with approved applications having higher income (mean $79.2K vs $47.2K) and lower debt-to-income ratios (mean 24.4% vs 36.0%).

2. The optimal LDA projection direction $\mathbf{w} = \begin{bmatrix} 0.2040 \\ -0.3367 \end{bmatrix}$ assigns a positive weight to income and a negative weight to debt-to-income ratio, confirming that higher income contributes positively to approval while higher debt-to-income ratio contributes negatively.

3. With the given prior probabilities $P(y=1) = 0.3$ and $P(y=0) = 0.7$, the classification threshold is 2.8022.

4. For a new applicant with income $55K and debt-to-income ratio 25%, the model predicts a denial of credit, though the decision is extremely close (projected value 2.7995 vs. threshold 2.8022), with even tiny changes in either feature potentially changing the outcome.

5. While debt-to-income ratio has a higher weight in the decision function, income contributes more to overall class separation due to the larger difference between approved and denied applications on this feature.

This LDA model provides a principled statistical approach to credit approval decisions based on the patterns observed in historical data while incorporating prior beliefs about approval rates. 