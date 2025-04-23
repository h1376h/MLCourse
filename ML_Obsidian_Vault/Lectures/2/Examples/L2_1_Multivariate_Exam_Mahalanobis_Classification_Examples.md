# Mahalanobis Distance and Classification Examples

This document provides examples and key concepts on Mahalanobis distance and its applications in classification, which are essential tools in machine learning, multivariate statistics, and pattern recognition.

## Key Concepts and Formulas

The Mahalanobis distance is a measure of the distance between a point and a distribution. Unlike Euclidean distance, it accounts for the correlations between variables and is scale-invariant.

### Mahalanobis Distance Formula

The squared Mahalanobis distance from a point $\mathbf{x}$ to a distribution with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ is:

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

Where:
- $\mathbf{x}$ = Point vector
- $\boldsymbol{\mu}$ = Mean vector of the distribution
- $\boldsymbol{\Sigma}$ = Covariance matrix of the distribution
- $\boldsymbol{\Sigma}^{-1}$ = Inverse of the covariance matrix

## Example 1: Mahalanobis Distance and Classification

### Problem Statement
You are building a binary classifier for a machine learning problem. The two classes follow multivariate normal distributions with the same covariance matrix but different means:

Class 1: $\mathbf{X} \sim \mathcal{N}\left(\begin{bmatrix} 2 \\ 4 \end{bmatrix}, \begin{bmatrix} 5 & 1 \\ 1 & 3 \end{bmatrix}\right)$

Class 2: $\mathbf{X} \sim \mathcal{N}\left(\begin{bmatrix} 5 \\ 6 \end{bmatrix}, \begin{bmatrix} 5 & 1 \\ 1 & 3 \end{bmatrix}\right)$

a) Calculate the Mahalanobis distance between the two class means.
b) For a new observation $\mathbf{x} = (3, 5)$, determine the class it belongs to using the minimum Mahalanobis distance classifier.
c) Show that the decision boundary based on the Mahalanobis distance is a straight line, and find its equation.

### Solution

#### Part a: Calculating the Mahalanobis distance between class means

The Mahalanobis distance between two points $\mathbf{x}_1$ and $\mathbf{x}_2$ with respect to covariance matrix $\boldsymbol{\Sigma}$ is:

$$d_M(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{(\mathbf{x}_1 - \mathbf{x}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_1 - \mathbf{x}_2)}$$

First, we need to find $\boldsymbol{\Sigma}^{-1}$:

$$|\boldsymbol{\Sigma}| = 5 \times 3 - 1 \times 1 = 15 - 1 = 14$$

$$\boldsymbol{\Sigma}^{-1} = \frac{1}{14} \begin{bmatrix} 3 & -1 \\ -1 & 5 \end{bmatrix} = \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix}$$

The difference between the class means is:
$$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix} - \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

Now we calculate the squared Mahalanobis distance:
$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} -3 \\ -2 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} -9/14 + 2/14 \\ 3/14 - 10/14 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} -7/14 \\ -7/14 \end{bmatrix}$$

$$d_M^2(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = 3 \times (7/14) + 2 \times (7/14) = 21/14 + 14/14 = 35/14 = 2.5$$

Therefore, the Mahalanobis distance is:
$$d_M(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \sqrt{2.5} \approx 1.58$$

#### Part b: Classifying a new observation using minimum Mahalanobis distance

For a new observation $\mathbf{x} = (3, 5)$, we need to calculate the Mahalanobis distance to each class mean and assign it to the class with the smaller distance.

Distance to Class 1 mean:
$$\mathbf{x} - \boldsymbol{\mu}_1 = \begin{bmatrix} 3 \\ 5 \end{bmatrix} - \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 3/14 - 1/14 \\ -1/14 + 5/14 \end{bmatrix} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 2/14 \\ 4/14 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = 1 \times (2/14) + 1 \times (4/14) = 2/14 + 4/14 = 6/14 = 3/7 \approx 0.429$$

Distance to Class 2 mean:
$$\mathbf{x} - \boldsymbol{\mu}_2 = \begin{bmatrix} 3 \\ 5 \end{bmatrix} - \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} -2 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} \begin{bmatrix} -2 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} -6/14 + 1/14 \\ 2/14 - 5/14 \end{bmatrix} = \begin{bmatrix} -2 & -1 \end{bmatrix} \begin{bmatrix} -5/14 \\ -3/14 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_2) = (-2) \times (-5/14) + (-1) \times (-3/14) = 10/14 + 3/14 = 13/14 \approx 0.929$$

Since $d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) < d_M^2(\mathbf{x}, \boldsymbol{\mu}_2)$ (0.429 < 0.929), we classify $\mathbf{x}$ as belonging to Class 1.

#### Part c: Finding the decision boundary equation

For the minimum Mahalanobis distance classifier with equal covariance matrices, the decision boundary is a hyperplane equidistant from both means in the Mahalanobis distance sense.

The decision boundary satisfies:
$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_1) = d_M^2(\mathbf{x}, \boldsymbol{\mu}_2)$$

This expands to:
$$(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_1) = (\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_2)$$

After expanding and simplifying, this becomes:
$$2(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2)$$

Let's compute this equation for our specific case:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} = \begin{bmatrix} -3 & -2 \end{bmatrix} \begin{bmatrix} 3/14 & -1/14 \\ -1/14 & 5/14 \end{bmatrix} = \begin{bmatrix} -9/14 + 2/14 & 3/14 - 10/14 \end{bmatrix} = \begin{bmatrix} -7/14 & -7/14 \end{bmatrix}$$

$$(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix} + \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 7 \\ 10 \end{bmatrix}$$

The right side of the equation is:
$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2) = \begin{bmatrix} -7/14 & -7/14 \end{bmatrix} \begin{bmatrix} 7 \\ 10 \end{bmatrix} = -7/14 \times 7 - 7/14 \times 10 = -49/14 - 70/14 = -119/14$$

So the decision boundary equation is:
$$2 \times \begin{bmatrix} -7/14 & -7/14 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -119/14$$

$$\begin{bmatrix} -1 & -1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -119/28$$

$$-x_1 - x_2 = -119/28$$

Simplifying:
$$x_1 + x_2 = 119/28 \approx 4.25$$

Therefore, the decision boundary is the straight line $x_1 + x_2 = 4.25$. Any point $(x_1, x_2)$ where $x_1 + x_2 > 4.25$ will be classified as Class 2, and any point where $x_1 + x_2 < 4.25$ will be classified as Class 1.

## Example 2: Outlier Detection with Mahalanobis Distance

### Problem Statement
A quality control engineer is monitoring a manufacturing process where two key parameters are measured: tensile strength ($X_1$) and thickness ($X_2$). Historical data shows that these parameters follow a bivariate normal distribution with:

$$\boldsymbol{\mu} = \begin{bmatrix} 120 \\ 5.5 \end{bmatrix}, \boldsymbol{\Sigma} = \begin{bmatrix} 25 & 2.5 \\ 2.5 & 0.64 \end{bmatrix}$$

a) Calculate the Mahalanobis distance for a product with measurements $\mathbf{x} = (130, 6.0)$.
b) If the engineer wants to flag products with Mahalanobis distances that would occur with less than 5% probability in the normal distribution, what threshold should be used?
c) Is the product in part (a) considered an outlier based on this threshold?

### Solution

#### Part a: Calculating the Mahalanobis distance

First, we need to find $\boldsymbol{\Sigma}^{-1}$:

$$|\boldsymbol{\Sigma}| = 25 \times 0.64 - 2.5 \times 2.5 = 16 - 6.25 = 9.75$$

$$\boldsymbol{\Sigma}^{-1} = \frac{1}{9.75} \begin{bmatrix} 0.64 & -2.5 \\ -2.5 & 25 \end{bmatrix} = \begin{bmatrix} 0.066 & -0.256 \\ -0.256 & 2.564 \end{bmatrix}$$

Now, we calculate the Mahalanobis distance for the product:
$$\mathbf{x} - \boldsymbol{\mu} = \begin{bmatrix} 130 \\ 6.0 \end{bmatrix} - \begin{bmatrix} 120 \\ 5.5 \end{bmatrix} = \begin{bmatrix} 10 \\ 0.5 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}) = \begin{bmatrix} 10 & 0.5 \end{bmatrix} \begin{bmatrix} 0.066 & -0.256 \\ -0.256 & 2.564 \end{bmatrix} \begin{bmatrix} 10 \\ 0.5 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}) = \begin{bmatrix} 10 & 0.5 \end{bmatrix} \begin{bmatrix} 0.66 - 0.128 \\ -2.56 + 1.282 \end{bmatrix} = \begin{bmatrix} 10 & 0.5 \end{bmatrix} \begin{bmatrix} 0.532 \\ -1.278 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}) = 10 \times 0.532 + 0.5 \times (-1.278) = 5.32 - 0.639 = 4.681$$

Therefore, the Mahalanobis distance is:
$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{4.681} \approx 2.164$$

#### Part b: Determining the threshold for outlier detection

For a bivariate normal distribution, the squared Mahalanobis distance follows a chi-square distribution with 2 degrees of freedom.

To find the threshold that corresponds to a 5% probability in the upper tail, we need the 95th percentile of the chi-square distribution with 2 degrees of freedom, which is approximately 5.991.

Therefore, the engineer should flag products with a squared Mahalanobis distance greater than 5.991, or a Mahalanobis distance greater than $\sqrt{5.991} \approx 2.448$.

#### Part c: Determining if the product is an outlier

The product has a squared Mahalanobis distance of 4.681, which is less than the threshold of 5.991.

Alternatively, the Mahalanobis distance is 2.164, which is less than the threshold of 2.448.

Therefore, the product is not considered an outlier based on the 5% significance level.

## Example 3: Discriminant Analysis with Three Classes

### Problem Statement
A botanist is studying three species of plants and has measured two characteristics: leaf width ($X_1$) and length ($X_2$). The measurements for each species follow multivariate normal distributions with the following parameters:

Species A: $\boldsymbol{\mu}_A = \begin{bmatrix} 3 \\ 8 \end{bmatrix}$, $\boldsymbol{\Sigma} = \begin{bmatrix} 1.2 & 0.4 \\ 0.4 & 2.0 \end{bmatrix}$

Species B: $\boldsymbol{\mu}_B = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, $\boldsymbol{\Sigma} = \begin{bmatrix} 1.2 & 0.4 \\ 0.4 & 2.0 \end{bmatrix}$

Species C: $\boldsymbol{\mu}_C = \begin{bmatrix} 4 \\ 10 \end{bmatrix}$, $\boldsymbol{\Sigma} = \begin{bmatrix} 1.2 & 0.4 \\ 0.4 & 2.0 \end{bmatrix}$

a) A leaf is found with measurements $\mathbf{x} = (4, 9)$. Classify it using the minimum Mahalanobis distance criterion.
b) Describe the decision boundaries between the three species.
c) If the prior probabilities of species A, B, and C are 0.5, 0.3, and 0.2 respectively, how would this affect the classification decision?

### Solution

#### Part a: Classifying using minimum Mahalanobis distance

First, we need to find $\boldsymbol{\Sigma}^{-1}$:

$$|\boldsymbol{\Sigma}| = 1.2 \times 2.0 - 0.4 \times 0.4 = 2.4 - 0.16 = 2.24$$

$$\boldsymbol{\Sigma}^{-1} = \frac{1}{2.24} \begin{bmatrix} 2.0 & -0.4 \\ -0.4 & 1.2 \end{bmatrix} = \begin{bmatrix} 0.8929 & -0.1786 \\ -0.1786 & 0.5357 \end{bmatrix}$$

Now we calculate the squared Mahalanobis distance to each species:

Distance to Species A:
$$\mathbf{x} - \boldsymbol{\mu}_A = \begin{bmatrix} 4 \\ 9 \end{bmatrix} - \begin{bmatrix} 3 \\ 8 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_A) = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 0.8929 & -0.1786 \\ -0.1786 & 0.5357 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_A) = 0.8929 - 0.1786 - 0.1786 + 0.5357 = 1.0714$$

Distance to Species B:
$$\mathbf{x} - \boldsymbol{\mu}_B = \begin{bmatrix} 4 \\ 9 \end{bmatrix} - \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} -1 \\ 3 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_B) = \begin{bmatrix} -1 & 3 \end{bmatrix} \begin{bmatrix} 0.8929 & -0.1786 \\ -0.1786 & 0.5357 \end{bmatrix} \begin{bmatrix} -1 \\ 3 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_B) = (-1)(0.8929)(-1) + (-1)(-0.1786)(3) + (3)(-0.1786)(-1) + (3)(0.5357)(3) = 0.8929 + 0.5358 + 0.5358 + 4.8213 = 6.7858$$

Distance to Species C:
$$\mathbf{x} - \boldsymbol{\mu}_C = \begin{bmatrix} 4 \\ 9 \end{bmatrix} - \begin{bmatrix} 4 \\ 10 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_C) = \begin{bmatrix} 0 & -1 \end{bmatrix} \begin{bmatrix} 0.8929 & -0.1786 \\ -0.1786 & 0.5357 \end{bmatrix} \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

$$d_M^2(\mathbf{x}, \boldsymbol{\mu}_C) = (0)(0.8929)(0) + (0)(-0.1786)(-1) + (-1)(-0.1786)(0) + (-1)(0.5357)(-1) = 0 + 0 + 0 + 0.5357 = 0.5357$$

Comparing the distances:
- To Species A: 1.0714
- To Species B: 6.7858
- To Species C: 0.5357

Since the Mahalanobis distance to Species C is the smallest, we classify the leaf as belonging to Species C.

#### Part b: Describing decision boundaries

The decision boundaries between pairs of classes (with equal covariance matrices) are hyperplanes that are equidistant from the class means in the Mahalanobis distance sense.

For species A and B, the decision boundary is:
$$2(\boldsymbol{\mu}_A - \boldsymbol{\mu}_B)^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = (\boldsymbol{\mu}_A - \boldsymbol{\mu}_B)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_A + \boldsymbol{\mu}_B)$$

Similarly for the boundaries between species A and C, and between species B and C.

The overall decision regions are formed by these three hyperplanes, creating a partition of the feature space into three regions, each corresponding to one of the species.

#### Part c: Effect of prior probabilities

When prior probabilities are considered, we modify the classification rule to use the Bayes decision rule, which minimizes the expected error rate.

Instead of just comparing Mahalanobis distances, we compute:
$$\ln(P(C_i)) - \frac{1}{2}d_M^2(\mathbf{x}, \boldsymbol{\mu}_i)$$

for each class $i$, and classify to the class with the highest value.

For our example:
- For Species A: $\ln(0.5) - \frac{1}{2}(1.0714) = -0.6931 - 0.5357 = -1.2288$
- For Species B: $\ln(0.3) - \frac{1}{2}(6.7858) = -1.2040 - 3.3929 = -4.5969$
- For Species C: $\ln(0.2) - \frac{1}{2}(0.5357) = -1.6094 - 0.2679 = -1.8773$

Since Species A has the highest value (-1.2288), we would classify the leaf as belonging to Species A when taking prior probabilities into account.

This demonstrates how prior probabilities can change the classification decision, especially when one class has a much higher prior probability than others.

## Related Topics

- [[L2_1_Mahalanobis_Distance|Mahalanobis Distance]]: In-depth coverage of Mahalanobis distance
- [[L2_1_Multivariate_Normal_Examples|Multivariate Normal Examples]]: Examples of multivariate normal distributions
- [[L2_1_Covariance_Examples|Covariance Examples]]: Understanding covariance structures
- [[L2_1_Linear_Discriminant_Analysis|Linear Discriminant Analysis]]: Classification techniques using Mahalanobis distance
- [[L2_1_Outlier_Detection|Outlier Detection]]: Statistical methods for identifying anomalies 