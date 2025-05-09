# Question 5: Logistic Regression for Tumor Classification

## Problem Statement
Consider a medical dataset with tumor features and diagnostic outcomes. Each patient has data on age (years) and tumor size (mm), with the target variable $y$ indicating whether the tumor is malignant (1) or benign (0).

| Age (years) | Tumor Size (mm) | $y$ (Malignant) |
|-------------|-----------------|-----------------|
| 15          | 20              | 0               |
| 65          | 30              | 0               |
| 30          | 50              | 1               |
| 90          | 20              | 1               |
| 44          | 35              | 0               |
| 20          | 70              | 1               |
| 50          | 40              | 1               |
| 36          | 25              | 0               |

A logistic regression model is being trained on this dataset to predict whether tumors are malignant or benign based on age and tumor size.

The model uses the sigmoid function:
$$g(z) = \frac{1}{1+e^{-z}}$$

And the hypothesis function:
$$h_\theta(x) = g(\theta^T x)$$

The cost function used for training is:
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}\log h_\theta(x^{(i)}) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))]$$

### Task
1. Starting with initial parameters $\theta_0 = 0$, $\theta_1 = 0$, and $\theta_2 = 0$, calculate the initial cost $J(\theta)$ for this dataset.
2. Calculate the first two iterations of gradient descent using the following update rule and a learning rate $\alpha = 0.01$:
   $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$
   Where:
   $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
3. For the same initial parameters, calculate the first two iterations of stochastic gradient descent using a single randomly selected training example at each step with learning rate $\alpha = 0.1$. Show all calculations.
4. Explain the decision boundary equation $\theta^T x = 0$ in the context of logistic regression. What does it represent geometrically?
5. Using the final optimized parameters $\theta_0 = -136.95$, $\theta_1 = 1.1$, and $\theta_2 = 2.2$, derive the equation of the decision boundary for this model.
6. The final optimized parameters for this model are $\theta_0 = -136.95$, $\theta_1 = 1.1$, and $\theta_2 = 2.2$. For a new patient with age 50 years and tumor size 30mm, calculate the predicted probability of the tumor being malignant and provide the classification.
7. Explain how the coefficients $\theta_1 = 1.1$ and $\theta_2 = 2.2$ can be interpreted in this medical context.
8. Conceptually, how would increasing and decreasing the learning rate affect the training process?

## Understanding the Problem
Logistic regression is a binary classification method that models the probability that a given input belongs to a certain class. In this medical context, we're using logistic regression to predict whether a tumor is malignant (y=1) or benign (y=0) based on the patient's age and tumor size.

The dataset consists of 8 patients, with 4 benign cases and 4 malignant cases. The feature matrix has shape (8, 2), representing 8 examples with 2 features each (age and tumor size).

The core of logistic regression is the sigmoid function, which maps any real-valued number to the range [0,1], making it suitable for representing probabilities:

![Sigmoid Function](../Images/L5_2_Quiz_5/sigmoid_function.png)

The sigmoid function has the following important properties:
- When z = 0: g(0) = 1/(1+e^0) = 1/2 = 0.5
- As z → +∞: g(z) → 1
- As z → -∞: g(z) → 0

This is crucial for understanding the decision boundary - when the model's raw output ($\theta^T x$) is exactly 0, the predicted probability is exactly 0.5.

For this binary classification problem, we use maximum likelihood estimation, which leads to the cost function given in the problem statement. This cost function (also known as the log loss or cross-entropy loss) penalizes confident but incorrect predictions heavily.

## Solution

### Step 1: Calculate the Initial Cost
With initial parameters $\theta = [0, 0, 0]$, the model predicts a probability of 0.5 for all examples, regardless of their features. This is because $\theta^T x = 0$ for all examples, and $g(0) = 0.5$.

Let's calculate the cost for each example step by step:

For example 1 (y=0):
- $z_1 = \theta^T x_1 = 0 \cdot 1 + 0 \cdot 15 + 0 \cdot 20 = 0$
- $h(x_1) = g(z_1) = g(0) = 0.5$
- Cost = $-\log(1-h(x_1)) = -\log(0.5) = 0.6931$

Similarly, for all examples:
- Example 1 (y=0): Cost = 0.6931
- Example 2 (y=0): Cost = 0.6931
- Example 3 (y=1): Cost = 0.6931
- Example 4 (y=1): Cost = 0.6931
- Example 5 (y=0): Cost = 0.6931
- Example 6 (y=1): Cost = 0.6931
- Example 7 (y=1): Cost = 0.6931
- Example 8 (y=0): Cost = 0.6931

The sum of individual costs is 5.5452, and with 8 examples, the average cost is:
$J(\theta) = \frac{1}{8} \cdot 5.5452 = 0.6931$

This initial cost of 0.6931 (which equals $-\log(0.5)$) makes intuitive sense because the model is giving a 50% probability to each class, and we have a balanced dataset with equal numbers of positive and negative examples.

### Step 2: Gradient Descent Iterations
Gradient descent works by iteratively updating the parameters in the direction of steepest descent of the cost function.

#### Iteration 1:
First, we calculate the predictions for each example using the current parameters $\theta = [0, 0, 0]$:
- For all examples, $h(x) = 0.5$ as calculated in Step 1

Next, we compute the errors (prediction - actual):
- Example 1 (y=0): error = 0.5 - 0 = 0.5
- Example 2 (y=0): error = 0.5 - 0 = 0.5
- Example 3 (y=1): error = 0.5 - 1 = -0.5
- Example 4 (y=1): error = 0.5 - 1 = -0.5
- Example 5 (y=0): error = 0.5 - 0 = 0.5
- Example 6 (y=1): error = 0.5 - 1 = -0.5
- Example 7 (y=1): error = 0.5 - 1 = -0.5
- Example 8 (y=0): error = 0.5 - 0 = 0.5

Then we calculate the gradients:

For $\theta_0$:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{8} \cdot (0.5 + 0.5 - 0.5 - 0.5 + 0.5 - 0.5 - 0.5 + 0.5) = \frac{1}{8} \cdot 0 = 0$$

For $\theta_1$ (Age coefficient):
$$\frac{\partial J}{\partial \theta_1} = \frac{1}{8} \cdot (0.5 \cdot 15 + 0.5 \cdot 65 - 0.5 \cdot 30 - 0.5 \cdot 90 + 0.5 \cdot 44 - 0.5 \cdot 20 - 0.5 \cdot 50 + 0.5 \cdot 36) = \frac{1}{8} \cdot (-15) = -1.875$$

For $\theta_2$ (Tumor Size coefficient):
$$\frac{\partial J}{\partial \theta_2} = \frac{1}{8} \cdot (0.5 \cdot 20 + 0.5 \cdot 30 - 0.5 \cdot 50 - 0.5 \cdot 20 + 0.5 \cdot 35 - 0.5 \cdot 70 - 0.5 \cdot 40 + 0.5 \cdot 25) = \frac{1}{8} \cdot (-35) = -4.375$$

Now we update the parameters using the learning rate $\alpha = 0.01$:
- $\theta_0 = 0 - 0.01 \cdot 0 = 0$
- $\theta_1 = 0 - 0.01 \cdot (-1.875) = 0.01875$
- $\theta_2 = 0 - 0.01 \cdot (-4.375) = 0.04375$

After the first iteration, the new parameters are $\theta = [0, 0.01875, 0.04375]$ and the new cost is $J(\theta) = 1.080547$.

#### Iteration 2:
With updated parameters $\theta = [0, 0.01875, 0.04375]$, we calculate new predictions:

Example 1: $z_1 = 0 + 0.01875 \cdot 15 + 0.04375 \cdot 20 = 1.1562$, $h(x_1) = 0.7607$
Example 2: $z_2 = 0 + 0.01875 \cdot 65 + 0.04375 \cdot 30 = 2.5312$, $h(x_2) = 0.9263$
Example 3: $z_3 = 0 + 0.01875 \cdot 30 + 0.04375 \cdot 50 = 2.7500$, $h(x_3) = 0.9399$
Example 4: $z_4 = 0 + 0.01875 \cdot 90 + 0.04375 \cdot 20 = 2.5625$, $h(x_4) = 0.9284$
Example 5: $z_5 = 0 + 0.01875 \cdot 44 + 0.04375 \cdot 35 = 2.3563$, $h(x_5) = 0.9134$
Example 6: $z_6 = 0 + 0.01875 \cdot 20 + 0.04375 \cdot 70 = 3.4375$, $h(x_6) = 0.9689$
Example 7: $z_7 = 0 + 0.01875 \cdot 50 + 0.04375 \cdot 40 = 2.6875$, $h(x_7) = 0.9363$
Example 8: $z_8 = 0 + 0.01875 \cdot 36 + 0.04375 \cdot 25 = 1.7687$, $h(x_8) = 0.8543$

We then compute the new errors:
- Example 1 (y=0): error = 0.7607 - 0 = 0.7607
- Example 2 (y=0): error = 0.9263 - 0 = 0.9263
- Example 3 (y=1): error = 0.9399 - 1 = -0.0601
- Example 4 (y=1): error = 0.9284 - 1 = -0.0716
- Example 5 (y=0): error = 0.9134 - 0 = 0.9134
- Example 6 (y=1): error = 0.9689 - 1 = -0.0311
- Example 7 (y=1): error = 0.9363 - 1 = -0.0637
- Example 8 (y=0): error = 0.8543 - 0 = 0.8543

And calculate the new gradients:

For $\theta_0$:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{8} \cdot (0.7607 + 0.9263 - 0.0601 - 0.0716 + 0.9134 - 0.0311 - 0.0637 + 0.8543) = 0.4035$$

For $\theta_1$ (Age coefficient):
$$\frac{\partial J}{\partial \theta_1} = \frac{1}{8} \cdot (\text{sum of errors} \cdot \text{ages}) = 16.3139$$

For $\theta_2$ (Tumor Size coefficient):
$$\frac{\partial J}{\partial \theta_2} = \frac{1}{8} \cdot (\text{sum of errors} \cdot \text{tumor sizes}) = 10.8956$$

Now we update the parameters again:
- $\theta_0 = 0 - 0.01 \cdot 0.4035 = -0.004035$
- $\theta_1 = 0.01875 - 0.01 \cdot 16.3139 = -0.144389$
- $\theta_2 = 0.04375 - 0.01 \cdot 10.8956 = -0.065206$

After the second iteration, the parameters are $\theta = [-0.004035, -0.144389, -0.065206]$ and the cost is $J(\theta) = 4.902499$.

![Cost Function over Gradient Descent Iterations](../Images/L5_2_Quiz_5/gradient_descent_cost.png)

The cost increases during these first iterations because the initial parameter updates move in a direction that temporarily increases the cost. This happens because the model is being pushed toward predicting nearly all examples as one class (evident from the high prediction values in iteration 2). With more iterations and proper learning rate tuning, the cost would eventually decrease as the model finds better parameters.

### Step 3: Stochastic Gradient Descent Iterations
Stochastic Gradient Descent (SGD) updates parameters using a single randomly selected example at each iteration, rather than the entire dataset.

#### SGD Iteration 1:
First, we randomly select an example. In this case, we selected example 7 (index 6):
- Features: $x = [1, 50, 40]$ (intercept, age, tumor size)
- Target: $y = 1$ (malignant)

With initial parameters $\theta = [0, 0, 0]$, we calculate:
- $z = \theta^T x = 0 \cdot 1 + 0 \cdot 50 + 0 \cdot 40 = 0$
- $h(x) = g(z) = g(0) = 0.5$
- Error = $h(x) - y = 0.5 - 1 = -0.5$

The gradients based on this single example are:
- $\frac{\partial J}{\partial \theta_0} = -0.5 \cdot 1 = -0.5$
- $\frac{\partial J}{\partial \theta_1} = -0.5 \cdot 50 = -25$
- $\frac{\partial J}{\partial \theta_2} = -0.5 \cdot 40 = -20$

Updating the parameters with learning rate $\alpha = 0.1$:
- $\theta_0 = 0 - 0.1 \cdot (-0.5) = 0.05$
- $\theta_1 = 0 - 0.1 \cdot (-25) = 2.5$
- $\theta_2 = 0 - 0.1 \cdot (-20) = 2$

The new parameters after the first SGD iteration are $\theta = [0.05, 2.5, 2]$ and the cost (evaluated on the full dataset) is $J(\theta) = 17.269788$.

#### SGD Iteration 2:
For the second iteration, we randomly selected example 5 (index 4):
- Features: $x = [1, 44, 35]$ (intercept, age, tumor size)
- Target: $y = 0$ (benign)

With updated parameters $\theta = [0.05, 2.5, 2]$, we calculate:
- $z = \theta^T x = 0.05 \cdot 1 + 2.5 \cdot 44 + 2 \cdot 35 = 0.05 + 110 + 70 = 180.05$
- $h(x) = g(z) = g(180.05) \approx 1.0$ (sigmoid of a large positive number approaches 1)
- Error = $h(x) - y = 1 - 0 = 1$

The gradients based on this single example are:
- $\frac{\partial J}{\partial \theta_0} = 1 \cdot 1 = 1$
- $\frac{\partial J}{\partial \theta_1} = 1 \cdot 44 = 44$
- $\frac{\partial J}{\partial \theta_2} = 1 \cdot 35 = 35$

Updating the parameters:
- $\theta_0 = 0.05 - 0.1 \cdot 1 = -0.05$
- $\theta_1 = 2.5 - 0.1 \cdot 44 = -1.9$
- $\theta_2 = 2 - 0.1 \cdot 35 = -1.5$

The parameters after the second SGD iteration are $\theta = [-0.05, -1.9, -1.5]$ and the cost is $J(\theta) = 17.269388$.

The SGD approach shows high variance in parameter updates, which is characteristic of this method. The first iteration made the model strongly predict malignant tumors (pushing the parameters to large positive values), while the second iteration corrected in the opposite direction after encountering a benign example. This zigzagging behavior is typical of SGD, especially with a relatively high learning rate.

### Step 4: Decision Boundary Explanation
The decision boundary in logistic regression is defined by the equation $\theta^T x = 0$, which corresponds to the set of points where the predicted probability is exactly 0.5.

To understand why, recall the sigmoid function:
- When $\theta^T x > 0$, we get $g(\theta^T x) > 0.5$, so we predict class 1 (malignant)
- When $\theta^T x < 0$, we get $g(\theta^T x) < 0.5$, so we predict class 0 (benign)
- When $\theta^T x = 0$, we get $g(\theta^T x) = 0.5$, which is the threshold point

Geometrically, the decision boundary is a hyperplane in the feature space. In our case with two features (age and tumor size), it's a line that separates the feature space into two regions: one where the model predicts benign tumors and another where it predicts malignant tumors.

### Step 5: Decision Boundary with Final Parameters
Given the final optimized parameters $\theta = [-136.95, 1.1, 2.2]$, we can derive the decision boundary equation step by step:

1. The decision boundary is defined by: $\theta_0 + \theta_1 \cdot \text{Age} + \theta_2 \cdot \text{Tumor Size} = 0$
2. Substituting our parameters: $-136.95 + 1.1 \cdot \text{Age} + 2.2 \cdot \text{Tumor Size} = 0$
3. Solving for Tumor Size: $2.2 \cdot \text{Tumor Size} = -(-136.95 + 1.1 \cdot \text{Age})$
4. Dividing both sides by 2.2: $\text{Tumor Size} = -(-136.95 + 1.1 \cdot \text{Age})/2.2$
5. Simplifying: $\text{Tumor Size} = 62.25 - 0.5 \cdot \text{Age}$

This equation gives us the line in the feature space that separates the benign and malignant predictions.

![Decision Boundary for Tumor Classification](../Images/L5_2_Quiz_5/decision_boundary.png)

The decision boundary shows that there's a trade-off between age and tumor size in determining malignancy. For each additional 2 years of age, the decision boundary lowers the tumor size threshold by 1mm. This means that for older patients, even smaller tumors might be classified as malignant, while younger patients would need larger tumors to receive a malignant classification.

### Step 6: Prediction for New Patient
For a new patient with Age=50 years and Tumor Size=30mm, using the parameters $\theta = [-136.95, 1.1, 2.2]$, we can calculate:

1. Compute $z = \theta^T x$:
   * $z = \theta_0 \cdot 1 + \theta_1 \cdot \text{Age} + \theta_2 \cdot \text{Tumor Size}$
   * $z = -136.95 \cdot 1 + 1.1 \cdot 50 + 2.2 \cdot 30$
   * $z = -136.95 + 55 + 66$
   * $z = -15.95$

2. Compute probability $P(y=1|x) = g(z)$:
   * $P(y=1|x) = 1 / (1 + e^{-z})$
   * $P(y=1|x) = 1 / (1 + e^{-(-15.95)})$
   * $P(y=1|x) = 1 / (1 + e^{15.95})$
   * $P(y=1|x) = 1 / (1 + 8.45 \times 10^6)$
   * $P(y=1|x) \approx 0.00000012$ (extremely close to 0)

3. Make classification decision:
   * Since $P(y=1|x) = 0.00000012 < 0.5$, we classify this tumor as benign (y=0).

The very low probability indicates high confidence in this prediction. This is because the point (Age=50, Tumor Size=30) is far below the decision boundary line in the feature space, which requires a tumor size of $62.25 - 0.5 \cdot 50 = 37.25$ mm for a 50-year-old patient to be classified as malignant.

### Step 7: Interpretation of Coefficients
The coefficients in logistic regression represent the change in log-odds of the outcome for a one-unit increase in the corresponding feature, holding other features constant.

$\theta_1 = 1.1$ (Age coefficient):
- For each additional year of age, the log-odds of a tumor being malignant increase by 1.1, holding tumor size constant.
- This corresponds to an odds ratio of $e^{1.1} = 3.0$, meaning the odds of malignancy are multiplied by 3 for each year increase in age.

$\theta_2 = 2.2$ (Tumor Size coefficient):
- For each additional millimeter of tumor size, the log-odds of a tumor being malignant increase by 2.2, holding age constant.
- This corresponds to an odds ratio of $e^{2.2} = 9.03$, meaning the odds of malignancy are multiplied by 9 for each mm increase in tumor size.

Since $\theta_2 > \theta_1$, tumor size has a stronger effect on the probability of malignancy than age. Specifically, the effect of tumor size is approximately 2 times stronger than the effect of age ($2.2/1.1 = 2$). 

This suggests that clinicians should pay particular attention to the size of the tumor when assessing malignancy risk, although both factors are important predictors.

![Probability Surface for Tumor Classification](../Images/L5_2_Quiz_5/probability_surface.png)

The 3D probability surface shows how the probability of malignancy changes across different combinations of age and tumor size. The steep gradient in the tumor size direction visually confirms its stronger influence on the prediction.

### Step 8: Effect of Learning Rate
The learning rate $\alpha$ controls the step size in parameter updates during gradient descent optimization.

**Increasing the learning rate:**
- Advantages:
  * Faster convergence if the rate is well-tuned
  * Requires fewer iterations to reach the optimum
  * Can escape local minima more easily
- Disadvantages:
  * Risk of overshooting the minimum and divergence if too large
  * May oscillate around the minimum without reaching it
  * Can cause numerical instability

**Decreasing the learning rate:**
- Advantages:
  * More stable and reliable convergence
  * Less sensitive to noise in the data
  * Better precision near the optimum
- Disadvantages:
  * Slower progress, requiring more iterations
  * May get stuck in local minima or plateau regions
  * Very small rates may make progress imperceptibly slow

![Effect of Learning Rate on Convergence](../Images/L5_2_Quiz_5/learning_rate_effect.png)

The plot shows how different learning rates affect convergence. With a very small learning rate (0.001), convergence is slow but steady. With a moderate rate (0.01), we see faster progress. Higher rates (0.1, 0.5) can lead to faster initial progress but may cause oscillations or even divergence.

## Visual Explanations

### Cost Function Surface
![Logistic Regression Cost Function Surface](../Images/L5_2_Quiz_5/cost_function_surface.png)

The cost function surface shows how the cost varies with different values of $\theta_1$ and $\theta_2$ (with $\theta_0$ fixed). The convex shape confirms that the logistic regression cost function has a single global minimum, making it amenable to gradient-based optimization methods.

### Tumor Dataset Visualization
![Tumor Classification Dataset](../Images/L5_2_Quiz_5/dataset_visualization.png)

The scatter plot shows the distribution of benign (blue circles) and malignant (red X's) tumors in the feature space. There's a clear pattern where malignant tumors tend to be larger and/or occur in older patients, which the logistic regression model aims to capture.

## Key Insights

### Mathematical Foundations
- Logistic regression uses the sigmoid function to map linear combinations of features to probabilities
- The decision boundary is determined by where the model predicts exactly 0.5 probability
- The cross-entropy loss function heavily penalizes confident but wrong predictions
- Gradient descent and its variants (like SGD) optimize the parameters by iteratively following the negative gradient

### Practical Applications
- In medical diagnostics, logistic regression provides interpretable results through its coefficients
- The odds ratios (3.0 for age and 9.0 for tumor size) quantify the influence of each factor
- For the patient with age 50 and tumor size 30mm, the model confidently predicts a benign tumor
- The decision boundary equation (Tumor Size = 62.25 - 0.5 × Age) provides a simple rule for clinicians

### Optimization Techniques
- Batch gradient descent uses all examples for each update, providing stable but slower updates
- Stochastic gradient descent uses a single random example per update, providing faster but noisier updates
- The learning rate must be carefully tuned to balance convergence speed and stability
- Initial iterations may sometimes increase the cost before finding the descent direction

## Conclusion
- The initial cost with zero parameters is 0.6931, which is exactly -log(0.5), reflecting equal probabilities for all examples
- Gradient descent updates move the parameters in the direction that minimizes the cost function, though early iterations may temporarily increase it
- Stochastic gradient descent shows higher variance in updates, reflecting its use of individual examples
- The decision boundary equation (Tumor Size = 62.25 - 0.5 × Age) creates a line that separates benign and malignant regions
- For a new patient with age 50 and tumor size 30mm, the model predicts a benign tumor with high confidence (probability ≈ 0)
- Tumor size has a stronger influence on malignancy prediction than age (odds ratios of 9.0 vs 3.0)
- The learning rate must be carefully chosen to ensure efficient convergence without oscillation or divergence

Logistic regression provides a powerful yet interpretable approach to medical classification problems. The ability to directly interpret coefficients as log-odds makes it particularly valuable in healthcare, where understanding the model's reasoning is often as important as its accuracy.
