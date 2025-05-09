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

The core of logistic regression is the sigmoid function, which maps any real-valued number to the range [0,1], making it suitable for representing probabilities:

![Sigmoid Function](../Images/L5_2_Quiz_5/sigmoid_function.png)

The sigmoid function has the property that $g(0) = 0.5$. This is crucial for understanding the decision boundary - when the model's raw output ($\theta^T x$) is exactly 0, the predicted probability is exactly 0.5.

For this binary classification problem, we use maximum likelihood estimation, which leads to the cost function given in the problem statement. This cost function (also known as the log loss or cross-entropy loss) penalizes confident but incorrect predictions heavily.

## Solution

### Step 1: Calculate the Initial Cost
With the initial parameters $\theta = [0, 0, 0]$, the model predicts a probability of 0.5 for all examples, regardless of their features. This is because $\theta^T x = 0$ for all examples, and $g(0) = 0.5$.

For each training example, the cost is calculated as:
$$-[y^{(i)}\log(0.5) + (1-y^{(i)})\log(0.5)]$$

Since $\log(0.5) = -\log(2) \approx -0.693$, the cost for each example is:
$$-[y^{(i)} \cdot (-0.693) + (1-y^{(i)}) \cdot (-0.693)] = 0.693$$

Averaging over all 8 examples, the initial cost is $J(\theta) = 0.693$.

Detailed calculation:
```
Predictions h(x) with initial θ: [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
Individual costs: [0.693 0.693 0.693 0.693 0.693 0.693 0.693 0.693]
Average cost: 0.693
```

### Step 2: Gradient Descent Iterations
Gradient descent works by iteratively updating the parameters in the direction of steepest descent of the cost function.

For logistic regression, the gradient of the cost function with respect to parameter $\theta_j$ is:
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

#### Iteration 1:
Initial predictions: $h_\theta(x) = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]$

Errors (prediction - actual): $[0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5]$

Gradients:
- $\frac{\partial J}{\partial \theta_0} = \frac{1}{8}(0.5 + 0.5 - 0.5 - 0.5 + 0.5 - 0.5 - 0.5 + 0.5) \cdot 1 = 0$
- $\frac{\partial J}{\partial \theta_1} = \frac{1}{8}(0.5 \cdot 15 + 0.5 \cdot 65 - 0.5 \cdot 30 - 0.5 \cdot 90 + 0.5 \cdot 44 - 0.5 \cdot 20 - 0.5 \cdot 50 + 0.5 \cdot 36) = -1.875$
- $\frac{\partial J}{\partial \theta_2} = \frac{1}{8}(0.5 \cdot 20 + 0.5 \cdot 30 - 0.5 \cdot 50 - 0.5 \cdot 20 + 0.5 \cdot 35 - 0.5 \cdot 70 - 0.5 \cdot 40 + 0.5 \cdot 25) = -4.375$

Updated parameters:
- $\theta_0 = 0 - 0.01 \cdot 0 = 0$
- $\theta_1 = 0 - 0.01 \cdot (-1.875) = 0.01875$
- $\theta_2 = 0 - 0.01 \cdot (-4.375) = 0.04375$

New cost: $J(\theta) = 1.081$

#### Iteration 2:
New predictions with updated $\theta = [0, 0.01875, 0.04375]$:
$h_\theta(x) = [0.761, 0.926, 0.940, 0.928, 0.913, 0.969, 0.936, 0.854]$

Errors: $[0.761, 0.926, -0.060, -0.072, 0.913, -0.031, -0.064, 0.854]$

Gradients:
- $\frac{\partial J}{\partial \theta_0} = 0.404$
- $\frac{\partial J}{\partial \theta_1} = 16.314$
- $\frac{\partial J}{\partial \theta_2} = 10.896$

Updated parameters:
- $\theta_0 = 0 - 0.01 \cdot 0.404 = -0.004$
- $\theta_1 = 0.01875 - 0.01 \cdot 16.314 = -0.144$
- $\theta_2 = 0.04375 - 0.01 \cdot 10.896 = -0.065$

New cost: $J(\theta) = 4.902$

![Cost Function over Gradient Descent Iterations](../Images/L5_2_Quiz_5/gradient_descent_cost.png)

The cost increases during these first iterations because the initial parameter updates move in a direction that temporarily increases the cost. This is common in the early stages of optimization, especially with non-normalized features. With more iterations and proper learning rate tuning, the cost would eventually decrease.

### Step 3: Stochastic Gradient Descent Iterations
Stochastic Gradient Descent (SGD) updates parameters using a single randomly selected example at each iteration, rather than the entire dataset.

#### Iteration 1:
Randomly selected example: index 6 (Age=50, Tumor_Size=40, y=1)

Prediction: $h_\theta(x^{(6)}) = 0.5$

Error: $0.5 - 1 = -0.5$

Gradients:
- $\frac{\partial J}{\partial \theta_0} = -0.5 \cdot 1 = -0.5$
- $\frac{\partial J}{\partial \theta_1} = -0.5 \cdot 50 = -25$
- $\frac{\partial J}{\partial \theta_2} = -0.5 \cdot 40 = -20$

Updated parameters:
- $\theta_0 = 0 - 0.1 \cdot (-0.5) = 0.05$
- $\theta_1 = 0 - 0.1 \cdot (-25) = 2.5$
- $\theta_2 = 0 - 0.1 \cdot (-20) = 2$

#### Iteration 2:
Randomly selected example: index 3 (Age=90, Tumor_Size=20, y=1)

With updated parameters $\theta = [0.05, 2.5, 2]$, the prediction is:
$h_\theta(x^{(3)}) = g(0.05 + 2.5 \cdot 90 + 2 \cdot 20) = g(265) \approx 1$

Error: $1 - 1 = 0$

Since the error is zero, the gradients are all zero, and the parameters remain unchanged.

The SGD approach shows high variance in parameter updates, which is characteristic of this method. SGD can converge faster in early iterations but may oscillate more around the optimum.

### Step 4: Decision Boundary Explanation
The decision boundary in logistic regression is defined by the equation $\theta^T x = 0$, which corresponds to the set of points where the predicted probability is exactly 0.5.

To understand why, recall that:
- When $\theta^T x > 0$, then $g(\theta^T x) > 0.5$, so we predict class 1 (malignant)
- When $\theta^T x < 0$, then $g(\theta^T x) < 0.5$, so we predict class 0 (benign)
- When $\theta^T x = 0$, then $g(\theta^T x) = 0.5$, which is the threshold point

Geometrically, the decision boundary is a hyperplane in the feature space. In our case with two features (age and tumor size), it's a line that separates the feature space into two regions: one where the model predicts benign tumors and another where it predicts malignant tumors.

### Step 5: Decision Boundary with Final Parameters
Given the final optimized parameters $\theta = [-136.95, 1.1, 2.2]$, the decision boundary equation is:
$$\theta_0 + \theta_1 \cdot \text{Age} + \theta_2 \cdot \text{Tumor Size} = 0$$
$$-136.95 + 1.1 \cdot \text{Age} + 2.2 \cdot \text{Tumor Size} = 0$$

Rearranging to express Tumor Size in terms of Age:
$$2.2 \cdot \text{Tumor Size} = 136.95 - 1.1 \cdot \text{Age}$$
$$\text{Tumor Size} = 62.25 - 0.5 \cdot \text{Age}$$

This equation gives us the line in the feature space that separates the benign and malignant predictions.

![Decision Boundary for Tumor Classification](../Images/L5_2_Quiz_5/decision_boundary.png)

The decision boundary shows that there's a trade-off between age and tumor size in determining malignancy. For older patients, even smaller tumors might be classified as malignant, while younger patients would need larger tumors to receive a malignant classification.

### Step 6: Prediction for New Patient
For a new patient with Age=50 years and Tumor Size=30mm, using the parameters $\theta = [-136.95, 1.1, 2.2]$:

$$z = \theta^T x = -136.95 + 1.1 \cdot 50 + 2.2 \cdot 30 = -136.95 + 55 + 66 = -15.95$$

The predicted probability is:
$$P(y=1|x) = g(z) = \frac{1}{1+e^{-(-15.95)}} = \frac{1}{1+e^{15.95}} \approx 0$$

Since the probability is significantly less than 0.5, we classify this tumor as benign (y=0).

The very low probability indicates high confidence in this prediction. This is because the point (50, 30) is far below the decision boundary line in the feature space.

### Step 7: Interpretation of Coefficients
The coefficients in logistic regression represent the change in log-odds of the outcome for a one-unit increase in the corresponding feature, holding other features constant.

- $\theta_1 = 1.1$: For each additional year of age, the log-odds of a tumor being malignant increase by 1.1, holding tumor size constant. This corresponds to an odds ratio of $e^{1.1} = 3.0$, meaning the odds of malignancy are multiplied by 3 for each year increase in age.

- $\theta_2 = 2.2$: For each additional millimeter of tumor size, the log-odds of a tumor being malignant increase by 2.2, holding age constant. This corresponds to an odds ratio of $e^{2.2} = 9.0$, meaning the odds of malignancy are multiplied by 9 for each mm increase in tumor size.

Since $\theta_2 > \theta_1$, tumor size has a stronger effect on the probability of malignancy than age. This suggests that clinicians should pay particular attention to the size of the tumor when assessing malignancy risk.

![Probability Surface for Tumor Classification](../Images/L5_2_Quiz_5/probability_surface.png)

The 3D probability surface shows how the probability of malignancy changes across different combinations of age and tumor size. The steep gradient in the tumor size direction visually confirms its stronger influence on the prediction.

### Step 8: Effect of Learning Rate
The learning rate $\alpha$ controls the step size in parameter updates during gradient descent optimization.

**Increasing the learning rate:**
- Faster convergence if well-tuned
- Risk of overshooting the minimum and divergence if too large
- May oscillate around the minimum without reaching it

**Decreasing the learning rate:**
- More stable and reliable convergence
- Slower progress, requiring more iterations
- May get stuck in local minima or plateau regions
- Very small rates may make progress imperceptibly slow

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
- The initial cost with zero parameters is 0.693, which is exactly -log(0.5), reflecting equal probabilities for all examples
- Gradient descent updates move the parameters in the direction that minimizes the cost function, though early iterations may temporarily increase it
- Stochastic gradient descent shows higher variance in updates, reflecting its use of individual examples
- The decision boundary equation (Tumor Size = 62.25 - 0.5 × Age) creates a line that separates benign and malignant regions
- For a new patient with age 50 and tumor size 30mm, the model predicts a benign tumor with high confidence
- Tumor size has a stronger influence on malignancy prediction than age (odds ratios of 9.0 vs 3.0)
- The learning rate must be carefully chosen to ensure efficient convergence without oscillation or divergence

Logistic regression provides a powerful yet interpretable approach to medical classification problems. The ability to directly interpret coefficients as log-odds makes it particularly valuable in healthcare, where understanding the model's reasoning is often as important as its accuracy.
