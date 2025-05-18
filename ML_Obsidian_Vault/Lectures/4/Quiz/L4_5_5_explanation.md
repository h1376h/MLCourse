# Question 5: Perceptron Learning Algorithm

## Problem Statement
Consider the perceptron learning algorithm with the following update rule:

$$w_{t+1} = w_t + \eta y_i x_i$$

if $y_i (w_t^T x_i) \leq 0$ (misclassification), and $w_{t+1} = w_t$ otherwise.

### Task
1. How does this update rule differ from the standard gradient descent update for logistic regression?
2. Why might this update rule converge faster than gradient descent for linearly separable data?
3. For a misclassified point with $x = [1, 2, 1]^T$ (including bias term), $y = 1$, and current weights $w = [0, 1, -1]^T$, calculate the updated weights using $\eta = 0.5$
4. Will this update always reduce the number of misclassified points? Explain why or why not.

## Understanding the Problem
The perceptron is one of the simplest types of artificial neural networks and forms the foundation for more complex models. It implements a binary classifier that makes predictions using a linear decision boundary. The perceptron learning algorithm updates the weights of the model when it encounters misclassified points, attempting to find a separating hyperplane for linearly separable data.

The core of the problem is understanding how the perceptron's update rule works, how it differs from other optimization methods like gradient descent for logistic regression, and what its limitations are in terms of guaranteeing error reduction.

## Solution

### Task 1: Comparing Perceptron Update Rule with Gradient Descent for Logistic Regression

#### Perceptron Update Rule:
$$w_{t+1} = w_t + \eta y_i x_i \quad \text{(if } y_i(w_t^T x_i) \leq 0 \text{)}$$

#### Gradient Descent Update for Logistic Regression:
$$w_{t+1} = w_t + \eta (y_i - \sigma(w_t^T x_i)) x_i$$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

The key differences are:
1. **Conditional vs. Unconditional**: The perceptron only updates when a point is misclassified, while logistic regression updates for every point.
2. **Magnitude of Update**: The perceptron update size is fixed by $\eta y_i x_i$ for misclassified points, whereas in logistic regression, the update is scaled by how far the prediction is from the true label.
3. **Error Function**: The perceptron directly minimizes misclassification errors, while logistic regression minimizes cross-entropy loss that approximates the 0-1 loss function.

### Task 2: Why Perceptron Might Converge Faster for Linearly Separable Data

When data is linearly separable, the perceptron convergence theorem guarantees that the algorithm will find a perfect decision boundary in a finite number of updates. The key advantages are:

1. **Focused Updates**: Perceptron only updates when necessary (misclassifications), ignoring correctly classified points and making larger steps toward the solution.
2. **Direct Optimization**: It directly optimizes the decision boundary position rather than a surrogate loss function.

![Convergence Comparison](../Images/L4_5_Quiz_5/convergence_comparison.png)

The convergence plots from our code execution show that the perceptron (left) achieves zero misclassifications after just a few epochs, while logistic regression (right) continues to optimize its loss function even after finding a reasonable decision boundary.

![Decision Boundaries at Epoch 0](../Images/L4_5_Quiz_5/decision_boundaries_epoch_0.png)

At the beginning (epoch 0), both algorithms start with all weights set to zero. The decision regions are clearly visualized, with light blue indicating Class +1 regions and light red indicating Class -1 regions.

![Decision Boundaries at Epoch 5](../Images/L4_5_Quiz_5/decision_boundaries_epoch_5.png)

After just 5 epochs, the perceptron boundary (left) is already very close to the true boundary, while logistic regression (right) takes more iterations to approach the optimal solution. The decision regions are now fully colored throughout the entire plot area, making it easy to see the classification boundaries.

![Decision Boundaries at Final Epoch](../Images/L4_5_Quiz_5/decision_boundaries_epoch_19.png)

By epoch 19 (final), both algorithms have found good decision boundaries, but the perceptron reaches this point faster. The improved visualization shows clear and consistent coloring of the decision regions across the entire feature space.

### Task 3: Calculate Updated Weights for the Specific Example

Given information:
- Point: $x = [1, 2, 1]^T$ (including bias term)
- Label: $y = 1$
- Current weights: $w = [0, 1, -1]^T$
- Learning rate: $\eta = 0.5$

Step 1: Check if the point is misclassified by computing $y(w^T x)$:
$$w^T x = 0 \times 1 + 1 \times 2 + (-1) \times 1 = 2 - 1 = 1$$
$$y(w^T x) = 1 \times 1 = 1 > 0$$

Since $y(w^T x) > 0$, the point is correctly classified. Therefore, no update is needed:
$$w_{t+1} = w_t = [0, 1, -1]^T$$

![Specific Example Visualization 2D](../Images/L4_5_Quiz_5/specific_example_2d.png)

The 2D visualization confirms that the point lies in the correctly classified region (blue area). The decision boundary is already oriented such that the point is on the correct side. The updated visualization includes clearer labels and improved region coloring to better distinguish the positive and negative classification regions.

![3D Weight Update Visualization](../Images/L4_5_Quiz_5/specific_example_update.png)

The 3D visualization provides a spatial perspective of how the weight vector and decision boundary would change if an update were needed. In this case, since the point is correctly classified, the weight vector remains unchanged. The visualization shows the original weights (blue arrow) and the decision boundary plane (blue surface), with better visibility and clarity compared to previous versions.

### Task 4: Will This Update Always Reduce the Number of Misclassified Points?

No, the perceptron update will not always reduce the total number of misclassified points. While it guarantees that the specific misclassified point used for the update will be correctly classified if it's the only point in the dataset, this improvement can come at the expense of misclassifying other points that were previously correctly classified.

![Non-Reduction Example](../Images/L4_5_Quiz_5/non_reduction_example.png)

In our example:
- Before update (left): Both points are misclassified (2 total misclassifications)
- After updating with respect to the first point (right): The first point becomes correctly classified, but the second point remains misclassified (1 total misclassification)

The improved visualization uses distinct markers for each point and consistent coloring of the decision regions, making it easier to see how the update affects each point's classification status. This demonstrates that while a single update might reduce the overall number of misclassifications, it's not guaranteed to do so for every possible dataset configuration. This is why perceptron learning may oscillate on non-linearly separable data.

## Visual Explanations

### Perceptron vs. Logistic Regression Convergence
![Convergence Comparison](../Images/L4_5_Quiz_5/convergence_comparison.png)

This visualization compares the convergence behavior of perceptron and logistic regression:
- The left plot shows the number of misclassifications per epoch for perceptron learning
- The right plot shows the binary cross-entropy loss per epoch for logistic regression

Notice how the perceptron achieves zero misclassifications rapidly, while logistic regression continues to optimize its loss function incrementally.

### Decision Boundary Evolution
![Decision Boundaries Evolution Sequence](../Images/L4_5_Quiz_5/decision_boundaries_epoch_19.png)

These visualizations show how the decision boundaries evolve over time:
- The perceptron (left) makes discrete jumps toward the optimal boundary when it encounters misclassified points
- Logistic regression (right) makes continuous small adjustments toward the optimal boundary
- The enhanced visualization now clearly shows the complete decision regions throughout the entire feature space, with consistent coloring and improved legend placement

### 3D Visualization of Weight Update Process
![3D Weight Update](../Images/L4_5_Quiz_5/specific_example_update.png)

This 3D visualization shows how the weight vector and decision boundary change after a perceptron update. For correctly classified points (as in Task 3), no update occurs. For misclassified points, the weight vector would move in the direction that pushes the decision boundary to correctly classify the point. The improved visualization uses better camera angles and clearer labels to make the 3D relationships more apparent.

## Key Insights

### Theoretical Foundations
- The perceptron converges in a finite number of steps if the data is linearly separable (Novikoff's theorem)
- The perceptron directly minimizes the 0-1 loss function, which counts misclassifications
- The perceptron's update rule is closely related to stochastic gradient descent on a hinge loss

### Practical Implications
- The perceptron works well for linearly separable data but can fail to converge for non-linearly separable data
- Unlike logistic regression, perceptron only provides binary predictions, not probability estimates
- The simplicity of the perceptron update rule makes it computationally efficient

### Limitations and Extensions
- The perceptron cannot solve the XOR problem or other non-linearly separable problems
- More advanced variants like the voted perceptron or kernel perceptron can handle more complex data
- The perceptron update rule forms the basis for more complex neural network training algorithms

## Conclusion
- The perceptron update rule differs from logistic regression's gradient descent update by being conditional and having a fixed update magnitude that doesn't depend on prediction confidence.
- Perceptron convergence is typically faster for linearly separable data because it makes direct, focused updates only when needed.
- For the given example with $x = [1, 2, 1]^T$, $y = 1$, $w = [0, 1, -1]^T$, and $\eta = 0.5$, no update is needed since the point is already correctly classified.
- The perceptron update does not guarantee a reduction in the total number of misclassifications across all data points, though it will correct the single misclassification it's updating for. 