# Question 10: Pocket Algorithm Applications

## Problem Statement
Consider a simple 2D dataset with points that are not linearly separable, and you need to apply the Pocket Algorithm.

### Task
1. Explain the goal of the Pocket Algorithm in one sentence
2. If after 100 iterations, your Pocket weights are $w = [3, -1, 2]^T$ (including bias term), write the equation of the corresponding decision boundary
3. For a perceptron with learning rate $\eta = 0.1$, calculate the weight update for a misclassified point $x = [2, 1]^T$ with true label $y = 1$
4. Why does the Pocket Algorithm perform better than the standard Perceptron for non-separable data? Explain in one sentence

## Understanding the Problem
The Pocket Algorithm is an extension of the standard Perceptron algorithm specifically designed to handle data that is not linearly separable. When data classes overlap or cannot be perfectly separated by a straight line (or hyperplane in higher dimensions), the standard Perceptron may never converge. The Pocket Algorithm addresses this limitation by keeping track of the best-performing weights found so far during training, ensuring we obtain the most effective linear classifier possible for challenging datasets.

## Solution

### Step 1: Understand the goal of the Pocket Algorithm
The Pocket Algorithm's primary goal is to find the best possible linear decision boundary that minimizes the number of misclassifications for non-separable data by keeping track of the weights that give the highest classification accuracy.

This approach differs from the standard Perceptron in a critical way:
- The standard Perceptron updates weights whenever it encounters a misclassified point but keeps only the final weights
- The Pocket Algorithm also updates weights but maintains a separate set of "best weights" that achieved the highest accuracy over all training data

![Non-separable Dataset](../Images/L4_4_Quiz_10/non_separable_dataset.png)

### Step 2: Write the decision boundary equation
Given pocket weights $w = [3, -1, 2]^T$ where the first element is the bias term:
- Bias: $w_0 = 3$
- First feature weight: $w_1 = -1$
- Second feature weight: $w_2 = 2$

The decision boundary equation is:
$$w_1 x_1 + w_2 x_2 + w_0 = 0$$

Substituting the given weights:
$$-1 \cdot x_1 + 2 \cdot x_2 + 3 = 0$$

Rearranging to standard form:
$$-x_1 + 2x_2 + 3 = 0$$

This equation represents a line in 2D space that serves as the decision boundary. Points on one side are classified as positive (Class 1), and points on the other side as negative (Class 2).

### Step 3: Calculate weight update for a misclassified point
For a perceptron with learning rate $\eta = 0.1$, when a point $x = [2, 1]^T$ with true label $y = 1$ is misclassified, the weight update is:

$$\Delta w = \eta \cdot y \cdot x$$

For the given values:
$$\Delta w = 0.1 \cdot 1 \cdot [2, 1]^T = [0.2, 0.1]^T$$

Note that for the full weight vector including bias, we would use the augmented input $[1, 2, 1]^T$ (with a 1 prepended for the bias), resulting in:
$$\Delta w = 0.1 \cdot 1 \cdot [1, 2, 1]^T = [0.1, 0.2, 0.1]^T$$

This update would be added to the current weights to get the new weights.

### Step 4: Implement and test the Pocket Algorithm
To demonstrate the Pocket Algorithm's effectiveness, we implemented it on a synthetic non-separable dataset:

```python
def pocket_algorithm(X, y, max_iterations=1000, learning_rate=0.1):
    # Add a column of ones for the bias term
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Initialize weights randomly
    weights = np.random.randn(X_with_bias.shape[1])
    
    # Initialize pocket weights and best accuracy
    pocket_weights = weights.copy()
    best_accuracy = 0.0
    
    accuracy_history = []
    best_accuracy_history = []
    
    for iteration in range(max_iterations):
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X_shuffled = X_with_bias[indices]
        y_shuffled = y[indices]
        
        # Loop through all samples
        for i in range(len(X_shuffled)):
            # Make prediction
            prediction = np.sign(np.dot(X_shuffled[i], weights))
            
            # Update weights if misclassified
            if prediction != y_shuffled[i]:
                weights += learning_rate * y_shuffled[i] * X_shuffled[i]
        
        # Calculate current accuracy on the whole dataset
        predictions = np.sign(np.dot(X_with_bias, weights))
        accuracy = np.mean(predictions == y)
        accuracy_history.append(accuracy)
        
        # Update pocket weights if accuracy improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            pocket_weights = weights.copy()
        
        best_accuracy_history.append(best_accuracy)
    
    return pocket_weights, weights, accuracy_history, best_accuracy_history
```

Running this algorithm on our dataset for 100 iterations yielded:
- Pocket weights: $[0.6783, -0.1395, -0.0859]^T$
- Final standard Perceptron weights: $[0.4683, -0.0904, -0.0600]^T$
- Best accuracy (Pocket): 87.5%
- Final accuracy (Standard Perceptron): 86.0%

![Pocket Accuracy](../Images/L4_4_Quiz_10/pocket_accuracy.png)

The graph shows how the standard Perceptron accuracy (blue) fluctuates while the Pocket algorithm (red) maintains the best accuracy found.

### Step 5: Compare decision boundaries
To visualize the difference between the Pocket Algorithm and the standard Perceptron, we plotted their respective decision boundaries:

![Pocket Decision Boundary](../Images/L4_4_Quiz_10/pocket_decision_boundary.png)

![Perceptron Decision Boundary](../Images/L4_4_Quiz_10/perceptron_decision_boundary.png)

The comparison reveals that the Pocket Algorithm achieves better classification performance:
- Pocket Algorithm: 50/400 misclassified points (87.5% accuracy)
- Standard Perceptron: 56/400 misclassified points (86.0% accuracy)

![Boundary Comparison](../Images/L4_4_Quiz_10/boundary_comparison.png)

### Step 6: Classify a new point
To demonstrate practical application, we classified a new point $(3, 3)$ using both algorithms:

![New Point Classification](../Images/L4_4_Quiz_10/new_point_classification.png)

Both algorithms classified this point as Class 1, but this may not always be the case, especially for points near the decision boundary.

### Step 7: Connect to bias-variance tradeoff
The Pocket Algorithm relates to the bias-variance tradeoff in machine learning:

1. **Standard Perceptron** may exhibit higher variance:
   - It's sensitive to the ordering of training examples
   - May not converge for non-separable data, leading to unstable results
   - Final weights depend heavily on the most recently processed examples

2. **Pocket Algorithm** reduces variance at a potential small cost to bias:
   - By keeping the best weights, it's less affected by the specific ordering of examples
   - Ensures a more stable solution by optimizing over the entire dataset
   - Effectively implements a form of early stopping based on validation performance

## Key Insights

### Theoretical Understanding
- The Pocket Algorithm is a straightforward yet effective modification to the Perceptron
- It addresses the non-convergence issue of the standard Perceptron for non-separable data
- It can be viewed as implementing a form of empirical risk minimization

### Practical Advantages
- The Pocket Algorithm generally achieves higher accuracy on non-separable datasets
- It provides more stable results across different training runs
- The implementation requires minimal additional computation or memory
- It can be considered a "best-effort" approach for linear classification when perfect separation is impossible

### Implementation Considerations
- The algorithm requires tracking accuracy across the entire dataset after each epoch
- Like the standard Perceptron, it still depends on the learning rate and initialization
- The maximum number of iterations becomes an important parameter since the algorithm does not naturally converge

## Conclusion
1. The goal of the Pocket Algorithm is to find the best possible linear decision boundary for non-separable data by keeping track of the weights that correctly classify the most training examples.

2. For the given pocket weights $w = [3, -1, 2]^T$, the decision boundary equation is $-x_1 + 2x_2 + 3 = 0$.

3. For a perceptron with learning rate $\eta = 0.1$, the weight update for a misclassified point $x = [2, 1]^T$ with true label $y = 1$ would be $\Delta w = [0.1, 0.2, 0.1]^T$ (including bias update).

4. The Pocket Algorithm performs better than the standard Perceptron for non-separable data because it keeps track of the best-performing weights throughout training rather than just returning the final weights, which may be suboptimal due to the inherent oscillation of the Perceptron in non-separable scenarios. 