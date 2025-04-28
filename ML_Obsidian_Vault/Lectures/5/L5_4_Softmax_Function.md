# Softmax Function

## Definition
The softmax function, also known as the normalized exponential function, is a generalization of the sigmoid function for multiple classes. It converts a vector of real numbers into a probability distribution. For a vector $\mathbf{z} = (z_1, z_2, \ldots, z_K)$, the softmax function is defined as:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

This transforms each element $z_i$ to a value in the range $(0, 1)$, with the sum of all values equal to 1.

## Properties
- Outputs a probability distribution (all values sum to 1)
- Preserves the order of inputs (if $z_i > z_j$, then $\text{softmax}(z_i) > \text{softmax}(z_j)$)
- Scale invariant to a constant added to all inputs
- Differentiable with respect to all inputs
- Sensitivity to large differences in inputs due to exponential function
- Non-linear transformation that emphasizes larger input values

## Role in Multi-class Logistic Regression
The softmax function is the foundation of multinomial logistic regression (softmax regression). It extends binary logistic regression to problems with multiple class labels. In this context:

$$P(y=i|\mathbf{x}) = \frac{e^{\mathbf{w}_i^T \mathbf{x} + b_i}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

where:
- $\mathbf{w}_i$ is the weight vector for class $i$
- $\mathbf{x}$ is the feature vector
- $b_i$ is the bias term for class $i$
- $P(y=i|\mathbf{x})$ is the probability that the sample belongs to class $i$
- $K$ is the total number of classes

The predicted class is the one with the highest probability: $\hat{y} = \arg\max_i P(y=i|\mathbf{x})$.

## Relation to Sigmoid Function
For binary classification ($K=2$), the softmax function is equivalent to the sigmoid function:

When $K=2$, we have:
$$P(y=1|\mathbf{x}) = \frac{e^{\mathbf{w}_1^T \mathbf{x} + b_1}}{e^{\mathbf{w}_1^T \mathbf{x} + b_1} + e^{\mathbf{w}_2^T \mathbf{x} + b_2}}$$

If we define $\mathbf{w} = \mathbf{w}_1 - \mathbf{w}_2$ and $b = b_1 - b_2$, we get:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Which is exactly the sigmoid function.

## Mathematical Properties
The Jacobian matrix of the softmax function has a simple form:
$$\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \text{softmax}(z_i)(\delta_{ij} - \text{softmax}(z_j))$$

where $\delta_{ij}$ is the Kronecker delta.

## Cross-Entropy Loss
The softmax function is typically used with the cross-entropy loss function for multi-class classification:

$$L = -\sum_{i=1}^{K} y_i \log(\text{softmax}(z_i))$$

where $y_i$ is 1 if the true class is $i$ and 0 otherwise (one-hot encoding).

## Numerical Stability
Direct computation of softmax can lead to numerical overflow due to the exponential function. A common trick is to subtract the maximum value before computing the exponential:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}$$

This doesn't change the result but prevents numerical issues.

## Implementation
Here's a numerically stable implementation of the softmax function in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    # Subtract max for numerical stability
    shifted_z = z - np.max(z)
    # Calculate exponentials
    exp_values = np.exp(shifted_z)
    # Normalize to get probabilities
    return exp_values / np.sum(exp_values)

# Example with 3 classes
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print(f"Input scores: {scores}")
print(f"Softmax probabilities: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")

# Visualization for 3 classes
def plot_softmax():
    # Create a range of values for two classes (third will be fixed)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Fixed value for third class
    z3 = 0
    
    # Calculate softmax for each point
    Z = np.zeros((100, 100, 3))
    for i in range(100):
        for j in range(100):
            Z[i, j] = softmax([X[i, j], Y[i, j], z3])
    
    # Plot probability of first class
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z[:, :, 0], 20, cmap='viridis')
    plt.colorbar(label='P(class 1)')
    plt.title('Softmax Probability for Class 1 (with class 3 fixed at 0)')
    plt.xlabel('Score for class 1')
    plt.ylabel('Score for class 2')
    plt.grid(True)
    plt.show()

plot_softmax()
```

## Applications in Neural Networks
- Output layer of neural networks for multi-class classification
- Attention mechanisms in transformer architectures
- Reinforcement learning for action probability distributions
- Temperature scaling in softmax can control distribution sharpness

## Related Concepts
- [[L5_1_Sigmoid_Function|Sigmoid Function]] (binary case)
- [[L5_4_Multi_Class_Extension|Multi-Class Classification]]
- [[L5_4_Multi_Class_MLE|Multi-Class MLE]]
- [[L5_4_Multi_Class_Cross_Entropy|Multi-Class Cross Entropy]]
- [[L4_6_Softmax_Regression|Softmax Regression in Linear Classifiers]]

## Historical Context
The softmax function was introduced in the context of regression problems by statistician David Cox in 1958. It gained popularity in machine learning with the rise of artificial neural networks in the 1980s and has become a standard component in modern deep learning architectures.

## References
1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. Chapter 4.3.4.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Chapter 6.2.2.
3. Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B, 20(2), 215-242. 