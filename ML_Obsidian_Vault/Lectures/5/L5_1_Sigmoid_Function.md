# Sigmoid Function

## Definition
The sigmoid function, also known as the logistic function, is a mathematical function with an S-shaped curve (sigmoid curve) that maps any real-valued number to a value between 0 and 1. It is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z$ is any real number, and $e$ is the base of the natural logarithm.

## Properties
- The sigmoid function is bounded between 0 and 1
- It is monotonically increasing
- It is differentiable
- Its derivative has a simple form: $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- It is symmetric around the point $(0, 0.5)$
- It approaches 0 as $z \to -\infty$ and approaches 1 as $z \to \infty$

## Role in Logistic Regression
In logistic regression, the sigmoid function serves as the activation function that transforms a linear combination of features into a probability value:

$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

where:
- $w$ is the weight vector
- $x$ is the feature vector
- $b$ is the bias term
- $P(y=1|x)$ is the probability that the sample belongs to the positive class

The sigmoid function allows us to interpret the output as a probability, which is essential for binary classification problems. When the model's output is greater than 0.5, we classify the input as belonging to the positive class, otherwise to the negative class.

## Decision Boundary
The decision boundary in logistic regression is defined by the points where the sigmoid function equals 0.5, which occurs when the input to the sigmoid function is 0:

$$\sigma(z) = 0.5 \Rightarrow z = 0 \Rightarrow w^Tx + b = 0$$

This creates a linear decision boundary in the feature space.

## Limitations
- Saturates for large positive or negative inputs, which can lead to vanishing gradients
- Not zero-centered, which can cause zig-zagging dynamics during gradient descent
- In modern deep learning, other activation functions like ReLU often outperform sigmoid for hidden layers

## Implementation
Here's a simple implementation of the sigmoid function in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Visualize the sigmoid function
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid_values)
plt.grid(True)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
plt.show()
```

## Related Concepts
- [[L5_1_Logistic_Regression_Model|Logistic Regression Model]]
- [[L5_1_Binary_Classification|Binary Classification]]
- [[L5_1_Decision_Boundary|Decision Boundary]]
- [[L5_4_Softmax_Function|Softmax Function]] (multi-class generalization)
- [[L4_3_Sigmoid_Function|Sigmoid in Linear Classifiers]]

## Historical Context
The sigmoid function was first introduced in the context of logistic regression by statistician Joseph Berkson in 1944, though the logistic function itself was described much earlier by Pierre François Verhulst in 1838 as a model of population growth.

## References
1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. Chapter 4.3.2.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer. Chapter 4.4. 