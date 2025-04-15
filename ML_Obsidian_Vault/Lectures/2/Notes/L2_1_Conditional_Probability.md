# Conditional Probability

## Overview
Conditional probability is a measure of the probability of an event occurring given that another event has already occurred. It is a foundational concept in probability theory and has significant applications in machine learning, particularly in Bayesian methods, graphical models, and classification problems.

## Conditional Probability Formula
The conditional probability of event $A$ given event $B$ is defined as:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

where:
- $P(A|B)$ is the probability of event $A$ occurring given that event $B$ has occurred
- $P(A \cap B)$ is the probability of both events $A$ and $B$ occurring
- $P(B)$ is the probability of event $B$ occurring, with $P(B) > 0$

### Properties
- **Non-negativity**: $P(A|B) \geq 0$
- **Normalization**: $P(\Omega|B) = 1$ where $\Omega$ is the sample space
- **Additivity**: For mutually exclusive events $A_1$ and $A_2$:
  $$P(A_1 \cup A_2|B) = P(A_1|B) + P(A_2|B)$$
- **Multiplication Rule**: $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$

## Key Concepts

### Independence
Events $A$ and $B$ are independent if and only if:
$$P(A|B) = P(A)$$

This means that the occurrence of event $B$ does not affect the probability of event $A$.

### Conditional Independence
Events $A$ and $B$ are conditionally independent given event $C$ if:
$$P(A \cap B|C) = P(A|C)P(B|C)$$

### Bayes' Theorem
Bayes' theorem allows us to compute the conditional probability of an event $A$ given an event $B$ based on the conditional probability of $B$ given $A$:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

where:
- $P(A)$ is the prior probability of $A$
- $P(B|A)$ is the likelihood of $B$ given $A$
- $P(A|B)$ is the posterior probability of $A$ given $B$
- $P(B)$ is the marginal probability of $B$

### Law of Total Probability
For a partition $\{B_1, B_2, \ldots, B_n\}$ of the sample space:
$$P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)$$

### Chain Rule
The chain rule (or multiplication rule) can be derived from the definition of conditional probability:

$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdot \ldots \cdot P(A_n|A_1 \cap A_2 \cap \ldots \cap A_{n-1})$$

This is particularly useful when modeling sequences of events.

## Conditional Probability for Random Variables

### Discrete Random Variables
For discrete random variables $X$ and $Y$:
$$P(X = x|Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}$$

### Continuous Random Variables
For continuous random variables $X$ and $Y$:
$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

where $f_{X|Y}$ is the conditional PDF and $f_Y$ is the marginal PDF of $Y$.

### Conditional Expectation
The conditional expectation of $X$ given $Y = y$ is:
$$E[X|Y = y] = \begin{cases}
\sum_x x P(X = x|Y = y) & \text{(discrete case)} \\
\int x f_{X|Y}(x|y) dx & \text{(continuous case)}
\end{cases}$$

### Law of Total Expectation
$$E[X] = E[E[X|Y]]$$

### Conditional Variance
$$\text{Var}(X|Y) = E[(X - E[X|Y])^2|Y]$$

### Law of Total Variance
$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$

## Applications in Machine Learning

### Naive Bayes Classifiers
Naive Bayes classifiers apply Bayes' theorem with an assumption of conditional independence between features given the class. For a classification problem with features $\mathbf{X} = (X_1, X_2, \ldots, X_n)$ and class variable $Y$:

$$P(Y=y|\mathbf{X}) \propto P(Y=y) \prod_{i=1}^{n} P(X_i|Y=y)$$

### Probabilistic Graphical Models
In models like Bayesian networks, conditional probability distributions define relationships between random variables, allowing for efficient representation of joint distributions.

### Diagnostic Reasoning
In diagnostic applications (e.g., medical diagnosis), we often calculate:
- **Sensitivity**: $P(\text{positive test}|\text{disease})$
- **Specificity**: $P(\text{negative test}|\text{no disease})$
- **Positive Predictive Value**: $P(\text{disease}|\text{positive test})$
- **Negative Predictive Value**: $P(\text{no disease}|\text{negative test})$
- **Likelihood Ratio**: $\frac{P(\text{test result}|\text{disease})}{P(\text{test result}|\text{no disease})}$

### Hidden Markov Models
In HMMs, the probability of a sequence of observations $\mathbf{X}$ given hidden states $\mathbf{Z}$ is:
$$P(\mathbf{X}|\mathbf{Z}) = \prod_{t=1}^T P(X_t|Z_t)$$

### Markov Chains
For a Markov chain, the probability of the next state depends only on the current state:
$$P(X_{t+1}|X_1, \ldots, X_t) = P(X_{t+1}|X_t)$$

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Fundamental concepts of probability theory
- [[L2_1_Joint_Probability|Joint Probability]]: Working with multiple events
- [[L2_1_Independence|Independence]]: When events don't influence each other
- [[L2_1_Bayesian_Inference|Bayesian Inference]]: Using conditional probability for inference

## Examples
For practical examples and applications of conditional probability, see:
- [[L2_1_Conditional_Probability_Examples|Conditional Probability Examples]]: Medical diagnosis, spam filtering, Naive Bayes classification, and more
- [[L2_1_Bayesian_Networks_Examples|Bayesian Networks Examples]]: Working with conditional dependencies
- [[L2_1_HMM_Examples|Hidden Markov Model Examples]]: Sequence modeling applications 