# Lecture 2.2: Information Theory and Entropy Quiz

## Overview
This quiz contains 4 questions from different topics covered in section 2.2 of the lectures on Information Theory and Entropy.

## Question 1

### Problem Statement
Consider a discrete random variable X with the following probability distribution:
- P(X = 1) = 0.2
- P(X = 2) = 0.3
- P(X = 3) = 0.4
- P(X = 4) = 0.1

#### Task
1. Calculate the entropy H(X) of this distribution
2. What would be the entropy if the distribution were uniform over these four values?
3. If we have another random variable Y with the same possible values but is deterministic (i.e., one outcome has probability 1), what is its entropy?
4. Explain why the uniform distribution has maximum entropy among all distributions over the same set of values

## Question 2

### Problem Statement
Consider two distributions P and Q over the same discrete random variable X:
- P: P(X = 0) = 0.7, P(X = 1) = 0.3
- Q: Q(X = 0) = 0.5, Q(X = 1) = 0.5

#### Task
1. Calculate the KL divergence D_KL(P||Q)
2. Calculate the KL divergence D_KL(Q||P)
3. Calculate the cross-entropy H(P, Q)
4. Explain why D_KL(P||Q) ≠ D_KL(Q||P) and what this means in practice

## Question 3

### Problem Statement
Consider two discrete random variables X and Y with the following joint probability distribution:

|       | Y = 0 | Y = 1 |
|-------|-------|-------|
| X = 0 | 0.3   | 0.2   |
| X = 1 | 0.1   | 0.4   |

#### Task
1. Calculate the entropy of X, H(X)
2. Calculate the entropy of Y, H(Y)
3. Calculate the joint entropy H(X, Y)
4. Calculate the mutual information I(X; Y) and interpret what it means about the relationship between X and Y

## Question 4

### Problem Statement
In a classification problem, we have a binary classifier that outputs the probability of the positive class. The true labels are y and the predicted probabilities are ŷ.

#### Task
1. Write down the formula for the cross-entropy loss between the true labels and predicted probabilities
2. If we have 4 samples with true labels y = [1, 0, 1, 0] and predicted probabilities ŷ = [0.8, 0.3, 0.6, 0.2], calculate the cross-entropy loss
3. Calculate the KL divergence between the true distribution and predicted distribution
4. Explain how minimizing the cross-entropy loss relates to maximum likelihood estimation 