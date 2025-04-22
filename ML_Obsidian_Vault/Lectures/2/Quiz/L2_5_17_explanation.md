# Question 17: Naive Bayes Parameter Estimation

### Problem Statement
We have a training set consisting of samples and their labels. All samples come from one of two classes, 0 and 1. Samples are two dimensional vectors. The input data is the form $\{X1, X2, Y\}$ where $X1$ and $X2$ are the two values for the input vector and $Y$ is the label for this sample.

After learning the parameters of a Naive Bayes classifier we arrived at the following table:

Table 1: Naive Bayes conditional probabilities

| | $Y = 0$ | $Y = 1$ |
|:---:|:---:|:---:|
| $X1$ | $P(X1 = 1\|Y = 0) = 1/5$ | $P(X1 = 1\|Y = 1) = 3/8$ |
| $X2$ | $P(X2 = 1\|Y = 0) = 1/3$ | $P(X2 = 1\|Y = 1) = 3/4$ |

#### Task
Denote by $w_1$ the probability of class 1 (that is $w_1 = P(Y = 1)$). If we know that the likelihood of the following two samples: $\{1,0,1\},\{0,1,0\}$ given our Naive Bayes model is $1/180$, what is the value of $w_1$? You do not need to derive an explicit value for $w_1$. It is enough to write a (correct...) equation that has $w_1$ as the only unknown and that when solved would provide the value of $w_1$. Simplify as best as you can.

## Understanding the Problem

We have a Naive Bayes classifier with the following conditional probabilities:
- $P(X1=1|Y=0) = 1/5$
- $P(X1=1|Y=1) = 3/8$
- $P(X2=1|Y=0) = 1/3$
- $P(X2=1|Y=1) = 3/4$

We need to compute the likelihood of samples {1,0,1} and {0,1,0} and use the constraint that this likelihood equals 1/180 to find $w_1 = P(Y=1)$.

## Naive Bayes Model

Naive Bayes assumes that features X1 and X2 are conditionally independent given the class Y.
This means: $P(X1,X2|Y) = P(X1|Y) \times P(X2|Y)$

The complete model is:
$P(X1,X2,Y) = P(Y) \times P(X1|Y) \times P(X2|Y)$

### Visual Explanation: Naive Bayes Graphical Model
The Naive Bayes model assumes that features X1 and X2 are conditionally independent given the class Y.
In a graphical model, Y would directly influence both X1 and X2, but there's no direct connection between X1 and X2.
Y → X1
Y → X2

## Solution

### Step 1: Calculating Probabilities for the Given Samples

For sample {1,0,1}, we have X1=1, X2=0, Y=1:
$P(X1=1,X2=0,Y=1) = P(Y=1) \times P(X1=1|Y=1) \times P(X2=0|Y=1)$
$= w_1 \times 0.375 \times (1 - 0.75)$
$= w_1 \times 0.375 \times 0.25$
$= w_1 \times 0.09375$

For sample {0,1,0}, we have X1=0, X2=1, Y=0:
$P(X1=0,X2=1,Y=0) = P(Y=0) \times P(X1=0|Y=0) \times P(X2=1|Y=0)$
$= (1 - w_1) \times (1 - 0.2) \times 0.3333333333333333$
$= (1 - w_1) \times 0.8 \times 0.3333333333333333$
$= (1 - w_1) \times 0.26666666666666666$

Numerically:
- $P(X1=1,X2=0,Y=1) = w_1 \times 0.09375$
- $P(X1=0,X2=1,Y=0) = (1 - w_1) \times 0.26666666666666666$

### Step 2: Setting up the Equation

We're told that the likelihood of the two samples is 1/180.
This means: $P(X1=1,X2=0,Y=1) \times P(X1=0,X2=1,Y=0) = 1/180$

Substituting our expressions:
$[w_1 \times 0.09375] \times [(1 - w_1) \times 0.26666666666666666] = 1/180$

This gives us the equation:
$0.025 \times w_1 \times (1 - w_1) = 0.00555555555555556$

Expanded equation:
$-0.025 \times w_1^2 + 0.025 \times w_1 = 0.00555555555555556$

### Step 3: Solving for $w_1$

Rearranging into standard form: $ax^2 + bx + c = 0$

$-0.025 \times w_1^2 + 0.025 \times w_1 - 0.00555555555555556 = 0$

Coefficients:
- $a = -0.0250000000000000$
- $b = 0.0250000000000000$
- $c = -0.00555555555555556$

Solutions: $[0.333333333333333, 0.666666666666667]$

The value of $w_1 = P(Y=1)$ is $0.3333333333333333$, which can be expressed as the fraction $\frac{1}{3}$.

### Step 4: Verification

Likelihood with $w_1 = 0.3333333333333333$: $0.005555555555555556$
Expected likelihood: $0.005555555555555556$
Difference: $0.0$

Verification successful!

### Likelihood Function Explanation

The likelihood function plotted against different values of $w_1$ would show a parabolic shape with two roots. Only one ($w_1 = \frac{1}{3}$) is our solution because:
1. The parabolic shape is characteristic of the product of two linear functions in $w_1$ and $(1-w_1)$
2. This is exactly what we get in the Naive Bayes model when multiplying the probabilities of two different samples

## Key Insights

### Theoretical Foundations
- **Naive Bayes Assumption**: Features are conditionally independent given the class. This simplifies the joint probability calculations significantly.
- **Prior Probability**: The parameter $w_1 = P(Y=1)$ represents our belief about the class distribution before seeing any features.
- **Likelihood Constraint**: When given a constraint on the joint probability of multiple samples, this leads to an equation that can be solved for the unknown parameter.

### Mathematical Techniques
- **Quadratic Equation**: The product of probabilities for two different classes (involving $w_1$ and $(1-w_1)$) leads to a quadratic equation.
- **Solution Validation**: When solving quadratic equations in the context of probabilities, we must ensure that the solution lies in the range [0,1] and verify it matches our constraints.

### Practical Applications
- **Parameter Estimation**: In practice, we often estimate the prior probabilities from training data frequencies, but this problem demonstrates how to determine them from other constraints.
- **Model Completeness**: A Naive Bayes model requires both the conditional probabilities $P(X|Y)$ and the prior probabilities $P(Y)$ to make predictions.

## Conclusion

We determined that the prior probability of class 1 is $w_1 = P(Y=1) = \frac{1}{3}$. This was obtained by setting up joint probabilities for the given samples using the Naive Bayes model, then solving a quadratic equation based on the constraint that the likelihood equals 1/180.

The solution process demonstrates how the Naive Bayes assumption of conditional independence allows us to factorize complex joint probabilities into simpler terms, making the model both computationally efficient and mathematically tractable.

The fraction $\frac{1}{3}$ represents the prior probability of class 1, meaning that before considering any features, we believe there's a one-third chance that a randomly selected sample belongs to class 1. 