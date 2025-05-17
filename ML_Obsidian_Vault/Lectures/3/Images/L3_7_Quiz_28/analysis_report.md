# Question 28: Analysis Results

## 1. Best Model for Each Regularization Level

| Regularization (λ) | Best Model | Test Error |
|:------------------:|:----------:|:----------:|
| 0.0001 | Cubic Model | 0.074577 |
| 0.1 | Cubic Model | 0.077704 |
| 10 | Degree 10 Polynomial | 0.393536 |
| 1000 | Degree 10 Polynomial | 1.665080 |

## 2. Bias-Variance Tradeoff for Degree 10 Polynomial

| Regularization (λ) | Training Error | Test Error | Difference (Approx. Variance) |
|:------------------:|:--------------:|:----------:|:-----------------------------:|
| 0.0001 | 0.102009 | 0.153439 | 0.051430 |
| 0.1 | 0.102710 | 0.134788 | 0.032078 |
| 10 | 0.241666 | 0.393536 | 0.151869 |
| 1000 | 1.306516 | 1.665080 | 0.358565 |

## 3. Effect of Regularization on Different Models

| Model | λ=0.0001 Test Error | λ=1000 Test Error | Relative Change |
|:-----:|:-------------------:|:------------------:|:---------------:|
| Linear Model | 7.259539 | 36.081896 | 397.03% |
| Cubic Model | 0.074577 | 5.314785 | 7026.59% |
| Degree 10 Polynomial | 0.153439 | 1.665080 | 985.17% |

## 4. Optimal Regularization for Cubic Model

| Regularization (λ) | Test Error |
|:------------------:|:----------:|
| 0.0001 | 0.074577 |
| 0.1 | 0.077704 |
| 10 | 0.476300 |
| 1000 | 5.314785 |

Optimal λ for cubic model: **0.0001**

## 5. Degree 10 Polynomial with Low Regularization

Training Error: 0.102009
Test Error: 0.153439
Ratio (Test/Train): 1.50x higher

## 6. Best Overall Combination

Best overall model: **Cubic Model** with λ = **0.0001**
Test Error: 0.074577
