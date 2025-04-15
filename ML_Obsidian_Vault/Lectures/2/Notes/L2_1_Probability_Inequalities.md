# Probability Inequalities

Probability inequalities provide bounds on probabilities and expectations, which are crucial for theoretical analysis and practical applications in machine learning.

## Basic Inequalities

### Markov's Inequality
- **Statement**: For any non-negative random variable X and a > 0,
  P(X ≥ a) ≤ E[X]/a
- **Applications**: 
  - Bounding tail probabilities
  - Proving other inequalities
  - Analyzing algorithm performance

### Chebyshev's Inequality
- **Statement**: For any random variable X with finite mean μ and variance σ²,
  P(|X - μ| ≥ kσ) ≤ 1/k²
- **Applications**:
  - Bounding deviation from mean
  - Proving weak law of large numbers
  - Confidence interval construction

### Jensen's Inequality
- **Statement**: For a convex function φ and random variable X,
  φ(E[X]) ≤ E[φ(X)]
- **Applications**:
  - Proving other inequalities
  - Analyzing loss functions
  - Information theory

## Advanced Inequalities

### Hoeffding's Inequality
- **Statement**: For independent random variables X₁,...,Xₙ bounded by [a,b],
  P(|X̄ - E[X̄]| ≥ t) ≤ 2e^(-2nt²/(b-a)²)
- **Applications**:
  - Concentration of measure
  - Statistical learning theory
  - Confidence intervals

### Bernstein's Inequality
- **Statement**: For independent random variables with bounded variance,
  P(|X̄ - E[X̄]| ≥ t) ≤ 2e^(-nt²/(2σ² + 2Mt/3))
- **Applications**:
  - Tighter bounds than Hoeffding
  - Variance-aware concentration

### McDiarmid's Inequality
- **Statement**: For function f with bounded differences,
  P(|f(X) - E[f(X)]| ≥ t) ≤ 2e^(-2t²/∑cᵢ²)
- **Applications**:
  - Analyzing algorithm stability
  - Generalization bounds
  - Concentration of measure

## Concentration Inequalities

### Chernoff Bounds
- **Statement**: For independent Bernoulli random variables,
  P(X ≥ (1+δ)μ) ≤ e^(-μδ²/3) for 0 < δ < 1
- **Applications**:
  - Tail bounds for sums
  - Analyzing randomized algorithms
  - Statistical testing

### Azuma's Inequality
- **Statement**: For martingales with bounded differences,
  P(|Xₙ - X₀| ≥ t) ≤ 2e^(-t²/(2∑cᵢ²))
- **Applications**:
  - Analyzing dependent random variables
  - Concentration of measure
  - Martingale theory

## Applications in Machine Learning

1. **Generalization Bounds**
   - Bounding generalization error
   - Analyzing model complexity
   - Understanding overfitting

2. **Algorithm Analysis**
   - Runtime analysis
   - Performance guarantees
   - Convergence rates

3. **Statistical Learning Theory**
   - VC dimension bounds
   - Rademacher complexity
   - PAC learning

4. **Confidence Intervals**
   - Constructing confidence intervals
   - Hypothesis testing
   - Statistical inference

## Practical Considerations

### Choosing Appropriate Bounds
- Consider distribution assumptions
- Evaluate tightness of bounds
- Balance between generality and precision

### Computational Aspects
- Numerical stability
- Implementation considerations
- Approximation methods

## Related Topics
- [[L2_1_Concentration_Inequalities|Concentration Inequalities]]: More specialized concentration results
- [[L2_1_Limit_Theorems|Limit Theorems]]: Asymptotic behavior and limits
- [[L2_1_Examples|Probability Examples]]: Practical applications of inequalities 