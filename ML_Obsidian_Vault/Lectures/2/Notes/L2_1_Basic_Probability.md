# Basic Probability

Basic probability provides the foundation for quantifying uncertainty and making predictions in machine learning.

## Probability Space
- **Sample Space ($\Omega$)**: The set of all possible outcomes of a random experiment
- **Events**: Subsets of the sample space representing specific outcomes or collections of outcomes
- **Probability Measure ($P$)**: A function that assigns a value between 0 and 1 to events

## Axioms of Probability (Kolmogorov's Axioms)
1. **Non-negativity**: $P(A) \geq 0$ for any event $A$
2. **Normalization**: $P(\Omega) = 1$ (the probability of the entire sample space is 1)
3. **Additivity**: For mutually exclusive (disjoint) events $A$ and $B$, $P(A \cup B) = P(A) + P(B)$

## Basic Properties
- **Complement Rule**: $P(A^c) = 1 - P(A)$
- **Inclusion-Exclusion Principle**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Monotonicity**: If $A \subseteq B$, then $P(A) \leq P(B)$
- **Probability of Empty Set**: $P(\emptyset) = 0$
- **Probability of Finite Union**: 
  $$P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i=1}^n P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \cdots + (-1)^{n+1} P\left(\bigcap_{i=1}^n A_i\right)$$

## Random Variables
- **Definition**: A function $X: \Omega \rightarrow \mathbb{R}$ that maps outcomes from the sample space to real numbers
- **Discrete Random Variables**: Take on countable values (e.g., number of heads in coin flips)
- **Continuous Random Variables**: Take on uncountable values in an interval (e.g., height, weight)
- **Mixed Random Variables**: Have both discrete and continuous components

## Related Topics
- [[L2_1_PMF_PDF_CDF|PMF, PDF, and CDF]]: Understanding probability distributions and their functions
- [[L2_1_Joint_Probability|Joint Probability]]: Working with multiple random variables
- [[L2_1_Conditional_Probability|Conditional Probability]]: Probability of events given other events
- [[L2_1_Independence|Independence]]: When events don't influence each other
- [[L2_1_Probability_Distributions|Probability Distributions]]: Common probability distribution families
- [[L2_1_Examples|Probability Examples]]: Practical applications and examples 