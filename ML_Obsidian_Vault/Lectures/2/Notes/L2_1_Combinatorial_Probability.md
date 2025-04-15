# Combinatorial Probability

Combinatorial probability deals with counting the number of possible outcomes in probability problems and calculating probabilities based on these counts.

## Fundamental Counting Principles

### Multiplication Principle
If there are $n_1$ ways to do the first task, $n_2$ ways to do the second task, and so on, then the total number of ways to perform all tasks is:
$$n_1 \times n_2 \times \cdots \times n_k$$

### Addition Principle
If two tasks are mutually exclusive (cannot be done simultaneously), and there are $n_1$ ways to do the first task and $n_2$ ways to do the second task, then the total number of ways to do either task is:
$$n_1 + n_2$$

## Permutations

### Permutations of Distinct Objects
The number of ways to arrange $n$ distinct objects is:
$$n! = n \times (n-1) \times \cdots \times 2 \times 1$$

### Permutations of a Subset
The number of ways to arrange $r$ objects from $n$ distinct objects is:
$$P(n,r) = \frac{n!}{(n-r)!}$$

### Permutations with Repetition
The number of distinct arrangements of $n$ objects where there are $n_1$ identical objects of type 1, $n_2$ identical objects of type 2, ..., and $n_k$ identical objects of type $k$ is:
$$\frac{n!}{n_1! n_2! \cdots n_k!}$$

## Combinations

### Combinations Without Repetition
The number of ways to choose $r$ objects from $n$ distinct objects without regard to order is:
$$C(n,r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

### Combinations With Repetition
The number of ways to choose $r$ objects from $n$ distinct objects where repetition is allowed and order doesn't matter is:
$$\binom{n+r-1}{r} = \frac{(n+r-1)!}{r!(n-1)!}$$

## Probability Applications

### Classical Probability
For equally likely outcomes, the probability of an event $A$ is:
$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

### Examples
1. **Dice Problems**: Probability of getting a sum of 7 when rolling two dice
2. **Card Problems**: Probability of getting a flush in poker
3. **Birthday Problem**: Probability that at least two people share the same birthday
4. **Urn Problems**: Probability of drawing certain colored balls from an urn

## Advanced Topics

### Multinomial Coefficients
The number of ways to divide $n$ distinct objects into $k$ groups of sizes $n_1, n_2, \ldots, n_k$ is:
$$\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! n_2! \cdots n_k!}$$

### Stirling Numbers
- **First Kind**: Number of ways to arrange $n$ objects into $k$ cycles
- **Second Kind**: Number of ways to partition $n$ objects into $k$ non-empty subsets

### Catalan Numbers
The number of valid sequences of $n$ pairs of parentheses is:
$$C_n = \frac{1}{n+1}\binom{2n}{n}$$

## Applications in Machine Learning

1. **Feature Selection**: Counting possible feature combinations
2. **Model Selection**: Enumerating possible model configurations
3. **Combinatorial Optimization**: Solving problems with discrete solution spaces
4. **Graph Theory**: Counting paths and cycles in graphs
5. **Natural Language Processing**: Counting word combinations and sequences

## Related Topics
- [[L2_1_Basic_Probability|Basic Probability]]: Foundation of probability theory
- [[L2_1_Discrete_Distributions|Discrete Distributions]]: Probability distributions for discrete outcomes
- [[L2_1_Examples|Probability Examples]]: Practical applications of probability 