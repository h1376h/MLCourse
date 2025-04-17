# Question 21: Probability Multiple Choice

## Problem Statement
For each question, select the single best answer.

## Task
1. Which of the following random variables is most likely to be modeled using a Poisson distribution?
   A) The number of customers arriving at a store in one hour
   B) The height of adult males in a population
   C) The time until a radioactive particle decays
   D) The proportion of defective items in a manufacturing batch

2. The property of a statistical estimator that states its expected value equals the true parameter value is called:
   A) Consistency
   B) Efficiency
   C) Unbiasedness
   D) Sufficiency

3. When calculating the probability of a union of three events $P(A \cup B \cup C)$, which of the following must be subtracted to avoid double-counting?
   A) $P(A \cap B)$, $P(B \cap C)$, and $P(A \cap C)$
   B) Only $P(A \cap B \cap C)$
   C) $P(A \cap B)$, $P(B \cap C)$, $P(A \cap C)$, and then add $P(A \cap B \cap C)$
   D) The expected value of all three events

## Solution

### Question 21.1: Poisson Distribution Applications

The correct answer is **A) The number of customers arriving at a store in one hour**.

#### Key properties of the Poisson distribution:
- Models the **number of events** occurring in a fixed interval
- Events occur independently of each other
- Events occur at a constant average rate
- The probability of two events occurring at exactly the same time is zero

#### Analysis of each option:

**A) The number of customers arriving at a store in one hour:**
- Customer arrivals occur randomly and independently
- We're counting the number of events (arrivals) in a fixed interval (one hour)
- There's typically a constant average rate of arrivals
- This is a classic example of the Poisson distribution

![[q21_1_distributions.png|Distribution Models]]

**B) The height of adult males in a population:**
- Heights are continuous measurements, not counts of events
- Heights follow a normal (Gaussian) distribution
- The normal distribution is symmetric and bell-shaped
- Characterized by mean $\mu$ and standard deviation $\sigma$

**C) The time until a radioactive particle decays:**
- This measures waiting time until an event occurs, not count of events
- Follows an exponential distribution, not Poisson
- The exponential distribution models the time between events in a Poisson process
- If events follow $\text{Poisson}(\lambda)$, time between events follows $\text{Exponential}(\lambda)$

**D) The proportion of defective items in a manufacturing batch:**
- This represents a fixed number of trials (batch size) with success/failure outcomes
- Follows a binomial distribution with parameters $n$ (batch size) and $p$ (defect probability)
- Not appropriate for Poisson, which models rare events over continuous intervals

#### Mathematical formulation:
For a Poisson random variable $X$ with rate parameter $\lambda$ (average number of events per interval):

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad \text{for } k = 0, 1, 2, \ldots$$

- Mean: $E[X] = \lambda$
- Variance: $\text{Var}(X) = \lambda$

![[q21_1_poisson_comparison.png|Poisson distributions with different parameters]]

### Question 21.2: Properties of Statistical Estimators

The correct answer is **C) Unbiasedness**.

#### Properties of statistical estimators:

**1. Unbiasedness: $E[\hat{\theta}] = \theta$**
- An estimator $\hat{\theta}$ is unbiased if its expected value equals the true parameter value $\theta$
- Mathematically: $E[\hat{\theta}] = \theta$
- Example: Sample mean is an unbiased estimator of population mean
- The estimator's average value over many samples equals the parameter we're trying to estimate

![[q21_2_unbiasedness.png|Unbiased vs. Biased Estimators]]

**2. Consistency: Estimator converges to true value as sample size increases**
- A consistent estimator gets closer to the true parameter value as sample size grows
- Mathematically: $\hat{\theta} \to \theta$ as $n \to \infty$
- Example: The sample variance is a consistent estimator of population variance

**3. Efficiency: Has minimum variance among all unbiased estimators**
- An efficient estimator has the smallest variance among all unbiased estimators
- Less dispersion means more precise estimates
- Example: Sample mean is the most efficient estimator for the population mean in normal distributions

**4. Sufficiency: Captures all information about the parameter in the sample**
- A sufficient statistic contains all the information in the sample about the parameter
- No other statistic calculated from the same sample can add information about the parameter
- Example: Sample mean is sufficient for the population mean in normal distributions with known variance

#### Mathematical foundation:
From our simulation results:
- For unbiased sample mean: Expected values approximately 9.99 (true param = 10)
- For biased sample maximum: Expected values approximately 14.85 (true param = 10)

This demonstrates that the sample mean is unbiased ($E[\bar{X}] = \mu$), while the sample maximum is biased ($E[X_{max}] > \mu$).

![[q21_2_estimator_properties.png|Statistical Estimator Properties]]

### Question 21.3: Inclusion-Exclusion Principle

The correct answer is **C) $P(A \cap B)$, $P(B \cap C)$, $P(A \cap C)$, and then add $P(A \cap B \cap C)$**.

#### The Inclusion-Exclusion Principle for Three Events:

$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

![[q21_3_sets.png|Sets visualization]]

#### Step-by-step explanation:

**1. We start by adding the individual probabilities: $P(A) + P(B) + P(C)$**
- This counts every element that's in at least one of the sets
- But elements in intersections get counted multiple times
- Elements in exactly two sets are counted twice
- Elements in all three sets are counted three times

**2. To correct for double-counting, we subtract the pairwise intersections:**
- Subtract $P(A \cap B)$ to remove double-counting of elements in both A and B
- Subtract $P(A \cap C)$ to remove double-counting of elements in both A and C
- Subtract $P(B \cap C)$ to remove double-counting of elements in both B and C

**3. But now we have a new problem with the triple intersection:**
- Elements in $A \cap B \cap C$ were initially counted 3 times (in steps A, B, and C)
- Then they were subtracted 3 times (in steps $A \cap B$, $A \cap C$, and $B \cap C$)
- So they've effectively been completely removed, when they should be counted once
- To fix this, we need to add back $P(A \cap B \cap C)$

![[q21_3_inclusion_exclusion_calculation.png|Inclusion-Exclusion Calculation]]

#### Numerical example:
If:
- $P(A) = 0.5$
- $P(B) = 0.4$
- $P(C) = 0.3$
- $P(A \cap B) = 0.2$
- $P(A \cap C) = 0.15$
- $P(B \cap C) = 0.1$
- $P(A \cap B \cap C) = 0.05$

Then:
$$P(A \cup B \cup C) = 0.5 + 0.4 + 0.3 - 0.2 - 0.15 - 0.1 + 0.05 = 0.8$$

#### Answer analysis:
- **A)** Only subtracting $P(A \cap B)$ would account for just one pairwise intersection, leaving other overlaps double-counted
- **B)** Only subtracting $P(A \cap B \cap C)$ would not address double-counting in any of the pairwise intersections
- **C)** Subtracting all pairwise intersections and then adding back the triple intersection correctly accounts for all cases
- **D)** The expected value is not relevant to the inclusion-exclusion principle, which deals with set operations

## Summary of Answers
1. A) The number of customers arriving at a store in one hour
2. C) Unbiasedness  
3. C) $P(A \cap B)$, $P(B \cap C)$, $P(A \cap C)$, and then add $P(A \cap B \cap C)$ 