"""
Conjugate Priors Examples

This script demonstrates various conjugate prior relationships in Bayesian inference.
It provides examples of Beta-Binomial, Gamma-Poisson, and Normal-Normal conjugacy.
"""

import beta_binomial_conjugacy
import gamma_poisson_conjugacy
import normal_normal_conjugacy
import sequential_updating

if __name__ == "__main__":
    print("\n==== Beta-Binomial Conjugacy Example ====\n")
    beta_binomial_conjugacy.beta_binomial_example()
    
    print("\n==== Gamma-Poisson Conjugacy Example ====\n")
    gamma_poisson_conjugacy.gamma_poisson_example()
    
    print("\n==== Normal-Normal Conjugacy Example ====\n")
    normal_normal_conjugacy.normal_normal_example()
    
    print("\n==== Sequential Updating Example ====\n")
    sequential_updating.sequential_updating_example() 