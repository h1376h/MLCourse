# Lecture 5.2: Hard Margin and Soft Margin SVMs Quiz

## Overview
This quiz contains 4 questions covering different topics from section 5.2 of the lectures on Hard Margin SVM, Soft Margin SVM, Slack Variables, and Regularization Trade-offs.

## Question 1

### Problem Statement
Consider the limitations of hard margin SVMs and the motivation for soft margin SVMs.

#### Task
1. Explain what "hard margin" means in the context of SVMs
2. What are the main limitations of hard margin SVMs in practical applications?
3. Give an example of a dataset where hard margin SVM would fail and explain why
4. How does the soft margin approach address these limitations?

For a detailed explanation of this problem, see [Question 1: Hard vs Soft Margin](L5_2_1_explanation.md).

## Question 2

### Problem Statement
Consider the mathematical formulation of soft margin SVMs with slack variables.

#### Task
1. Write the primal optimization problem for soft margin SVM including slack variables
2. Explain the role of slack variables $\xi_i$ in the formulation
3. What does the constraint $\xi_i \geq 0$ ensure?
4. Interpret the meaning of different values of $\xi_i$:
   - $\xi_i = 0$
   - $0 < \xi_i < 1$
   - $\xi_i = 1$
   - $\xi_i > 1$

For a detailed explanation of this problem, see [Question 2: Slack Variables and Mathematical Formulation](L5_2_2_explanation.md).

## Question 3

### Problem Statement
Consider the regularization parameter C in soft margin SVMs and its effect on the bias-variance tradeoff.

#### Task
1. Explain the role of the parameter C in the soft margin SVM objective function
2. What happens to the model when:
   - C approaches infinity
   - C approaches zero
3. How does C control the bias-variance tradeoff in SVMs?
4. Describe a systematic approach to select the optimal value of C
5. What is the relationship between C and the number of support vectors?

For a detailed explanation of this problem, see [Question 3: Regularization Parameter C](L5_2_3_explanation.md).

## Question 4

### Problem Statement
Consider the hinge loss function and its relationship to the SVM optimization problem.

#### Task
1. Write the mathematical expression for the hinge loss function
2. Plot the hinge loss function and compare it with:
   - 0-1 loss (classification error)
   - Logistic loss
   - Squared loss
3. Explain why the hinge loss is well-suited for SVMs
4. How does the hinge loss relate to the soft margin formulation with slack variables?
5. What are the advantages and disadvantages of hinge loss compared to other loss functions?

For a detailed explanation of this problem, see [Question 4: Hinge Loss Function](L5_2_4_explanation.md).
