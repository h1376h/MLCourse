# Lecture 3.4: Multiple Linear Regression Quiz

## Overview
This quiz contains 20 questions covering different topics from section 3.4 of the lectures on Multiple Linear Regression. These questions test your understanding of extending linear regression to multiple variables, matrix formulations, feature engineering, categorical variables, and nonlinear relationships.

## Question 1

### Problem Statement
Consider a multiple linear regression model with 3 features and 4 observations:

| $x_1$ | $x_2$ | $x_3$ | $y$ |
|-------|-------|-------|-----|
| 2     | 5     | 1     | 12  |
| 3     | 2     | 0     | 7   |
| 1     | 4     | 2     | 11  |
| 4     | 3     | 1     | 13  |

The multiple linear regression model is given by:
$$y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + \epsilon$$

#### Task
1. Write down the design matrix $\boldsymbol{X}$ and the target vector $\boldsymbol{y}$ for this dataset
2. Express the multiple linear regression model in matrix form using $\boldsymbol{X}$, $\boldsymbol{w}$, and $\boldsymbol{y}$
3. Write the normal equation for finding the optimal weights $\boldsymbol{w}$
4. Without calculating the actual values, describe the dimensions of $\boldsymbol{X}^T\boldsymbol{X}$ and $\boldsymbol{X}^T\boldsymbol{y}$ in this example

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Matrix Formulation of Multiple Linear Regression](L3_4_1_explanation.md).

## Question 2

### Problem Statement
You are building a multiple linear regression model to predict housing prices and have collected the following features:
- $x_1$: Size of the house (in square meters)
- $x_2$: Number of bedrooms
- $x_3$: Size of the house (in square feet)
- $x_4$: Number of bathrooms
- $x_5$: Year built

Note that 1 square meter equals approximately 10.764 square feet.

#### Task
1. [🔍] Identify which features in this dataset might cause multicollinearity and explain why
2. Describe two methods to detect multicollinearity in a dataset
3. Propose two approaches to address the multicollinearity in this dataset
4. Explain what would happen to the coefficient estimates and their standard errors if you were to ignore the multicollinearity

For a detailed explanation of this problem, including methods for detecting and addressing multicollinearity, see [Question 2: Multicollinearity in Housing Price Prediction](L3_4_2_explanation.md).

## Question 3

### Problem Statement
You are analyzing the factors that influence crop yield. Your dataset includes the following variables:
- $x_1$: Amount of fertilizer (kg/hectare)
- $x_2$: Amount of water (liters/day)
- $x_3$: Average daily temperature (°C)
- $y$: Crop yield (tons/hectare)

Initial analysis suggests that:
1. More fertilizer generally increases yield, but the effect depends on water amount
2. Higher temperatures improve yield up to a point, after which they become harmful
3. The effect of water on yield diminishes as more water is added

#### Task
1. Propose a multiple regression model that includes appropriate interaction terms between fertilizer and water to capture their joint effect
2. Suggest a feature transformation for temperature to model the diminishing returns and eventual negative impact
3. Propose a feature transformation for water to capture the diminishing returns effect
4. Write the complete equation for your proposed model including all main effects, interaction terms, and transformed features

For a detailed explanation of this problem, including effective feature engineering for agricultural data, see [Question 3: Feature Engineering for Crop Yield Prediction](L3_4_3_explanation.md).

## Question 4

### Problem Statement
You are building a model to predict a car's fuel efficiency (mpg) based on the following data:

| Car Model | Engine Type | Transmission | Weight (kg) | MPG |
|-----------|-------------|--------------|-------------|-----|
| Model A   | Hybrid      | Automatic    | 1200        | 45  |
| Model B   | Gasoline    | Manual       | 1500        | 32  |
| Model C   | Diesel      | Automatic    | 1800        | 28  |
| Model D   | Gasoline    | Automatic    | 1400        | 30  |
| Model E   | Hybrid      | Manual       | 1300        | 42  |
| Model F   | Diesel      | Manual       | 1700        | 30  |

You want to use multiple linear regression with weight as a continuous predictor and engine type and transmission as categorical predictors.

#### Task
1. Create appropriate dummy variables for the categorical predictors in this dataset
2. Write down the design matrix $\boldsymbol{X}$ using these dummy variables and the weight feature
3. Write the full regression equation with coefficients for all variables (using general coefficient symbols like $w_0$, $w_1$, etc.)
4. Explain how to interpret the coefficient of one of your dummy variables in this model

For a detailed explanation of this problem, including proper encoding of categorical variables, see [Question 4: Dummy Variables for Car Efficiency Prediction](L3_4_4_explanation.md).

## Question 5

### Problem Statement
You are studying the relationship between a student's hours of study and their exam score. You have collected data from 6 students:

| Hours of Study ($x$) | Exam Score ($y$) |
|----------------------|------------------|
| 1                    | 45               |
| 2                    | 50               |
| 3                    | 60               |
| 4                    | 65               |
| 5                    | 68               |
| 6                    | 70               |

Initial analysis suggests that the relationship is not strictly linear, but follows a curve that flattens as study hours increase.

#### Task
1. Propose a polynomial regression model of degree 2 (quadratic) to fit this data
2. Write down the design matrix $\boldsymbol{X}$ for this polynomial regression
3. Express the model in both expanded form and matrix form
4. Explain how this polynomial model helps capture the diminishing returns effect of additional study hours

For a detailed explanation of this problem, including polynomial modeling of learning curves, see [Question 5: Polynomial Regression for Learning Curves](L3_4_5_explanation.md).

## Question 6

### Problem Statement
You are developing a model to predict temperature across a city based on data collected from several weather stations. You have data from 4 stations at different locations $(x_1, x_2)$ where $x_1$ and $x_2$ represent the coordinates on a map:

| Station | Location $(x_1, x_2)$ | Temperature (°C) |
|---------|------------------------|------------------|
| A       | (1, 1)                 | 22               |
| B       | (4, 2)                 | 25               |
| C       | (2, 5)                 | 20               |
| D       | (5, 5)                 | 23               |

You decide to use radial basis functions (RBFs) to model the temperature at any location in the city, using the known station locations as centers.

#### Task
1. Define a radial basis function and explain why it might be useful for this spatial prediction problem
2. Using a Gaussian RBF with $\sigma = 1.5$, calculate the value of the basis function $\phi_A(x)$ for the location $(3, 3)$ using station A as the center
3. Write the complete model equation using all four stations as centers for the RBFs
4. Explain the role of the parameter $\sigma$ in the Gaussian RBF and how changing it would affect your predictions

For a detailed explanation of this problem, including basis function calculations, see [Question 6: Radial Basis Functions for Spatial Modeling](L3_4_6_explanation.md).

## Question 7

### Problem Statement
You're working on a linear regression problem with an increasing number of features. The dataset has 100 training examples.

#### Task
1. Explain what the "curse of dimensionality" refers to in the context of multiple linear regression
2. Calculate how many parameters need to be estimated for the following models:
   a. A linear regression model with 10 features
   b. A linear regression model with 10 features and all possible 2-way interaction terms
   c. A polynomial regression model of degree 3 with 5 original features
3. Describe two practical problems that arise when the number of parameters approaches or exceeds the number of training examples
4. Suggest two methods to address these problems without collecting more data

For a detailed explanation of this problem, including challenges with high-dimensional data, see [Question 7: The Curse of Dimensionality in Regression](L3_4_7_explanation.md).

## Question 8

### Problem Statement
You are a data scientist for an e-commerce company. Your task is to design a multiple regression model to predict customer spending based on the following information:

- Customer age (years)
- Account age (months)
- Average monthly website visits
- Number of previous purchases
- Average product rating from customer
- Customer location (urban, suburban, rural)
- Customer gender (male, female, non-binary)
- Device type used for shopping (mobile, desktop, tablet)

The company wants to use this model to identify high-value customers and personalize marketing strategies.

#### Task
1. Identify which variables would be treated as continuous and which would require dummy variable encoding
2. For the categorical variables, explain how many dummy variables would be needed for each and why
3. Suggest at least two interaction terms that might be meaningful to include in the model and explain your reasoning
4. Describe two potential limitations or challenges with this multiple regression approach for predicting customer spending

For a detailed explanation of this problem, including effective feature engineering for e-commerce data, see [Question 8: Applied Multiple Regression for Customer Analytics](L3_4_8_explanation.md).

## Question 9

### Problem Statement
Consider the following regression model:
$$y = 5 + 2x_1 - 3x_2 + 4x_1x_2 + \epsilon$$

where $x_1x_2$ is an interaction term.

#### Task
1. What is the effect of increasing $x_1$ by one unit on $y$ when $x_2 = 0$?
2. What is the effect of increasing $x_1$ by one unit on $y$ when $x_2 = 2$?
3. Draw a simple sketch showing how the effect of $x_1$ on $y$ changes for different values of $x_2$
4. Explain in simple terms why interaction terms are important in regression models

For a detailed explanation of this problem, including visual interpretation of interaction effects, see [Question 9: Interpreting Interaction Terms](L3_4_9_explanation.md).

## Question 10

### Problem Statement
You have collected data on five houses:

| House | Size ($m^2$) | Age (years) | Location | Price ($1000) |
|-------|--------------|-------------|----------|---------------|
| 1     | 150          | 10          | Urban    | 300           |
| 2     | 100          | 15          | Rural    | 180           |
| 3     | 120          | 5           | Urban    | 350           |
| 4     | 200          | 20          | Rural    | 220           |
| 5     | 180          | 8           | Urban    | 380           |

#### Task
1. Create a scatter plot of house prices vs. size (by hand, roughly to scale)
2. Identify visually whether there appears to be a different price-size relationship for urban vs. rural houses
3. Write down a regression equation that includes size, location (as a dummy variable), and an interaction between size and location
4. Based on the data pattern, explain what you would expect the sign (positive or negative) of the interaction term coefficient to be

For a detailed explanation of this problem, including visual data analysis, see [Question 10: Visualizing Interactions in Housing Data](L3_4_10_explanation.md).

## Question 11

### Problem Statement
Consider a regression problem with a single feature $x$ where the true relationship is:
$$y = \sin(x) + \epsilon \quad \text{for } x \in [0, 2\pi]$$

You want to approximate this using polynomials of different degrees.

#### Task
1. Suggest a third-degree polynomial model to approximate this relationship
2. Write down the design matrix for this model if you have observations at $x = 0, \pi/2, \pi, 3\pi/2, 2\pi$
3. Sketch how you expect the following to perform in approximating a sine curve (by hand):
   a. Linear model (degree 1)
   b. Cubic model (degree 3)
   c. 9th degree polynomial
4. Explain why very high-degree polynomials might not be ideal despite their flexibility

For a detailed explanation of this problem, including polynomial approximation of nonlinear functions, see [Question 11: Polynomial Approximation of Sine Function](L3_4_11_explanation.md).

## Question 12

### Problem Statement
A researcher wants to analyze the effectiveness of three different teaching methods (A, B, and C) on student performance. They collected test scores from 15 students, 5 for each method. They also recorded the number of hours each student studied.

| Student | Teaching Method | Study Hours | Test Score |
|---------|----------------|-------------|------------|
| 1       | A              | 2           | 75         |
| 2       | A              | 3           | 82         |
| 3       | A              | 1           | 68         |
| 4       | A              | 4           | 90         |
| 5       | A              | 2           | 78         |
| 6       | B              | 2           | 80         |
| 7       | B              | 3           | 85         |
| 8       | B              | 1           | 72         |
| 9       | B              | 4           | 94         |
| 10      | B              | 2           | 82         |
| 11      | C              | 2           | 70         |
| 12      | C              | 3           | 75         |
| 13      | C              | 1           | 65         |
| 14      | C              | 4           | 84         |
| 15      | C              | 2           | 71         |

#### Task
1. Create the necessary dummy variables to represent the teaching methods
2. Write a multiple regression equation that predicts test scores based on teaching method and study hours
3. If method A is the reference category, what would a positive coefficient for the method B dummy variable indicate?
4. Would it be useful to include an interaction between study hours and teaching method? Explain why or why not.

For a detailed explanation of this problem, including analysis of educational data, see [Question 12: Educational Methods Regression Analysis](L3_4_12_explanation.md).

## Question 13

### Problem Statement
Evaluate whether each of the following statements is TRUE or FALSE. Justify your answer with a brief explanation.

#### Task
1. [🔍] In a multiple linear regression model, if features $x_1$ and $x_2$ are perfectly correlated (correlation coefficient = 1), then $(\boldsymbol{X}^T\boldsymbol{X})$ will be singular (non-invertible).
2. When encoding a categorical variable with $k$ categories using dummy variables, you always need exactly $k$ dummy variables.
3. [🔍] Adding a polynomial term (e.g., $x^2$) to a regression model always improves the model's fit to the training data.
4. [🔍] In multiple linear regression, the normal equation $\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$ provides the global minimum of the sum of squared errors cost function.
5. [🔍] If a predictor variable has no effect on the response, its coefficient in a multiple regression model will always be exactly zero.
6. [🔍] In a multiple regression model with interaction terms, the coefficient of a main effect (e.g., $x_1$) represents the effect of that variable when all interacting variables are zero.
7. Radial basis functions are useful only for problems with exactly two input dimensions.
8. [🔍] The curse of dimensionality refers exclusively to computational complexity issues when fitting models with many features.

For a detailed explanation of these true/false statements, see [Question 13: Core Concepts in Multiple Linear Regression](L3_4_13_explanation.md).

## Question 14

### Problem Statement
Match each concept in Column A with the most appropriate description in Column B.

#### Task
Match each item in Column A with the correct item from Column B:

**Column A:**
1. Multicollinearity
2. Dummy variables
3. Interaction terms
4. Design matrix
5. Basis function expansion
6. Normal equations
7. The hat matrix
8. Polynomial regression

**Column B:**
A. $\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T$ which projects $\boldsymbol{y}$ onto the column space of $\boldsymbol{X}$
B. A technique to transform input features into higher-dimensional space using fixed non-linear functions
C. Binary variables created to represent categorical predictors in regression models
D. A method to fit nonlinear relationships by including powers of input features
E. A set of linear equations whose solution provides the optimal regression coefficients
F. The problem where two or more predictor variables are highly correlated
G. Terms that capture how the effect of one variable depends on the value of another variable
H. A matrix containing all predictor variables (including the intercept column) for all observations

For a detailed explanation of this matching problem, see [Question 14: Key Concepts in Multiple Linear Regression](L3_4_14_explanation.md).

## Question 15

### Problem Statement
Complete each statement with the appropriate term or mathematical expression.

#### Task
Fill in each blank with the appropriate term or expression:

1. [🔍] In a multiple linear regression model with $d$ features, the design matrix $\boldsymbol{X}$ has dimensions $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
2. [🔍] The closed-form solution to the least squares problem in matrix form is given by $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
3. [🔍] When there is perfect multicollinearity among predictors, the matrix $\boldsymbol{X}^T\boldsymbol{X}$ becomes $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
4. If a categorical variable has $k$ levels, we typically create $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$ dummy variables to represent it.
5. [🔍] A polynomial regression model of degree 3 with a single input variable $x$ can be written as $y = \_\_\_\_\_\_\_\_\_\_\_\_\_\_ + \epsilon$.
6. A Gaussian radial basis function can be expressed as $\phi(\boldsymbol{x}) = \_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
7. [🔍] The "curse of dimensionality" in regression refers to problems that arise when $\_\_\_\_\_\_\_\_\_\_\_\_\_\_$.
8. [🔍] In matrix form, the predictions of a linear regression model can be written as $\hat{\boldsymbol{y}} = \_\_\_\_\_\_\_\_\_\_\_\_\_\_$.

For a detailed explanation of these fill-in-the-blank questions, see [Question 15: Mathematical Foundations of Multiple Regression](L3_4_15_explanation.md).

## Question 16

### Problem Statement
Select the best answer for each of the following questions.

#### Task
For each question, select the single best answer from the given options:

1. [📚] Which of the following is NOT a valid way to address multicollinearity in a regression model?
   A. Remove one of the correlated variables
   B. Combine the correlated variables into a single feature
   C. Add more training examples
   D. Square all the input features
   E. Use regularization techniques
   
2. When creating dummy variables for a categorical predictor with 4 levels, how many dummy variables are typically used?
   A. 1
   B. 2
   C. 3
   D. 4
   E. 5
   
3. [📚] What does the interaction term $x_1 \times x_2$ in a regression model capture?
   A. The sum of the effects of $x_1$ and $x_2$
   B. How the effect of $x_1$ changes based on the value of $x_2$
   C. The average effect of $x_1$ and $x_2$
   D. The direct causal relationship between $x_1$ and $x_2$
   E. The correlation between $x_1$ and $x_2$
   
4. [📚] Which of the following is a key advantage of polynomial regression over standard linear regression?
   A. Always produces models with lower test error
   B. Always requires fewer training examples
   C. Can capture nonlinear relationships in the data
   D. Always produces simpler models
   E. Eliminates the need for feature selection
   
5. What is the primary purpose of using radial basis functions in regression?
   A. To eliminate multicollinearity
   B. To reduce the number of features
   C. To capture similarities between data points based on their distance
   D. To ensure all features have equal importance
   E. To guarantee a closed-form solution exists
   
6. [📚] As the degree of a polynomial regression model increases:
   A. Training error always decreases
   B. Test error always decreases
   C. The coefficients always become smaller
   D. The model becomes more interpretable
   E. The number of required training examples decreases
   
7. Which statement about the normal equations solution $\boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$ is TRUE?
   A. It provides the unique global minimum of the cost function only when $\boldsymbol{X}^T\boldsymbol{X}$ is invertible
   B. It always provides a unique solution regardless of the properties of $\boldsymbol{X}$
   C. It requires fewer computations than iterative methods for very large datasets
   D. It is robust to outliers in the data
   E. It automatically prevents overfitting
   
8. In the context of the curse of dimensionality, which statement is TRUE?
   A. Adding more features always improves model performance
   B. As the number of features increases, the amount of data needed to generalize accurately grows exponentially
   C. The curse of dimensionality only affects classification problems, not regression
   D. Using polynomial features eliminates the curse of dimensionality
   E. The curse of dimensionality refers to the difficulty of visualizing high-dimensional data

For a detailed explanation of these multiple-choice questions, see [Question 16: Multiple Choice on Multiple Regression](L3_4_16_explanation.md).

## Question 17

### Problem Statement
Consider the generalized linear model approach where we use basis functions to transform the input data:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 \phi_1(\boldsymbol{x}) + \ldots w_m \phi_m(\boldsymbol{x})$$

Where $\{\phi_1(\boldsymbol{x}), \ldots, \phi_m(\boldsymbol{x})\}$ is a set of basis functions.

#### Task
1. [📚] Define what basis functions are and explain their role in extending linear regression to capture non-linear relationships
2. [📚] For each of the following basis function types, write down their mathematical formulation and describe a scenario where they would be particularly useful:
   a. Polynomial basis functions
   b. Gaussian radial basis functions
   c. Sigmoid basis functions
3. [📚] If you have a dataset with input features $\boldsymbol{x} \in \mathbb{R}^2$ and want to fit a quadratic model, write down all the basis functions you would need
4. [📚] Explain how the choice of basis functions affects the bias-variance tradeoff in your model

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Basis Functions in Generalized Linear Models](L3_4_17_explanation.md).

## Question 18

### Problem Statement
Consider a multiple linear regression problem with $n$ observations and $d$ features. We want to fit the model:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d$$

using matrix operations.

#### Task
1. [📚] Write down the design matrix $\boldsymbol{X}$ and show how to incorporate the intercept term $w_0$ in this matrix
2. [📚] Express the prediction of the model in matrix form using $\boldsymbol{X}$ and $\boldsymbol{w}$
3. [📚] Write down the cost function (sum of squared errors) in matrix notation
4. [📚] Derive the gradient of the cost function with respect to $\boldsymbol{w}$ in matrix form
5. [📚] By setting the gradient to zero, derive the normal equations and the closed-form solution for the optimal weights $\boldsymbol{w}$

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Matrix Formulation of Multiple Linear Regression](L3_4_18_explanation.md).

## Question 19

### Problem Statement
In linear regression, the pseudo-inverse is an important concept, especially when dealing with singular or non-invertible matrices.

#### Task
1. [📚] Define what the pseudo-inverse is in the context of linear regression and write down its formula
2. [📚] Explain when the pseudo-inverse becomes necessary in linear regression instead of the normal inverse
3. [📚] What are two specific scenarios in linear regression that would lead to a non-invertible $\boldsymbol{X}^T\boldsymbol{X}$ matrix?
4. [📚] Describe how the pseudo-inverse can be calculated using Singular Value Decomposition (SVD)
5. [📚] Explain the relationship between ridge regression and the pseudo-inverse approach for handling non-invertible matrices

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Pseudo-Inverse in Linear Regression](L3_4_19_explanation.md).

## Question 20

### Problem Statement
In generalized linear models, we use basis functions to transform the input data:

$$f(\boldsymbol{x}; \boldsymbol{w}) = w_0 + w_1\phi_1(\boldsymbol{x}) + w_2\phi_2(\boldsymbol{x}) + \cdots + w_m\phi_m(\boldsymbol{x})$$

where $\{\phi_1(\boldsymbol{x}), \ldots, \phi_m(\boldsymbol{x})\}$ is a set of basis functions.

#### Task
1. [📚] Explain how generalized linear models extend basic linear regression while still preserving the linear optimization techniques
2. [📚] For a univariate input $x$, write down the specific basis functions for:
   a) Linear regression
   b) Polynomial regression of degree 3
   c) Gaussian radial basis functions with centers at $c_1=1$, $c_2=2$, and $c_3=3$ with width $\sigma=0.5$
3. [📚] Describe the key advantages and disadvantages of using:
   a) Polynomial basis functions
   b) Radial basis functions
4. [📚] Given a dataset with highly non-linear patterns, explain which basis functions you would recommend and why

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Generalized Linear Models and Basis Functions](L3_4_19_explanation.md). 