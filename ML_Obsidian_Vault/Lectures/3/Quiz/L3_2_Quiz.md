# Lecture 3.2: Simple Linear Regression Quiz

## Overview
This quiz contains 15 questions from different topics covered in section 3.2 of the lectures on Simple Linear Regression.

## Question 1

### Problem Statement
Consider a simple linear regression model for predicting house prices based on house size (in square feet). The following data points are observed:

| House Size (x) | Price (y) in $1000s |
|----------------|---------------------|
| 1000           | 150                 |
| 1500           | 200                 |
| 2000           | 250                 |
| 2500           | 300                 |
| 3000           | 350                 |

#### Task
1. Find the least squares estimates for the slope ($\beta_1$) and intercept ($\beta_0$) of the linear regression model
2. Interpret the meaning of the slope coefficient in the context of this problem
3. Calculate the prediction for a house with 1800 square feet
4. Calculate the residuals for each data point and the residual sum of squares (RSS)

For a detailed explanation of this problem, including step-by-step calculations and interpretations, see [Question 1: Simple Linear Regression for House Prices](L3_2_1_explanation.md).

## Question 2

### Problem Statement
Professor Statistics wants to help first-year students understand how study time affects exam performance. She collected data from 5 students who tracked their study hours before the final exam:

| Hours Studying (x) | Exam Score (y) |
|--------------------|----------------|
| 2                  | 65             |
| 3                  | 70             |
| 5                  | 85             |
| 7                  | 90             |
| 8                  | 95             |

Professor Statistics believes this is a perfect opportunity to demonstrate simple linear regression and wants you to help create a "Study Hour Predictor" for future students.

#### Task
1. Calculate the average study time ($\bar{x}$) and average exam score ($\bar{y}$)
2. Calculate the covariance between study hours and exam scores:
   $$\text{Cov}(x,y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$
3. Calculate the variance of study hours:
   $$\text{Var}(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$
4. Using these values, compute the "magic formula" (the slope $\beta_1$ and intercept $\beta_0$) for predicting exam scores:
   $$\beta_1 = \frac{\text{Cov}(x,y)}{\text{Var}(x)} \quad \text{and} \quad \beta_0 = \bar{y} - \beta_1\bar{x}$$

For a detailed explanation of this problem, including step-by-step calculations and statistical interpretations, see [Question 2: Study Time and Exam Performance Regression](L3_2_2_explanation.md).

## Question 3

### Problem Statement
Given a simple linear regression model with the following equation: $\hat{y} = 3 + 2x$

#### Task
1. Calculate the predicted value $\hat{y}$ when $x = 4$
2. If the actual observed value when $x = 4$ is $y = 12$, what is the residual?

For a detailed explanation of this problem, see [Question 3: Basic Prediction and Residual Calculation](L3_2_3_explanation.md).

## Question 4

### Problem Statement
For a simple linear regression model, you're given that the mean of $x$ is $\bar{x} = 5$, the mean of $y$ is $\bar{y} = 10$, and the slope coefficient is $\beta_1 = 2$.

#### Task
1. Calculate the intercept term $\beta_0$ of the regression line
2. Write down the complete regression equation

For a detailed explanation of this problem, see [Question 4: Finding the Intercept from Means](L3_2_4_explanation.md). 

## Question 5

### Problem Statement
You're the owner of a beachside ice cream shop and want to predict daily sales based on temperature. You've tracked your sales (in hundreds of dollars) on days with different temperatures:

| Temperature (°C) | Ice Cream Sales ($100s) |
|------------------|-------------------------|
| 15               | 4                       |
| 18               | 5                       |
| 20               | 6                       |
| 25               | 8                       |
| 30               | 11                      |
| 35               | 15                      |

You notice something interesting: on hotter days, not only do sales increase, but they seem to increase more dramatically. You wonder if this pattern affects your sales prediction model.

#### Task
1. Create a simple linear regression model to predict ice cream sales based on temperature (find the equation: $\text{Sales} = \beta_0 + \beta_1 \times \text{Temperature}$)
2. Draw a quick sketch showing your sales predictions vs. actual sales (or calculate and plot the residuals)
3. What do you notice about the pattern of errors? Does your model consistently under-predict or over-predict at certain temperatures?
4. If your model isn't perfectly predicting sales on very hot days, explain which assumption of linear regression might be violated and suggest a transformation that might improve your model

For a detailed explanation of this problem, including regression diagnostics and transformation techniques, see [Question 5: Ice Cream Sales Prediction](L3_2_5_explanation.md).

## Question 6

### Problem Statement
In a psychological study, researchers are investigating the relationship between hours of sleep (x) and cognitive test performance (y). They collect the following data from 6 participants:

| Hours of Sleep (x) | Cognitive Test Score (y) |
|-------------------|--------------------------|
| 5                 | 65                       |
| 6                 | 70                       |
| 7                 | 80                       |
| 8                 | 85                       |
| 9                 | 88                       |
| 10                | 90                       |

#### Task
1. Calculate the correlation coefficient (r) between hours of sleep and cognitive test scores
2. Calculate the coefficient of determination ($R^2$) and interpret its meaning in this context
3. If the standard deviation of hours of sleep is 1.8 hours and the standard deviation of cognitive test scores is 9.6 points, find the slope ($\beta_1$) of the regression line using the correlation coefficient
4. What proportion of the variance in cognitive test scores can be explained by hours of sleep?

For a detailed explanation of this problem, including statistical calculations and interpretations, see [Question 6: Sleep and Cognitive Performance](L3_2_6_explanation.md).

## Question 7

### Problem Statement
A restaurant owner wants to understand how outdoor temperature affects ice water consumption. Data was collected on 5 different days:

| Temperature (x) in °C | Water Consumed (y) in liters |
|----------------------|-----------------------------|
| 20                    | 35                          |
| 25                    | 45                          |
| 30                    | 60                          |
| 35                    | 80                          |
| 40                    | 95                          |

#### Task
1. Sketch a scatter plot of the data points. Does a linear relationship seem appropriate?
2. Calculate the mean of x and y values.
3. Using the least squares method, find the linear equation that best fits this data.
4. If tomorrow's forecast is 28°C, how many liters of water should the restaurant prepare?
5. Calculate the residual for the day when temperature was 30°C.

For a detailed explanation of this problem, see [Question 7: Temperature and Water Consumption](L3_2_7_explanation.md). 

## Question 8

### Problem Statement
A data scientist wants to understand the error distribution in a linear regression model. They collect the following dataset relating advertising expenditure (in $1000s) to product sales (in units):

| Advertising (x) | Sales (y) |
|----------------|-----------|
| 1              | 20        |
| 3              | 40        |
| 5              | 50        |
| 7              | 65        |
| 9              | 80        |

After fitting a linear regression model, they obtain the following predicted values for sales: 25, 38, 50, 63, 75.

#### Task
1. Calculate the residuals for each observation
2. Compute the mean, variance, and standard deviation of the residuals
3. Plot (or describe) a visualization that could help assess if the errors follow a Gaussian distribution
4. Verify if the residuals sum to zero (or very close to zero), and explain why this is an expected property in linear regression with an intercept term

For a detailed explanation of this problem, including residual analysis and error distribution assessment, see [Question 8: Error Distribution in Linear Regression](L3_2_8_explanation.md).

## Question 9

### Problem Statement
An e-commerce company wants to understand the relationship between the time users spend on their website (in minutes) and the amount they spend on purchases (in dollars). They collect data from 5 random user sessions:

| Time on Site (x) in minutes | Purchase Amount (y) in $ |
|----------------------------|--------------------------|
| 2                          | 15                       |
| 5                          | 35                       |
| 8                          | 40                       |
| 12                         | 60                       |
| 15                         | 75                       |

#### Task
1. Compute the least squares estimates for the slope ($\beta_1$) and intercept ($\beta_0$) using the following formulas:
   $$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
   $$\beta_0 = \bar{y} - \beta_1\bar{x}$$
2. Calculate the predicted purchase amount for a user who spends 10 minutes on the site
3. For each of the original data points, compute the squared error $(y_i - \hat{y}_i)^2$
4. Calculate the total Mean Squared Error (MSE) for this model

For a detailed explanation of this problem, including regression calculations and model evaluation, see [Question 9: E-commerce Website Regression Analysis](L3_2_9_explanation.md).

## Question 10

### Problem Statement
A plant biologist is studying the effect of light exposure (hours per day) on plant growth (in cm). The following data was collected over a week:

| Light Exposure (x) in hours | Plant Growth (y) in cm |
|----------------------------|------------------------|
| 4                          | 2.1                    |
| 6                          | 3.4                    |
| 8                          | 4.7                    |
| 10                         | 5.9                    |
| 12                         | 7.2                    |

#### Task
1. Find the least squares estimates for the slope ($\beta_1$) and intercept ($\beta_0$)
2. What is the expected growth when the plant receives 9 hours of light?
3. If you increase light exposure by 2 hours, how much additional growth would you expect?
4. Calculate the $R^2$ value for this model and interpret what it means about the relationship between light exposure and plant growth.

For a detailed explanation of this problem, see [Question 10: Light Exposure and Plant Growth](L3_2_10_explanation.md).

## Question 11

### Problem Statement
You are given the following data on study hours and exam scores for 6 students:

| Student | Study Hours (x) | Exam Score (y) |
|---------|----------------|----------------|
| A       | 2              | 50             |
| B       | 3              | 60             |
| C       | 5              | 70             |
| D       | 7              | 80             |
| E       | 8              | 90             |
| F       | 10             | 95             |

Professor Andrew's formula for estimating the expected exam score is: Score = 40 + 5.5 × (Study Hours).

#### Task
1. Calculate the predicted score for each student using Professor Andrew's formula
2. Calculate the residual for each student (the difference between actual and predicted score)
3. Calculate the Residual Sum of Squares (RSS)
4. If a new student studies for 6 hours, what would be their predicted exam score according to this model?

For a detailed explanation of this problem, including prediction calculations and residual analysis, see [Question 11: Study Hours and Exam Scores Prediction](L3_2_11_explanation.md).

## Question 12

### Problem Statement
Given the sample data points (1, 2), (2, 4), and (3, 6):

#### Task
1. Using only these three points, calculate the means $x̄$ and $ȳ$
2. Find the slope of the simple linear regression model by hand
3. Find the intercept of the model
4. Write down the resulting equation for predicting y from x

For a detailed explanation of this problem, see [Question 12: Simple Linear Regression Calculation](L3_2_12_explanation.md).

## Question 13

### Problem Statement
Consider a simple linear regression model where we want to predict fuel efficiency (miles per gallon) based on car weight (in thousands of pounds). We have the following data from 5 different car models:

| Car Model | Weight (x) in 1000 lbs | Fuel Efficiency (y) in MPG |
|-----------|--------------------------|----------------------------|
| A         | 2.5                      | 30                         |
| B         | 3.0                      | 25                         |
| C         | 3.5                      | 23                         |
| D         | 4.0                      | 20                         |
| E         | 4.5                      | 18                         |

#### Task
1. Calculate the means $\bar{x}$ and $\bar{y}$, and the sample covariance and variance needed for the regression coefficients
2. Determine the least squares estimates for $\beta_0$ and $\beta_1$
3. Interpret the meaning of the slope coefficient in the context of this problem
4. Calculate the coefficient of determination $R^2$ using the formula:
   $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
   and explain what it tells us about the relationship between car weight and fuel efficiency

For a detailed explanation of this problem, including regression analysis and model interpretation, see [Question 13: Car Weight and Fuel Efficiency](L3_2_13_explanation.md).

## Question 14

### Problem Statement
A medical researcher is studying the relationship between daily vitamin C intake (in mg) and the duration of common cold symptoms (in days). The data collected from 6 patients is as follows:

| Patient | Vitamin C Intake (x) in mg | Cold Duration (y) in days |
|---------|----------------------------|---------------------------|
| 1       | 50                         | 7                         |
| 2       | 100                        | 6                         |
| 3       | 150                        | 5                         |
| 4       | 200                        | 4                         |
| 5       | 250                        | 3                         |
| 6       | 300                        | 3                         |

#### Task
1. Calculate the least squares estimates for the slope and intercept of the linear regression model
2. Write the equation of the linear regression line
3. If we define the "effectiveness" of vitamin C as the reduction in cold duration (in days) for each additional 100mg of vitamin C, what is the effectiveness according to this model?
4. Using the model, predict the cold duration for a patient with a vitamin C intake of 175mg

For a detailed explanation of this problem, including medical data analysis and interpretation, see [Question 14: Vitamin C and Cold Duration](L3_2_14_explanation.md).

## [⭐] Question 15

### Problem Statement
A researcher is investigating the relationship between age and glucose levels in patients. The data collected from 6 subjects is as follows:

| Subject | Age (x) | Glucose Level (y) |
|---------|---------|-------------------|
| 1       | 43      | 99                |
| 2       | 21      | 65                |
| 3       | 25      | 79                |
| 4       | 42      | 75                |
| 5       | 57      | 87                |
| 6       | 59      | 81                |

#### Task
1. Derive a simple linear regression equation to predict glucose level based on age
2. Calculate the correlation coefficient between age and glucose level
3. Using your derived regression equation, predict the glucose level for a 55-year-old subject
4. Calculate the coefficient of determination (R²) and interpret what percentage of the variation in glucose levels can be explained by age

#### Solution Approaches
This problem can be tackled using multiple approaches:

For a detailed explanation using standard regression analysis methods, see [Question 15: Age and Glucose Level Prediction](L3_2_15_explanation.md).

For an alternative approach using calculus and cost function minimization, see [Alternative Solution: Analytical Derivation](L3_2_15_analytical_explanation.md).

For a simplified, exam-friendly approach with computational shortcuts, see [Alternative Solution: Quick Method](L3_2_15_quick_solution.md).
