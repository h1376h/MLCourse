# Lecture 9.3: Regression Evaluation Metrics Quiz

## Overview
This quiz contains 25 comprehensive questions covering regression evaluation metrics, including MSE, RMSE, MAE, R-squared, adjusted R-squared, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Consider a regression model that predicts house prices based on square footage.

**Training Results:**
- Actual prices: $200K, $250K, $300K, $350K, $400K
- Predicted prices: $210K, $245K, $295K, $355K, $395K

#### Task
1. Calculate the Mean Squared Error (MSE)
2. Calculate the Root Mean Squared Error (RMSE)
3. Calculate the Mean Absolute Error (MAE)
4. Which metric is most sensitive to outliers? Explain why
5. If you had to choose one metric to report to stakeholders, which would you pick and why?

For a detailed explanation of this question, see [Question 1: Basic Regression Metrics](L9_3_1_explanation.md).

## Question 2

### Problem Statement
You're evaluating a model that predicts student exam scores based on study hours.

**Test Results:**
- Actual scores: 75, 82, 68, 90, 85, 78, 92, 80
- Predicted scores: 78, 80, 70, 88, 87, 75, 90, 82

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. Calculate the Mean Absolute Percentage Error (MAPE)
5. Which prediction has the largest absolute error?

For a detailed explanation of this question, see [Question 2: Student Score Prediction](L9_3_2_explanation.md).

## Question 3

### Problem Statement
Consider a model that predicts daily sales for a retail store.

**Weekly Results:**
- Actual sales: $1200, $1350, $1100, $1400, $1250, $1300, $1150
- Predicted sales: $1180, $1320, $1120, $1380, $1270, $1280, $1180

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within $100 of actual sales?
5. If the store manager wants to understand prediction accuracy in dollar terms, which metric would be most useful?

For a detailed explanation of this question, see [Question 3: Retail Sales Prediction](L9_3_3_explanation.md).

## Question 4

### Problem Statement
You're comparing two models for predicting car prices.

**Model A Performance:**
- MSE: 2500
- MAE: 45
- R-squared: 0.85

**Model B Performance:**
- MSE: 2800
- MAE: 42
- R-squared: 0.88

#### Task
1. Which model has lower prediction variance?
2. Which model has lower average absolute error?
3. Which model explains more variance in the target variable?
4. If you had to choose one model, which would you select and why?
5. Calculate the RMSE for both models

For a detailed explanation of this question, see [Question 4: Model Comparison](L9_3_4_explanation.md).

## Question 5

### Problem Statement
Consider a model that predicts customer satisfaction scores (1-10 scale).

**Results:**
- Actual scores: 7, 8, 6, 9, 7, 8, 6, 7, 8, 9
- Predicted scores: 7.2, 7.8, 6.1, 8.9, 7.1, 7.9, 6.2, 7.3, 7.7, 8.8

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 0.5 points of actual scores?
5. For a satisfaction scale, which metric is most interpretable?

For a detailed explanation of this question, see [Question 5: Customer Satisfaction Prediction](L9_3_5_explanation.md).

## Question 6

### Problem Statement
You have a model that predicts monthly electricity consumption.

**Quarterly Results:**
- Actual consumption: 450 kWh, 520 kWh, 380 kWh, 600 kWh
- Predicted consumption: 460 kWh, 510 kWh, 390 kWh, 590 kWh

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. Calculate the Mean Absolute Percentage Error (MAPE)
5. If the utility company wants to understand prediction accuracy in percentage terms, which metric would be most useful?

For a detailed explanation of this question, see [Question 6: Electricity Consumption Prediction](L9_3_6_explanation.md).

## Question 7

### Problem Statement
Consider a model that predicts delivery times for online orders.

**Sample Results:**
- Actual times: 2.5, 3.0, 1.8, 4.2, 2.9, 3.5, 2.1, 3.8 days
- Predicted times: 2.6, 2.9, 1.9, 4.0, 3.1, 3.4, 2.2, 3.7 days

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 0.5 days of actual delivery times?
5. For delivery time predictions, which metric would customers find most understandable?

For a detailed explanation of this question, see [Question 7: Delivery Time Prediction](L9_3_7_explanation.md).

## Question 8

### Problem Statement
You're evaluating a model that predicts stock prices.

**Daily Results:**
- Actual prices: $45.20, $46.80, $44.50, $47.10, $45.90
- Predicted prices: $45.50, $46.50, $44.80, $46.80, $46.20

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. Calculate the Mean Absolute Percentage Error (MAPE)
5. For stock price predictions, which metric would investors find most useful?

For a detailed explanation of this question, see [Question 8: Stock Price Prediction](L9_3_8_explanation.md).

## Question 9

### Problem Statement
Consider a model that predicts website traffic based on marketing spend.

**Monthly Results:**
- Actual traffic: 1500, 1800, 1200, 2000, 1600, 1900, 1400, 1700
- Predicted traffic: 1550, 1750, 1250, 1950, 1650, 1850, 1450, 1680

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 100 visitors of actual traffic?
5. For marketing planning, which metric would be most practical?

For a detailed explanation of this question, see [Question 9: Website Traffic Prediction](L9_3_9_explanation.md).

## Question 10

### Problem Statement
You have a model that predicts product ratings based on customer reviews.

**Sample Results:**
- Actual ratings: 4.2, 3.8, 4.5, 3.9, 4.1, 4.3, 3.7, 4.0
- Predicted ratings: 4.1, 3.9, 4.4, 4.0, 4.2, 4.2, 3.8, 4.1

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 0.2 points of actual ratings?
5. For product rating predictions, which metric would be most meaningful to customers?

For a detailed explanation of this question, see [Question 10: Product Rating Prediction](L9_3_10_explanation.md).

## Question 11

### Problem Statement
Consider a model that predicts employee productivity scores.

**Department Results:**
- Sales: 85, 78, 92, 88, 76
- Marketing: 82, 85, 79, 88, 84
- Engineering: 90, 87, 93, 85, 89

**Predicted Scores:**
- Sales: 87, 80, 90, 86, 78
- Marketing: 84, 83, 81, 86, 82
- Engineering: 88, 89, 91, 87, 88

#### Task
1. Calculate the MSE for each department
2. Calculate the overall MSE across all departments
3. Which department has the most accurate predictions?
4. Calculate the RMSE for the worst-performing department
5. How would you use this information to improve the model?

For a detailed explanation of this question, see [Question 11: Employee Productivity Prediction](L9_3_11_explanation.md).

## Question 12

### Problem Statement
You're evaluating a model that predicts customer lifetime value (CLV).

**Results by Customer Segment:**
- High-value: Actual $5000, Predicted $4800
- Medium-value: Actual $2500, Predicted $2600
- Low-value: Actual $800, Predicted $850

#### Task
1. Calculate the MSE for each segment
2. Calculate the overall MSE
3. Calculate the MAE for each segment
4. Which segment has the most accurate predictions?
5. If you had to improve one segment, which would you focus on and why?

For a detailed explanation of this question, see [Question 12: Customer Lifetime Value Prediction](L9_3_12_explanation.md).

## Question 13

### Problem Statement
Consider a model that predicts monthly revenue for different store locations.

**Store Performance:**
- Downtown: Actual $50K, Predicted $48K
- Mall: Actual $35K, Predicted $37K
- Suburban: Actual $25K, Predicted $26K
- Airport: Actual $40K, Predicted $38K

#### Task
1. Calculate the MSE for each store
2. Calculate the overall MSE
3. Calculate the MAE for each store
4. Which store has the most accurate predictions?
5. If you had to improve one store's predictions, which would you choose?

For a detailed explanation of this question, see [Question 13: Store Revenue Prediction](L9_3_13_explanation.md).

## Question 14

### Problem Statement
You have a model that predicts project completion times.

**Project Results:**
- Small projects: Actual 2 weeks, Predicted 2.2 weeks
- Medium projects: Actual 6 weeks, Predicted 5.8 weeks
- Large projects: Actual 12 weeks, Predicted 11.5 weeks

#### Task
1. Calculate the MSE for each project size
2. Calculate the overall MSE
3. Calculate the MAE for each project size
4. Which project size has the most accurate predictions?
5. For project planning, which metric would be most useful?

For a detailed explanation of this question, see [Question 14: Project Timeline Prediction](L9_3_14_explanation.md).

## Question 15

### Problem Statement
Consider a model that predicts customer wait times at a restaurant.

**Time Slot Results:**
- Lunch (12-2 PM): Actual 15 min, Predicted 16 min
- Dinner (6-8 PM): Actual 25 min, Predicted 24 min
- Late night (9-11 PM): Actual 10 min, Predicted 11 min

#### Task
1. Calculate the MSE for each time slot
2. Calculate the overall MSE
3. Calculate the MAE for each time slot
4. Which time slot has the most accurate predictions?
5. For restaurant management, which metric would be most practical?

For a detailed explanation of this question, see [Question 15: Restaurant Wait Time Prediction](L9_3_15_explanation.md).

## Question 16

### Problem Statement
You're evaluating a model that predicts quarterly profits for different business units.

**Business Unit Results:**
- Manufacturing: Actual $100K, Predicted $98K
- Sales: Actual $150K, Predicted $152K
- R&D: Actual -$20K, Predicted -$18K
- Support: Actual $30K, Predicted $32K

#### Task
1. Calculate the MSE for each business unit
2. Calculate the overall MSE
3. Calculate the MAE for each business unit
4. Which business unit has the most accurate predictions?
5. How would you handle the negative profit predictions in your evaluation?

For a detailed explanation of this question, see [Question 16: Business Unit Profit Prediction](L9_3_16_explanation.md).

## Question 17

### Problem Statement
Consider a model that predicts daily temperature based on weather patterns.

**Weekly Results:**
- Monday: Actual 72°F, Predicted 74°F
- Tuesday: Actual 68°F, Predicted 67°F
- Wednesday: Actual 75°F, Predicted 73°F
- Thursday: Actual 70°F, Predicted 71°F
- Friday: Actual 78°F, Predicted 76°F

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 3°F of actual temperature?
5. For weather forecasting, which metric would be most meaningful to the public?

For a detailed explanation of this question, see [Question 17: Temperature Prediction](L9_3_17_explanation.md).

## Question 18

### Problem Statement
You have a model that predicts monthly subscription renewals.

**Results by Plan Type:**
- Basic: Actual 85%, Predicted 87%
- Premium: Actual 92%, Predicted 90%
- Enterprise: Actual 78%, Predicted 80%

#### Task
1. Calculate the MSE for each plan type
2. Calculate the overall MSE
3. Calculate the MAE for each plan type
4. Which plan type has the most accurate predictions?
5. For subscription management, which metric would be most useful?

For a detailed explanation of this question, see [Question 18: Subscription Renewal Prediction](L9_3_18_explanation.md).

## Question 19

### Problem Statement
Consider a model that predicts customer support ticket resolution times.

**Priority Level Results:**
- Low: Actual 2 hours, Predicted 2.5 hours
- Medium: Actual 8 hours, Predicted 7.5 hours
- High: Actual 24 hours, Predicted 22 hours
- Critical: Actual 4 hours, Predicted 4.5 hours

#### Task
1. Calculate the MSE for each priority level
2. Calculate the overall MSE
3. Calculate the MAE for each priority level
4. Which priority level has the most accurate predictions?
5. For support management, which metric would be most important?

For a detailed explanation of this question, see [Question 19: Support Ticket Resolution Prediction](L9_3_19_explanation.md).

## Question 20

### Problem Statement
You're evaluating a model that predicts quarterly market share for different products.

**Product Results:**
- Product A: Actual 25%, Predicted 26%
- Product B: Actual 35%, Predicted 33%
- Product C: Actual 20%, Predicted 21%
- Product D: Actual 20%, Predicted 20%

#### Task
1. Calculate the MSE for each product
2. Calculate the overall MSE
3. Calculate the MAE for each product
4. Which product has the most accurate predictions?
5. For market analysis, which metric would be most valuable?

For a detailed explanation of this question, see [Question 20: Market Share Prediction](L9_3_20_explanation.md).

## Question 21

### Problem Statement
Consider a model that predicts daily website conversion rates.

**Weekly Results:**
- Monday: Actual 2.5%, Predicted 2.7%
- Tuesday: Actual 3.1%, Predicted 3.0%
- Wednesday: Actual 2.8%, Predicted 2.9%
- Thursday: Actual 3.2%, Predicted 3.1%
- Friday: Actual 2.9%, Predicted 2.8%

#### Task
1. Calculate the MSE
2. Calculate the RMSE
3. Calculate the MAE
4. What percentage of predictions are within 0.3% of actual conversion rates?
5. For marketing optimization, which metric would be most practical?

For a detailed explanation of this question, see [Question 21: Conversion Rate Prediction](L9_3_21_explanation.md).

## Question 22

### Problem Statement
You have a model that predicts monthly customer acquisition costs.

**Channel Results:**
- Social Media: Actual $50, Predicted $52
- Google Ads: Actual $75, Predicted $73
- Email Marketing: Actual $30, Predicted $31
- Referral: Actual $25, Predicted $26

#### Task
1. Calculate the MSE for each channel
2. Calculate the overall MSE
3. Calculate the MAE for each channel
4. Which channel has the most accurate predictions?
5. For budget planning, which metric would be most useful?

For a detailed explanation of this question, see [Question 22: Customer Acquisition Cost Prediction](L9_3_22_explanation.md).

## Question 23

### Problem Statement
Consider a model that predicts quarterly inventory turnover rates.

**Product Category Results:**
- Electronics: Actual 4.2, Predicted 4.0
- Clothing: Actual 6.8, Predicted 7.0
- Home & Garden: Actual 3.5, Predicted 3.7
- Books: Actual 8.1, Predicted 7.9

#### Task
1. Calculate the MSE for each category
2. Calculate the overall MSE
3. Calculate the MAE for each category
4. Which category has the most accurate predictions?
5. For inventory management, which metric would be most important?

For a detailed explanation of this question, see [Question 23: Inventory Turnover Prediction](L9_3_23_explanation.md).

## Question 24

### Problem Statement
You're evaluating a model that predicts monthly customer retention rates.

**Segment Results:**
- New customers: Actual 65%, Predicted 67%
- Returning customers: Actual 85%, Predicted 83%
- Loyal customers: Actual 95%, Predicted 94%
- At-risk customers: Actual 45%, Predicted 47%

#### Task
1. Calculate the MSE for each segment
2. Calculate the overall MSE
3. Calculate the MAE for each segment
4. Which segment has the most accurate predictions?
5. For customer success planning, which metric would be most valuable?

For a detailed explanation of this question, see [Question 24: Customer Retention Prediction](L9_3_24_explanation.md).

## Question 25

### Problem Statement
Consider a model that predicts daily energy consumption for a building.

**Seasonal Results:**
- Spring: Actual 120 kWh, Predicted 118 kWh
- Summer: Actual 180 kWh, Predicted 182 kWh
- Fall: Actual 110 kWh, Predicted 112 kWh
- Winter: Actual 160 kWh, Predicted 158 kWh

#### Task
1. Calculate the MSE for each season
2. Calculate the overall MSE
3. Calculate the MAE for each season
4. Which season has the most accurate predictions?
5. For energy management, which metric would be most practical?

For a detailed explanation of this question, see [Question 25: Seasonal Energy Consumption Prediction](L9_3_25_explanation.md).
