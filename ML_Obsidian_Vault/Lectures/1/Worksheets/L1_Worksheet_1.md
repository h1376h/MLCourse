# Learning Problem Analysis Worksheet

## Introduction

This worksheet helps you practice identifying the core components of a well-posed learning problem according to Tom Mitchell's definition:

> "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

## Practice Exercise

For each scenario below, identify:
1. Task (T): What the ML system is trying to accomplish
2. Experience (E): What data the system learns from
3. Performance Measure (P): How we evaluate success

### Scenario 1: Email Organization System
An ML system that categorizes incoming emails into folders like "Work," "Personal," "Promotions," etc.

**Your Analysis:**
- T: 
- E: 
- P: 

### Scenario 2: Sentiment Analysis Tool
A system designed to analyze customer reviews and determine if they are positive, negative, or neutral.

**Your Analysis:**
- T: 
- E: 
- P: 

### Scenario 3: Energy Consumption Forecasting
A model that predicts the electricity usage of a building for the next 24 hours.

**Your Analysis:**
- T: 
- E: 
- P: 

### Scenario 4: Medical Image Segmentation
A system that identifies and outlines different organs in medical CT scans.

**Your Analysis:**
- T: 
- E: 
- P: 

### Scenario 5: Fraud Detection System
A model that identifies potentially fraudulent credit card transactions.

**Your Analysis:**
- T: 
- E: 
- P: 

## Sample Solution

### Scenario 1 Solution:
- T: Categorizing emails into appropriate folders
- E: A dataset of emails with their correct folder classifications
- P: Accuracy of folder classifications on new emails

(Solutions for remaining scenarios intentionally left blank for learning purposes)

## Creating Your Own Examples

Think of a real-world problem that could be solved using machine learning. Define:

1. A clear description of the problem
2. Task (T): What would the ML system do?
3. Experience (E): What data would it learn from?
4. Performance (P): How would you measure its success?

## Quiz Example

### Problem Statement
A ride-sharing company wants to develop a machine learning system to predict how long it will take a driver to complete a trip from pickup to dropoff. Identify the Task (T), Experience (E), and Performance Measure (P) for this learning problem.

### Solution

**Ride-sharing Trip Duration Prediction**

**Task (T)**: Predicting the duration (in minutes) of a trip from pickup to dropoff location.

**Experience (E)**:
- Historical trip data including:
  - GPS coordinates of pickup and dropoff locations
  - Actual trip durations of completed rides
  - Time of day and day of week
  - Weather conditions during trips
  - Traffic conditions during trips
  - Driver experience level
  - Vehicle type
  - Route taken

**Performance Measure (P)**:
- Mean Absolute Error (MAE) between predicted and actual trip durations
- Percentage of predictions within 2 minutes of actual duration
- Customer satisfaction ratings related to arrival time accuracy

**Explanation**:
This is a regression problem where the system learns to predict a continuous value (time duration). The model would analyze patterns in historical trip data to make predictions about future trips. The performance is measured by how close the predictions are to the actual trip durations, which directly impacts customer experience and operational efficiency.

## Discussion Questions

1. For which of the scenarios would collecting Experience (E) be most challenging? Why?
2. Which Performance measures (P) might be misleading in real-world applications?
3. Are any of these learning problems poorly posed? How could you improve them?
4. What ethical considerations might arise in collecting Experience (E) for these scenarios? 