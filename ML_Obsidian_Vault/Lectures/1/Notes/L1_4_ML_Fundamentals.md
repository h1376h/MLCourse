# Machine Learning Fundamentals

## 1. Types of Learning

### Supervised Learning
- **Regression**: Predicting continuous values
- **Classification**: Predicting discrete class labels

### Unsupervised Learning
- **Clustering**: Grouping similar instances together
- **Dimensionality Reduction**: Reducing features while preserving information
- **Association**: Discovering relationships between variables

### Semi-supervised Learning
- Uses both labeled and unlabeled data for training
- Helpful when labeled data is limited or expensive to obtain

### Reinforcement Learning
- Learning through reward/feedback for actions
- Agent learns optimal behavior through environment interaction
- Applications: game playing, robotics, autonomous systems

### Active Learning
- Selectively choosing data points to be labeled for training
- Algorithm identifies the most informative instances to label
- Reduces labeling effort while maximizing model performance
- Useful when labeling data is expensive or time-consuming

### Online Learning
- Model updates incrementally as new data arrives
- Adapts to changing patterns over time
- Doesn't require storing all historical data
- Important for streaming data and real-time applications

## 2. Is This a Learning Problem?
A problem can be approached through machine learning when:
- A pattern exists
- We do not know it mathematically
- We have data on it

## 3. Generalization
The ability of a model to perform well on previously unseen data.
- Underfitting: High bias, model too simple to capture patterns
- Overfitting: High variance, model captures noise in training data
- Regularization techniques help balance bias and variance 

## 4. Well-posed Learning Problem
According to Tom Mitchell: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

This definition provides a framework for understanding and evaluating machine learning systems:
- **Task (T)**: What the ML system is trying to accomplish
- **Experience (E)**: What data the system learns from
- **Performance (P)**: How we evaluate success

### Examples and Exercises
- [[Examples/L1_4_Well_Posed_Examples|Basic Examples]] - 10 examples of well-posed learning problems
- [[Examples/L1_2_Application_Examples|Real-world Applications]] - Detailed analysis of ML applications
- [[Examples/L1_4_Problem_Identification|Learning Problem Identification]] - Examples of what makes a good ML problem
- [[Examples/L1_Worksheet_1|Practice Worksheet]] - Exercises to practice identifying T, E, and P 