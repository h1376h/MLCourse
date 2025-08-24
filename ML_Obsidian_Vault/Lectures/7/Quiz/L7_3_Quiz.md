# Lecture 7.3: Random Forest Deep Dive Quiz

## Overview
This quiz contains 31 questions covering Random Forest fundamentals, including tree diversity mechanisms, feature subsampling, voting strategies, out-of-bag estimation, feature importance analysis, and practical applications.

## Question 1

### Problem Statement
Random Forest combines bagging with feature subsampling to create diverse trees.

#### Task
1. Explain how Random Forest creates diversity among trees
2. Describe the relationship between Random Forest and bagging
3. Why is feature subsampling important in Random Forest?
4. How does Random Forest differ from simple bagging of decision trees?
5. Calculate the probability that two trees will have completely different feature sets if each tree randomly selects 3 features from a total of 10 features

For a detailed explanation of this question, see [Question 1: Random Forest Foundations](L7_3_1_explanation.md).

## Question 2

### Problem Statement
Feature subsampling in Random Forest affects tree diversity and performance.

#### Task
1. If you have $20$ features and consider $5$ at each split, what is the probability a specific feature is used?
2. How does this probability change if you increase the number of features considered?
3. What is the tradeoff between feature subsampling and tree performance?
4. How do you choose the optimal number of features to consider?
5. Using the formula $P(\text{feature used}) = 1 - \left(\frac{n-1}{n}\right)^k$ where $n$ is total features and $k$ is features per split, calculate the probability for $n=20$ and $k=5$

For a detailed explanation of this question, see [Question 2: Feature Subsampling Analysis](L7_3_2_explanation.md).

## Question 3

### Problem Statement
Random Forest uses different voting strategies for making predictions.

#### Task
1. What is the difference between hard voting and soft voting?
2. When would you prefer soft voting over hard voting?
3. How does Random Forest handle probability estimates?
4. What is the advantage of ensemble voting over single tree predictions?
5. For a 3-tree ensemble with predictions $[0.8, 0.6, 0.9]$, calculate both hard voting result (threshold 0.5) and soft voting average

For a detailed explanation of this question, see [Question 3: Voting Strategies](L7_3_3_explanation.md).

## Question 4

### Problem Statement
Out-of-bag estimation provides internal validation for Random Forest without cross-validation.

#### Task
1. How does out-of-bag estimation work?
2. What is the advantage of OOB estimation over cross-validation?
3. When might OOB estimation not be reliable?
4. How does OOB estimation help with model selection?
5. For a dataset with 1000 samples, calculate the expected number of OOB samples per tree using the formula: $\text{OOB samples} = n \times (1 - \frac{1}{n})^n$ where $n$ is the dataset size

For a detailed explanation of this question, see [Question 4: Out-of-Bag Estimation](L7_3_4_explanation.md).

## Question 5

### Problem Statement
Feature importance in Random Forest measures variable significance for predictions.

#### Task
1. How is feature importance calculated in Random Forest?
2. Why is Random Forest feature importance more reliable than single tree importance?
3. What are the limitations of feature importance measures?
4. How can you use feature importance for feature selection?
5. Given feature importance scores $[0.4, 0.3, 0.2, 0.1]$, calculate the cumulative importance percentage and determine how many features are needed to reach a high cumulative importance threshold (e.g., 80% or 90%)

For a detailed explanation of this question, see [Question 5: Feature Importance Analysis](L7_3_5_explanation.md).

## Question 6

### Problem Statement
Analyze a Random Forest with $5$ trees for a binary classification problem.

**Sample 1:** Tree predictions $[0, 1, 0, 1, 0]$ with confidences $[0.8, 0.6, 0.9, 0.7, 0.85]$
**Sample 2:** Tree predictions $[1, 1, 1, 1, 1]$ with confidences $[0.95, 0.88, 0.92, 0.89, 0.91]$
**Sample 3:** Tree predictions $[0, 1, 0, 1, 0]$ with confidences $[0.55, 0.65, 0.45, 0.75, 0.60]$

#### Task
1. What are the final predictions using hard voting?
2. What are the final predictions using soft voting?
3. Which sample has the highest confidence in the ensemble prediction?
4. If you needed high confidence, which sample would you trust most?
5. Calculate the variance of predictions for each sample and explain which sample shows the highest disagreement among trees

For a detailed explanation of this question, see [Question 6: Ensemble Prediction Analysis](L7_3_6_explanation.md).

## Question 7

### Problem Statement
Design a Random Forest configuration for a dataset with $1000$ samples, $25$ features, binary classification, and maximum $100$ trees.

**Note:** Common rules: $\text{max\_features} = \sqrt{\text{total\_features}}$ or $\log_2(\text{total\_features})$. OOB stabilizes around $50$ trees.

#### Task
1. Calculate the optimal number of features to consider at each split
2. Determine the minimum number of trees needed for reliable OOB estimation
3. If you want $95\%$ confidence that a feature is selected at least once, how many trees do you need?
4. What would be the expected number of unique features used across all trees?
5. Derive the general formula for calculating the probability that a feature is never selected across all trees, and explain how this probability decreases as the number of trees increases

For a detailed explanation of this question, see [Question 7: Configuration Optimization](L7_3_7_explanation.md).

## Question 8

### Problem Statement
Analyze Random Forest performance with $50$ trees, $10$ features per split (out of $100$ total), and bootstrap sample size $632$ ($63.2\%$ of $1000$ total samples).

**Note:** Bootstrap probability ≈ $63.2\%$ for large datasets. OOB = $36.8\%$ per tree.

#### Task
1. What is the probability that a specific sample appears in the bootstrap sample for a given tree?
2. How many trees, on average, will not contain a specific sample (OOB trees)?
3. If you increase features per split to $20$, how does this affect tree diversity?
4. Calculate the expected number of trees that will use a specific feature at least once.
5. Calculate the probability that a specific feature is used in exactly 25 out of 50 trees using the binomial probability formula

For a detailed explanation of this question, see [Question 8: Performance Analysis](L7_3_8_explanation.md).

## Question 9

### Problem Statement
Compare three Random Forest configurations for a dataset with $20$ total features:

**Configuration A:** $100$ trees, $5$ features per split, $\text{max-depth} = 10$

**Configuration B:** $50$ trees, $10$ features per split, $\text{max-depth} = 15$

**Configuration C:** $200$ trees, $3$ features per split, $\text{max-depth} = 8$

#### Task
1. Which configuration will likely have the highest tree diversity?
2. Which configuration will be fastest to train? (Training speed $\propto$ trees × depth × features per split)
3. Which configuration will likely have the lowest variance in predictions? (Variance: more trees = lower variance, deeper trees = higher variance)
4. If you have limited memory, which configuration would you choose? (Memory $\propto 2^{\text{depth}} \times \text{trees}$)
5. Calculate the training time ratio between the fastest and slowest configurations, assuming each tree takes 2 seconds to train

For a detailed explanation of this question, see [Question 9: Configuration Comparison](L7_3_9_explanation.md).

## Question 10

### Problem Statement
Consider a $2$D classification problem with features $X$ and $Y$. A Random Forest has $3$ trees:

**Tree 1:** Splits on $X$ at $x = 5$, then $Y$ at $y = 3$
**Tree 2:** Splits on $Y$ at $y = 4$, then $X$ at $x = 6$  
**Tree 3:** Splits on $X$ at $x = 4$, then $Y$ at $y = 2$

#### Task
1. Draw the decision boundaries for each tree
2. What is the prediction for point $(3, 1)$?
3. What is the prediction for point $(7, 5)$?
4. How does the ensemble decision boundary differ from individual tree boundaries?
5. Calculate the area of the region where all three trees agree on the same class prediction

**Note:** Follow each tree's decision path. Ensemble uses majority voting from all trees.

For a detailed explanation of this question, see [Question 10: Decision Boundary Analysis](L7_3_10_explanation.md).

## Question 11

### Problem Statement
Investigate fraud detection using a Random Forest with $7$ trees. Each tree gives fraud probabilities for $4$ suspicious transactions:

**Transaction A:** $[0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3]$
**Transaction B:** $[0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7]$
**Transaction C:** $[0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4]$
**Transaction D:** $[0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]$

#### Task
1. Calculate the ensemble fraud probability for each transaction
2. If fraud threshold is $0.5$, which transactions are flagged as suspicious?
3. Which transaction shows the highest disagreement among trees (highest variance)?
4. If you could only investigate $2$ transactions, which would you prioritize and why?
5. Calculate the standard deviation of fraud probabilities for each transaction and rank them by uncertainty

For a detailed explanation of this question, see [Question 11: Fraud Detection Game](L7_3_11_explanation.md).

## Question 12

### Problem Statement
Build a Random Forest for medical diagnosis with $500$ patients, $30$ medical features, binary diagnosis (Healthy/Sick), and maximum $50$ trees due to computational limits.

#### Task
1. Calculate the optimal number of features per split for maximum diversity
2. If you want each feature to be used in at least $80\%$ of trees, how many trees do you need?
3. Design a feature sampling strategy that ensures rare but important features aren't ignored
4. What's the trade-off between your diversity strategy and individual tree performance?

For a detailed explanation of this question, see [Question 12: Tree Diversity Challenge](L7_3_12_explanation.md).

## Question 13

### Problem Statement
Three Random Forest configurations compete for best performance on a dataset with $1000$ samples and $20$ features:

**Forest Alpha:** $100$ trees, $4$ features per split, $\text{max\_depth} = 8$
**Forest Beta:** $50$ trees, $8$ features per split, $\text{max\_depth} = 12$
**Forest Gamma:** $200$ trees, $3$ features per split, $\text{max\_depth} = 6$

#### Task
1. Which forest will have the highest tree diversity? Calculate the diversity metric
2. If each tree takes $2$ seconds to train, which forest trains fastest?
3. Which forest will likely have the most stable predictions (lowest variance)?
4. If memory is limited to $1000$ tree nodes total, which forest fits best?
5. Calculate the expected number of unique features used across all trees for each forest configuration

For a detailed explanation of this question, see [Question 13: Battle Royale Analysis](L7_3_13_explanation.md).

## Question 14

### Problem Statement
Analyze customer churn data with Random Forest feature importance scores:

**Feature Importance Rankings:**
1. Monthly_Charges: $0.45$
2. Contract_Length: $0.28$
3. Internet_Service: $0.15$
4. Payment_Method: $0.08$
5. Gender: $0.04$

#### Task
1. If you remove the bottom $40\%$ of features, which ones remain?
2. What percentage of total importance do the top $3$ features represent?
3. If you want to reduce features to $60\%$ of original, which features would you keep?
4. Design a feature selection strategy that preserves $90\%$ of importance while reducing features

For a detailed explanation of this question, see [Question 14: Feature Importance Treasure Hunt](L7_3_14_explanation.md).

## Question 15

### Problem Statement
Analyze Random Forest performance over time with $75$ trees showing these accuracy trends:

**Training History:**
- Week 1: $85\%$ accuracy with $25$ trees
- Week 2: $87\%$ accuracy with $50$ trees  
- Week 3: $89\%$ accuracy with $75$ trees

#### Task
1. If the trend continues linearly, what accuracy would you expect with $100$ trees?
2. If you want $92\%$ accuracy, how many trees would you need?
3. What's the accuracy improvement per additional tree based on this data?
4. If each tree takes $3$ minutes to train, how long would it take to reach $92\%$ accuracy?
5. Calculate the correlation coefficient between number of trees and accuracy, and determine if the relationship is statistically significant

For a detailed explanation of this question, see [Question 15: Performance Time Machine](L7_3_15_explanation.md).

## Question 16

### Problem Statement
Solve a Random Forest puzzle with incomplete information:

**Known Information:**
- Dataset: $800$ samples, $15$ features
- Random Forest: $60$ trees, $4$ features per split
- OOB accuracy: $82\%$
- Individual tree accuracy range: $65\%$ - $78\%$

#### Task
1. What's the probability a specific feature is used in a given tree?
2. How many trees, on average, will not contain a specific sample?
3. If you increase features per split to $6$, how does this affect tree diversity?
4. What's the minimum number of trees needed for reliable OOB estimation?
5. Derive the general formula for calculating the expected number of trees that will use a specific feature at least once, and explain how this expectation changes with different feature sampling strategies

For a detailed explanation of this question, see [Question 16: Puzzle Box Solution](L7_3_16_explanation.md).

## Question 17

### Problem Statement
Create visual representations of Random Forest decision boundaries with $4$ trees for a $2$D classification problem:

**Tree 1:** $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Tree 2:** $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B  
**Tree 3:** $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Tree 4:** $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B

#### Task
1. Draw the decision boundary for each tree on a coordinate grid ($X$: $0$-$8$, $Y$: $0$-$8$)
2. Color-code the regions: Class A = Blue, Class B = Red
3. What's the ensemble prediction for point $(4, 3)$?
4. Which tree creates the most interesting geometric pattern?
5. Calculate the percentage of the grid area where the ensemble prediction differs from any individual tree prediction

For a detailed explanation of this question, see [Question 17: Art Gallery Visualization](L7_3_17_explanation.md).

## Question 18

### Problem Statement
Manage a restaurant using Random Forest to predict daily customer count with $6$ trees based on weather, day of week, and special events.

**Daily Predictions ($6$ trees):**
- Monday: $[45, 52, 48, 50, 47, 49]$
- Tuesday: $[38, 42, 40, 41, 39, 43]$
- Wednesday: $[55, 58, 56, 57, 54, 59]$
- Thursday: $[62, 65, 63, 64, 61, 66]$

#### Task
1. Calculate the ensemble prediction and confidence interval for each day
2. If you need to prepare food for $95\%$ of predicted customers, how much should you prepare each day?
3. Which day shows the highest prediction uncertainty (variance)?
4. If you can only staff for $3$ days, which days would you prioritize?
5. Calculate the coefficient of variation (CV = standard deviation/mean) for each day to measure relative uncertainty

For a detailed explanation of this question, see [Question 18: Restaurant Optimization](L7_3_18_explanation.md).

## Question 19

### Problem Statement
Evaluate sports team player performance using Random Forest with $8$ trees and $5$ skills:

**Player Evaluation Scores ($8$ trees):**
**Player Alpha:** $[85, 88, 87, 86, 89, 84, 87, 88]$
**Player Beta:** $[92, 89, 91, 90, 88, 93, 89, 91]$
**Player Gamma:** $[78, 82, 80, 79, 81, 77, 80, 79]$

#### Task
1. Calculate each player's ensemble score and consistency rating
2. If you need $2$ players and value consistency over peak performance, who do you choose?
3. Which player has the highest "upside potential" (highest individual tree score)?
4. Design a scoring system that weights ensemble average ($70\%$) and consistency ($30\%$)
5. Calculate the Sharpe ratio for each player using the formula: $\text{Sharpe Ratio} = \frac{\text{Mean Score}}{\text{Standard Deviation}}$

For a detailed explanation of this question, see [Question 19: Sports Team Strategy](L7_3_19_explanation.md).

## Question 20

### Problem Statement
Escape a Random Forest escape room by solving ensemble puzzles with $4$ doors controlled by different trees:

**Tree Predictions:**
- Door 1: $[0.8, 0.7, 0.9, 0.8, 0.7]$ ($5$ trees)
- Door 2: $[0.3, 0.4, 0.2, 0.3, 0.4]$ ($5$ trees)
- Door 3: $[0.6, 0.7, 0.5, 0.6, 0.7]$ ($5$ trees)
- Door 4: $[0.9, 0.8, 0.9, 0.8, 0.9]$ ($5$ trees)

You need $3$ safe doors to escape.

#### Task
1. Calculate the ensemble safety probability for each door
2. If "safe" means probability $> 0.6$, which doors can you use?
3. What's the probability that you can escape (at least $3$ safe doors)?
4. If you can only check $2$ doors, which combination maximizes escape probability?
5. Calculate the probability of successful escape if you randomly choose 3 doors without checking their probabilities first

For a detailed explanation of this question, see [Question 20: Escape Room Puzzles](L7_3_20_explanation.md).

## Question 21

### Problem Statement
Predict rainfall probability using Random Forest with $10$ trees considering temperature, humidity, pressure, and wind speed.

**Daily Rainfall Predictions ($10$ trees):**
- Day 1: $[0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2]$
- Day 2: $[0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7]$
- Day 3: $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4]$

#### Task
1. Calculate ensemble rainfall probability and uncertainty for each day
2. If you need to issue a rain warning for probabilities $> 0.5$, which days get warnings?
3. Which day has the most reliable prediction (lowest variance)?
4. If you can only make one prediction, which day would you be most confident about?
5. Calculate the entropy of predictions for each day using $H = -\sum p_i \log_2(p_i)$ to measure prediction uncertainty

For a detailed explanation of this question, see [Question 21: Weather Station System](L7_3_21_explanation.md).

## Question 22

### Problem Statement
Predict stock performance using Random Forest with $12$ trees evaluating stocks based on market indicators.

**Stock Performance Predictions ($12$ trees):**
**Tech Stock:** $[0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8]$
**Energy Stock:** $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4]$
**Healthcare Stock:** $[0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5]$

#### Task
1. Calculate ensemble performance score and consistency for each stock
2. If you can invest in $2$ stocks and want to minimize risk, which do you choose?
3. Which stock shows the highest potential return (highest ensemble score)?
4. Design a risk-adjusted scoring system: $\text{Score} = \text{Ensemble\_Score} \times (1 - \text{Variance})$
5. Calculate the Value at Risk (VaR) at 95% confidence level for each stock using the 5th percentile of predictions

For a detailed explanation of this question, see [Question 22: Investment Portfolio Strategy](L7_3_22_explanation.md).

## Question 23

### Problem Statement
Investigate fraud detection using Random Forest with $7$ trees giving fraud probabilities for $4$ suspicious transactions:

**Transaction A:** $[0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3]$
**Transaction B:** $[0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7]$
**Transaction C:** $[0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4]$
**Transaction D:** $[0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]$

#### Task
1. Calculate the ensemble fraud probability for each transaction
2. If fraud threshold is $0.5$, which transactions are flagged as suspicious?
3. Which transaction shows the highest disagreement among trees (highest variance)?
4. If you could only investigate $2$ transactions, which would you prioritize and why?
5. Calculate the confidence interval (95%) for each transaction's fraud probability using the t-distribution

For a detailed explanation of this question, see [Question 23: Detective Game Analysis](L7_3_23_explanation.md).

## Question 24

### Problem Statement
Build a Random Forest for medical diagnosis with $500$ patients, $30$ medical features, binary diagnosis (Healthy/Sick), and maximum $50$ trees due to computational limits.

#### Task
1. Calculate the optimal number of features per split for maximum diversity
2. If you want each feature to be used in at least $80\%$ of trees, how many trees do you need?
3. Design a feature sampling strategy that ensures rare but important features aren't ignored
4. What's the trade-off between your diversity strategy and individual tree performance?
5. Derive the general formula for calculating the expected number of unique features used across all trees, and explain how this expectation changes with different numbers of trees and feature sampling strategies

For a detailed explanation of this question, see [Question 24: Diversity Challenge Strategy](L7_3_24_explanation.md).

## Question 25

### Problem Statement
Compare three Random Forest configurations competing for best performance on a dataset with $1000$ samples and $20$ features:

**Forest Alpha:** $100$ trees, $4$ features per split, $\text{max\_depth} = 8$
**Forest Beta:** $50$ trees, $8$ features per split, $\text{max\_depth} = 12$
**Forest Gamma:** $200$ trees, $3$ features per split, $\text{max\_depth} = 6$

#### Task
1. Which forest will have the highest tree diversity? Calculate the diversity metric
2. If each tree takes $2$ seconds to train, which forest trains fastest?
3. Which forest will likely have the most stable predictions (lowest variance)?
4. If memory is limited to $1000$ tree nodes total, which forest fits best?
5. Calculate the expected number of trees that will use a specific feature at least once for each configuration

For a detailed explanation of this question, see [Question 25: Battle Royale Comparison](L7_3_25_explanation.md).

## Question 26

### Problem Statement
Random Forest combines bagging with feature subsampling to create diverse trees.

#### Task
1. Explain how Random Forest creates diversity among its trees
2. Describe the relationship between Random Forest and bagging
3. Why is feature subsampling important in Random Forest?
4. How does Random Forest differ from simple bagging of decision trees?

For a detailed explanation of this question, see [Question 26: Random Forest Foundations](L7_3_26_explanation.md).

## Question 27

### Problem Statement
Feature subsampling in Random Forest affects tree diversity and performance.

#### Task
1. For a dataset with $d=64$ total features, what would be a commonly recommended number of features $m$ to consider at each split?
2. What is the primary goal of limiting the number of features considered at each split?
3. What happens to the Random Forest algorithm if you set the number of features to consider at each split, $m$, equal to the total number of features, $d$?
4. How does using a very small $m$ (e.g., $m=1$) impact the bias and variance of the individual trees in the forest?

For a detailed explanation of this question, see [Question 27: Feature Subsampling Analysis](L7_3_27_explanation.md).

## Question 28

### Problem Statement
Random Forest uses different voting strategies for making predictions.

#### Task
1. What is the difference between hard voting (majority vote) and soft voting (averaging probabilities)?
2. When would you prefer soft voting over hard voting?
3. A Random Forest with 5 trees is used for a binary classification task. The individual trees predict the following probabilities for class 1: $[0.8, 0.4, 0.45, 0.9, 0.6]$. What is the final prediction using hard voting (with a 0.5 threshold) and soft voting?
4. What is the advantage of ensemble voting over a single tree's prediction?

For a detailed explanation of this question, see [Question 28: Voting Strategies](L7_3_28_explanation.md).

## Question 29

### Problem Statement
Out-of-bag (OOB) estimation provides an internal validation metric for Random Forest.

#### Task
1. What is a major advantage of using OOB estimation over traditional cross-validation?
2. For a large dataset, approximately what percentage of the data is out-of-bag for any given tree?
3. If a single data point is used to test 35 out of 100 trees in the forest (i.e., it was OOB for those 35 trees), how is its OOB prediction calculated?

For a detailed explanation of this question, see [Question 29: Out-of-Bag Estimation](L7_3_29_explanation.md).

## Question 30

### Problem Statement
Feature importance in Random Forest measures how significant each variable is for making predictions.

#### Task
1. Briefly explain one common method for calculating feature importance in a Random Forest
2. Why is feature importance from a Random Forest generally considered more reliable than importance from a single decision tree?
3. You have the following feature importances: Feature A: $0.55$, Feature B: $0.25$, Feature C: $0.15$, Feature D: $0.05$. What percentage of the model's predictive power is captured by Features A and B combined?
4. If you wanted to build a simpler model, which feature would you consider removing first and why?

For a detailed explanation of this question, see [Question 30: Feature Importance Analysis](L7_3_30_explanation.md).

## Question 31

### Problem Statement
Compare a Bagging ensemble of deep decision trees with a Random Forest ensemble.

#### Task
1. What is the key algorithmic difference in how trees are constructed in Random Forest versus in Bagging?
2. In a dataset with one very strong, dominant predictor feature and several moderately useful features, which of the two ensembles would likely build more diverse trees? Explain why.
3. Which of the two methods is designed to more effectively reduce the correlation between the trees in the ensemble?
4. If both ensembles use the same number of trees, which one would you expect to have lower variance in its predictions? Why?

For a detailed explanation of this question, see [Question 31: Bagging vs. Random Forest](L7_3_31_explanation.md).