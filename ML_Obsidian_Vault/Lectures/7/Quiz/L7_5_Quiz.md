# Lecture 7.5: Random Forest Deep Dive Quiz

## Overview
This quiz contains 34 comprehensive questions covering Random Forest as an ensemble method, including tree diversity mechanisms, feature subsampling strategies, bagging integration, voting strategies (hard vs soft), out-of-bag estimation, feature importance analysis, ensemble prediction analysis, configuration optimization, decision boundary visualization, and practical applications across various domains. Topics include mathematical foundations of ensemble diversity, probability calculations for feature selection, voting mechanism comparisons, OOB validation strategies, feature importance computation, and real-world Random Forest deployment scenarios.

## Question 1

### Problem Statement
Random Forest combines bagging with feature subsampling.

#### Task
1. How does Random Forest create diversity among trees?
2. What is the relationship between Random Forest and bagging?
3. Why is feature subsampling important in Random Forest?
4. How does Random Forest differ from a simple bagging of decision trees?

For a detailed explanation of this question, see [Question 1: Random Forest Foundations](L7_5_1_explanation.md).

## Question 2

### Problem Statement
Feature subsampling in Random Forest affects tree diversity.

#### Task
1. If you have $20$ features and consider $5$ at each split, what is the probability a specific feature is used?
2. How does this probability change if you increase the number of features considered?
3. What is the tradeoff between feature subsampling and tree performance?
4. How do you choose the optimal number of features to consider?

For a detailed explanation of this question, see [Question 2: Feature Subsampling Analysis](L7_5_2_explanation.md).

## Question 3

### Problem Statement
Random Forest uses different voting strategies for predictions.

#### Task
1. What is the difference between hard voting and soft voting?
2. When would you prefer soft voting over hard voting?
3. How does Random Forest handle probability estimates?
4. What is the advantage of ensemble voting over single tree predictions?

For a detailed explanation of this question, see [Question 3: Voting Strategies](L7_5_3_explanation.md).

## Question 4

### Problem Statement
Out-of-bag estimation provides internal validation for Random Forest.

#### Task
1. How does out-of-bag estimation work?
2. What is the advantage of OOB estimation over cross-validation?
3. When might OOB estimation not be reliable?
4. How does OOB estimation help with model selection?

For a detailed explanation of this question, see [Question 4: Out-of-Bag Estimation](L7_5_4_explanation.md).

## Question 5

### Problem Statement
Feature importance in Random Forest measures variable significance.

#### Task
1. How is feature importance calculated in Random Forest?
2. Why is Random Forest feature importance more reliable than single tree importance?
3. What are the limitations of feature importance measures?
4. How can you use feature importance for feature selection?

For a detailed explanation of this question, see [Question 5: Feature Importance Analysis](L7_5_5_explanation.md).

## Question 6

### Problem Statement
Analyze a Random Forest with $5$ trees for a binary classification problem.

#### Task
Given the following predictions from $5$ trees for $3$ test samples:

**Sample 1:**
- Tree 1: Class 0 (confidence: $0.8$)
- Tree 2: Class 1 (confidence: $0.6$)
- Tree 3: Class 0 (confidence: $0.9$)
- Tree 4: Class 1 (confidence: $0.7$)
- Tree 5: Class 0 (confidence: $0.85$)

**Sample 2:**
- Tree 1: Class 1 (confidence: $0.95$)
- Tree 2: Class 1 (confidence: $0.88$)
- Tree 3: Class 1 (confidence: $0.92$)
- Tree 4: Class 1 (confidence: $0.89$)
- Tree 5: Class 1 (confidence: $0.91$)

**Sample 3:**
- Tree 1: Class 0 (confidence: $0.55$)
- Tree 2: Class 1 (confidence: $0.65$)
- Tree 3: Class 0 (confidence: $0.45$)
- Tree 4: Class 1 (confidence: $0.75$)
- Tree 5: Class 0 (confidence: $0.60$)

1. What are the final predictions using hard voting?
2. What are the final predictions using soft voting?
3. Which sample has the highest confidence in the ensemble prediction?
4. If you needed to make a decision with high confidence, which sample would you trust most?

For a detailed explanation of this question, see [Question 6: Ensemble Prediction Analysis](L7_5_6_explanation.md).

## Question 7

### Problem Statement
Design a Random Forest configuration for a specific problem.

#### Task
You have a dataset with:
- $1000$ samples
- $25$ features
- Binary classification problem
- Computational budget allows maximum $100$ trees

**Design Requirements:**
1. Calculate the optimal number of features to consider at each split
2. Determine the minimum number of trees needed for reliable OOB estimation
3. If you want $95\%$ confidence that a feature is selected at least once, how many trees do you need?
4. What would be the expected number of unique features used across all trees?

For a detailed explanation of this question, see [Question 7: Configuration Optimization](L7_5_7_explanation.md).

## Question 8

### Problem Statement
Analyze Random Forest performance under different conditions.

#### Task
Consider a Random Forest with the following characteristics:
- $50$ trees
- $10$ features per split (out of $100$ total features)
- Bootstrap sample size = $632$ samples ($63.2\%$ of $1000$ total samples)

**Questions:**
1. What is the probability that a specific sample appears in the bootstrap sample for a given tree?
2. How many trees, on average, will not contain a specific sample (OOB trees)?
3. If you increase the number of features per split to $20$, how does this affect tree diversity?
4. Calculate the expected number of trees that will use a specific feature at least once.

For a detailed explanation of this question, see [Question 8: Performance Analysis](L7_5_8_explanation.md).

## Question 9

### Problem Statement
Compare different Random Forest configurations.

#### Task
You have three Random Forest configurations:

**Configuration A:** $100$ trees, $5$ features per split, $\text{max\_depth} = 10$
**Configuration B:** $50$ trees, $10$ features per split, $\text{max\_depth} = 15$  
**Configuration C:** $200$ trees, $3$ features per split, $\text{max\_depth} = 8$

**Questions:**
1. Which configuration will likely have the highest tree diversity?
2. Which configuration will be fastest to train?
3. Which configuration will likely have the lowest variance in predictions?
4. If you have limited memory, which configuration would you choose?

For a detailed explanation of this question, see [Question 9: Configuration Comparison](L7_5_9_explanation.md).

## Question 10

### Problem Statement
Analyze Random Forest decision boundaries.

#### Task
Consider a $2$D classification problem with features $X$ and $Y$. You have a Random Forest with $3$ trees:

**Tree 1:** Splits on $X$ at $x = 5$, then $Y$ at $y = 3$
**Tree 2:** Splits on $Y$ at $y = 4$, then $X$ at $x = 6$  
**Tree 3:** Splits on $X$ at $x = 4$, then $Y$ at $y = 2$

**Questions:**
1. Draw the decision boundaries for each tree
2. What is the prediction for point $(3, 1)$?
3. What is the prediction for point $(7, 5)$?
4. How does the ensemble decision boundary differ from individual tree boundaries?

For a detailed explanation of this question, see [Question 10: Decision Boundary Analysis](L7_5_10_explanation.md).

## Question 11

### Problem Statement
Create a "Random Forest Detective" game to solve a mystery using ensemble predictions.

#### Task
You're investigating a fraud case and have a Random Forest with $7$ trees. Each tree gives a fraud probability for $4$ suspicious transactions:

**Transaction A:** $[0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3]$
**Transaction B:** $[0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7]$
**Transaction C:** $[0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4]$
**Transaction D:** $[0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]$

**Your Mission:**
1. Calculate the ensemble fraud probability for each transaction
2. If fraud threshold is $0.5$, which transactions are flagged as suspicious?
3. Which transaction shows the highest disagreement among trees (highest variance)?
4. If you could only investigate $2$ transactions, which would you prioritize and why?

For a detailed explanation of this question, see [Question 11: Fraud Detection Game](L7_5_11_explanation.md).

## Question 12

### Problem Statement
Design a "Tree Diversity Challenge" to maximize Random Forest performance.

#### Task
You're building a Random Forest for a medical diagnosis system with $30$ features. Your goal is to maximize tree diversity while maintaining individual tree quality.

**Challenge Setup:**
- Dataset: $500$ patients, $30$ medical features
- Target: Binary diagnosis (Healthy/Sick)
- Constraint: Maximum $50$ trees due to computational limits

**Your Strategy:**
1. Calculate the optimal number of features per split for maximum diversity
2. If you want each feature to be used in at least $80\%$ of trees, how many trees do you need?
3. Design a feature sampling strategy that ensures rare but important features aren't ignored
4. What's the trade-off between your diversity strategy and individual tree performance?

For a detailed explanation of this question, see [Question 12: Tree Diversity Challenge](L7_5_12_explanation.md).

## Question 13

### Problem Statement
Create a "Random Forest Battle Royale" comparing different ensemble strategies.

#### Task
Three Random Forest configurations are competing for the best performance on a dataset with $1000$ samples and $20$ features:

**Forest Alpha:** $100$ trees, $4$ features per split, $\text{max\_depth} = 8$
**Forest Beta:** $50$ trees, $8$ features per split, $\text{max\_depth} = 12$
**Forest Gamma:** $200$ trees, $3$ features per split, $\text{max\_depth} = 6$

**Battle Questions:**
1. Which forest will have the highest tree diversity? Calculate the diversity metric
2. If each tree takes $2$ seconds to train, which forest trains fastest?
3. Which forest will likely have the most stable predictions (lowest variance)?
4. If memory is limited to $1000$ tree nodes total, which forest fits best?

For a detailed explanation of this question, see [Question 13: Battle Royale Analysis](L7_5_13_explanation.md).

## Question 14

### Problem Statement
Design a "Feature Importance Treasure Hunt" using Random Forest insights.

#### Task
You're analyzing customer churn data and your Random Forest reveals these feature importance scores:

**Feature Importance Rankings:**
1. Monthly_Charges: $0.45$
2. Contract_Length: $0.28$
3. Internet_Service: $0.15$
4. Payment_Method: $0.08$
5. Gender: $0.04$

**Your Investigation:**
1. If you remove the bottom $40\%$ of features, which ones remain?
2. What percentage of total importance do the top $3$ features represent?
3. If you want to reduce features to $60\%$ of original, which features would you keep?
4. Design a feature selection strategy that preserves $90\%$ of importance while reducing features

For a detailed explanation of this question, see [Question 14: Feature Importance Treasure Hunt](L7_5_14_explanation.md).

## Question 15

### Problem Statement
Create a "Random Forest Time Machine" to predict past and future performance.

#### Task
You're analyzing a Random Forest's performance over time. The forest has $75$ trees and shows these accuracy trends:

**Training History:**
- Week 1: $85\%$ accuracy with $25$ trees
- Week 2: $87\%$ accuracy with $50$ trees  
- Week 3: $89\%$ accuracy with $75$ trees

**Your Predictions:**
1. If the trend continues linearly, what accuracy would you expect with $100$ trees?
2. If you want $92\%$ accuracy, how many trees would you need?
3. What's the accuracy improvement per additional tree based on this data?
4. If each tree takes $3$ minutes to train, how long would it take to reach $92\%$ accuracy?

For a detailed explanation of this question, see [Question 15: Performance Time Machine](L7_5_15_explanation.md).

## Question 16

### Problem Statement
Design a "Random Forest Puzzle Box" with missing information.

#### Task
You're given a Random Forest puzzle with incomplete information. Solve the missing pieces:

**Known Information:**
- Dataset: $800$ samples, $15$ features
- Random Forest: $60$ trees, $4$ features per split
- OOB accuracy: $82\%$
- Individual tree accuracy range: $65\%$ - $78\%$

**Missing Pieces to Find:**
1. What's the probability a specific feature is used in a given tree?
2. How many trees, on average, will not contain a specific sample?
3. If you increase features per split to $6$, how does this affect tree diversity?
4. What's the minimum number of trees needed for reliable OOB estimation?

For a detailed explanation of this question, see [Question 16: Puzzle Box Solution](L7_5_16_explanation.md).

## Question 17

### Problem Statement
Create a "Random Forest Art Gallery" visualizing ensemble decision boundaries.

#### Task
You're an artist creating visual representations of Random Forest decision boundaries. Your forest has $4$ trees for a $2$D classification problem:

**Tree 1:** $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Tree 2:** $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B  
**Tree 3:** $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Tree 4:** $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B

**Artistic Challenge:**
1. Draw the decision boundary for each tree on a coordinate grid ($X$: $0$-$8$, $Y$: $0$-$8$)
2. Color-code the regions: Class A = Blue, Class B = Red
3. What's the ensemble prediction for point $(4, 3)$?
4. Which tree creates the most interesting geometric pattern?

For a detailed explanation of this question, see [Question 17: Art Gallery Visualization](L7_5_17_explanation.md).

## Question 18

### Problem Statement
Design a "Random Forest Restaurant" menu optimization system.

#### Task
You're managing a restaurant and using Random Forest to predict daily customer count. Your forest has $6$ trees and predicts based on weather, day of week, and special events.

**Daily Predictions ($6$ trees):**
- Monday: $[45, 52, 48, 50, 47, 49]$
- Tuesday: $[38, 42, 40, 41, 39, 43]$
- Wednesday: $[55, 58, 56, 57, 54, 59]$
- Thursday: $[62, 65, 63, 64, 61, 66]$

**Restaurant Planning:**
1. Calculate the ensemble prediction and confidence interval for each day
2. If you need to prepare food for $95\%$ of predicted customers, how much should you prepare each day?
3. Which day shows the highest prediction uncertainty (variance)?
4. If you can only staff for $3$ days, which days would you prioritize?

For a detailed explanation of this question, see [Question 18: Restaurant Optimization](L7_5_18_explanation.md).

## Question 19

### Problem Statement
Create a "Random Forest Sports Team" draft strategy.

#### Task
You're a sports team manager using Random Forest to evaluate player performance. Your forest has $8$ trees and evaluates players on $5$ skills:

**Player Evaluation Scores ($8$ trees):**
**Player Alpha:** $[85, 88, 87, 86, 89, 84, 87, 88]$
**Player Beta:** $[92, 89, 91, 90, 88, 93, 89, 91]$
**Player Gamma:** $[78, 82, 80, 79, 81, 77, 80, 79]$

**Draft Strategy:**
1. Calculate each player's ensemble score and consistency rating
2. If you need $2$ players and value consistency over peak performance, who do you choose?
3. Which player has the highest "upside potential" (highest individual tree score)?
4. Design a scoring system that weights ensemble average ($70\%$) and consistency ($30\%$)

For a detailed explanation of this question, see [Question 19: Sports Team Strategy](L7_5_19_explanation.md).

## Question 20

### Problem Statement
Design a "Random Forest Escape Room" with ensemble logic puzzles.

#### Task
You're trapped in a Random Forest escape room! To escape, you must solve ensemble puzzles using only the information provided.

**Room Setup:**
- $4$ doors, each controlled by a different Random Forest tree
- Each tree gives a probability of the door being safe
- You need $3$ safe doors to escape

**Tree Predictions:**
- Door 1: $[0.8, 0.7, 0.9, 0.8, 0.7]$ ($5$ trees)
- Door 2: $[0.3, 0.4, 0.2, 0.3, 0.4]$ ($5$ trees)
- Door 3: $[0.6, 0.7, 0.5, 0.6, 0.7]$ ($5$ trees)
- Door 4: $[0.9, 0.8, 0.9, 0.8, 0.9]$ ($5$ trees)

**Escape Challenge:**
1. Calculate the ensemble safety probability for each door
2. If "safe" means probability $> 0.6$, which doors can you use?
3. What's the probability that you can escape (at least $3$ safe doors)?
4. If you can only check $2$ doors, which combination maximizes escape probability?

For a detailed explanation of this question, see [Question 20: Escape Room Puzzles](L7_5_20_explanation.md).

## Question 21

### Problem Statement
Create a "Random Forest Weather Station" prediction system.

#### Task
You're a meteorologist using Random Forest to predict rainfall probability. Your forest has $10$ trees and considers temperature, humidity, pressure, and wind speed.

**Daily Rainfall Predictions ($10$ trees):**
- Day 1: $[0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2]$
- Day 2: $[0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7]$
- Day 3: $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4]$

**Weather Forecasting:**
1. Calculate ensemble rainfall probability and uncertainty for each day
2. If you need to issue a rain warning for probabilities $> 0.5$, which days get warnings?
3. Which day has the most reliable prediction (lowest variance)?
4. If you can only make one prediction, which day would you be most confident about?

For a detailed explanation of this question, see [Question 21: Weather Station System](L7_5_21_explanation.md).

## Question 22

### Problem Statement
Design a "Random Forest Investment Portfolio" optimization strategy.

#### Task
You're a financial advisor using Random Forest to predict stock performance. Your forest has $12$ trees and evaluates stocks based on market indicators.

**Stock Performance Predictions ($12$ trees):**
**Tech Stock:** $[0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8]$
**Energy Stock:** $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4]$
**Healthcare Stock:** $[0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5]$

**Portfolio Strategy:**
1. Calculate ensemble performance score and consistency for each stock
2. If you can invest in $2$ stocks and want to minimize risk, which do you choose?
3. Which stock shows the highest potential return (highest ensemble score)?
4. Design a risk-adjusted scoring system: $\text{Score} = \text{Ensemble\_Score} \times (1 - \text{Variance})$

For a detailed explanation of this question, see [Question 22: Investment Portfolio Strategy](L7_5_22_explanation.md).

## Question 23

### Problem Statement
Create a "Random Forest Detective" game to solve a mystery using ensemble predictions.

#### Task
You're investigating a fraud case and have a Random Forest with $7$ trees. Each tree gives a fraud probability for $4$ suspicious transactions:

**Transaction A:** $[0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3]$
**Transaction B:** $[0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7]$
**Transaction C:** $[0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4]$
**Transaction D:** $[0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]$

**Your Mission:**
1. Calculate the ensemble fraud probability for each transaction
2. If fraud threshold is $0.5$, which transactions are flagged as suspicious?
3. Which transaction shows the highest disagreement among trees (highest variance)?
4. If you could only investigate $2$ transactions, which would you prioritize and why?

For a detailed explanation of this question, see [Question 23: Detective Game Analysis](L7_5_23_explanation.md).

## Question 24

### Problem Statement
Design a "Tree Diversity Challenge" to maximize Random Forest performance.

#### Task
You're building a Random Forest for a medical diagnosis system with $30$ features. Your goal is to maximize tree diversity while maintaining individual tree quality.

**Challenge Setup:**
- Dataset: $500$ patients, $30$ medical features
- Target: Binary diagnosis (Healthy/Sick)
- Constraint: Maximum $50$ trees due to computational limits

**Your Strategy:**
1. Calculate the optimal number of features per split for maximum diversity
2. If you want each feature to be used in at least $80\%$ of trees, how many trees do you need?
3. Design a feature sampling strategy that ensures rare but important features aren't ignored
4. What's the trade-off between your diversity strategy and individual tree performance?

For a detailed explanation of this question, see [Question 24: Diversity Challenge Strategy](L7_5_24_explanation.md).

## Question 25

### Problem Statement
Create a "Random Forest Battle Royale" comparing different ensemble strategies.

#### Task
Three Random Forest configurations are competing for the best performance on a dataset with $1000$ samples and $20$ features:

**Forest Alpha:** $100$ trees, $4$ features per split, $\text{max\_depth} = 8$
**Forest Beta:** $50$ trees, $8$ features per split, $\text{max\_depth} = 12$
**Forest Gamma:** $200$ trees, $3$ features per split, $\text{max\_depth} = 6$

**Battle Questions:**
1. Which forest will have the highest tree diversity? Calculate the diversity metric
2. If each tree takes $2$ seconds to train, which forest trains fastest?
3. Which forest will likely have the most stable predictions (lowest variance)?
4. If memory is limited to $1000$ tree nodes total, which forest fits best?

For a detailed explanation of this question, see [Question 25: Battle Royale Comparison](L7_5_25_explanation.md).

## Question 26

### Problem Statement
Design a "Feature Importance Treasure Hunt" using Random Forest insights.

#### Task
You're analyzing customer churn data and your Random Forest reveals these feature importance scores:

**Feature Importance Rankings:**
1. Monthly_Charges: $0.45$
2. Contract_Length: $0.28$
3. Internet_Service: $0.15$
4. Payment_Method: $0.08$
5. Gender: $0.04$

**Your Investigation:**
1. If you remove the bottom $40\%$ of features, which ones remain?
2. What percentage of total importance do the top $3$ features represent?
3. If you want to reduce features to $60\%$ of original, which features would you keep?
4. Design a feature selection strategy that preserves $90\%$ of importance while reducing features

For a detailed explanation of this question, see [Question 26: Feature Importance Hunt](L7_5_26_explanation.md).

## Question 27

### Problem Statement
Create a "Random Forest Time Machine" to predict past and future performance.

#### Task
You're analyzing a Random Forest's performance over time. The forest has $75$ trees and shows these accuracy trends:

**Training History:**
- Week 1: $85\%$ accuracy with $25$ trees
- Week 2: $87\%$ accuracy with $50$ trees  
- Week 3: $89\%$ accuracy with $75$ trees

**Your Predictions:**
1. If the trend continues linearly, what accuracy would you expect with $100$ trees?
2. If you want $92\%$ accuracy, how many trees would you need?
3. What's the accuracy improvement per additional tree based on this data?
4. If each tree takes $3$ minutes to train, how long would it take to reach $92\%$ accuracy?

For a detailed explanation of this question, see [Question 27: Performance Time Machine](L7_5_27_explanation.md).

## Question 28

### Problem Statement
Design a "Random Forest Puzzle Box" with missing information.

#### Task
You're given a Random Forest puzzle with incomplete information. Solve the missing pieces:

**Known Information:**
- Dataset: $800$ samples, $15$ features
- Random Forest: $60$ trees, $4$ features per split
- OOB accuracy: $82\%$
- Individual tree accuracy range: $65\%$ - $78\%$

**Missing Pieces to Find:**
1. What's the probability a specific feature is used in a given tree?
2. How many trees, on average, will not contain a specific sample?
3. If you increase features per split to $6$, how does this affect tree diversity?
4. What's the minimum number of trees needed for reliable OOB estimation?

For a detailed explanation of this question, see [Question 28: Puzzle Box Solution](L7_5_28_explanation.md).

## Question 29

### Problem Statement
Create a "Random Forest Art Gallery" visualizing ensemble decision boundaries.

#### Task
You're an artist creating visual representations of Random Forest decision boundaries. Your forest has $4$ trees for a $2$D classification problem:

**Tree 1:** $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Tree 2:** $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B  
**Tree 3:** $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Tree 4:** $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B

**Artistic Challenge:**
1. Draw the decision boundary for each tree on a coordinate grid ($X$: $0$-$8$, $Y$: $0$-$8$)
2. Color-code the regions: Class A = Blue, Class B = Red
3. What's the ensemble prediction for point $(4, 3)$?
4. Which tree creates the most interesting geometric pattern?

For a detailed explanation of this question, see [Question 29: Art Gallery Visualization](L7_5_29_explanation.md).

## Question 30

### Problem Statement
Design a "Random Forest Restaurant" menu optimization system.

#### Task
You're managing a restaurant and using Random Forest to predict daily customer count. Your forest has $6$ trees and predicts based on weather, day of week, and special events.

**Daily Predictions ($6$ trees):**
- Monday: $[45, 52, 48, 50, 47, 49]$
- Tuesday: $[38, 42, 40, 41, 39, 43]$
- Wednesday: $[55, 58, 56, 57, 54, 59]$
- Thursday: $[62, 65, 63, 64, 61, 66]$

**Restaurant Planning:**
1. Calculate the ensemble prediction and confidence interval for each day
2. If you need to prepare food for $95\%$ of predicted customers, how much should you prepare each day?
3. Which day shows the highest prediction uncertainty (variance)?
4. If you can only staff for $3$ days, which days would you prioritize?

For a detailed explanation of this question, see [Question 30: Restaurant Optimization](L7_5_30_explanation.md).

## Question 31

### Problem Statement
Create a "Random Forest Sports Team" draft strategy.

#### Task
You're a sports team manager using Random Forest to evaluate player performance. Your forest has $8$ trees and evaluates players on $5$ skills:

**Player Evaluation Scores ($8$ trees):**
**Player Alpha:** $[85, 88, 87, 86, 89, 84, 87, 88]$
**Player Beta:** $[92, 89, 91, 90, 88, 93, 89, 91]$
**Player Gamma:** $[78, 82, 80, 79, 81, 77, 80, 79]$

**Draft Strategy:**
1. Calculate each player's ensemble score and consistency rating
2. If you need $2$ players and value consistency over peak performance, who do you choose?
3. Which player has the highest "upside potential" (highest individual tree score)?
4. Design a scoring system that weights ensemble average ($70\%$) and consistency ($30\%$)

For a detailed explanation of this question, see [Question 31: Sports Team Strategy](L7_5_31_explanation.md).

## Question 32

### Problem Statement
Design a "Random Forest Escape Room" with ensemble logic puzzles.

#### Task
You're trapped in a Random Forest escape room! To escape, you must solve ensemble puzzles using only the information provided.

**Room Setup:**
- $4$ doors, each controlled by a different Random Forest tree
- Each tree gives a probability of the door being safe
- You need $3$ safe doors to escape

**Tree Predictions:**
- Door 1: $[0.8, 0.7, 0.9, 0.8, 0.7]$ ($5$ trees)
- Door 2: $[0.3, 0.4, 0.2, 0.3, 0.4]$ ($5$ trees)
- Door 3: $[0.6, 0.7, 0.5, 0.6, 0.7]$ ($5$ trees)
- Door 4: $[0.9, 0.8, 0.9, 0.8, 0.9]$ ($5$ trees)

**Escape Challenge:**
1. Calculate the ensemble safety probability for each door
2. If "safe" means probability $> 0.6$, which doors can you use?
3. What's the probability that you can escape (at least $3$ safe doors)?
4. If you can only check $2$ doors, which combination maximizes escape probability?

For a detailed explanation of this question, see [Question 32: Escape Room Puzzles](L7_5_32_explanation.md).

## Question 33

### Problem Statement
Create a "Random Forest Weather Station" prediction system.

#### Task
You're a meteorologist using Random Forest to predict rainfall probability. Your forest has $10$ trees and considers temperature, humidity, pressure, and wind speed.

**Daily Rainfall Predictions ($10$ trees):**
- Day 1: $[0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2]$
- Day 2: $[0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7]$
- Day 3: $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4]$

**Weather Forecasting:**
1. Calculate ensemble rainfall probability and uncertainty for each day
2. If you need to issue a rain warning for probabilities $> 0.5$, which days get warnings?
3. Which day has the most reliable prediction (lowest variance)?
4. If you can only make one prediction, which day would you be most confident about?

For a detailed explanation of this question, see [Question 33: Weather Station System](L7_5_33_explanation.md).

## Question 34

### Problem Statement
Design a "Random Forest Investment Portfolio" optimization strategy.

#### Task
You're a financial advisor using Random Forest to predict stock performance. Your forest has $12$ trees and evaluates stocks based on market indicators.

**Stock Performance Predictions ($12$ trees):**
**Tech Stock:** $[0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8]$
**Energy Stock:** $[0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4]$
**Healthcare Stock:** $[0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5]$

**Portfolio Strategy:**
1. Calculate ensemble performance score and consistency for each stock
2. If you can invest in $2$ stocks and want to minimize risk, which do you choose?
3. Which stock shows the highest potential return (highest ensemble score)?
4. Design a risk-adjusted scoring system: $\text{Score} = \text{Ensemble\_Score} \times (1 - \text{Variance})$

For a detailed explanation of this question, see [Question 34: Investment Portfolio Strategy](L7_5_34_explanation.md).
