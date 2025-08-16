# Lecture 7.5: Random Forest Deep Dive Quiz

## Overview
This quiz tests your understanding of Random Forest as an ensemble method, including tree diversity, feature subsampling, bagging integration, and voting strategies.

## Question 1

### Problem Statement
Random Forest combines bagging with feature subsampling.

#### Task
1. How does Random Forest create diversity among trees?
2. What is the relationship between Random Forest and bagging?
3. Why is feature subsampling important in Random Forest?
4. How does Random Forest differ from a simple bagging of decision trees?

**Answer**:
1. Random Forest creates diversity through bootstrap sampling and random feature selection at each split
2. Random Forest is an extension of bagging that adds feature subsampling for additional diversity
3. Feature subsampling prevents all trees from using the same features, creating more diverse trees
4. Random Forest uses feature subsampling while simple bagging uses all available features

## Question 2

### Problem Statement
Feature subsampling in Random Forest affects tree diversity.

#### Task
1. If you have 20 features and consider 5 at each split, what is the probability a specific feature is used?
2. How does this probability change if you increase the number of features considered?
3. What is the tradeoff between feature subsampling and tree performance?
4. How do you choose the optimal number of features to consider?

**Answer**:
1. Probability = 5/20 = 0.25 or 25%
2. Higher number of features considered increases the probability but reduces diversity
3. Tradeoff: fewer features = more diversity but potentially worse individual tree performance
4. Optimal choice: typically √(number of features) for classification, number of features/3 for regression

## Question 3

### Problem Statement
Random Forest uses different voting strategies for predictions.

#### Task
1. What is the difference between hard voting and soft voting?
2. When would you prefer soft voting over hard voting?
3. How does Random Forest handle probability estimates?
4. What is the advantage of ensemble voting over single tree predictions?

**Answer**:
1. Hard voting counts class predictions, soft voting averages class probabilities
2. Prefer soft voting when you need probability estimates or confidence scores
3. Random Forest averages probability estimates from all trees for each class
4. Ensemble voting reduces variance, is more robust to individual tree errors, and provides better generalization

## Question 4

### Problem Statement
Out-of-bag estimation provides internal validation for Random Forest.

#### Task
1. How does out-of-bag estimation work?
2. What is the advantage of OOB estimation over cross-validation?
3. When might OOB estimation not be reliable?
4. How does OOB estimation help with model selection?

**Answer**:
1. OOB estimation uses trees where a sample was not in the bootstrap sample to make predictions for that sample
2. OOB is faster (no separate validation) and uses all data for training
3. OOB might not be reliable with very small datasets or when the number of trees is small
4. OOB helps select optimal parameters (number of trees, features per split) without external validation

## Question 5

### Problem Statement
Feature importance in Random Forest measures variable significance.

#### Task
1. How is feature importance calculated in Random Forest?
2. Why is Random Forest feature importance more reliable than single tree importance?
3. What are the limitations of feature importance measures?
4. How can you use feature importance for feature selection?

**Answer**:
1. Feature importance is calculated by averaging impurity reduction across all trees when that feature is used for splitting
2. Random Forest importance is more reliable because it averages across many trees, reducing individual tree biases
3. Limitations: correlation between features can inflate importance, importance doesn't indicate causality
4. Feature importance can be used to select top features or remove low-importance features to reduce dimensionality

## Question 6

### Problem Statement
Analyze a Random Forest with 5 trees for a binary classification problem.

#### Task
Given the following predictions from 5 trees for 3 test samples:

**Sample 1:**
- Tree 1: Class 0 (confidence: 0.8)
- Tree 2: Class 1 (confidence: 0.6)
- Tree 3: Class 0 (confidence: 0.9)
- Tree 4: Class 1 (confidence: 0.7)
- Tree 5: Class 0 (confidence: 0.85)

**Sample 2:**
- Tree 1: Class 1 (confidence: 0.95)
- Tree 2: Class 1 (confidence: 0.88)
- Tree 3: Class 1 (confidence: 0.92)
- Tree 4: Class 1 (confidence: 0.89)
- Tree 5: Class 1 (confidence: 0.91)

**Sample 3:**
- Tree 1: Class 0 (confidence: 0.55)
- Tree 2: Class 1 (confidence: 0.65)
- Tree 3: Class 0 (confidence: 0.45)
- Tree 4: Class 1 (confidence: 0.75)
- Tree 5: Class 0 (confidence: 0.60)

1. What are the final predictions using hard voting?
2. What are the final predictions using soft voting?
3. Which sample has the highest confidence in the ensemble prediction?
4. If you needed to make a decision with high confidence, which sample would you trust most?

**Answer**:
1. Hard voting: Sample 1: Class 0 (3 votes), Sample 2: Class 1 (5 votes), Sample 3: Class 0 (3 votes)
2. Soft voting: Sample 1: Class 0 (avg confidence 0.77), Sample 2: Class 1 (avg confidence 0.91), Sample 3: Class 0 (avg confidence 0.60)
3. Sample 2 has the highest confidence (0.91)
4. Sample 2 - highest agreement among trees and highest confidence

## Question 7

### Problem Statement
Design a Random Forest configuration for a specific problem.

#### Task
You have a dataset with:
- 1000 samples
- 25 features
- Binary classification problem
- Computational budget allows maximum 100 trees

**Design Requirements:**
1. Calculate the optimal number of features to consider at each split
2. Determine the minimum number of trees needed for reliable OOB estimation
3. If you want 95% confidence that a feature is selected at least once, how many trees do you need?
4. What would be the expected number of unique features used across all trees?

**Answer**:
1. Optimal features per split: √25 = 5 features
2. Minimum trees for reliable OOB: typically 50-100, so 100 trees is sufficient
3. For 95% confidence: P(feature not selected) = (20/25)^100 ≈ 0.0001, so 100 trees gives >99.99% confidence
4. Expected unique features: Each tree uses 5 features, with 100 trees and 25 total features, most features will be used multiple times

## Question 8

### Problem Statement
Analyze Random Forest performance under different conditions.

#### Task
Consider a Random Forest with the following characteristics:
- 50 trees
- 10 features per split (out of 100 total features)
- Bootstrap sample size = 632 samples (63.2% of 1000 total samples)

**Questions:**
1. What is the probability that a specific sample appears in the bootstrap sample for a given tree?
2. How many trees, on average, will not contain a specific sample (OOB trees)?
3. If you increase the number of features per split to 20, how does this affect tree diversity?
4. Calculate the expected number of trees that will use a specific feature at least once.

**Answer**:
1. P(sample in bootstrap) = 1 - (1 - 1/1000)^1000 ≈ 1 - e^(-1) ≈ 0.632
2. OOB trees = 50 × (1 - 0.632) ≈ 18.4 trees
3. Increasing features reduces diversity as more trees will use the same features
4. P(feature used in tree) = 1 - (90/100)^10 ≈ 0.651, so expected trees using feature = 50 × 0.651 ≈ 32.6 trees

## Question 9

### Problem Statement
Compare different Random Forest configurations.

#### Task
You have three Random Forest configurations:

**Configuration A:** 100 trees, 5 features per split, max_depth=10
**Configuration B:** 50 trees, 10 features per split, max_depth=15  
**Configuration C:** 200 trees, 3 features per split, max_depth=8

**Questions:**
1. Which configuration will likely have the highest tree diversity?
2. Which configuration will be fastest to train?
3. Which configuration will likely have the lowest variance in predictions?
4. If you have limited memory, which configuration would you choose?

**Answer**:
1. Configuration C (3 features per split creates highest diversity)
2. Configuration B (fewer trees, moderate depth)
3. Configuration C (most trees, lowest variance)
4. Configuration B (moderate number of trees, moderate depth)

## Question 10

### Problem Statement
Analyze Random Forest decision boundaries.

#### Task
Consider a 2D classification problem with features X and Y. You have a Random Forest with 3 trees:

**Tree 1:** Splits on X at x=5, then Y at y=3
**Tree 2:** Splits on Y at y=4, then X at x=6  
**Tree 3:** Splits on X at x=4, then Y at y=2

**Questions:**
1. Draw the decision boundaries for each tree
2. What is the prediction for point (3, 1)?
3. What is the prediction for point (7, 5)?
4. How does the ensemble decision boundary differ from individual tree boundaries?

**Answer**:
1. Tree boundaries create rectangular regions; ensemble combines these regions
2. Point (3,1): Tree 1: Class depends on Y split, Tree 2: Class depends on X split, Tree 3: Class depends on Y split
3. Point (7,5): Tree 1: Class depends on Y split, Tree 2: Class depends on X split, Tree 3: Class depends on Y split
4. Ensemble boundary is smoother and more complex, combining the rectangular regions from all trees

## Question 11

### Problem Statement
Create a "Random Forest Detective" game to solve a mystery using ensemble predictions.

#### Task
You're investigating a fraud case and have a Random Forest with 7 trees. Each tree gives a fraud probability for 4 suspicious transactions:

**Transaction A:** [0.1, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3]
**Transaction B:** [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7]
**Transaction C:** [0.4, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4]
**Transaction D:** [0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]

**Your Mission:**
1. Calculate the ensemble fraud probability for each transaction
2. If fraud threshold is 0.5, which transactions are flagged as suspicious?
3. Which transaction shows the highest disagreement among trees (highest variance)?
4. If you could only investigate 2 transactions, which would you prioritize and why?

**Answer**:
1. A: 0.23, B: 0.80, C: 0.49, D: 0.17
2. Only Transaction B (0.80 > 0.5)
3. Transaction C has highest variance (0.4-0.6 range)
4. Prioritize B (clear fraud) and C (high uncertainty needs investigation)

## Question 12

### Problem Statement
Design a "Tree Diversity Challenge" to maximize Random Forest performance.

#### Task
You're building a Random Forest for a medical diagnosis system with 30 features. Your goal is to maximize tree diversity while maintaining individual tree quality.

**Challenge Setup:**
- Dataset: 500 patients, 30 medical features
- Target: Binary diagnosis (Healthy/Sick)
- Constraint: Maximum 50 trees due to computational limits

**Your Strategy:**
1. Calculate the optimal number of features per split for maximum diversity
2. If you want each feature to be used in at least 80% of trees, how many trees do you need?
3. Design a feature sampling strategy that ensures rare but important features aren't ignored
4. What's the trade-off between your diversity strategy and individual tree performance?

**Answer**:
1. Optimal features per split: √30 ≈ 5-6 features
2. P(feature not used) = (25/30)^50 ≈ 0.0001, so 50 trees gives >99.99% coverage
3. Use weighted sampling or ensure minimum feature usage per tree
4. Higher diversity may reduce individual tree accuracy but improves ensemble robustness

## Question 13

### Problem Statement
Create a "Random Forest Battle Royale" comparing different ensemble strategies.

#### Task
Three Random Forest configurations are competing for the best performance on a dataset with 1000 samples and 20 features:

**Forest Alpha:** 100 trees, 4 features per split, max_depth=8
**Forest Beta:** 50 trees, 8 features per split, max_depth=12
**Forest Gamma:** 200 trees, 3 features per split, max_depth=6

**Battle Questions:**
1. Which forest will have the highest tree diversity? Calculate the diversity metric
2. If each tree takes 2 seconds to train, which forest trains fastest?
3. Which forest will likely have the most stable predictions (lowest variance)?
4. If memory is limited to 1000 tree nodes total, which forest fits best?

**Answer**:
1. Forest Gamma (3 features per split creates highest diversity)
2. Forest Beta (50 trees × 2s = 100s vs Alpha: 200s, Gamma: 400s)
3. Forest Gamma (most trees = lowest variance)
4. Forest Alpha: 100×8=800 nodes, Beta: 50×12=600 nodes, Gamma: 200×6=1200 nodes. Beta fits best.

## Question 14

### Problem Statement
Design a "Feature Importance Treasure Hunt" using Random Forest insights.

#### Task
You're analyzing customer churn data and your Random Forest reveals these feature importance scores:

**Feature Importance Rankings:**
1. Monthly_Charges: 0.45
2. Contract_Length: 0.28
3. Internet_Service: 0.15
4. Payment_Method: 0.08
5. Gender: 0.04

**Your Investigation:**
1. If you remove the bottom 40% of features, which ones remain?
2. What percentage of total importance do the top 3 features represent?
3. If you want to reduce features to 60% of original, which features would you keep?
4. Design a feature selection strategy that preserves 90% of importance while reducing features

**Answer**:
1. Remove Gender (0.04) and Payment_Method (0.08) - keep top 3 features
2. Top 3: 0.45 + 0.28 + 0.15 = 0.88 (88% of total importance)
3. Keep Monthly_Charges and Contract_Length (0.45 + 0.28 = 0.73, which is >60%)
4. Keep top 2 features (Monthly_Charges + Contract_Length = 0.73) and add Internet_Service (0.15) to reach 0.88 (90% of 0.98 total)

## Question 15

### Problem Statement
Create a "Random Forest Time Machine" to predict past and future performance.

#### Task
You're analyzing a Random Forest's performance over time. The forest has 75 trees and shows these accuracy trends:

**Training History:**
- Week 1: 85% accuracy with 25 trees
- Week 2: 87% accuracy with 50 trees  
- Week 3: 89% accuracy with 75 trees

**Your Predictions:**
1. If the trend continues linearly, what accuracy would you expect with 100 trees?
2. If you want 92% accuracy, how many trees would you need?
3. What's the accuracy improvement per additional tree based on this data?
4. If each tree takes 3 minutes to train, how long would it take to reach 92% accuracy?

**Answer**:
1. Linear trend: +2% per 25 trees, so 100 trees = 89% + 2% = 91%
2. Need 92% - 89% = 3% improvement, so 3% ÷ 2% × 25 = 37.5 additional trees. Total: 75 + 38 = 113 trees
3. 2% improvement per 25 trees = 0.08% improvement per tree
4. 38 additional trees × 3 minutes = 114 minutes = 1 hour 54 minutes

## Question 16

### Problem Statement
Design a "Random Forest Puzzle Box" with missing information.

#### Task
You're given a Random Forest puzzle with incomplete information. Solve the missing pieces:

**Known Information:**
- Dataset: 800 samples, 15 features
- Random Forest: 60 trees, 4 features per split
- OOB accuracy: 82%
- Individual tree accuracy range: 65% - 78%

**Missing Pieces to Find:**
1. What's the probability a specific feature is used in a given tree?
2. How many trees, on average, will not contain a specific sample?
3. If you increase features per split to 6, how does this affect tree diversity?
4. What's the minimum number of trees needed for reliable OOB estimation?

**Answer**:
1. P(feature used) = 4/15 = 0.267 (26.7%)
2. P(sample in bootstrap) ≈ 0.632, so OOB trees = 60 × (1-0.632) ≈ 22 trees
3. Increasing to 6 features reduces diversity (more trees use same features)
4. Minimum trees for reliable OOB: typically 50-100, so 60 trees is sufficient

## Question 17

### Problem Statement
Create a "Random Forest Art Gallery" visualizing ensemble decision boundaries.

#### Task
You're an artist creating visual representations of Random Forest decision boundaries. Your forest has 4 trees for a 2D classification problem:

**Tree 1:** X ≤ 3 → Class A, X > 3 → Class B
**Tree 2:** Y ≤ 2 → Class A, Y > 2 → Class B  
**Tree 3:** X ≤ 5 AND Y ≤ 4 → Class A, otherwise Class B
**Tree 4:** X + Y ≤ 6 → Class A, X + Y > 6 → Class B

**Artistic Challenge:**
1. Draw the decision boundary for each tree on a coordinate grid (X: 0-8, Y: 0-8)
2. Color-code the regions: Class A = Blue, Class B = Red
3. What's the ensemble prediction for point (4, 3)?
4. Which tree creates the most interesting geometric pattern?

**Answer**:
1. Tree 1: Vertical line at X=3, Tree 2: Horizontal line at Y=2, Tree 3: Rectangle, Tree 4: Diagonal line
2. Point (4,3): Tree 1: B, Tree 2: B, Tree 3: A, Tree 4: A → Ensemble: A (2 votes) vs B (2 votes) → Tie
3. Tree 4 creates diagonal boundary (most interesting pattern)

## Question 18

### Problem Statement
Design a "Random Forest Restaurant" menu optimization system.

#### Task
You're managing a restaurant and using Random Forest to predict daily customer count. Your forest has 6 trees and predicts based on weather, day of week, and special events.

**Daily Predictions (6 trees):**
- Monday: [45, 52, 48, 50, 47, 49]
- Tuesday: [38, 42, 40, 41, 39, 43]
- Wednesday: [55, 58, 56, 57, 54, 59]
- Thursday: [62, 65, 63, 64, 61, 66]

**Restaurant Planning:**
1. Calculate the ensemble prediction and confidence interval for each day
2. If you need to prepare food for 95% of predicted customers, how much should you prepare each day?
3. Which day shows the highest prediction uncertainty (variance)?
4. If you can only staff for 3 days, which days would you prioritize?

**Answer**:
1. Monday: 48.5 ± 2.5, Tuesday: 40.5 ± 2.0, Wednesday: 56.5 ± 2.0, Thursday: 63.5 ± 2.0
2. 95% confidence: Monday: 51, Tuesday: 42, Wednesday: 58, Thursday: 65
3. Monday has highest variance (45-52 range)
4. Prioritize Thursday (highest demand), Wednesday (high demand), Monday (moderate demand)

## Question 19

### Problem Statement
Create a "Random Forest Sports Team" draft strategy.

#### Task
You're a sports team manager using Random Forest to evaluate player performance. Your forest has 8 trees and evaluates players on 5 skills:

**Player Evaluation Scores (8 trees):**
**Player Alpha:** [85, 88, 87, 86, 89, 84, 87, 88]
**Player Beta:** [92, 89, 91, 90, 88, 93, 89, 91]
**Player Gamma:** [78, 82, 80, 79, 81, 77, 80, 79]

**Draft Strategy:**
1. Calculate each player's ensemble score and consistency rating
2. If you need 2 players and value consistency over peak performance, who do you choose?
3. Which player has the highest "upside potential" (highest individual tree score)?
4. Design a scoring system that weights ensemble average (70%) and consistency (30%)

**Answer**:
1. Alpha: 86.75 ± 1.8, Beta: 90.38 ± 1.6, Gamma: 79.63 ± 1.6
2. Choose Alpha and Gamma (most consistent performers)
3. Player Beta has highest upside (93 from one tree)
4. Weighted scores: Alpha: 86.75×0.7 + (10-1.8)×0.3 = 60.73 + 2.46 = 63.19, Beta: 63.27, Gamma: 55.74

## Question 20

### Problem Statement
Design a "Random Forest Escape Room" with ensemble logic puzzles.

#### Task
You're trapped in a Random Forest escape room! To escape, you must solve ensemble puzzles using only the information provided.

**Room Setup:**
- 4 doors, each controlled by a different Random Forest tree
- Each tree gives a probability of the door being safe
- You need 3 safe doors to escape

**Tree Predictions:**
- Door 1: [0.8, 0.7, 0.9, 0.8, 0.7] (5 trees)
- Door 2: [0.3, 0.4, 0.2, 0.3, 0.4] (5 trees)
- Door 3: [0.6, 0.7, 0.5, 0.6, 0.7] (5 trees)
- Door 4: [0.9, 0.8, 0.9, 0.8, 0.9] (5 trees)

**Escape Challenge:**
1. Calculate the ensemble safety probability for each door
2. If "safe" means probability > 0.6, which doors can you use?
3. What's the probability that you can escape (at least 3 safe doors)?
4. If you can only check 2 doors, which combination maximizes escape probability?

**Answer**:
1. Door 1: 0.78, Door 2: 0.32, Door 3: 0.62, Door 4: 0.86
2. Doors 1, 3, and 4 are safe (>0.6)
3. P(escape) = P(at least 3 safe) = 1 (all 3 safe doors available)
4. Check Doors 1 and 4 (highest probabilities, 0.78 and 0.86)

## Question 21

### Problem Statement
Create a "Random Forest Weather Station" prediction system.

#### Task
You're a meteorologist using Random Forest to predict rainfall probability. Your forest has 10 trees and considers temperature, humidity, pressure, and wind speed.

**Daily Rainfall Predictions (10 trees):**
- Day 1: [0.2, 0.3, 0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2]
- Day 2: [0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7]
- Day 3: [0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4]

**Weather Forecasting:**
1. Calculate ensemble rainfall probability and uncertainty for each day
2. If you need to issue a rain warning for probabilities > 0.5, which days get warnings?
3. Which day has the most reliable prediction (lowest variance)?
4. If you can only make one prediction, which day would you be most confident about?

**Answer**:
1. Day 1: 0.21 ± 0.07, Day 2: 0.71 ± 0.07, Day 3: 0.41 ± 0.07
2. Only Day 2 gets rain warning (0.71 > 0.5)
3. All days have similar variance (0.07), but Day 2 has highest confidence due to high probability
4. Day 2 - highest probability (0.71) with consistent predictions across trees

## Question 22

### Problem Statement
Design a "Random Forest Investment Portfolio" optimization strategy.

#### Task
You're a financial advisor using Random Forest to predict stock performance. Your forest has 12 trees and evaluates stocks based on market indicators.

**Stock Performance Predictions (12 trees):**
**Tech Stock:** [0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8]
**Energy Stock:** [0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4]
**Healthcare Stock:** [0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5]

**Portfolio Strategy:**
1. Calculate ensemble performance score and consistency for each stock
2. If you can invest in 2 stocks and want to minimize risk, which do you choose?
3. Which stock shows the highest potential return (highest ensemble score)?
4. Design a risk-adjusted scoring system: Score = Ensemble_Score × (1 - Variance)

**Answer**:
1. Tech: 0.80 ± 0.07, Energy: 0.40 ± 0.07, Healthcare: 0.60 ± 0.07
2. Choose Tech and Healthcare (highest scores with reasonable consistency)
3. Tech Stock has highest potential return (0.80)
4. Risk-adjusted scores: Tech: 0.80 × (1-0.07) = 0.744, Energy: 0.40 × 0.93 = 0.372, Healthcare: 0.60 × 0.93 = 0.558
