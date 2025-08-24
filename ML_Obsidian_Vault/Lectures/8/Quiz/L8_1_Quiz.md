# Lecture 8.1: Foundations of Feature Selection Quiz

## Overview
This quiz contains 20 questions covering the foundations of feature selection, including motivations, the curse of dimensionality, different selection approaches, and practical applications. All questions are designed to be solvable using pen and paper with concrete examples and calculations.

## Question 1

### Problem Statement
Feature selection is a critical step in the machine learning pipeline that affects multiple aspects of model development.

#### Task
1. What are the three main benefits of feature selection?
2. How does feature selection improve model interpretability?
3. Why is feature selection important for real-time applications?
4. If a model takes $5$ minutes to train with $100$ features, estimate training time with $25$ features (assume linear scaling)
5. Calculate the memory reduction when reducing features from $1000$ to $100$ (assume each feature uses $8$ bytes per sample)
6. A neural network has training time proportional to $O(n^2 \cdot d)$ where $n$ is samples and $d$ is features. If training with $1000$ samples and $50$ features takes $2$ hours, calculate training time for $1000$ samples and $10$ features. What's the percentage improvement?
7. Design a feature selection strategy for a weather prediction model that must run on a smartphone with limited battery 
life

For a detailed explanation of this question, see [Question 1: Feature Selection Fundamentals](L8_1_1_explanation.md).

## Question 2

### Problem Statement
The curse of dimensionality affects model performance as the number of features increases, particularly for distance-based algorithms.

#### Task
1. What is the curse of dimensionality in one sentence?
2. How does the curse affect nearest neighbor algorithms?
3. What happens to the volume of a hypercube as dimensions increase?
4. If you have $1000$ samples in $2$D, how many samples would you need in $10$D for similar density?
5. Calculate the ratio of volume to surface area for a unit hypercube in $2$D vs $10$D
6. In a unit hypercube, the distance between two random points follows $E[d] = \sqrt{\frac{d}{6}}$ where $d$ is dimensions. Calculate the expected distance in $2$D, $5$D, and $10$D. What happens to the ratio of maximum to expected distance as dimensions increase?

For a detailed explanation of this question, see [Question 2: Curse of Dimensionality](L8_1_2_explanation.md).

## Question 3

### Problem Statement
Feature selection approaches can be categorized as supervised (using labels) or unsupervised (without labels), each with different advantages and applications.

#### Task
1. What is the main advantage of supervised feature selection?
2. When would you use unsupervised feature selection?
3. How do you measure feature relevance in unsupervised scenarios?
4. If you have $1000$ samples with $50$ features, how many possible feature subsets exist?
5. Calculate the number of feature subsets with exactly $10$ features from $50$ total features
6. For a dataset with $n$ features, the number of possible feature combinations is $2^n - 1$. If you want to evaluate at least $80\%$ of all possible combinations, what's the minimum number of features you can handle if you can evaluate $1000$ combinations per second and have $1$ hour total?

For a detailed explanation of this question, see [Question 3: Supervised vs Unsupervised Selection](L8_1_3_explanation.md).

## Question 4

### Problem Statement
Feature selection and feature extraction are different approaches to dimensionality reduction with distinct trade-offs.

#### Task
1. What is the key difference between selection and extraction?
2. Which approach preserves original feature interpretability?
3. When would you choose extraction over selection?
4. If you transform features using PCA, is this selection or extraction?
5. Compare the interpretability and computational cost of both approaches
6. PCA reduces dimensions by finding eigenvectors. If you have $100$ features and want to retain $95\%$ of variance, and the eigenvalues are $\lambda_1 = 50$, $\lambda_2 = 30$, $\lambda_3 = 15$, $\lambda_4 = 5$, how many principal components do you need? Calculate the cumulative variance explained.

For a detailed explanation of this question, see [Question 4: Selection vs Extraction](L8_1_4_explanation.md).

## Question 5

### Problem Statement
Consider a dataset with 1000 samples and 100 features where only 20 features are truly relevant to the target variable.

#### Task
1. What percentage of features are irrelevant in this dataset?
2. If each irrelevant feature adds $1\%$ noise, what's the total noise level?
3. How would this affect model performance and training time?
4. What's the signal-to-noise ratio with all features vs relevant features only?
5. Calculate the probability of selecting only relevant features by random chance if you pick $20$ features
6. If the signal strength is $S = 20 \cdot \sigma_s^2$ and noise is $N = 80 \cdot \sigma_n^2$, calculate the SNR. If $\sigma_s^2 = 4$ and $\sigma_n^2 = 1$, what's the SNR improvement when using only relevant features?

For a detailed explanation of this question, see [Question 5: Irrelevant Features Impact](L8_1_5_explanation.md).

## Question 6

### Problem Statement
The search space for feature selection grows exponentially with the number of features, making exhaustive search impractical for large feature sets.

#### Task
1. If you have $10$ features, how many possible feature subsets exist?
2. How many subsets have exactly $5$ features?
3. What's the growth rate of the search space (express as a function of $n$)?
4. If evaluating each subset takes $1$ second, how long would exhaustive search take for $20$ features?
5. Calculate the number of subsets with $3$-$7$ features from $20$ total features
6. A greedy forward selection algorithm evaluates features one by one. If you have $50$ features and each evaluation takes $0.1$ seconds, calculate total time for greedy vs exhaustive search. What's the speedup factor?

For a detailed explanation of this question, see [Question 6: Search Space Complexity](L8_1_6_explanation.md).

## Question 7

### Problem Statement
Feature selection can improve model generalization by reducing overfitting and managing the bias-variance trade-off.

#### Task
1. How does having too many features lead to overfitting?
2. What is the relationship between features and model complexity?
3. How does feature selection help with the bias-variance trade-off?
4. If a model overfits with $100$ features, what would happen with $10$ features?
5. Compare the generalization error before and after feature selection using a concrete example
6. The generalization error can be modeled as $E = \text{Bias}^2 + \text{Variance} + \text{Noise}$. If Bias $= 0.1$, Variance $= 0.3$, and Noise $= 0.05$ with $100$ features, and reducing to $20$ features changes Bias to $0.2$ and Variance to $0.1$, calculate the total error change. Is this improvement?

For a detailed explanation of this question, see [Question 7: Overfitting and Generalization](L8_1_7_explanation.md).

## Question 8

### Problem Statement
Different domains have different feature selection requirements based on their specific constraints and goals.

#### Task
1. What are the key considerations for medical diagnosis features?
2. How do financial applications differ from image recognition in feature selection?
3. What's important for real-time sensor data feature selection?
4. Compare feature selection needs for text vs numerical data
5. Which domain would benefit most from interpretable features and why?
6. In medical diagnosis, false positive rate (FPR) and false negative rate (FNR) are critical. If feature A has FPR $= 0.05$, FNR $= 0.10$ and feature B has FPR $= 0.02$, FNR $= 0.15$, which is better? Calculate the total error rate for each and justify your choice.

For a detailed explanation of this question, see [Question 8: Domain-Specific Considerations](L8_1_8_explanation.md).

## Question 9

### Problem Statement
Feature selection affects different types of machine learning algorithms differently, requiring tailored approaches.

#### Task
1. How does feature selection affect linear models (e.g., linear regression)?
2. How does it affect tree-based models (e.g., decision trees)?
3. How does it affect neural networks?
4. Which algorithm type benefits most from feature selection and why?
5. Compare the impact on different algorithm families using specific examples
6. For a decision tree, the information gain is $IG(S,A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$. If you have $3$ features with IG values $[0.8, 0.6, 0.3]$, and you want to select the top $2$, what's the total information gain? What percentage of maximum possible IG do you retain?

For a detailed explanation of this question, see [Question 9: Algorithm-Specific Effects](L8_1_9_explanation.md).

## Question 10

### Problem Statement
Consider a dataset with 500 samples and 50 features where features 1-10 are highly relevant, 11-30 are moderately relevant, and 31-50 are irrelevant.

#### Task
1. What percentage of features are highly relevant?
2. If you select only the top $20$ features, what's the coverage of relevant information?
3. How would you measure the quality of your selection?
4. What's the optimal number of features for this dataset?
5. Calculate the information retention with different feature counts $(10, 20, 30, 40, 50)$
6. If highly relevant features contribute $60\%$ of total information, moderately relevant contribute $35\%$, and irrelevant contribute $5\%$, calculate the information retention when selecting: (a) top $10$ features, (b) top $20$ features, (c) top $30$ features. Which gives the best information-to-feature ratio?

For a detailed explanation of this question, see [Question 10: Feature Relevance Analysis](L8_1_10_explanation.md).

## Question 11

### Problem Statement
Feature selection can be viewed as a search and optimization problem with multiple objectives and constraints.

#### Task
1. What is the objective function for feature selection?
2. What are the constraints in this optimization problem?
3. How do you balance multiple objectives (e.g., accuracy vs feature count)?
4. If you want to maximize accuracy while minimizing features, how do you formulate this mathematically?
5. Compare different optimization approaches (greedy, genetic, exhaustive)
6. Formulate the multi-objective optimization problem: maximize accuracy $A$ while minimizing features $F$. If accuracy follows $A = 0.8 + 0.1 \cdot \log(F)$ for $F \geq 5$, and you want to maximize $A - 0.05 \cdot F$, what's the optimal number of features? Calculate the objective value at $F=5$, $F=10$, and $F=15$.

For a detailed explanation of this question, see [Question 11: Optimization Formulation](L8_1_11_explanation.md).

## Question 12

### Problem Statement
The relationship between features and target variables determines selection effectiveness and requires appropriate measurement techniques.

#### Task
1. How do you measure feature-target relationships for numerical data?
2. What types of relationships are hard to detect with simple correlation?
3. How do you handle non-linear relationships in feature selection?
4. If a feature has no linear correlation but high mutual information, what does this suggest?
5. Compare different relationship measures (correlation, mutual information, chi-square) with examples
6. Calculate the Pearson correlation coefficient for these data points: Feature $X = [1, 2, 3, 4, 5]$, Target $Y = [2, 4, 5, 4, 6]$. If the correlation threshold is $0.7$, would this feature be selected? What about if you use the absolute correlation value?

For a detailed explanation of this question, see [Question 12: Feature-Target Relationships](L8_1_12_explanation.md).

## Question 13

### Problem Statement
Feature selection affects model robustness and stability, particularly when dealing with noisy or changing data.

#### Task
1. How does feature selection improve model stability?
2. What happens to model performance with noisy features?
3. How does selection affect cross-validation results?
4. If a model is unstable with 100 features, what would happen with 20 features?
5. Compare stability before and after selection using a concrete scenario
6. Model stability can be measured by the standard deviation of cross-validation scores. If a model with $100$ features has CV scores $[0.75, 0.78, 0.72, 0.76, 0.74]$ and with $20$ features has scores $[0.73, 0.74, 0.72, 0.73, 0.74]$, calculate the stability improvement. What's the percentage reduction in standard deviation?

For a detailed explanation of this question, see [Question 13: Model Stability and Robustness](L8_1_13_explanation.md).

## Question 14

### Problem Statement
The cost of feature acquisition affects selection decisions, requiring careful analysis of cost-benefit trade-offs.

#### Task
1. How do you balance feature cost vs performance improvement?
2. What's the trade-off between expensive and cheap features?
3. How do you optimize the cost-performance ratio?
4. If feature A costs $\$10$ and improves accuracy by $2\%$, while feature B costs $\$100$ and improves by $5\%$, which is better?
5. Calculate the cost-effectiveness (improvement per dollar) of different feature sets
6. You have a budget of $\$500$ and three feature sets: Set 1 (cost $\$200$, accuracy $85\%$), Set 2 (cost $\$300$, accuracy $87\%$), Set 3 (cost $\$400$, accuracy $89\%$). Calculate the cost-effectiveness ratio (accuracy improvement per dollar) for each set. Which gives the best value for money?

For a detailed explanation of this question, see [Question 14: Feature Cost Analysis](L8_1_14_explanation.md).

## Question 15

### Problem Statement
Feature redundancy occurs when multiple features provide similar information, leading to multicollinearity and reduced model performance.

#### Task
1. What is feature redundancy and why is it problematic?
2. How do you detect multicollinearity between features?
3. What's the difference between redundancy and irrelevance?
4. If two features have correlation $0.95$, what should you do and why?
5. How does redundancy affect model interpretability and performance?
6. Calculate the variance inflation factor (VIF) for a feature. If feature X has correlation $0.9$ with other features, what's the VIF? If the threshold is VIF $> 5$, should this feature be removed? What about if correlation is $0.8$?

For a detailed explanation of this question, see [Question 15: Feature Redundancy and Multicollinearity](L8_1_15_explanation.md).

## Question 16

### Problem Statement
Statistical significance testing helps determine if feature selection results are reliable and not due to chance.

#### Task
1. What is statistical significance in feature selection?
2. How do you test if a selected feature is truly important?
3. What's the role of p-values in feature selection?
4. If a feature improves accuracy by $0.5\%$, how do you know it's significant?
5. Compare different significance testing approaches (t-test, permutation test, bootstrap)
6. In a permutation test, you randomly shuffle labels $1000$ times. If the original accuracy improvement is $0.05$ and only $25$ out of $1000$ permutations give improvement $\geq 0.05$, what's the p-value? Is this significant at $\alpha = 0.05$? What about at $\alpha = 0.01$?

For a detailed explanation of this question, see [Question 16: Statistical Significance in Selection](L8_1_16_explanation.md).

## Question 17

### Problem Statement
Different data types require different feature selection strategies due to their unique characteristics and challenges.

#### Task
1. How does feature selection differ for text data vs numerical data?
2. What special considerations exist for image data feature selection?
3. How do you handle time series features in selection?
4. If you have mixed data types, what selection approach would you use?
5. Compare selection strategies across different data modalities with examples
6. For text data, TF-IDF scores are calculated as $TF\text{-}IDF = TF \times \log(\frac{N}{DF})$ where $N$ is total documents and $DF$ is document frequency. If you have $1000$ documents and a word appears in $100$ documents with frequency $5$ in one document, calculate its TF-IDF score. If the threshold is $2.0$, would this word be selected as a feature?

For a detailed explanation of this question, see [Question 17: Data Type-Specific Strategies](L8_1_17_explanation.md).

## Question 18

### Problem Statement
Ensemble feature selection methods combine multiple selection approaches for more robust and reliable results.

#### Task
1. What is ensemble feature selection and how does it work?
2. What are the advantages of combining multiple selection methods?
3. How do you aggregate results from different selection approaches?
4. If three methods select different feature sets, how do you decide which features to keep?
5. Compare ensemble vs single method selection with pros and cons
6. Three selection methods give feature importance scores $[0.8, 0.6, 0.9]$ for feature A and $[0.7, 0.8, 0.5]$ for feature B. If you use weighted voting with weights $[0.4, 0.3, 0.3]$, which feature has higher ensemble score? Calculate the final scores and rank the features.

For a detailed explanation of this question, see [Question 18: Ensemble Feature Selection](L8_1_18_explanation.md).

## Question 19

### Problem Statement
You're playing a game where you must select features to maximize model performance under specific constraints.

**Rules:**
- You have $100$ total features
- Only $15$ are truly useful
- Each useful feature gives $+10$ points
- Each useless feature gives $-2$ points
- You must select exactly $20$ features

#### Task
1. What's your best possible score?
2. What's your worst possible score?
3. If you randomly select $20$ features, what's your expected score?
4. What strategy would you use to maximize your score?
5. Calculate the probability of getting a positive score with random selection
6. If you can identify useful features with $80\%$ accuracy ($20\%$ false positive rate), what's your expected score using this imperfect selection method? Compare this to random selection and perfect selection.

For a detailed explanation of this question, see [Question 19: Feature Selection Strategy Game](L8_1_19_explanation.md).

## Question 20

### Problem Statement
You need to decide whether to use feature selection for your project and determine the optimal approach.

**Considerations:**
- Dataset: $500$ samples, $30$ features
- Goal: Interpretable model for business users
- Time constraint: $2$ weeks total
- Performance requirement: $80\%$ accuracy minimum
- Available methods: Correlation-based, mutual information, recursive feature elimination

#### Task
1. Should you use feature selection? Yes/No and why?
2. What type of selection would you choose and why?
3. How many features would you aim to keep and how do you justify this number?
4. What's your biggest risk in this decision?
5. Design a step-by-step feature selection strategy for this project
6. If correlation-based selection takes $1$ hour, mutual information takes $3$ hours, and RFE takes $8$ hours, and you have $40$ hours total for the entire project, calculate the time allocation. What percentage of total time should be spent on feature selection vs model training and evaluation?

For a detailed explanation of this question, see [Question 20: Practical Feature Selection Decision](L8_1_20_explanation.md).
