# Machine Learning Cheatsheet and "How To" Guide (Lectures 5–11 Combined)

## 📋 Quick Reference Cheatsheet

### Lecture 5: Support Vector Machines (SVM)

**Primal Formulation (Hard/Soft Margin):**
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{w}||^2 \color{red}{+ C\sum_{i=1}^n \xi_i}$$
$$\text{subject to: } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \color{red}{- \xi_i}, \quad \color{red}{\xi_i \geq 0}$$

> **Note:** The red parts (slack variables `ξ_i`, regularization term `C∑ξ_i`, and constraint `≥ 1 - ξ_i`) are for **soft margin** only. For **hard margin**, remove these red parts and set `C = ∞`.

**Dual Formulation:**
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{subject to: } \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \color{red}{\leq C}$$

> **Note:** The upper bound `≤ C` on Lagrange multipliers is for **soft margin** only. For **hard margin**, remove this upper bound (set `C = ∞`).

**Optimal Weight Vector (from Lagrangian derivative):**
$$\mathbf{w}^* = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

**Decision Function:**
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

**Functional Margin:** $\hat{\gamma}_i = y_i(\mathbf{w}^T\mathbf{x}_i + b)$
**Geometric Margin:** $\gamma_i = \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{||\mathbf{w}||}$
**Margin Width:** $\frac{2}{||\mathbf{w}||}$

**Hinge Loss:** $L_h(y, f(x)) = \max(0, 1 - y \cdot f(x))$
**ε-insensitive Loss:** $L_ε(y, f(x)) = \max(0, |y - f(x)| - ε)$

**Mercer's Theorem:**
A function $K(\mathbf{x}, \mathbf{z})$ is a valid kernel if and only if:
1. **Symmetry:** $K(\mathbf{x}, \mathbf{z}) = K(\mathbf{z}, \mathbf{x})$
2. **Positive Semi-definite:** For any finite set of points $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, the Gram matrix $G_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite (all eigenvalues ≥ 0)

**Common Kernels:**
- **Linear:** $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$
- **Polynomial:** $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$
- **RBF/Gaussian:** $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$
- **Sigmoid:** $K(\mathbf{x}, \mathbf{z}) = \tanh(\kappa \mathbf{x}^T\mathbf{z} + \theta)$

---

### Lecture 6: Decision Trees

**Entropy (Multi-class):**
$$H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

**Information Gain:**
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**Gain Ratio:**
$$\text{Gain Ratio}(S, A) = \frac{IG(S, A)}{\text{Split Info}(S, A)}$$

**Split Information:**
$$\text{Split Info}(S, A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

**Classification Error:**
$$Error(S) = 1 - \max_i(p_i)$$

**Cost-complexity pruning:** $R_\alpha(T) = R(T) + \alpha|T|$

**Tie-Breaking Criteria (in order of preference):**
When multiple splits have the same information gain or gain ratio:
1. **Higher balance ratio** (more balanced splits)
2. **Lower subset variance** (more uniform distribution)
3. **Lower size entropy** (more balanced proportions)
4. **More pure nodes** (better class separation)
5. **Fewer total subsets** (simpler tree structure)

---

### Lecture 7: Ensemble Methods

**Bootstrap Sampling:**
- **Sample size**: Same as original dataset
- **Expected unique samples**: $n \times (1 - 1/e) \approx 0.632n$
- **Out-of-bag samples**: $n \times (1 - 1/n)^n \approx 0.368n$

**Random Forest Feature Subsampling:**
- **Feature subsampling**: $\sqrt{p}$ or $\log_2(p)$ features per split
- **Feature selection probability**: $P(\text{feature used}) = 1 - \left(\frac{n-1}{n}\right)^k$

**AdaBoost:**
- **Weak learner weight**: $\alpha_t = \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
- **Sample weight update**: $w_i^{(t+1)} = w_i^{(t)} \times e^{-\alpha_t y_i h_t(x_i)}$
- **Final prediction**: $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$
- **Training error bound**: $E_{train} \leq \prod_{t=1}^T 2\sqrt{\epsilon_t(1-\epsilon_t)}$

**AdaBoost Stopping Conditions:**
- **Maximum iterations reached**: $t = T_{max}$
- **Perfect classification**: $\epsilon_t = 0$ (no misclassifications)
- **Weak learner error ≥ 0.5**: $\epsilon_t \geq 0.5$ (worse than random)
- **Convergence**: $|\alpha_t| < \text{tolerance}$ (negligible contribution)
- **Validation performance**: No improvement on validation set for $k$ iterations

**Combination Strategies:**
- **Simple averaging**: $\frac{1}{T}\sum_{t=1}^T h_t(x)$
- **Weighted averaging**: $\sum_{t=1}^T w_t h_t(x)$
- **Majority voting**: $\text{sign}\left(\sum_{t=1}^T h_t(x)\right)$

---

### Lecture 8: Feature Engineering and Selection

**Search Space Size:**
- **Total subsets**: $2^n - 1$ (excluding empty set)
- **Subsets with k features**: $\binom{n}{k} = \frac{n!}{k!(n-k)!}$
- **Subsets with k to m features**: $\sum_{i=k}^m \binom{n}{i}$

**Pearson Correlation:**
$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X) \text{Var}(Y)}}$$

**Mutual Information:**
$$I(X;Y) = \sum_{x,y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

**Alternative Forms of Mutual Information:**
$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

**Conditional Mutual Information:**
$$I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log\left(\frac{p(x,y|z)}{p(x|z)p(y|z)}\right)$$

**Curse of Dimensionality:**
- **Sample density**: Decreases as $1/n^d$ where $d$ is dimensions
- **Expected distance**: $E[d] = \sqrt{d/6}$ for unit hypercube
- **Volume ratio**: $S/V = 2d$ (increases with dimensions)

**Feature Selection Methods:**

**Filter Methods:**
- **Definition**: Select features based on statistical measures (correlation, mutual information, chi-square)
- **Pros**: Fast, independent of learning algorithm, generalizable
- **Cons**: Ignores feature interactions, may miss important features
- **Examples**: Correlation, mutual information, chi-square test, ANOVA F-test

**Wrapper Methods:**
- **Definition**: Use the learning algorithm itself to evaluate feature subsets
- **Pros**: Considers feature interactions, optimized for specific algorithm
- **Cons**: Computationally expensive, prone to overfitting
- **Examples**: Forward selection, backward elimination, recursive feature elimination (RFE)

**Embedded Methods:**
- **Definition**: Feature selection is built into the learning algorithm
- **Pros**: Efficient, considers feature interactions, less overfitting
- **Cons**: Algorithm-specific, may not be interpretable
- **Examples**: Lasso (L1 regularization), Ridge (L2 regularization), Elastic Net, Random Forest feature importance

---

### Lecture 9: Model Evaluation and Validation

**Classification Metrics:**
- **Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$
- **F1 Score**: $\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Regression Metrics:**
- **MSE**: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE**: $\text{RMSE} = \sqrt{\text{MSE}}$
- **MAE**: $\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **R²**: $\text{R²} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

**Statistical Tests:**
- **Standard Error**: $\text{SE} = \sqrt{\frac{p(1-p)}{n}}$
- **Z-score**: $z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\text{SE}_1^2 + \text{SE}_2^2}}$

**Validation Methods:**
- **K-fold CV**: Each sample used $(k-1)$ times for training, $1$ time for testing
- **Bootstrap**: $P(\text{sample not selected}) = (1 - \frac{1}{n})^n$

---

### Lecture 10: Handling Imbalanced Data

**Imbalance Ratios:**
- **Imbalance Ratio**: $\text{IR} = \frac{\text{Majority Class Count}}{\text{Minority Class Count}}$
- **Minority Class Percentage**: $\text{Minority \%} = \frac{\text{Minority Count}}{\text{Total Count}} \times 100\%$

**Evaluation Metrics for Imbalanced Data:**
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall/Sensitivity**: $\text{Recall} = \frac{TP}{TP + FN}$
- **Specificity**: $\text{Specificity} = \frac{TN}{TN + FP}$
- **F1 Score**: $\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Fβ Score**: $\text{Fβ} = \frac{(1 + β²) \times \text{Precision} \times \text{Recall}}{β² \times \text{Precision} + \text{Recall}}$
- **Balanced Accuracy**: $\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$
- **G-Mean**: $\text{G-Mean} = \sqrt{\text{Sensitivity} \times \text{Specificity}}$

**Cost-Sensitive Learning:**
- **Cost Matrix**: $C_{ij}$ = cost of predicting class $i$ when true class is $j$
- **Weighted Loss**: $\text{Weighted Loss} = \sum_{i,j} C_{ij} \times \text{Confusion}_{ij}$
- **Class Weights**: $\text{Weight}_i = \frac{\text{Total Samples}}{\text{Number of Classes} \times \text{Class}_i \text{ Count}}$

**Imbalance Levels:**
- **Mild Imbalance**: IR < 3:1 (minority > 25%)
- **Moderate Imbalance**: 3:1 ≤ IR ≤ 10:1 (10% ≤ minority ≤ 25%)
- **Severe Imbalance**: IR > 10:1 (minority < 10%)

---

### Lecture 11: Clustering and Unsupervised Learning

**Core Distance Metrics:**

**Euclidean Distance:**
$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Manhattan Distance (L1):**
$$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Cosine Similarity:**
$$\cos(\theta) = \frac{x \cdot y}{\|x\| \|y\|} = \frac{\sum_{i=1}^{n}x_i y_i}{\sqrt{\sum_{i=1}^{n}x_i^2} \sqrt{\sum_{i=1}^{n}y_i^2}}$$

**K-Means Clustering:**

**Objective Function (WCSS):**
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Centroid Update:**
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

**Hierarchical Clustering:**

**Single Linkage:**
$$d(C_1, C_2) = \min_{x \in C_1, y \in C_2} d(x,y)$$

**Complete Linkage:**
$$d(C_1, C_2) = \max_{x \in C_1, y \in C_2} d(x,y)$$

**Ward Distance:**
$$d(C_1, C_2) = \frac{|C_1||C_2|}{|C_1| + |C_2|} \|\mu_1 - \mu_2\|^2$$

---

## 🎯 Question Type Categories

### Lecture 5: SVM Question Types

1. **Linear Separability & Hyperplane Analysis**
2. **Primal vs Dual Formulation**
3. **Support Vector Identification**
4. **Soft Margin & Slack Variables**
5. **Kernel Trick & Feature Transformation**
6. **Multi-Class SVM**
7. **Support Vector Regression (SVR)**
8. **Computational Considerations**
9. **Loss Function Analysis**
10. **Parameter Tuning & Model Selection**

### Lecture 6: Decision Trees Question Types

1. **Tree Structure Analysis**
2. **Entropy and Information Gain Calculation**
3. **Decision Tree Algorithm Analysis (ID3, C4.5, CART)**
4. **Tree Pruning and Overfitting Analysis**
5. **Impurity Measures Comparison**
6. **Missing Values and Special Cases**
7. **Cost-Sensitive Learning and Business Applications**
8. **Cross-Validation and Model Selection**
9. **Decision Tree Visualization and Interpretation**
10. **Advanced Topics (Multi-output, Online Learning)**

### Lecture 7: Ensemble Methods Question Types

1. **Ensemble Performance Analysis**
2. **Bootstrap Sampling and Bagging**
3. **Random Forest Analysis**
4. **AdaBoost Algorithm Analysis**
5. **Gradient Boosting Fundamentals**
6. **Advanced Boosting Algorithms (XGBoost, LightGBM, CatBoost)**
7. **Ensemble Diversity and Combination**
8. **Bias-Variance Trade-off in Ensembles**
9. **Practical Ensemble Applications**
10. **Ensemble Model Selection and Validation**

### Lecture 8: Feature Engineering and Selection Question Types

1. **Feature Selection Fundamentals and Benefits**
2. **Curse of Dimensionality Analysis**
3. **Univariate Feature Selection Methods**
4. **Multivariate Feature Selection Methods**
5. **Filter Methods Analysis**
6. **Wrapper Methods and Search Strategies**
7. **Feature Engineering Techniques**
8. **Feature Selection Evaluation and Validation**
9. **Advanced Feature Selection Techniques**
10. **Practical Feature Selection Applications**

### Lecture 9: Model Evaluation and Validation Question Types

1. **Overfitting and Underfitting Detection**
2. **Classification Metrics Calculation**
3. **Regression Metrics Calculation**
4. **ROC Curve and AUC Analysis**
5. **Cross-Validation Analysis**
6. **Sampling Techniques Analysis**
7. **Bootstrap Analysis**
8. **Statistical Significance Testing**
9. **Model Comparison and Selection**
10. **Evaluation Best Practices and Pitfalls**

### Lecture 10: Handling Imbalanced Data Question Types

1. **Class Imbalance Detection and Analysis**
2. **Evaluation Metrics for Imbalanced Data**
3. **Random Oversampling Analysis**
4. **Random Undersampling Analysis**
5. **SMOTE Algorithm Analysis**
6. **Advanced Synthetic Methods Analysis**
7. **Hybrid and Ensemble Methods Analysis**
8. **Cost-Sensitive Learning Analysis**
9. **Method Selection and Comparison**
10. **Real-World Application Analysis**

### Lecture 11: Clustering and Unsupervised Learning Question Types

1. **Distance Metrics and Similarity Calculations**
2. **K-Means Algorithm Implementation**
3. **Hierarchical Clustering Analysis**
4. **DBSCAN Algorithm Analysis**
5. **Gaussian Mixture Models and EM**
6. **Clustering Evaluation Metrics**
7. **Advanced Clustering Techniques**
8. **Clustering Applications and Case Studies**
9. **Parameter Selection and Model Selection**
10. **Clustering Challenges and Solutions**

---

### Common Mistakes to Avoid:

**SVM:**
- **Forgetting constraints** in optimization problems
- **Mixing up signs** in margin calculations
- **Ignoring KKT conditions** in dual formulation
- **Not checking linear separability** before applying hard margin
- **Forgetting slack variables** in soft margin problems
- **Miscalculating kernel functions** - expand carefully
- **Not considering computational complexity** in algorithm choice

**Decision Trees:**
- **Forgetting log base 2** in entropy calculations
- **Mixing up information gain and gain ratio**
- **Not considering class imbalance** in impurity measures
- **Ignoring overfitting** in tree depth analysis
- **Forgetting cost considerations** in business applications
- **Not checking stopping criteria** in algorithm analysis
- **Miscalculating weighted averages** in split evaluation

**Ensemble Methods:**
- **Confusing bagging and boosting** - different purposes and mechanisms
- **Forgetting bootstrap sampling properties** - 63.2% unique samples
- **Miscalculating AdaBoost weights** - use correct formula with ln
- **Ignoring diversity requirements** - ensembles need diverse base learners
- **Not considering bias-variance trade-off** - choose ensemble based on base learner characteristics
- **Forgetting regularization** - especially in gradient boosting
- **Not checking weak learner requirements** - must be better than random

**Feature Engineering and Selection:**
- **Confusing correlation with causation** - correlation doesn't imply causation
- **Ignoring feature interactions** - univariate methods miss interactions
- **Not considering computational cost** - wrapper methods can be expensive
- **Forgetting validation** - always validate feature selection results
- **Overlooking domain knowledge** - expert insights are valuable
- **Not checking for overfitting** - feature selection can overfit
- **Ignoring feature stability** - unstable selection may not generalize

**Model Evaluation and Validation:**
- **Confusing precision and recall** - precision is TP/(TP+FP), recall is TP/(TP+FN)
- **Forgetting to square root RMSE** - RMSE = √MSE
- **Ignoring class imbalance** - accuracy can be misleading
- **Not considering statistical significance** - differences may not be meaningful
- **Overlooking data leakage** - using test data in training

**Handling Imbalanced Data:**
- **Using accuracy for imbalanced data** - accuracy is misleading
- **Ignoring business context** - different costs for different errors
- **Not considering computational cost** - some methods are expensive
- **Forgetting to validate** - always test on holdout set
- **Overlooking data quality** - noise affects synthetic methods
- **Not checking assumptions** - verify method assumptions
- **Ignoring interpretability** - some methods are black boxes

**Clustering and Unsupervised Learning:**
- **Using Euclidean distance for text data** - use Cosine similarity instead
- **Not normalizing features** - different scales affect clustering
- **Choosing K arbitrarily** - use elbow method or gap statistic
- **Ignoring cluster shapes** - K-means assumes spherical clusters
- **Not validating results** - always check clustering quality
- **Forgetting computational complexity** - some algorithms don't scale
- **Overlooking interpretability** - business context matters

### Time Management:
- **Simple calculations**: 2-3 minutes
- **Medium complexity**: 5-8 minutes  
- **Complex derivations**: 10-15 minutes
- **Multi-part problems**: 15-20 minutes

---

## 🎯 Quick Reference Decision Trees

### Which SVM to Use?
```
Is data linearly separable?
├─ Yes → Hard Margin SVM
└─ No → Soft Margin SVM
    ├─ Small C → More tolerance
    └─ Large C → Less tolerance
```

### Which Kernel to Choose?
```
What's the data structure?
├─ Linear → Linear kernel
├─ Polynomial patterns → Polynomial kernel
├─ Radial patterns → RBF kernel
└─ Unknown → Try RBF (default choice)
```

### Which Ensemble Method?
```
What's the base learner characteristic?
├─ High variance, low bias → Bagging (Random Forest)
├─ High bias, low variance → Boosting (AdaBoost, Gradient Boosting)
└─ Different algorithms → Stacking
```

### Which Feature Selection Method?
```
What are the requirements?
├─ Fast, general → Filter methods
├─ Accurate, specific → Wrapper methods
├─ Built-in selection → Embedded methods
└─ Real-time → Online selection
```

### Which Evaluation Metric?
```
Classification → Accuracy, Precision, Recall, F1
Regression → MSE, RMSE, MAE, R²
Imbalanced → Precision, Recall, F1, AUC
```

### Which Sampling Method for Imbalanced Data?
```
Imbalance Level:
├─ Mild (IR < 3) → No sampling needed
├─ Moderate (3 ≤ IR ≤ 10) → SMOTE or cost-sensitive
└─ Severe (IR > 10) → Advanced methods or ensemble
```

### Which Clustering Algorithm?
```
Data Characteristics:
├─ Spherical clusters → K-means, GMM
├─ Arbitrary shapes → DBSCAN, Spectral
├─ Hierarchical structure → Hierarchical clustering
└─ Probabilistic assignments → GMM, EM
```

### Which Distance Metric?
```
Data Type:
├─ Continuous numerical → Euclidean, Manhattan
├─ High-dimensional → Cosine similarity
├─ Categorical → Hamming, Jaccard
└─ Mixed types → Weighted combinations
```

---

## 📊 Statistics Reference

### Basic Statistical Formulas

**Variance:**
$$\text{Var}(X) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2 = \frac{1}{n}\sum_{i=1}^{n}x_i^2 - \bar{x}^2$$

**Sample Variance (Unbiased):**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Covariance:**
$$\text{Cov}(X,Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) = \frac{1}{n}\sum_{i=1}^{n}x_i y_i - \bar{x}\bar{y}$$

**Sample Covariance (Unbiased):**
$$s_{xy} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Standard Deviation:**
$$\sigma = \sqrt{\text{Var}(X)} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Sample Standard Deviation:**
$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
