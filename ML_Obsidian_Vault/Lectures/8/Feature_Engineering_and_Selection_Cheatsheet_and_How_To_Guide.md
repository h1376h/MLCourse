# Feature Engineering and Selection Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core Feature Selection Formulas

**Search Space Size:**
- **Total subsets**: $2^n - 1$ (excluding empty set)
- **Subsets with k features**: $\binom{n}{k} = \frac{n!}{k!(n-k)!}$
- **Subsets with k to m features**: $\sum_{i=k}^m \binom{n}{i}$

**Pearson Correlation:**
$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

**Mutual Information:**
$$I(X;Y) = \sum_{x,y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

**Chi-Square Statistic:**
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

**KL Divergence:**
$$D_{KL}(P||Q) = \sum_i p_i \log\left(\frac{p_i}{q_i}\right)$$

**Variance Inflation Factor (VIF):**
$$VIF = \frac{1}{1-R^2}$$

### Key Concepts

**Feature Selection Types:**
- **Filter Methods**: Independent of learning algorithm
- **Wrapper Methods**: Use learning algorithm for evaluation
- **Embedded Methods**: Built into learning algorithm

**Selection Approaches:**
- **Univariate**: Evaluate features individually
- **Multivariate**: Consider feature interactions
- **Sequential**: Forward/backward selection
- **Random**: Genetic algorithms, random search

**Evaluation Criteria:**
- **Distance Measures**: Class separability
- **Information Measures**: Information gain, entropy
- **Dependency Measures**: Correlation, independence
- **Consistency Measures**: Classification consistency
- **Stability Measures**: Selection consistency

---

## 🎯 Question Type 1: Feature Selection Fundamentals and Benefits

### How to Approach:

**Step 1: Identify Core Benefits**
- **Performance improvement**: Better accuracy, reduced overfitting
- **Computational efficiency**: Faster training and prediction
- **Interpretability**: Simpler models, clearer insights

**Step 2: Calculate Performance Improvements**
- **Training time**: Linear scaling with feature count
- **Memory usage**: Bytes per sample × number of features
- **Complexity reduction**: $O(n^d)$ where $d$ is feature count

**Step 3: Analyze Curse of Dimensionality**
- **Sample density**: Decreases exponentially with dimensions
- **Distance metrics**: Become less meaningful in high dimensions
- **Volume ratios**: Surface area to volume increases

**Step 4: Evaluate Practical Constraints**
- **Real-time requirements**: Inference speed limits
- **Memory constraints**: Storage limitations
- **Cost considerations**: Feature acquisition costs

**Step 5: Design Selection Strategy**
- **Problem-specific requirements**: Accuracy vs speed trade-offs
- **Resource constraints**: Computational and memory limits
- **Domain knowledge**: Expert insights for feature relevance

### Example Template:
```
Given: Dataset with [n] features, [m] samples, [constraints]
1. Core benefits:
   - Performance: [accuracy improvement] through [mechanism]
   - Efficiency: [time/memory] reduction by [percentage]
   - Interpretability: [simpler models] with [fewer features]
2. Computational analysis:
   - Training time: [original] → [reduced] ([improvement]%)
   - Memory usage: [original] bytes → [reduced] bytes
   - Complexity: O([original]) → O([reduced])
3. Curse of dimensionality:
   - Sample density: [calculation] samples per unit volume
   - Distance metrics: [less meaningful] in [dimensions]
   - Volume ratios: [surface/volume] increases with dimensions
4. Practical constraints:
   - Real-time: [speed requirement] → [max features]
   - Memory: [limit] → [max features]
   - Cost: [budget] → [feature selection strategy]
5. Strategy design: [recommended approach] because [justification]
```

---

## 🎯 Question Type 2: Curse of Dimensionality Analysis

### How to Approach:

**Step 1: Understand Dimensionality Effects**
- **Sample density**: Decreases as $1/n^d$ where $d$ is dimensions
- **Distance concentration**: All distances become similar
- **Volume ratios**: Surface area grows faster than volume

**Step 2: Calculate Sample Requirements**
- **Density preservation**: $n_2 = n_1 \times (d_2/d_1)^d$
- **Exponential growth**: Sample needs grow exponentially with dimensions
- **Practical limits**: When sample density becomes too low

**Step 3: Analyze Distance Metrics**
- **Expected distance**: $E[d] = \sqrt{d/6}$ for unit hypercube
- **Distance concentration**: Ratio of max to expected distance
- **Meaningful distances**: When distances become indistinguishable

**Step 4: Evaluate Volume Properties**
- **Hypercube volume**: $V = 1^d = 1$ (unit hypercube)
- **Surface area**: $S = 2d$ (grows linearly with dimensions)
- **Volume ratio**: $S/V = 2d$ (increases with dimensions)

**Step 5: Assess Algorithm Impact**
- **Nearest neighbors**: Performance degrades with dimensions
- **Clustering**: Distance-based methods become unreliable
- **Visualization**: Impossible beyond 3D without projection

### Example Template:
```
Given: [d]-dimensional space with [n] samples
1. Sample density analysis:
   - Density: [n] samples in [d]D space
   - Equivalent density: [n_equivalent] samples in 2D
   - Growth factor: [factor]× more samples needed
2. Distance metrics:
   - Expected distance: √([d]/6) = [value]
   - Max distance: [value] (diagonal of unit hypercube)
   - Concentration ratio: [max]/[expected] = [value]
3. Volume properties:
   - Volume: [1] (unit hypercube)
   - Surface area: [2d] (grows linearly)
   - Volume ratio: [2d] (increases with dimensions)
4. Algorithm impact:
   - Nearest neighbors: [performance degradation]
   - Clustering: [reliability issues]
   - Visualization: [dimensionality reduction needed]
5. Solutions: [feature selection/dimensionality reduction] because [reasoning]
```

---

## 🎯 Question Type 3: Univariate Feature Selection Methods

### How to Approach:

**Step 1: Understand Univariate Approach**
- **Individual evaluation**: Each feature evaluated independently
- **Computational efficiency**: $O(n)$ evaluations for $n$ features
- **Limitation**: Ignores feature interactions

**Step 2: Calculate Correlation Measures**
- **Pearson correlation**: Linear relationships only
- **Spearman correlation**: Monotonic relationships
- **Kendall's tau**: Ordinal relationships

**Step 3: Apply Information Measures**
- **Mutual information**: Captures non-linear relationships
- **Information gain**: Reduction in entropy
- **Chi-square**: Independence testing for categorical data

**Step 4: Set Selection Thresholds**
- **Correlation threshold**: $|r| > 0.3$ (moderate correlation)
- **Information threshold**: $I(X;Y) > 0.1$ (significant information)
- **P-value threshold**: $\alpha < 0.05$ (statistical significance)

**Step 5: Evaluate Selection Results**
- **Number of selected features**: Balance relevance and complexity
- **Performance validation**: Cross-validation with selected features
- **Stability assessment**: Consistency across data splits

### Example Template:
```
Given: Dataset with [n] features, target variable [Y]
1. Univariate approach:
   - Evaluations: [n] individual feature assessments
   - Complexity: O([n]) vs O([2^n]) for exhaustive search
   - Limitation: [ignores feature interactions]
2. Correlation analysis:
   - Pearson correlation: [formula] for linear relationships
   - Spearman correlation: [formula] for monotonic relationships
   - Threshold: |r| > [0.3] for selection
3. Information measures:
   - Mutual information: [formula] for non-linear relationships
   - Information gain: [entropy reduction] calculation
   - Chi-square: [independence test] for categorical data
4. Selection criteria:
   - Correlation: [threshold] → [number] features selected
   - Information: [threshold] → [number] features selected
   - Statistical: [p-value] → [number] features selected
5. Validation: [cross-validation] shows [performance improvement]
```

---

## 🎯 Question Type 4: Multivariate Feature Selection Methods

### How to Approach:

**Step 1: Identify When Univariate Fails**
- **Feature interactions**: Combined effects not captured individually
- **Redundancy**: Correlated features with similar information
- **Synergy**: Features that work better together than alone

**Step 2: Analyze Feature Interactions**
- **Pairwise correlations**: Identify redundant feature pairs
- **Interaction effects**: Features that enhance each other
- **Conditional independence**: Features independent given others

**Step 3: Handle Search Space Complexity**
- **Exponential growth**: $2^n - 1$ possible subsets
- **Heuristic search**: Forward/backward selection
- **Random search**: Genetic algorithms, simulated annealing

**Step 4: Apply Feature Clustering**
- **Correlation-based clustering**: Group similar features
- **Hierarchical clustering**: Build feature hierarchies
- **Representative selection**: Choose one feature per cluster

**Step 5: Evaluate Multivariate Performance**
- **Subset evaluation**: Use learning algorithm performance
- **Cross-validation**: Unbiased performance estimation
- **Stability analysis**: Consistency across different samples

### Example Template:
```
Given: [n] features with [interaction patterns]
1. Univariate limitations:
   - Feature interactions: [example] not captured individually
   - Redundancy: [correlated features] with [correlation value]
   - Synergy: [feature pair] works better together
2. Interaction analysis:
   - Pairwise correlations: [correlation matrix] analysis
   - Interaction effects: [combined performance] vs [individual]
   - Conditional independence: [features] independent given [others]
3. Search space:
   - Total subsets: 2^[n] - 1 = [number] possible combinations
   - Heuristic search: [forward/backward] selection
   - Random search: [genetic algorithm] with [parameters]
4. Feature clustering:
   - Correlation threshold: [value] for clustering
   - Number of clusters: [k] clusters identified
   - Representative selection: [one feature] per cluster
5. Performance evaluation: [learning algorithm] shows [improvement]
```

---

## 🎯 Question Type 5: Filter Methods Analysis

### How to Approach:

**Step 1: Understand Filter Characteristics**
- **Algorithm independence**: Evaluation independent of learning algorithm
- **Computational efficiency**: Fast evaluation of individual features
- **Generality**: Results applicable to multiple algorithms

**Step 2: Compare Filter vs Wrapper Methods**
- **Speed**: Filters much faster than wrappers
- **Accuracy**: Wrappers may find better feature subsets
- **Overfitting**: Filters less prone to overfitting

**Step 3: Apply Different Filter Criteria**
- **Correlation-based**: Linear relationships with target
- **Information-based**: Mutual information, entropy reduction
- **Statistical**: Chi-square, t-test, ANOVA
- **Distance-based**: Class separability measures

**Step 4: Set Appropriate Thresholds**
- **Correlation threshold**: $|r| > 0.3$ for moderate correlation
- **Information threshold**: $I(X;Y) > 0.1$ for significant information
- **P-value threshold**: $\alpha < 0.05$ for statistical significance

**Step 5: Evaluate Filter Performance**
- **Feature ranking**: Order features by relevance scores
- **Subset selection**: Choose top-k features or threshold-based
- **Validation**: Cross-validate with selected features

### Example Template:
```
Given: [n] features with [filter criteria] and [thresholds]
1. Filter characteristics:
   - Algorithm independence: [evaluation method] independent of [learning algorithm]
   - Computational efficiency: [O(n)] vs [O(2^n)] for wrappers
   - Generality: [results] applicable to [multiple algorithms]
2. Filter vs wrapper comparison:
   - Speed: [filters] [X] times faster than [wrappers]
   - Accuracy: [wrappers] may find [better] feature subsets
   - Overfitting: [filters] less prone to [overfitting]
3. Filter criteria:
   - Correlation: [formula] for [linear relationships]
   - Information: [mutual information] for [non-linear relationships]
   - Statistical: [chi-square] for [independence testing]
   - Distance: [class separability] measures
4. Threshold selection:
   - Correlation: |r| > [0.3] → [number] features selected
   - Information: I(X;Y) > [0.1] → [number] features selected
   - Statistical: p < [0.05] → [number] features selected
5. Performance evaluation: [cross-validation] shows [improvement]
```

---

## 🎯 Question Type 6: Wrapper Methods and Search Strategies

### How to Approach:

**Step 1: Understand Wrapper Characteristics**
- **Algorithm-specific**: Uses learning algorithm for evaluation
- **Computational cost**: Expensive due to model training
- **Overfitting risk**: May overfit to training data

**Step 2: Implement Search Strategies**
- **Exhaustive search**: Evaluate all possible subsets
- **Sequential search**: Forward/backward selection
- **Random search**: Genetic algorithms, simulated annealing
- **Heuristic search**: Beam search, branch and bound

**Step 3: Calculate Search Space Size**
- **Total subsets**: $2^n - 1$ for $n$ features
- **Subsets with k features**: $\binom{n}{k}$
- **Search efficiency**: Number of evaluations needed

**Step 4: Analyze Search Complexity**
- **Exhaustive search**: $O(2^n)$ evaluations
- **Forward selection**: $O(n^2)$ evaluations
- **Genetic algorithm**: $O(p \times g)$ where $p$ is population, $g$ is generations

**Step 5: Evaluate Wrapper Performance**
- **Cross-validation**: Unbiased performance estimation
- **Stability analysis**: Consistency across different samples
- **Computational cost**: Time and memory requirements

### Example Template:
```
Given: [n] features with [learning algorithm] and [search strategy]
1. Wrapper characteristics:
   - Algorithm-specific: [evaluation] using [learning algorithm]
   - Computational cost: [expensive] due to [model training]
   - Overfitting risk: [high] due to [algorithm-specific optimization]
2. Search strategies:
   - Exhaustive: [2^n - 1] evaluations for [n] features
   - Sequential: [forward/backward] selection
   - Random: [genetic algorithm] with [parameters]
   - Heuristic: [beam search] with [beam width]
3. Search space analysis:
   - Total subsets: [2^n - 1] = [number] possible combinations
   - Subsets with k features: [binomial coefficient] = [number]
   - Search efficiency: [evaluations] needed for [strategy]
4. Complexity analysis:
   - Exhaustive: O([2^n]) evaluations
   - Forward selection: O([n²]) evaluations
   - Genetic algorithm: O([p × g]) evaluations
5. Performance evaluation: [cross-validation] shows [improvement] with [cost]
```

---

## 🎯 Question Type 7: Feature Engineering Techniques

### How to Approach:

**Step 1: Identify Feature Types**
- **Numerical**: Continuous or discrete numerical values
- **Categorical**: Nominal or ordinal categories
- **Text**: String data requiring text processing
- **Temporal**: Time-series or date/time data

**Step 2: Apply Numerical Transformations**
- **Scaling**: Standardization, normalization, min-max scaling
- **Log transformation**: $log(x + 1)$ for skewed distributions
- **Polynomial features**: $x^2, x^3$ for non-linear relationships
- **Binning**: Discretization of continuous variables

**Step 3: Handle Categorical Features**
- **One-hot encoding**: Binary indicators for each category
- **Label encoding**: Numerical labels for categories
- **Target encoding**: Mean target value per category
- **Hash encoding**: Fixed-size representation for high cardinality

**Step 4: Process Text Features**
- **Bag of words**: Word frequency vectors
- **TF-IDF**: Term frequency-inverse document frequency
- **Word embeddings**: Dense vector representations
- **N-grams**: Sequence of n consecutive words

**Step 5: Create Interaction Features**
- **Pairwise interactions**: Product of two features
- **Ratio features**: Division of two features
- **Aggregation features**: Mean, sum, count by groups
- **Time-based features**: Lag, rolling statistics

### Example Template:
```
Given: Dataset with [feature types] and [target variable]
1. Feature types:
   - Numerical: [continuous/discrete] features
   - Categorical: [nominal/ordinal] features
   - Text: [string] features requiring [processing]
   - Temporal: [time-series] features
2. Numerical transformations:
   - Scaling: [standardization/normalization] for [reason]
   - Log transformation: log(x + 1) for [skewed distributions]
   - Polynomial: [x², x³] for [non-linear relationships]
   - Binning: [discretization] into [k] bins
3. Categorical handling:
   - One-hot encoding: [binary indicators] for [categories]
   - Label encoding: [numerical labels] for [categories]
   - Target encoding: [mean target] per [category]
   - Hash encoding: [fixed-size] for [high cardinality]
4. Text processing:
   - Bag of words: [word frequency] vectors
   - TF-IDF: [term frequency] × [inverse document frequency]
   - Word embeddings: [dense vectors] for [semantic meaning]
   - N-grams: [n-word] sequences
5. Interaction features: [pairwise products], [ratios], [aggregations]
```

---

## 🎯 Question Type 8: Feature Selection Evaluation and Validation

### How to Approach:

**Step 1: Choose Evaluation Criteria**
- **Performance-based**: Accuracy, AUC, F1-score
- **Stability-based**: Consistency across data splits
- **Efficiency-based**: Training time, memory usage
- **Interpretability-based**: Model complexity, feature importance

**Step 2: Implement Validation Strategy**
- **Cross-validation**: K-fold for unbiased estimation
- **Hold-out validation**: Separate test set
- **Nested cross-validation**: For hyperparameter tuning
- **Bootstrap validation**: Resampling-based estimation

**Step 3: Assess Feature Selection Stability**
- **Jaccard similarity**: Overlap between selected feature sets
- **Kuncheva index**: Stability measure for feature sets
- **Consistency across folds**: Percentage of features selected in all folds
- **Ranking stability**: Correlation of feature rankings

**Step 4: Analyze Performance Trade-offs**
- **Accuracy vs interpretability**: Simpler models vs better performance
- **Speed vs accuracy**: Faster training vs higher accuracy
- **Memory vs performance**: Lower memory usage vs better performance
- **Cost vs benefit**: Feature acquisition cost vs performance gain

**Step 5: Validate Selection Results**
- **Statistical significance**: Test if improvement is significant
- **Practical significance**: Is improvement meaningful in practice?
- **Robustness**: Performance across different datasets
- **Generalization**: Performance on unseen data

### Example Template:
```
Given: Feature selection results with [evaluation criteria] and [validation strategy]
1. Evaluation criteria:
   - Performance: [accuracy/AUC/F1] = [value]
   - Stability: [Jaccard similarity] = [value]
   - Efficiency: [training time] = [value]
   - Interpretability: [model complexity] = [value]
2. Validation strategy:
   - Cross-validation: [k]-fold with [metric]
   - Hold-out: [test set] with [performance]
   - Nested CV: [hyperparameter tuning] with [results]
   - Bootstrap: [resampling] with [confidence intervals]
3. Stability analysis:
   - Jaccard similarity: [overlap] between [feature sets]
   - Kuncheva index: [stability measure] = [value]
   - Consistency: [percentage] of features selected in all folds
   - Ranking correlation: [correlation] of [feature rankings]
4. Performance trade-offs:
   - Accuracy vs interpretability: [simpler models] vs [better performance]
   - Speed vs accuracy: [faster training] vs [higher accuracy]
   - Memory vs performance: [lower memory] vs [better performance]
5. Validation results: [statistical significance] and [practical significance]
```

---

## 🎯 Question Type 9: Advanced Feature Selection Techniques

### How to Approach:

**Step 1: Understand Embedded Methods**
- **Lasso regression**: L1 regularization for feature selection
- **Ridge regression**: L2 regularization for feature shrinkage
- **Elastic net**: Combination of L1 and L2 regularization
- **Decision trees**: Feature importance through splits

**Step 2: Apply Regularization Techniques**
- **L1 regularization**: $||w||_1 = \sum_i |w_i|$ promotes sparsity
- **L2 regularization**: $||w||_2^2 = \sum_i w_i^2$ prevents overfitting
- **Elastic net**: $\alpha||w||_1 + (1-\alpha)||w||_2^2$ combines both
- **Group lasso**: Regularization for grouped features

**Step 3: Implement Feature Importance Methods**
- **Permutation importance**: Performance decrease when feature is permuted
- **SHAP values**: Shapley additive explanations for feature contributions
- **Partial dependence plots**: Effect of individual features on predictions
- **Feature ablation**: Performance when feature is removed

**Step 4: Handle Multi-output Problems**
- **Multi-task learning**: Shared feature selection across tasks
- **Multi-label classification**: Feature selection for multiple labels
- **Multi-target regression**: Feature selection for multiple targets
- **Hierarchical selection**: Feature selection at different levels

**Step 5: Apply Online Feature Selection**
- **Streaming data**: Feature selection for data streams
- **Incremental learning**: Update feature selection with new data
- **Adaptive selection**: Adjust selection based on performance
- **Real-time selection**: Feature selection for real-time systems

### Example Template:
```
Given: [problem type] with [advanced techniques] and [constraints]
1. Embedded methods:
   - Lasso regression: [L1 regularization] for [sparsity]
   - Ridge regression: [L2 regularization] for [shrinkage]
   - Elastic net: [L1 + L2] for [balanced regularization]
   - Decision trees: [feature importance] through [splits]
2. Regularization techniques:
   - L1: ||w||₁ = Σ|wᵢ| promotes [sparsity]
   - L2: ||w||₂² = Σwᵢ² prevents [overfitting]
   - Elastic net: α||w||₁ + (1-α)||w||₂² combines [both]
   - Group lasso: [grouped regularization] for [structured features]
3. Feature importance:
   - Permutation: [performance decrease] when [feature permuted]
   - SHAP values: [Shapley explanations] for [feature contributions]
   - Partial dependence: [individual feature] effects on [predictions]
   - Feature ablation: [performance] when [feature removed]
4. Multi-output handling:
   - Multi-task: [shared selection] across [tasks]
   - Multi-label: [selection] for [multiple labels]
   - Multi-target: [selection] for [multiple targets]
   - Hierarchical: [selection] at [different levels]
5. Online selection: [streaming data], [incremental learning], [adaptive selection]
```

---

## 🎯 Question Type 10: Practical Feature Selection Applications

### How to Approach:

**Step 1: Understand Domain Requirements**
- **Business constraints**: Cost, time, interpretability requirements
- **Technical constraints**: Computational resources, memory limits
- **Regulatory constraints**: Compliance, explainability requirements
- **Performance requirements**: Accuracy, speed, reliability needs

**Step 2: Design Selection Pipeline**
- **Data preprocessing**: Cleaning, normalization, encoding
- **Feature engineering**: Creating new features, transformations
- **Selection methods**: Filter, wrapper, embedded approaches
- **Validation strategy**: Cross-validation, hold-out testing

**Step 3: Implement Cost-Benefit Analysis**
- **Feature acquisition cost**: Cost to obtain each feature
- **Computational cost**: Training and prediction time
- **Performance benefit**: Improvement in model performance
- **ROI calculation**: Return on investment for feature selection

**Step 4: Handle Real-world Challenges**
- **Missing data**: Imputation, deletion, or feature engineering
- **Noisy features**: Outlier detection and removal
- **High dimensionality**: Curse of dimensionality effects
- **Feature drift**: Changes in feature distributions over time

**Step 5: Monitor and Maintain Selection**
- **Performance monitoring**: Track model performance over time
- **Feature drift detection**: Monitor feature distributions
- **Selection updates**: Re-evaluate feature selection periodically
- **Model retraining**: Update models with new feature sets

### Example Template:
```
Given: [application domain] with [requirements] and [constraints]
1. Domain requirements:
   - Business: [cost/time/interpretability] constraints
   - Technical: [computational/memory] limits
   - Regulatory: [compliance/explainability] requirements
   - Performance: [accuracy/speed/reliability] needs
2. Selection pipeline:
   - Preprocessing: [cleaning/normalization/encoding]
   - Engineering: [new features/transformations]
   - Selection: [filter/wrapper/embedded] methods
   - Validation: [cross-validation/hold-out] testing
3. Cost-benefit analysis:
   - Acquisition cost: [cost] per [feature]
   - Computational cost: [time] for [training/prediction]
   - Performance benefit: [improvement] in [metric]
   - ROI: [return] on [investment]
4. Real-world challenges:
   - Missing data: [imputation/deletion/engineering]
   - Noisy features: [outlier detection/removal]
   - High dimensionality: [curse of dimensionality] effects
   - Feature drift: [distribution changes] over time
5. Monitoring: [performance tracking], [drift detection], [selection updates]
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify feature selection type** - filter, wrapper, or embedded
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### During Solution:
1. **Show all steps** - even if you can do mental math
2. **Use clear notation** - define variables explicitly
3. **Draw diagrams** - visualization helps understanding
4. **Check units** - ensure dimensional consistency
5. **Verify results** - plug back into original equations

### Common Mistakes to Avoid:
- **Confusing correlation with causation** - correlation doesn't imply causation
- **Ignoring feature interactions** - univariate methods miss interactions
- **Not considering computational cost** - wrapper methods can be expensive
- **Forgetting validation** - always validate feature selection results
- **Overlooking domain knowledge** - expert insights are valuable
- **Not checking for overfitting** - feature selection can overfit
- **Ignoring feature stability** - unstable selection may not generalize

### Time Management:
- **Simple calculations**: 2-3 minutes
- **Medium complexity**: 5-8 minutes  
- **Complex derivations**: 10-15 minutes
- **Multi-part problems**: 15-20 minutes

### Final Checklist:
- [ ] All parts of question addressed
- [ ] Units and signs correct
- [ ] Results make intuitive sense
- [ ] Diagrams labeled clearly
- [ ] Key steps explained

---

## 🎯 Quick Reference: Decision Trees

### Which Feature Selection Method?
```
What are the requirements?
├─ Fast, general → Filter methods
├─ Accurate, specific → Wrapper methods
├─ Built-in selection → Embedded methods
└─ Real-time → Online selection
```

### Which Evaluation Criterion?
```
What's the data type?
├─ Numerical → Correlation, mutual information
├─ Categorical → Chi-square, information gain
├─ Mixed → Multiple criteria
└─ Text → TF-IDF, word embeddings
```

### How Many Features to Select?
```
What's the dataset size?
├─ Small (<1000) → 10-50% of features
├─ Medium (1000-10000) → 5-20% of features
└─ Large (>10000) → 1-10% of features
```

### Which Search Strategy?
```
What's the feature count?
├─ Small (<20) → Exhaustive search
├─ Medium (20-100) → Sequential search
└─ Large (>100) → Random search
```

---

*This guide covers the most common Feature Engineering and Selection question types. Practice with each approach and adapt based on specific problem requirements. Remember: understanding the concepts is more important than memorizing formulas!*
