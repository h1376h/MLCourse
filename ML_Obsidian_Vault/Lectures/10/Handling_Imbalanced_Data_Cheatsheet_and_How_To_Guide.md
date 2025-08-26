# Handling Imbalanced Data Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core Imbalanced Data Formulas

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

**Sampling Methods:**
- **Oversampling Factor**: $\text{OS Factor} = \frac{\text{Target Majority Count}}{\text{Current Minority Count}}$
- **Undersampling Factor**: $\text{US Factor} = \frac{\text{Target Minority Count}}{\text{Current Majority Count}}$
- **SMOTE Ratio**: $\text{SMOTE Ratio} = \frac{\text{Desired Minority Count}}{\text{Current Minority Count}}$

**Cost-Sensitive Learning:**
- **Cost Matrix**: $C_{ij}$ = cost of predicting class $i$ when true class is $j$
- **Weighted Loss**: $\text{Weighted Loss} = \sum_{i,j} C_{ij} \times \text{Confusion}_{ij}$
- **Class Weights**: $\text{Weight}_i = \frac{\text{Total Samples}}{\text{Number of Classes} \times \text{Class}_i \text{ Count}}$

### Key Concepts

**Imbalance Levels:**
- **Mild Imbalance**: IR < 3:1 (minority > 25%)
- **Moderate Imbalance**: 3:1 ≤ IR ≤ 10:1 (10% ≤ minority ≤ 25%)
- **Severe Imbalance**: IR > 10:1 (minority < 10%)

**Sampling Techniques:**
- **Random Oversampling**: Duplicate minority samples
- **Random Undersampling**: Remove majority samples
- **SMOTE**: Generate synthetic minority samples
- **ADASYN**: Adaptive synthetic sampling
- **Borderline SMOTE**: Focus on decision boundaries

**Evaluation Approaches:**
- **Precision-Recall Curves**: Better than ROC for imbalanced data
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Metrics**: Account for class imbalance
- **Cost-Sensitive**: Weight errors by misclassification cost

---

## 🎯 Question Type 1: Class Imbalance Detection and Analysis

### How to Approach:

**Step 1: Calculate Imbalance Ratios**
- **Count samples**: Count samples in each class
- **Calculate IR**: Majority count / Minority count
- **Determine level**: Mild, moderate, or severe imbalance

**Step 2: Analyze Distribution**
- **Class percentages**: Percentage of each class
- **Sample counts**: Absolute numbers in each class
- **Visualization**: Bar chart or pie chart representation

**Step 3: Identify Problems**
- **Accuracy paradox**: High accuracy but poor minority performance
- **Bias towards majority**: Model favors majority class
- **Poor generalization**: Model doesn't learn minority patterns

**Step 4: Assess Impact**
- **Business impact**: Cost of missing minority cases
- **Model performance**: How imbalance affects predictions
- **Evaluation bias**: Why standard metrics fail

### Example Template:
```
Given: Dataset with [n] total samples, [majority_count] majority class, [minority_count] minority class
1. Imbalance analysis:
   - Total samples: [n]
   - Majority class: [majority_count] samples ([majority_percent]%)
   - Minority class: [minority_count] samples ([minority_percent]%)
   - Imbalance ratio: [majority_count]/[minority_count] = [IR]
2. Imbalance level:
   - Level: [Mild/Moderate/Severe] imbalance
   - Justification: IR = [IR] is [<3/3-10/>10]
3. Problem identification:
   - Accuracy paradox: [present/absent] - [explanation]
   - Bias towards majority: [present/absent] - [explanation]
   - Poor generalization: [present/absent] - [explanation]
4. Impact assessment:
   - Business impact: [high/medium/low] cost of missing minority cases
   - Model performance: [severely/partially/not] affected by imbalance
```

---

## 🎯 Question Type 2: Evaluation Metrics for Imbalanced Data

### How to Approach:

**Step 2: Calculate Balanced Metrics**
- **Precision**: Focus on positive predictive value
- **Recall**: Focus on sensitivity (true positive rate)
- **F1 Score**: Harmonic mean for balanced evaluation
- **Balanced Accuracy**: Average of sensitivity and specificity

**Step 3: Compare with Standard Metrics**
- **Accuracy**: Overall correctness (misleading for imbalance)
- **Precision-Recall**: Better than ROC for imbalanced data
- **F1 vs Accuracy**: Why F1 is more appropriate

**Step 4: Interpret Results**
- **High precision**: Few false positives
- **High recall**: Few false negatives
- **Balanced F1**: Good overall performance
- **Business context**: Which metric matters most

### Example Template:
```
Given: Confusion matrix with [TP], [TN], [FP], [FN] for imbalanced dataset
1. Standard metrics:
   - Accuracy = ([TP] + [TN]) / ([TP] + [TN] + [FP] + [FN]) = [value]
   - Precision = [TP] / ([TP] + [FP]) = [value]
   - Recall = [TP] / ([TP] + [FN]) = [value]
2. Balanced metrics:
   - F1 Score = 2 × [precision] × [recall] / ([precision] + [recall]) = [value]
   - Balanced Accuracy = ([recall] + [specificity]) / 2 = [value]
   - G-Mean = √([recall] × [specificity]) = [value]
3. Metric comparison:
   - Accuracy: [misleading/appropriate] because [reasoning]
   - F1 Score: [better/worse] than accuracy because [reasoning]
   - Precision-Recall: [better/worse] than ROC because [reasoning]
4. Business interpretation:
   - [High/Medium/Low] precision indicates [few/many] false positives
   - [High/Medium/Low] recall indicates [few/many] false negatives
   - Primary metric: [precision/recall/F1] based on [business requirement]
```

---

## 🎯 Question Type 3: Random Oversampling Analysis

### How to Approach:

**Step 1: Understand Oversampling Process**
- **Duplicate samples**: Copy minority class samples
- **Target balance**: Achieve desired class distribution
- **Oversampling factor**: How many times to duplicate

**Step 2: Calculate Oversampling Parameters**
- **Current distribution**: Original class counts
- **Target distribution**: Desired class counts
- **Oversampling factor**: Target minority / Current minority
- **New sample count**: Original minority × Oversampling factor

**Step 3: Analyze Advantages**
- **No information loss**: All original data preserved
- **Simple implementation**: Easy to understand and implement
- **Balanced dataset**: Equal representation of classes

**Step 4: Identify Limitations**
- **Overfitting**: Model memorizes duplicated samples
- **No new information**: No additional patterns learned
- **Computational cost**: Larger training dataset

### Example Template:
```
Given: Dataset with [majority_count] majority and [minority_count] minority samples
1. Oversampling analysis:
   - Current minority: [minority_count] samples
   - Target minority: [target_count] samples (to match majority)
   - Oversampling factor: [target_count]/[minority_count] = [factor]
   - New minority count: [minority_count] × [factor] = [new_count]
2. Process description:
   - Method: Random oversampling (duplication)
   - Samples to duplicate: [minority_count] original samples
   - Duplication count: [factor] times each sample
   - Final distribution: [majority_count] vs [new_count]
3. Advantages:
   - Information loss: [none] - all original data preserved
   - Implementation: [simple] - easy to understand
   - Balance: [achieved] - equal class representation
4. Limitations:
   - Overfitting risk: [high/medium/low] due to [duplication]
   - New information: [none] - no additional patterns
   - Computational cost: [increased] - [new_count] total samples
```

---

## 🎯 Question Type 4: Random Undersampling Analysis

### How to Approach:

**Step 1: Understand Undersampling Process**
- **Remove samples**: Randomly remove majority class samples
- **Target balance**: Achieve desired class distribution
- **Undersampling factor**: Fraction of majority to keep

**Step 2: Calculate Undersampling Parameters**
- **Current distribution**: Original class counts
- **Target distribution**: Desired class counts
- **Undersampling factor**: Target majority / Current majority
- **Samples to remove**: Current majority - Target majority

**Step 3: Analyze Advantages**
- **Reduced computational cost**: Smaller training dataset
- **Balanced dataset**: Equal representation of classes
- **No synthetic data**: All samples are real

**Step 4: Identify Limitations**
- **Information loss**: Valuable majority samples removed
- **Reduced training data**: Less data for model learning
- **Random selection**: May remove important patterns

### Example Template:
```
Given: Dataset with [majority_count] majority and [minority_count] minority samples
1. Undersampling analysis:
   - Current majority: [majority_count] samples
   - Target majority: [target_count] samples (to match minority)
   - Undersampling factor: [target_count]/[majority_count] = [factor]
   - Samples to remove: [majority_count] - [target_count] = [remove_count]
2. Process description:
   - Method: Random undersampling (removal)
   - Samples to keep: [target_count] majority samples
   - Removal strategy: Random selection
   - Final distribution: [target_count] vs [minority_count]
3. Advantages:
   - Computational cost: [reduced] - smaller dataset
   - Balance: [achieved] - equal class representation
   - Data quality: [real] - no synthetic samples
4. Limitations:
   - Information loss: [high/medium/low] - [remove_count] samples removed
   - Training data: [reduced] - less data for learning
   - Pattern loss: [possible] - important samples may be removed
```

---

## 🎯 Question Type 5: SMOTE Algorithm Analysis

### How to Approach:

**Step 1: Understand SMOTE Process**
- **K-nearest neighbors**: Find k nearest minority neighbors
- **Linear interpolation**: Create synthetic samples between neighbors
- **Random selection**: Choose random neighbor for interpolation
- **Synthetic generation**: Generate new minority samples

**Step 2: Calculate SMOTE Parameters**
- **K value**: Number of nearest neighbors (typically 5)
- **SMOTE ratio**: Desired minority / Current minority
- **Synthetic samples**: Number of new samples to generate
- **Interpolation factor**: Random factor between 0 and 1

**Step 3: Analyze Advantages**
- **New information**: Creates synthetic minority samples
- **No overfitting**: Avoids exact duplication
- **Better generalization**: Model learns minority patterns

**Step 4: Identify Limitations**
- **Noise sensitivity**: Sensitive to outliers
- **Boundary issues**: May create samples in majority region
- **Computational cost**: KNN calculation required

### Example Template:
```
Given: Minority class with [minority_count] samples, k=[k] nearest neighbors
1. SMOTE parameters:
   - Current minority: [minority_count] samples
   - Target minority: [target_count] samples
   - SMOTE ratio: [target_count]/[minority_count] = [ratio]
   - Synthetic samples needed: [target_count] - [minority_count] = [synthetic_count]
   - K nearest neighbors: [k]
2. SMOTE process:
   - For each minority sample: Find [k] nearest minority neighbors
   - Random selection: Choose random neighbor for interpolation
   - Linear interpolation: Create synthetic sample between original and neighbor
   - Interpolation factor: Random value between 0 and 1
3. Advantages:
   - New information: [yes] - synthetic samples created
   - Overfitting: [reduced] - no exact duplication
   - Generalization: [improved] - learns minority patterns
4. Limitations:
   - Noise sensitivity: [high/medium/low] - sensitive to outliers
   - Boundary issues: [possible] - may create samples in wrong region
   - Computational cost: [increased] - KNN calculation required
```

---

## 🎯 Question Type 6: Advanced Synthetic Methods Analysis

### How to Approach:

**Step 1: Understand Advanced Methods**
- **ADASYN**: Adaptive synthetic sampling based on difficulty
- **Borderline SMOTE**: Focus on decision boundary samples
- **Safe Level SMOTE**: Generate samples in safe regions
- **Data Augmentation**: Create variations of existing samples

**Step 2: Compare with Basic SMOTE**
- **ADASYN**: Weights samples by classification difficulty
- **Borderline SMOTE**: Only generates samples near decision boundary
- **Safe Level SMOTE**: Ensures synthetic samples are in minority region
- **Data Augmentation**: Domain-specific transformations

**Step 3: Analyze Method Selection**
- **Dataset characteristics**: Noise level, boundary complexity
- **Computational resources**: Processing time and memory
- **Domain knowledge**: Understanding of data distribution
- **Performance requirements**: Accuracy vs speed trade-offs

**Step 4: Evaluate Effectiveness**
- **Classification performance**: Impact on model accuracy
- **Generalization**: Performance on test data
- **Stability**: Consistency across different runs
- **Interpretability**: Understanding of synthetic samples

### Example Template:
```
Given: [method] for imbalanced dataset with [characteristics]
1. Method analysis:
   - Method: [ADASYN/Borderline SMOTE/Safe Level SMOTE/Data Augmentation]
   - Key feature: [adaptive sampling/boundary focus/safe regions/transformations]
   - Target samples: [difficult samples/boundary samples/safe samples/variations]
2. Comparison with SMOTE:
   - Sample selection: [weighted/selective/safe/transformed] vs [random]
   - Generation strategy: [adaptive/boundary-focused/safe/augmented] vs [interpolation]
   - Computational cost: [higher/similar/lower] than SMOTE
3. Method selection:
   - Dataset characteristics: [noisy/clean] - [method] is [appropriate/inappropriate]
   - Boundary complexity: [simple/complex] - [method] is [suitable/unsuitable]
   - Domain knowledge: [required/not required] for [method]
4. Effectiveness evaluation:
   - Classification performance: [improved/similar/worse] than SMOTE
   - Generalization: [better/similar/worse] on test data
   - Stability: [high/medium/low] consistency across runs
```

---

## 🎯 Question Type 7: Hybrid and Ensemble Methods Analysis

### How to Approach:

**Step 1: Understand Hybrid Methods**
- **SMOTEENN**: SMOTE + Edited Nearest Neighbors
- **SMOTETomek**: SMOTE + Tomek links removal
- **Combination approach**: Oversampling + undersampling
- **Cleaning phase**: Remove noisy or borderline samples

**Step 2: Analyze Ensemble Methods**
- **Multiple models**: Train different models on balanced subsets
- **Voting strategies**: Combine predictions from multiple models
- **Bagging**: Bootstrap sampling for balanced datasets
- **Boosting**: Sequential learning with sample weights

**Step 3: Calculate Ensemble Parameters**
- **Number of models**: How many models in ensemble
- **Sampling strategy**: How to create balanced subsets
- **Voting weights**: How to combine model predictions
- **Performance aggregation**: How to measure ensemble performance

**Step 4: Compare with Single Methods**
- **Performance**: Ensemble vs single model performance
- **Robustness**: Stability across different datasets
- **Computational cost**: Training time and memory requirements
- **Interpretability**: Understanding of ensemble decisions

### Example Template:
```
Given: [hybrid_method] with [n] models for imbalanced dataset
1. Hybrid method analysis:
   - Method: [SMOTEENN/SMOTETomek/Ensemble]
   - Components: [SMOTE + ENN/SMOTE + Tomek/Multiple models]
   - Cleaning phase: [yes/no] - [description of cleaning]
2. Ensemble parameters:
   - Number of models: [n]
   - Sampling strategy: [bootstrap/stratified/balanced]
   - Voting method: [majority/weighted/average]
   - Performance aggregation: [mean/median/max]
3. Process description:
   - Step 1: [oversampling/undersampling] to create balanced subsets
   - Step 2: [cleaning/removal] of [noisy/borderline] samples
   - Step 3: Train [n] models on [balanced/cleaned] data
   - Step 4: Combine predictions using [voting method]
4. Comparison with single methods:
   - Performance: [better/similar/worse] than [single_method]
   - Robustness: [higher/medium/lower] stability
   - Computational cost: [n] times [single_method] cost
   - Interpretability: [more/less] complex than single model
```

---

## 🎯 Question Type 8: Cost-Sensitive Learning Analysis

### How to Approach:

**Step 1: Define Cost Matrix**
- **False positive cost**: Cost of predicting positive when negative
- **False negative cost**: Cost of predicting negative when positive
- **True positive/negative cost**: Usually 0 (correct predictions)
- **Cost ratio**: FN cost / FP cost

**Step 2: Calculate Weighted Metrics**
- **Weighted accuracy**: Account for misclassification costs
- **Expected cost**: Average cost across all predictions
- **Cost-sensitive threshold**: Optimal decision threshold
- **Class weights**: Inverse of class frequencies

**Step 3: Analyze Algorithm Modifications**
- **Loss function**: Modify loss to include costs
- **Decision threshold**: Adjust threshold based on costs
- **Sample weights**: Weight samples by misclassification cost
- **Ensemble weights**: Weight models by cost performance

**Step 4: Compare with Sampling Methods**
- **Performance**: Cost-sensitive vs sampling performance
- **Interpretability**: Understanding of cost-based decisions
- **Flexibility**: Ability to adjust for different cost scenarios
- **Implementation**: Ease of implementation and tuning

### Example Template:
```
Given: Cost matrix with FP cost = [fp_cost] and FN cost = [fn_cost]
1. Cost matrix analysis:
   - False positive cost: [fp_cost] (predict positive when negative)
   - False negative cost: [fn_cost] (predict negative when positive)
   - Cost ratio: [fn_cost]/[fp_cost] = [ratio]
   - Expected cost: [fp_cost] × FP + [fn_cost] × FN = [expected_cost]
2. Weighted metrics:
   - Class weights: Majority = [majority_weight], Minority = [minority_weight]
   - Weighted accuracy: [weighted_accuracy]
   - Cost-sensitive threshold: [threshold] (instead of 0.5)
3. Algorithm modifications:
   - Loss function: [modified] to include misclassification costs
   - Decision threshold: [adjusted] based on cost ratio
   - Sample weights: [applied] based on class importance
4. Comparison with sampling:
   - Performance: [better/similar/worse] than [sampling_method]
   - Interpretability: [more/less] interpretable than sampling
   - Flexibility: [more/less] flexible for different scenarios
   - Implementation: [easier/harder] than sampling methods
```

---

## 🎯 Question Type 9: Method Selection and Comparison

### How to Approach:

**Step 1: Assess Dataset Characteristics**
- **Imbalance level**: Mild, moderate, or severe
- **Dataset size**: Small, medium, or large
- **Noise level**: Clean, moderate, or noisy data
- **Feature space**: Low or high dimensional

**Step 2: Consider Business Requirements**
- **Cost of false positives**: Business impact of FP errors
- **Cost of false negatives**: Business impact of FN errors
- **Interpretability**: Need for model explanation
- **Computational constraints**: Time and memory limits

**Step 3: Evaluate Method Performance**
- **Classification metrics**: Precision, recall, F1, AUC
- **Computational cost**: Training and prediction time
- **Stability**: Consistency across different runs
- **Generalization**: Performance on unseen data

**Step 4: Make Recommendation**
- **Primary method**: Best overall approach
- **Alternative methods**: Backup options
- **Justification**: Clear reasoning for selection
- **Implementation plan**: Step-by-step approach

### Example Template:
```
Given: [dataset_characteristics] with [business_requirements]
1. Dataset assessment:
   - Imbalance level: [mild/moderate/severe] (IR = [ratio])
   - Dataset size: [small/medium/large] ([n] samples)
   - Noise level: [clean/moderate/noisy]
   - Feature space: [low/high] dimensional
2. Business requirements:
   - False positive cost: [high/medium/low]
   - False negative cost: [high/medium/low]
   - Interpretability: [required/not required]
   - Computational constraints: [strict/moderate/none]
3. Method evaluation:
   - Random oversampling: [performance] - [pros/cons]
   - Random undersampling: [performance] - [pros/cons]
   - SMOTE: [performance] - [pros/cons]
   - Cost-sensitive: [performance] - [pros/cons]
4. Recommendation:
   - Primary method: [recommended_method]
   - Alternative: [backup_method]
   - Justification: [clear reasoning]
   - Implementation: [step-by-step plan]
```

---

## 🎯 Question Type 10: Real-World Application Analysis

### How to Approach:

**Step 1: Understand Domain Context**
- **Application area**: Fraud detection, medical diagnosis, etc.
- **Business impact**: Cost of different types of errors
- **Data characteristics**: Typical imbalance ratios and patterns
- **Regulatory requirements**: Compliance and interpretability needs

**Step 2: Analyze Specific Challenges**
- **Class imbalance**: Typical ratios in the domain
- **Data quality**: Noise, missing values, outliers
- **Feature engineering**: Domain-specific features
- **Evaluation criteria**: Business-relevant metrics

**Step 3: Design Solution Strategy**
- **Data preprocessing**: Handling domain-specific issues
- **Sampling strategy**: Appropriate method for the domain
- **Model selection**: Suitable algorithms for the problem
- **Evaluation approach**: Domain-relevant metrics

**Step 4: Assess Implementation**
- **Deployment considerations**: Real-time vs batch processing
- **Monitoring requirements**: Performance tracking and updates
- **Maintenance needs**: Model retraining and validation
- **Success metrics**: Business impact measurement

### Example Template:
```
Given: [domain] application with [specific_requirements]
1. Domain context:
   - Application: [fraud_detection/medical_diagnosis/customer_churn/etc.]
   - Business impact: [high/medium/low] cost of errors
   - Typical imbalance: [ratio] (majority:minority)
   - Regulatory needs: [compliance/interpretability] requirements
2. Domain challenges:
   - Class imbalance: [severe/moderate/mild] - [ratio]
   - Data quality: [clean/moderate/noisy] - [specific_issues]
   - Feature engineering: [simple/complex] - [domain_features]
   - Evaluation: [precision/recall/F1] most important
3. Solution strategy:
   - Preprocessing: [handling_missing_values/outlier_removal/feature_scaling]
   - Sampling: [SMOTE/undersampling/cost-sensitive] for [reason]
   - Model: [algorithm] because [domain_suitability]
   - Evaluation: [metrics] based on [business_impact]
4. Implementation assessment:
   - Deployment: [real-time/batch] processing
   - Monitoring: [continuous/periodic] performance tracking
   - Maintenance: [frequent/infrequent] model updates
   - Success: [business_metrics] for impact measurement
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify imbalance type** - mild, moderate, or severe
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### Common Mistakes to Avoid:
- **Using accuracy for imbalanced data** - accuracy is misleading
- **Ignoring business context** - different costs for different errors
- **Not considering computational cost** - some methods are expensive
- **Forgetting to validate** - always test on holdout set
- **Overlooking data quality** - noise affects synthetic methods
- **Not checking assumptions** - verify method assumptions
- **Ignoring interpretability** - some methods are black boxes

### Quick Reference Decision Trees:

**Which Sampling Method?**
```
Imbalance Level:
├─ Mild (IR < 3) → No sampling needed
├─ Moderate (3 ≤ IR ≤ 10) → SMOTE or cost-sensitive
└─ Severe (IR > 10) → Advanced methods or ensemble
```

**Which Evaluation Metric?**
```
Business Context:
├─ FP cost high → Precision
├─ FN cost high → Recall
├─ Both important → F1 Score
└─ Balanced view → Balanced Accuracy
```

**When to Use Cost-Sensitive?**
```
Requirements:
├─ Known costs → Cost-sensitive learning
├─ Unknown costs → Sampling methods
├─ Interpretability needed → Cost-sensitive
└─ Computational constraints → Sampling
```

---

*This guide covers the most common Handling Imbalanced Data question types. Practice with each approach and adapt based on specific problem requirements. Remember: proper handling of imbalanced data is crucial for real-world applications!*
