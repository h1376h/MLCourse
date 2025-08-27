# Clustering and Unsupervised Learning Cheatsheet and "How To" Guide for Pen & Paper Exams

## 📋 Quick Reference Cheatsheet

### Core Distance Metrics

**Euclidean Distance:**
$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Manhattan Distance (L1):**
$$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Cosine Similarity:**
$$\cos(\theta) = \frac{x \cdot y}{\|x\| \|y\|} = \frac{\sum_{i=1}^{n}x_i y_i}{\sqrt{\sum_{i=1}^{n}x_i^2} \sqrt{\sum_{i=1}^{n}y_i^2}}$$

### K-Means Clustering

**Objective Function (WCSS):**
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Centroid Update:**
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

### Hierarchical Clustering

**Single Linkage:**
$$d(C_1, C_2) = \min_{x \in C_1, y \in C_2} d(x,y)$$

**Complete Linkage:**
$$d(C_1, C_2) = \max_{x \in C_1, y \in C_2} d(x,y)$$

**Ward Distance:**
$$d(C_1, C_2) = \frac{|C_1||C_2|}{|C_1| + |C_2|} \|\mu_1 - \mu_2\|^2$$

### DBSCAN Parameters

**ε-neighborhood:**
$$N_\varepsilon(x) = \{y \in D : d(x,y) \leq \varepsilon\}$$

**Core Point:**
$$|N_\varepsilon(x)| \geq \text{MinPts}$$

### Gaussian Mixture Models

**GMM Probability:**
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

**Responsibility (E-step):**
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

### Clustering Evaluation Metrics

**Silhouette Coefficient:**
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

**Calinski-Harabasz Index:**
$$\text{CH} = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{n - k}{k - 1}$$

---

## 🎯 Question Type 1: Distance Metrics and Similarity Calculations

### How to Approach:

**Step 1: Identify Data Type**
- **Numerical**: Euclidean, Manhattan, Minkowski
- **Categorical**: Hamming, Jaccard distance
- **Mixed**: Weighted combinations
- **High-dimensional**: Cosine similarity

**Step 2: Choose Appropriate Metric**
- **Euclidean**: Default for continuous data
- **Manhattan**: Robust to outliers, grid-like data
- **Cosine**: Direction similarity, text data
- **Minkowski**: General Lp norm

**Step 3: Calculate Distances**
- **Vector operations**: Component-wise calculations
- **Normalization**: Scale features if needed
- **Dimensionality**: Handle different feature scales

### Example Template:
```
Given: Points [x] and [y] with [n] dimensions
1. Data type analysis:
   - Data type: [numerical/categorical/mixed]
   - Dimensions: [n] features
   - Scale: [same/different] scales
2. Metric selection:
   - Chosen metric: [Euclidean/Manhattan/Cosine/Minkowski]
   - Justification: [appropriate for data type]
3. Distance calculation:
   - Euclidean: √(Σ(xᵢ - yᵢ)²) = [calculation] = [value]
   - Manhattan: Σ|xᵢ - yᵢ| = [calculation] = [value]
   - Cosine: [x·y]/(||x||·||y||) = [calculation] = [value]
4. Interpretation:
   - Similarity level: [high/medium/low] based on [distance value]
```

---

## 🎯 Question Type 2: K-Means Algorithm Implementation

### How to Approach:

**Step 1: Initialize Centroids**
- **Random**: Choose k random points
- **K-means++**: Probabilistic selection
- **Manual**: Specify initial positions
- **Multiple runs**: Different initializations

**Step 2: Assignment Step**
- **Calculate distances**: Point to all centroids
- **Assign to nearest**: Minimum distance centroid
- **Handle ties**: Consistent tie-breaking rule
- **Empty clusters**: Reinitialize or reassign

**Step 3: Update Step**
- **Calculate new centroids**: Mean of assigned points
- **Check convergence**: Centroids stop moving
- **Iteration limit**: Maximum iterations
- **WCSS tracking**: Monitor objective function

### Example Template:
```
Given: Dataset with [n] points, K=[k] clusters, initial centroids [centroids]
1. Initialization:
   - K = [k] clusters
   - Initial centroids: [list of centroids]
   - Method: [random/K-means++/manual]
2. Assignment step:
   - Point [i]: distances to centroids = [distances]
   - Assignment: [centroid] (minimum distance)
   - All assignments: [list of assignments]
3. Update step:
   - New centroid [i]: mean of [assigned points] = [new centroid]
   - All new centroids: [list of new centroids]
   - Movement: [old centroid] → [new centroid] = [distance]
4. Convergence check:
   - WCSS: [old value] → [new value]
   - Converged: [yes/no] - [reasoning]
```

---

## 🎯 Question Type 3: Hierarchical Clustering Analysis

### How to Approach:

**Step 1: Build Distance Matrix**
- **Pairwise distances**: All point pairs
- **Symmetric matrix**: d(i,j) = d(j,i)
- **Diagonal zeros**: d(i,i) = 0
- **Update strategy**: After each merge

**Step 2: Choose Linkage Method**
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance between clusters
- **Average**: Mean distance between clusters
- **Ward**: Minimizes within-cluster variance

**Step 3: Perform Merging**
- **Find minimum**: Smallest distance in matrix
- **Merge clusters**: Combine two closest clusters
- **Update matrix**: Recalculate distances
- **Record merge**: Height in dendrogram

### Example Template:
```
Given: [n] points with distance matrix [matrix]
1. Distance matrix:
   - Size: [n] × [n] matrix
   - Minimum distance: [value] between [clusters]
   - Linkage method: [single/complete/average/ward]
2. Merging process:
   - Step 1: Merge [cluster A] and [cluster B] at height [distance]
   - New cluster: [merged cluster]
   - Updated matrix: [new distances]
   - Step 2: [continue process]
3. Dendrogram construction:
   - Tree structure: [hierarchical representation]
   - Merge heights: [list of heights]
   - Cut points: [different K values]
4. Cluster interpretation:
   - K=[2]: [cluster assignments]
   - K=[3]: [cluster assignments]
   - Optimal K: [recommended value] based on [criterion]
```

---

## 🎯 Question Type 4: DBSCAN Algorithm Analysis

### How to Approach:

**Step 1: Understand Parameters**
- **ε (epsilon)**: Neighborhood radius
- **MinPts**: Minimum points for core
- **Core points**: |N_ε(x)| ≥ MinPts
- **Border points**: Connected to core points
- **Noise points**: Neither core nor border

**Step 2: Calculate Neighborhoods**
- **ε-neighborhood**: Points within radius ε
- **Core point identification**: Count neighbors
- **Density estimation**: Local point density
- **Connectivity**: Core point connections

**Step 3: Form Clusters**
- **Start with core points**: Seed clusters
- **Expand clusters**: Add connected points
- **Border point assignment**: Assign to nearest core
- **Noise identification**: Unassigned points

### Example Template:
```
Given: Dataset with ε=[epsilon] and MinPts=[minpts]
1. Parameter analysis:
   - ε = [epsilon] (neighborhood radius)
   - MinPts = [minpts] (minimum core points)
   - Total points: [n]
2. Point classification:
   - Core points: [list] with [count] neighbors each
   - Border points: [list] connected to cores
   - Noise points: [list] unassigned
3. Cluster formation:
   - Cluster 1: [core points] + [border points] = [total]
   - Cluster 2: [core points] + [border points] = [total]
   - Total clusters: [number]
4. Result analysis:
   - Clusters found: [number] (automatically determined)
   - Noise points: [number] ([percentage]%)
   - Cluster shapes: [spherical/arbitrary] due to [density-based approach]
```

---

## 🎯 Question Type 5: Gaussian Mixture Models and EM

### How to Approach:

**Step 1: Understand GMM Structure**
- **K components**: Number of Gaussian distributions
- **Parameters**: μ_k, Σ_k, π_k for each component
- **Mixture model**: Weighted sum of Gaussians
- **Soft clustering**: Probabilistic assignments

**Step 2: E-Step (Expectation)**
- **Calculate responsibilities**: γ_ik for each point
- **Posterior probabilities**: P(z_k|x_i)
- **Soft assignments**: Probabilistic cluster membership
- **Log-likelihood**: Model fit measure

**Step 3: M-Step (Maximization)**
- **Update means**: Weighted average of points
- **Update covariances**: Weighted scatter matrices
- **Update mixing weights**: Proportion of each component
- **Parameter constraints**: Valid probability distributions

### Example Template:
```
Given: GMM with K=[k] components, initial parameters [params]
1. Model structure:
   - K = [k] Gaussian components
   - Parameters: μ_k, Σ_k, π_k for each component
   - Initial values: [list of initial parameters]
2. E-step (Expectation):
   - Responsibilities γ_ik: [matrix of responsibilities]
   - Log-likelihood: [value]
   - Soft assignments: [probabilistic cluster assignments]
3. M-step (Maximization):
   - Updated means: [new μ values]
   - Updated covariances: [new Σ values]
   - Updated weights: [new π values]
4. Convergence analysis:
   - Log-likelihood change: [old] → [new] = [improvement]
   - Parameter convergence: [yes/no]
   - Final clustering: [hard assignments based on max probability]
```

---

## 🎯 Question Type 6: Clustering Evaluation Metrics

### How to Approach:

**Step 1: Choose Evaluation Type**
- **Internal**: No ground truth (Silhouette, CH, DB)
- **External**: With ground truth (ARI, NMI, Purity)
- **Relative**: Compare different K values (Elbow, Gap)
- **Stability**: Consistency across runs

**Step 2: Calculate Internal Metrics**
- **Silhouette**: Individual point quality
- **Calinski-Harabasz**: Between/within cluster ratio
- **Davies-Bouldin**: Average cluster similarity
- **Interpretation**: Higher/lower is better

**Step 3: Calculate External Metrics**
- **Adjusted Rand Index**: Agreement with ground truth
- **Normalized Mutual Information**: Information overlap
- **Purity**: Percentage of dominant class
- **Range**: 0 (random) to 1 (perfect)

### Example Template:
```
Given: Clustering result with [n] points, K=[k] clusters, [ground_truth]
1. Evaluation type:
   - Internal metrics: [Silhouette/CH/DB] (no ground truth)
   - External metrics: [ARI/NMI/Purity] (with ground truth)
   - Relative metrics: [Elbow/Gap] (compare K values)
2. Internal metric calculation:
   - Silhouette coefficient: [value] (range: -1 to 1)
   - Calinski-Harabasz: [value] (higher is better)
   - Davies-Bouldin: [value] (lower is better)
3. External metric calculation:
   - Adjusted Rand Index: [value] (range: 0 to 1)
   - Normalized Mutual Information: [value] (range: 0 to 1)
   - Purity: [value] (range: 0 to 1)
4. Interpretation:
   - Clustering quality: [excellent/good/poor] based on [metrics]
   - Optimal K: [recommended value] based on [criterion]
   - Algorithm comparison: [algorithm A] vs [algorithm B] = [better/worse]
```

---

## 🎯 Question Type 7: Advanced Clustering Techniques

### How to Approach:

**Step 1: Spectral Clustering**
- **Similarity matrix**: Convert distances to similarities
- **Graph Laplacian**: L = D - A (degree - adjacency)
- **Eigendecomposition**: Find smallest eigenvectors
- **K-means on eigenvectors**: Final clustering

**Step 2: Mean Shift Clustering**
- **Kernel function**: Gaussian or other kernels
- **Mean shift vector**: Gradient ascent direction
- **Mode seeking**: Find local density maxima
- **Bandwidth selection**: Critical parameter

**Step 3: Affinity Propagation**
- **Message passing**: Responsibility and availability
- **Exemplars**: Cluster representatives
- **Preference parameter**: Controls cluster number
- **Automatic K**: No need to specify number

### Example Template:
```
Given: [advanced_method] for dataset with [characteristics]
1. Method analysis:
   - Algorithm: [Spectral/Mean Shift/Affinity Propagation]
   - Key principle: [graph-based/density-based/message-passing]
   - Parameters: [list of key parameters]
2. Algorithm steps:
   - Step 1: [description of first step]
   - Step 2: [description of second step]
   - Step 3: [description of third step]
   - Output: [clustering result]
3. Advantages:
   - [advantage 1]: [explanation]
   - [advantage 2]: [explanation]
   - [advantage 3]: [explanation]
4. Limitations:
   - [limitation 1]: [explanation]
   - [limitation 2]: [explanation]
   - [limitation 3]: [explanation]
5. Scalability:
   - Time complexity: O([complexity])
   - Space complexity: O([complexity])
   - Large dataset handling: [sampling/approximation/online]
```

---

## 🎯 Question Type 8: Clustering Applications and Case Studies

### How to Approach:

**Step 1: Understand Application Domain**
- **Business context**: Customer segmentation, marketing
- **Technical context**: Image processing, text mining
- **Scientific context**: Bioinformatics, astronomy
- **Requirements**: Accuracy, interpretability, scalability

**Step 2: Data Preprocessing**
- **Feature engineering**: Domain-specific features
- **Normalization**: Scale features appropriately
- **Dimensionality reduction**: PCA, feature selection
- **Outlier handling**: Remove or treat outliers

**Step 3: Algorithm Selection**
- **Data characteristics**: Shape, size, dimensionality
- **Business requirements**: Interpretability, speed
- **Domain knowledge**: Expected cluster structure
- **Evaluation criteria**: Success metrics

### Example Template:
```
Given: [application_domain] with [specific_requirements]
1. Domain analysis:
   - Application: [customer_segmentation/image_segmentation/document_clustering]
   - Business objectives: [list of objectives]
   - Success metrics: [accuracy/interpretability/speed]
2. Data preprocessing:
   - Features: [list of relevant features]
   - Normalization: [standardization/min-max/robust]
   - Dimensionality: [original] → [reduced] features
   - Outliers: [removed/treated] using [method]
3. Algorithm selection:
   - Chosen algorithm: [K-means/DBSCAN/hierarchical/GMM]
   - Justification: [appropriate for data characteristics]
   - Parameters: [list of key parameters]
4. Implementation results:
   - Clusters found: [number] with [characteristics]
   - Performance: [evaluation metrics]
   - Business insights: [interpretable results]
   - Deployment: [production_ready/needs_refinement]
```

---

## 🎯 Question Type 9: Parameter Selection and Model Selection

### How to Approach:

**Step 1: Choose Number of Clusters (K)**
- **Elbow method**: WCSS vs K plot
- **Gap statistic**: Compare to random data
- **Silhouette analysis**: Average silhouette vs K
- **Business context**: Domain knowledge

**Step 2: Algorithm-Specific Parameters**
- **K-means**: Initialization method, number of runs
- **DBSCAN**: ε and MinPts selection
- **GMM**: Covariance type, initialization
- **Hierarchical**: Linkage method, distance metric

**Step 3: Validation Strategy**
- **Cross-validation**: Stability across subsets
- **Bootstrap**: Resampling approach
- **Multiple runs**: Different initializations
- **External validation**: Ground truth if available

### Example Template:
```
Given: Dataset with [characteristics] and [evaluation_criteria]
1. K selection analysis:
   - Elbow method: K = [optimal_K] at [elbow_point]
   - Gap statistic: K = [optimal_K] with gap = [value]
   - Silhouette analysis: K = [optimal_K] with score = [value]
   - Business recommendation: K = [recommended_K]
2. Parameter tuning:
   - Algorithm: [K-means/DBSCAN/GMM/hierarchical]
   - Key parameters: [list of parameters]
   - Optimal values: [list of optimal values]
   - Tuning method: [grid_search/cross_validation/expertise]
3. Validation results:
   - Stability: [high/medium/low] across [validation_method]
   - Performance: [evaluation_metrics]
   - Robustness: [parameter_sensitivity_analysis]
4. Model comparison:
   - Algorithm A: [performance] with [pros/cons]
   - Algorithm B: [performance] with [pros/cons]
   - Recommendation: [best_algorithm] because [justification]
```

---

## 🎯 Question Type 10: Clustering Challenges and Solutions

### How to Approach:

**Step 1: Identify Common Challenges**
- **Curse of dimensionality**: High-dimensional data
- **Noise and outliers**: Data quality issues
- **Cluster shapes**: Non-spherical clusters
- **Scalability**: Large datasets
- **Interpretability**: Business understanding

**Step 2: Apply Appropriate Solutions**
- **Dimensionality reduction**: PCA, feature selection
- **Robust algorithms**: DBSCAN, robust K-means
- **Advanced methods**: Spectral clustering, GMM
- **Sampling strategies**: Mini-batch, online clustering
- **Visualization**: 2D/3D plots, dendrograms

**Step 3: Evaluate Solution Effectiveness**
- **Performance improvement**: Before vs after
- **Computational cost**: Time and memory trade-offs
- **Interpretability**: Business insights gained
- **Robustness**: Stability across different runs

### Example Template:
```
Given: Clustering challenge with [specific_problem]
1. Challenge identification:
   - Problem: [dimensionality/noise/shapes/scalability/interpretability]
   - Impact: [performance_degradation/computational_cost/business_insights]
   - Root cause: [data_characteristics/algorithm_limitations/requirements]
2. Solution implementation:
   - Approach: [dimensionality_reduction/robust_algorithm/advanced_method/sampling/visualization]
   - Specific method: [PCA/DBSCAN/spectral_clustering/mini_batch/plots]
   - Parameters: [list of solution parameters]
3. Effectiveness evaluation:
   - Performance: [before] → [after] = [improvement]
   - Computational cost: [time] and [memory] requirements
   - Interpretability: [business_insights_gained]
   - Robustness: [stability_improvement]
4. Best practices:
   - Data preprocessing: [essential_steps]
   - Algorithm selection: [decision_criteria]
   - Parameter tuning: [systematic_approach]
   - Evaluation: [comprehensive_metrics]
```

---

## 📝 General Exam Strategy

### Before Starting:
1. **Read the entire question** - identify all parts
2. **Identify clustering type** - partitional, hierarchical, density-based, model-based
3. **Plan your time** - allocate based on question complexity
4. **Gather formulas** - write down relevant equations

### Common Mistakes to Avoid:
- **Using Euclidean distance for text data** - use Cosine similarity instead
- **Not normalizing features** - different scales affect clustering
- **Choosing K arbitrarily** - use elbow method or gap statistic
- **Ignoring cluster shapes** - K-means assumes spherical clusters
- **Not validating results** - always check clustering quality
- **Forgetting computational complexity** - some algorithms don't scale
- **Overlooking interpretability** - business context matters

### Quick Reference Decision Trees:

**Which Clustering Algorithm?**
```
Data Characteristics:
├─ Spherical clusters → K-means, GMM
├─ Arbitrary shapes → DBSCAN, Spectral
├─ Hierarchical structure → Hierarchical clustering
└─ Probabilistic assignments → GMM, EM
```

**Which Distance Metric?**
```
Data Type:
├─ Continuous numerical → Euclidean, Manhattan
├─ High-dimensional → Cosine similarity
├─ Categorical → Hamming, Jaccard
└─ Mixed types → Weighted combinations
```

**How to Choose K?**
```
Method:
├─ Business knowledge → Domain expertise
├─ Elbow method → WCSS vs K plot
├─ Gap statistic → Compare to random data
└─ Silhouette analysis → Average silhouette vs K
```

**Which Evaluation Metric?**
```
Scenario:
├─ No ground truth → Internal (Silhouette, CH, DB)
├─ With ground truth → External (ARI, NMI, Purity)
├─ Compare K values → Relative (Elbow, Gap)
└─ Algorithm comparison → Multiple metrics
```

---

*This guide covers the most common Clustering and Unsupervised Learning question types. Practice with each approach and adapt based on specific problem requirements. Remember: clustering is about finding meaningful patterns in data!*
