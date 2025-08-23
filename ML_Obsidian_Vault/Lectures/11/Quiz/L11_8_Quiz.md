# Lecture 11.8: Clustering Applications and Case Studies Quiz

## Overview
This quiz contains 20 questions covering different topics from section 11.8 of the lectures on Clustering Applications and Case Studies, including image segmentation, customer segmentation, document clustering, anomaly detection, real-world applications, and practical implementation challenges.

## Question 1

### Problem Statement
Image segmentation using clustering divides an image into meaningful regions by grouping similar pixels.

#### Task
1. What is image segmentation and how does clustering apply to this problem?
2. What features can be used for pixel clustering in image segmentation?
3. Compare K-Means vs Mean Shift for image segmentation applications
4. How do you incorporate spatial information in addition to color information?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 1: Image Segmentation Fundamentals](L11_8_1_explanation.md).

## Question 2

### Problem Statement
Consider segmenting a color image using K-Means clustering in RGB space.

An image has pixels with the following RGB values: $(255,0,0)$, $(250,10,5)$, $(0,255,0)$, $(5,250,10)$, $(0,0,255)$, $(10,5,250)$

#### Task
1. Apply K-Means with $K=3$ to cluster these pixels into color regions
2. Calculate the initial centroids using K-Means++ initialization
3. Show one iteration of the algorithm
4. How would the results change if you used Lab color space instead of RGB?
5. Calculate the compression ratio achieved by this clustering. If each original RGB pixel uses $24$ bits ($8$ bits per channel) and the compressed version uses $2$ bits per pixel (to encode $3$ cluster IDs) plus the storage for $3$ centroids, what is the total compression ratio? Also calculate the quantization error as the sum of squared distances from each pixel to its assigned centroid.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 2: Color-Based Image Segmentation](L11_8_2_explanation.md).

## Question 3

### Problem Statement
Customer segmentation helps businesses understand distinct groups within their customer base for targeted marketing.

#### Task
1. What are the key objectives of customer segmentation using clustering?
2. List six customer attributes commonly used for segmentation
3. How would you handle categorical attributes like "Preferred Product Category" in clustering?
4. How do you translate clustering results into actionable business insights?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 3: Customer Segmentation Strategy](L11_8_3_explanation.md).

## Question 4

### Problem Statement
Consider a retail company with customer data for segmentation:

| Customer | Age | Income | Frequency | Recency | Monetary |
|----------|-----|--------|-----------|---------|----------|
| A        | 25  | 50k    | 12        | 5       | 1200     |
| B        | 45  | 80k    | 8         | 15      | 2400     |
| C        | 35  | 60k    | 20        | 2       | 3000     |
| D        | 55  | 90k    | 3         | 90      | 800      |

#### Task
1. Normalize the features for clustering analysis
2. What value of $K$ would you recommend for this segmentation?
3. Interpret the business meaning of each potential cluster
4. How would you validate the quality of this segmentation for marketing purposes?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 4: Customer Segmentation Case Study](L11_8_4_explanation.md).

## Question 5

### Problem Statement
Document clustering groups text documents by content similarity for organization and analysis.

#### Task
1. How do you represent text documents for clustering algorithms?
2. What preprocessing steps are essential for document clustering?
3. Why is Cosine similarity preferred over Euclidean distance for document clustering?
4. How do you handle documents of vastly different lengths?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 5: Document Clustering Fundamentals](L11_8_5_explanation.md).

## Question 6

### Problem Statement
Consider clustering a collection of news articles using TF-IDF representation.

Document corpus: ["stock market crash", "market prices fall", "weather forecast rain", "rainy weather expected"]

#### Task
1. Calculate the TF-IDF matrix for this corpus (assume simple tokenization)
2. Apply K-Means with $K=2$ to cluster these documents
3. Calculate the Cosine similarity between documents $1$ and $2$
4. Interpret the resulting clusters in terms of content themes

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 6: Text Clustering Example](L11_8_6_explanation.md).

## Question 7

### Problem Statement
Anomaly detection using clustering identifies unusual patterns by treating outliers as noise or rare clusters.

#### Task
1. How can clustering algorithms be used for anomaly detection?
2. Compare DBSCAN vs Isolation Forest for anomaly detection
3. What are the advantages of clustering-based anomaly detection?
4. How do you set parameters to optimize anomaly detection performance?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 7: Clustering for Anomaly Detection](L11_8_7_explanation.md).

## Question 8

### Problem Statement
Network intrusion detection systems use clustering to identify abnormal network traffic patterns.

#### Task
1. What network traffic features would be relevant for clustering?
2. How would you distinguish between normal traffic clusters and anomalous traffic?
3. What clustering algorithm would be most suitable for real-time intrusion detection?
4. How do you handle the evolving nature of network attack patterns?
5. Given network traffic data with features $[\text{packet\_size}, \text{duration}, \text{src\_bytes}, \text{dst\_bytes}]$: Normal traffic samples $[(64, 0.1, 1000, 500), (128, 0.2, 2000, 1000), (96, 0.15, 1500, 750)]$ and potential intrusion sample $(1500, 2.0, 50000, 100)$. Use DBSCAN with $\varepsilon=500$ and MinPts $=2$ to determine if the potential intrusion is an outlier. Calculate all distances and show the clustering process.

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 8: Network Intrusion Detection](L11_8_8_explanation.md).

## Question 9

### Problem Statement
Gene expression clustering helps identify functional gene groups and disease patterns in bioinformatics.

#### Task
1. What biological insights can be gained from clustering gene expression data?
2. What distance metrics are appropriate for gene expression profiles?
3. How do you handle missing values in gene expression datasets?
4. How do you validate biological significance of gene clusters?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 9: Gene Expression Clustering](L11_8_9_explanation.md).

## Question 10

### Problem Statement
Market basket analysis uses clustering to identify customer purchasing patterns.

Transaction data:
- Customer A: {bread, milk, eggs}
- Customer B: {bread, butter, jam}
- Customer C: {milk, cheese, yogurt}
- Customer D: {bread, milk, butter}

#### Task
1. How would you represent this transaction data for clustering?
2. What distance metric is appropriate for binary transaction data?
3. Apply hierarchical clustering to identify customer purchase patterns
4. How would you use clustering results to design marketing strategies?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 10: Market Basket Analysis](L11_8_10_explanation.md).

## Question 11

### Problem Statement
Recommendation systems use clustering to group users or items for collaborative filtering.

#### Task
1. How does clustering enhance collaborative filtering recommendation systems?
2. Compare user-based vs item-based clustering for recommendations
3. How do you handle the sparsity of user-item rating matrices?
4. What are the benefits and limitations of clustering-based recommendations?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 11: Clustering for Recommendations](L11_8_11_explanation.md).

## Question 12

### Problem Statement
Social media analysis uses clustering to identify communities and sentiment patterns.

#### Task
1. What features can be extracted from social media data for clustering?
2. How would you cluster users based on their social media behavior?
3. How can clustering help in viral content analysis?
4. What privacy and ethical considerations apply to social media clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 12: Social Media Clustering](L11_8_12_explanation.md).

## Question 13

### Problem Statement
Financial fraud detection systems use clustering to identify suspicious transaction patterns.

#### Task
1. What transaction features are indicative of fraudulent behavior?
2. How would you design a clustering-based fraud detection system?
3. How do you handle class imbalance (rare fraud cases) in clustering?
4. What are the challenges of real-time fraud detection using clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 13: Financial Fraud Detection](L11_8_13_explanation.md).

## Question 14

### Problem Statement
Medical diagnosis applications use clustering for patient stratification and treatment personalization.

#### Task
1. How can clustering help in personalized medicine?
2. What types of medical data are suitable for clustering analysis?
3. How would you cluster patients with similar disease progression patterns?
4. What validation approaches ensure clinical relevance of medical clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 14: Medical Clustering Applications](L11_8_14_explanation.md).

## Question 15

### Problem Statement
Smart city applications use clustering for traffic pattern analysis and urban planning.

#### Task
1. How would you cluster traffic sensors to identify congestion patterns?
2. What temporal considerations are important for traffic clustering?
3. How can clustering help in optimizing public transportation routes?
4. What real-time constraints affect clustering in smart city applications?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 15: Smart City Clustering](L11_8_15_explanation.md).

## Question 16

### Problem Statement
E-commerce platforms use clustering for product recommendation and inventory management.

#### Task
1. How would you cluster products based on customer behavior data?
2. What features would you use to cluster products for recommendation?
3. How can clustering help in demand forecasting for inventory management?
4. How do you handle seasonal patterns in product clustering?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 16: E-commerce Clustering Applications](L11_8_16_explanation.md).

## Question 17

### Problem Statement
Quality control in manufacturing uses clustering to identify defect patterns.

#### Task
1. What sensor data and measurements are relevant for quality control clustering?
2. How would you detect anomalous manufacturing processes using clustering?
3. How do you incorporate temporal dependencies in manufacturing clustering?
4. What are the cost implications of false positives vs false negatives in quality control?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 17: Manufacturing Quality Control](L11_8_17_explanation.md).

## Question 18

### Problem Statement
Climate data analysis uses clustering to identify weather patterns and climate zones.

#### Task
1. What meteorological variables would you use for climate clustering?
2. How would you handle the spatial and temporal nature of climate data?
3. How can clustering help in climate change analysis?
4. What validation approaches are appropriate for climate clustering results?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 18: Climate Data Clustering](L11_8_18_explanation.md).

## Question 19

### Problem Statement
Consider the practical challenges when implementing clustering solutions in production environments.

#### Task
1. What are the main computational challenges for large-scale clustering?
2. How do you handle concept drift in streaming clustering applications?
3. What monitoring and maintenance practices are important for production clustering systems?
4. How do you balance between clustering accuracy and computational efficiency?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 19: Production Clustering Challenges](L11_8_19_explanation.md).

## Question 20

### Problem Statement
Design a comprehensive clustering solution for a multi-faceted business problem.

**Scenario**: A multinational retail company wants to implement a unified clustering system for:
- Customer segmentation across different regions
- Product recommendation in their e-commerce platform  
- Fraud detection in payment systems
- Inventory optimization across stores

#### Task
1. How would you design a unified clustering architecture for these diverse applications?
2. What are the shared components and application-specific components?
3. How would you handle the different data types, scales, and requirements?
4. What evaluation metrics would you use to assess the success of each clustering application?

For a detailed explanation of this problem, including step-by-step solutions and key insights, see [Question 20: Comprehensive Clustering System Design](L11_8_20_explanation.md).
