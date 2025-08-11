# Lecture 8.4: Density-Based Clustering Quiz

## Overview
This quiz contains 5 questions covering different topics from section 8.4 of the lectures on Density-Based Clustering, including DBSCAN algorithm, core/border/noise points, parameters, and advantages/limitations.

## Question 1

### Problem Statement
Consider the following dataset with 8 points in 2D space:

| Point | $x_1$ | $x_2$ |
|-------|-------|-------|
| A     | 1     | 1     |
| B     | 1     | 2     |
| C     | 2     | 1     |
| D     | 2     | 2     |
| E     | 8     | 8     |
| F     | 8     | 9     |
| G     | 9     | 8     |
| H     | 9     | 9     |

#### Task
1. [🔍] If $\epsilon = 1.5$ and MinPts = 3, identify core points
2. [🔍] If $\epsilon = 1.5$ and MinPts = 3, identify border points
3. [🔍] If $\epsilon = 1.5$ and MinPts = 3, identify noise points
4. [🔍] How many clusters would DBSCAN find with these parameters?

For a detailed explanation of this question, see [Question 1: DBSCAN Point Classification](L8_4_1_explanation.md).

## Question 2

### Problem Statement
DBSCAN has two main parameters: $\epsilon$ (epsilon) and MinPts.

#### Task
1. [📚] What does the $\epsilon$ parameter represent in DBSCAN?
2. [📚] What does the MinPts parameter represent in DBSCAN?
3. [📚] How do you choose appropriate values for these parameters?
4. [📚] What happens if you set $\epsilon$ too small or too large?

For a detailed explanation of this question, see [Question 2: DBSCAN Parameters](L8_4_2_explanation.md).

## Question 3

### Problem Statement
DBSCAN classifies points into three categories: core, border, and noise points.

#### Task
1. [📚] Define what makes a point a "core point"
2. [📚] Define what makes a point a "border point"
3. [📚] Define what makes a point a "noise point"
4. [📚] How does DBSCAN use these classifications to form clusters?

For a detailed explanation of this question, see [Question 3: DBSCAN Point Types](L8_4_3_explanation.md).

## Question 4

### Problem Statement
Consider the advantages of DBSCAN over K-means clustering.

#### Task
1. [📚] **Advantage 1**: How does DBSCAN handle clusters of irregular shapes?
2. [📚] **Advantage 2**: How does DBSCAN handle outliers and noise?
3. [📚] **Advantage 3**: How does DBSCAN determine the number of clusters?
4. [📚] **Advantage 4**: What types of data distributions is DBSCAN particularly good for?

For a detailed explanation of this question, see [Question 4: DBSCAN Advantages](L8_4_4_explanation.md).

## Question 5

### Problem Statement
Consider the limitations and challenges of DBSCAN.

#### Task
1. [📚] **Limitation 1**: How does DBSCAN perform on high-dimensional data?
2. [📚] **Limitation 2**: What happens when clusters have different densities?
3. [📚] **Limitation 3**: How sensitive is DBSCAN to parameter selection?
4. [📚] **Limitation 4**: What computational challenges does DBSCAN face with large datasets?

For a detailed explanation of this question, see [Question 5: DBSCAN Limitations](L8_4_5_explanation.md).
