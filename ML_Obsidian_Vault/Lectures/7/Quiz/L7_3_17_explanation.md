# Question 17: Random Forest Decision Boundaries Visualization

## Problem Statement
Create visual representations of Random Forest decision boundaries with $4$ trees for a $2$D classification problem:

**Tree 1:** $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Tree 2:** $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B  
**Tree 3:** $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Tree 4:** $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B

### Task
1. Draw the decision boundary for each tree on a coordinate grid ($X$: $0$-$8$, $Y$: $0$-$8$)
2. Color-code the regions: Class A = Blue, Class B = Red
3. What's the ensemble prediction for point $(4, 3)$?
4. Which tree creates the most interesting geometric pattern?
5. Calculate the percentage of the grid area where the ensemble prediction differs from any individual tree prediction

## Understanding the Problem
This problem explores how Random Forest ensembles combine multiple decision trees to create complex decision boundaries. Each tree makes decisions based on different features or combinations of features, and the ensemble uses majority voting to make final predictions. The key insight is understanding how individual tree boundaries interact and how the ensemble boundary emerges from their combination.

The problem demonstrates:
- How different types of decision boundaries (linear, rectangular, diagonal) can be combined
- The relationship between individual tree decisions and ensemble predictions
- The geometric complexity that emerges from ensemble methods
- The concept of majority voting in ensemble learning

## Solution

### Step 1: Individual Tree Decision Boundaries
Each tree creates a distinct decision boundary based on its specific rules:

#### Tree 1: $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
This tree creates a **vertical decision boundary** at $X = 3$:
- **Left side** ($X \leq 3$): Class A (Blue)
- **Right side** ($X > 3$): Class B (Red)

![Tree 1 Decision Boundary](../Images/L7_3_Quiz_17/tree1_decision_boundary.png)

The boundary is a simple vertical line, creating two infinite half-planes. This represents a univariate split based solely on the $X$ coordinate.

#### Tree 2: $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B
This tree creates a **horizontal decision boundary** at $Y = 2$:
- **Bottom side** ($Y \leq 2$): Class A (Blue)
- **Top side** ($Y > 2$): Class B (Red)

![Tree 2 Decision Boundary](../Images/L7_3_Quiz_17/tree2_decision_boundary.png)

Similar to Tree 1, this creates a simple horizontal line boundary, but based on the $Y$ coordinate.

#### Tree 3: $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
This tree creates a **rectangular decision boundary**:
- **Inside rectangle** ($X \leq 5$ AND $Y \leq 4$): Class A (Blue)
- **Outside rectangle**: Class B (Red)

![Tree 3 Decision Boundary](../Images/L7_3_Quiz_17/tree3_decision_boundary.png)

This is the most interesting boundary because it creates a **bounded region** instead of infinite half-planes. The AND condition creates a rectangular area where both constraints must be satisfied.

#### Tree 4: $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B
This tree creates a **diagonal decision boundary**:
- **Below diagonal** ($X + Y \leq 6$): Class A (Blue)
- **Above diagonal** ($X + Y > 6$): Class B (Red)

![Tree 4 Decision Boundary](../Images/L7_3_Quiz_17/tree4_decision_boundary.png)

This boundary creates a diagonal line where $X + Y = 6$, demonstrating how linear combinations of features can create oblique decision boundaries.

### Step 2: Combined Visualization
All decision boundaries are shown together for comparison:

![All Trees Combined](../Images/L7_3_Quiz_17/all_trees_combined.png)

This combined view shows how the different boundaries intersect and create complex regions. The point $(4, 3)$ is marked with an orange star for analysis in the next step.

### Step 3: Ensemble Prediction for Point $(4, 3)$
Let's analyze how each tree classifies the point $(4, 3)$ with detailed step-by-step calculations:

#### Tree 1: $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Step-by-step calculation:**
1. **Given point**: $X = 4$, $Y = 3$
2. **Check condition**: $X \leq 3$
3. **Evaluation**: $4 \leq 3$? **False**
4. **Decision**: Since $4 > 3$, Tree 1 predicts **Class B**

#### Tree 2: $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B
**Step-by-step calculation:**
1. **Given point**: $X = 4$, $Y = 3$
2. **Check condition**: $Y \leq 2$
3. **Evaluation**: $3 \leq 2$? **False**
4. **Decision**: Since $3 > 2$, Tree 2 predicts **Class B**

#### Tree 3: $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Step-by-step calculation:**
1. **Given point**: $X = 4$, $Y = 3$
2. **Check first condition**: $X \leq 5$ → $4 \leq 5$? **True**
3. **Check second condition**: $Y \leq 4$ → $3 \leq 4$? **True**
4. **Apply AND logic**: True AND True = **True**
5. **Decision**: Since both conditions are satisfied, Tree 3 predicts **Class A**

#### Tree 4: $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B
**Step-by-step calculation:**
1. **Given point**: $X = 4$, $Y = 3$
2. **Calculate sum**: $X + Y = 4 + 3 = 7$
3. **Check condition**: $X + Y \leq 6$ → $7 \leq 6$? **False**
4. **Decision**: Since $7 > 6$, Tree 4 predicts **Class B**

#### Ensemble Voting Calculation
**Step 1: Collect all tree predictions**
- Tree 1: Class B
- Tree 2: Class B  
- Tree 3: Class A
- Tree 4: Class B

**Step 2: Count votes for each class**
- Class A votes: 1
- Class B votes: 3

**Step 3: Apply majority voting rule**
- If Class A votes > Class B votes → Final prediction: Class A
- If Class B votes > Class A votes → Final prediction: Class B
- If votes are equal → Random choice (tie)

**Step 4: Determine winner**
- Vote count: 1 vs 3
- **Winner: Class B**
- **Winning percentage**: $3/4 = 75.0\%$

**Final Result:**
**Ensemble prediction for point $(4, 3)$: Class B**
**Confidence**: $3/4$ votes (75.0%)

The ensemble correctly identifies that the majority of trees classify this point as Class B, demonstrating the robustness of ensemble methods through majority voting.

### Step 4: Most Interesting Geometric Pattern
Let's analyze the geometric characteristics of each tree in detail:

#### Tree 1: $X \leq 3 \rightarrow$ Class A, $X > 3 \rightarrow$ Class B
**Geometric characteristics:**
- **Boundary**: Vertical line at $X = 3$
- **Shape**: Infinite half-planes (left and right)
- **Complexity**: Simple univariate split
- **Direction**: Parallel to Y-axis

#### Tree 2: $Y \leq 2 \rightarrow$ Class A, $Y > 2 \rightarrow$ Class B
**Geometric characteristics:**
- **Boundary**: Horizontal line at $Y = 2$
- **Shape**: Infinite half-planes (bottom and top)
- **Complexity**: Simple univariate split
- **Direction**: Parallel to X-axis

#### Tree 3: $X \leq 5$ AND $Y \leq 4 \rightarrow$ Class A, otherwise Class B
**Geometric characteristics:**
- **Boundary**: Rectangle with corners at $(0,0)$, $(5,0)$, $(5,4)$, $(0,4)$
- **Shape**: Bounded rectangular region
- **Complexity**: Multivariate split with AND condition
- **Direction**: Creates enclosed area with finite boundaries

#### Tree 4: $X + Y \leq 6 \rightarrow$ Class A, $X + Y > 6 \rightarrow$ Class B
**Geometric characteristics:**
- **Boundary**: Diagonal line $X + Y = 6$
- **Shape**: Infinite half-planes (below and above diagonal)
- **Complexity**: Linear combination of features
- **Direction**: 45-degree angle (slope = -1)

#### Comparative Analysis
**Complexity ranking (from simple to complex):**
1. **Trees 1 & 2**: Simple linear boundaries (univariate splits)
2. **Tree 4**: Diagonal boundary (linear combination of features)
3. **Tree 3**: Rectangular boundary (multivariate with AND condition)

**Tree 3 creates the most interesting geometric pattern because:**

1. **BOUNDED REGION**: Unlike infinite half-planes, creates finite rectangular area
2. **MULTIVARIATE SPLIT**: Uses both X and Y coordinates simultaneously
3. **LOGICAL COMPLEXITY**: AND condition creates intersection of constraints
4. **PRACTICAL RELEVANCE**: Represents real-world scenarios with multiple conditions
5. **GEOMETRIC UNIQUENESS**: Only tree that creates enclosed classification region

**Mathematical representation:**
Tree 3 boundary: $\{(X,Y) \mid X \leq 5 \text{ AND } Y \leq 4\}$

This creates a **closed set** in 2D space, unlike the **open half-planes** of other trees.

### Step 5: Area Percentage Where Ensemble Differs
Let's calculate the percentage of the grid area where the ensemble prediction differs from any individual tree prediction:

#### Detailed Calculation Steps

**Step 1: Define ensemble decision function**
For each point $(X, Y)$, collect predictions from all 4 trees and apply majority voting:
- If $\geq 2$ trees predict Class B → Class B
- Otherwise → Class A

**Step 2: Generate ensemble decision for entire grid**
- Grid dimensions: $100 \times 100 = 10,000$ total points
- Each point gets classified by the ensemble

**Step 3: Calculate differences between ensemble and individual trees**
For each tree, compute: $|\text{ensemble\_prediction} - \text{tree\_prediction}|$
- Result: $0$ if same prediction, $1$ if different prediction

**Individual tree differences:**
- Tree 1 differences: $2,512$ points
- Tree 2 differences: $1,888$ points  
- Tree 3 differences: $744$ points
- Tree 4 differences: $494$ points

**Step 4: Find total area where ensemble differs from ANY individual tree**
Use logical OR operation: $\text{total\_diff} = \text{diff\_tree1} \text{ OR } \text{diff\_tree2} \text{ OR } \text{diff\_tree3} \text{ OR } \text{diff\_tree4}$

This identifies points where the ensemble differs from at least one tree.

**Step 5: Calculate percentage**
- Grid size: $100 \times 100 = 10,000$ total points
- Points with differences: $5,000$
- Percentage calculation: $(5,000 / 10,000) \times 100 = 50.00\%$

#### Final Result
**Grid size**: $100 \times 100 = 10,000$ total points
**Points where ensemble differs from any individual tree**: $5,000$
**Percentage of grid area**: $50.00\%$

#### Interpretation
This means that in $50.0\%$ of the feature space, the ensemble makes a different prediction than at least one of the individual trees. This demonstrates the ensemble's ability to create more nuanced decision boundaries that capture complex patterns beyond what any single tree can represent.

The high percentage indicates that the ensemble creates a significantly different decision boundary compared to individual trees, demonstrating the power of ensemble methods to create more sophisticated and accurate classifications.

![Ensemble Differences Analysis](../Images/L7_3_Quiz_17/ensemble_differences_analysis.png)

The analysis shows:
- **Top row**: Ensemble boundary and differences with Trees 1 and 2
- **Bottom row**: Differences with Trees 3 and 4, plus total differences
- **Yellow regions**: Areas where ensemble predictions differ from individual tree predictions
- **Black lines**: Ensemble decision boundaries
- **Colored lines**: Individual tree boundaries

## Key Insights

### Geometric Patterns in Decision Trees
- **Univariate splits** (Trees 1 & 2) create simple linear boundaries
- **Multivariate splits** (Tree 3) can create bounded regions
- **Linear combinations** (Tree 4) create oblique boundaries
- **AND conditions** create more complex, enclosed regions
- **OR conditions** would create union regions

### Ensemble Learning Principles
- **Majority voting** combines individual tree predictions
- **Diversity** among trees leads to more robust ensemble decisions
- **Geometric complexity** emerges from combining simple boundaries
- **Error reduction** occurs when trees make different types of mistakes

### Practical Applications
- **Feature engineering**: Different trees can focus on different aspects of the data
- **Robustness**: Ensemble methods are less sensitive to individual tree errors
- **Interpretability**: Individual trees remain interpretable while ensemble complexity captures subtle patterns
- **Scalability**: Trees can be trained in parallel, making ensembles computationally efficient

## Conclusion
- **Individual boundaries**: Each tree creates distinct geometric patterns from simple linear to complex rectangular
- **Ensemble complexity**: The combination creates a sophisticated decision boundary that differs from any single tree
- **Point classification**: Point $(4, 3)$ demonstrates how majority voting resolves conflicting predictions
- **Geometric interest**: Tree 3's rectangular boundary shows the complexity possible with multivariate conditions
- **Area difference**: 50% of the grid shows different classifications between ensemble and individual trees

The Random Forest ensemble successfully combines the strengths of different decision boundaries, creating a more nuanced and accurate classification system than any individual tree could achieve alone. This demonstrates the fundamental principle of ensemble learning: combining multiple weak learners to create a strong, robust classifier.
