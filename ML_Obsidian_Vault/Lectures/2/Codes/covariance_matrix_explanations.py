import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def explain_covariance_contours():
    """Print detailed explanations for the covariance matrix contours example."""
    print(f"\n{'='*80}")
    print(f"Example: Multivariate Gaussians with Different Covariance Matrices")
    print(f"{'='*80}")
    
    print(f"Function: f(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    
    print("\nStep-by-Step Solution:")
    
    steps = [
        {
            "title": "Understand the multivariate Gaussian PDF",
            "description": "The probability density function of a bivariate Gaussian with mean μ = (0,0) and covariance matrix Σ defines a surface whose contours we want to analyze.",
            "math": "For a bivariate Gaussian:\nf(x,y) = (1/√(2π|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])\nwhere Σ is the covariance matrix and |Σ| is its determinant."
        },
        {
            "title": "Analyze the quadratic form in the exponent",
            "description": "The key term that determines the shape of the contours is the quadratic form (x,y)ᵀ Σ⁻¹ (x,y), which creates elliptical level curves.",
            "math": "The exponent term -1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)] determines the contour shapes.\nFor constant density c, the contours satisfy:\n(x,y)ᵀ Σ⁻¹ (x,y) = -2log(c·√(2π|Σ|)) = constant"
        },
        {
            "title": "Consider different types of covariance matrices",
            "description": "Different covariance matrices lead to different shaped contours, which can be analyzed based on the eigenvalues and eigenvectors of Σ.",
            "math": "Case 1: Diagonal covariance Σ = [[σ₁², 0], [0, σ₂²]]\n  When σ₁² = σ₂² (scaled identity): Circular contours\n  When σ₁² ≠ σ₂² (different variances): Axis-aligned ellipses\n\nCase 2: Non-diagonal covariance Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]\n  When ρ ≠ 0: Rotated ellipses\n  When ρ > 0: Ellipses tilted along y = x direction\n  When ρ < 0: Ellipses tilted along y = -x direction"
        },
        {
            "title": "Draw the contours for each case",
            "description": "For each type of covariance matrix, we can sketch the resulting contours based on our analysis.",
            "math": "For diagonal covariance: The principal axes of the elliptical contours align with coordinate axes.\n  - The lengths of the semi-axes are proportional to √σ₁² and √σ₂².\n\nFor non-diagonal covariance: The principal axes of the elliptical contours align with the eigenvectors of Σ.\n  - The lengths of the semi-axes are proportional to the square roots of the eigenvalues.\n  - The correlation coefficient ρ determines the rotation angle of the ellipses."
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}: {step['title']}")
        print(f"{step['description']}")
        for line in step['math'].split('\n'):
            print(f"  {line}")
    
    print("\nContour Values:")
    print("c = varies based on probability density")
    
    print("\nKey Insights:")
    print("- The contour lines connect all points (x,y) where f(x,y) equals the contour value c.")
    print("- Each contour shape provides insight into the function's behavior in that region.")
    print("- This function produces elliptical contours, which indicate multivariate Gaussian distributions.")
    print("- The shape and orientation of the ellipses directly reflect the covariance structure.")
    
    print("\nCovariance Matrix Effects on Contour Shapes:")
    print("1. Diagonal covariance matrices:")
    print("   - Equal variances (σ₁² = σ₂²): Circular contours")
    print("   - Different variances (σ₁² ≠ σ₂²): Axis-aligned elliptical contours")
    print("   - Major/minor axes proportional to the square roots of the variances")
    
    print("\n2. Non-diagonal covariance matrices:")
    print("   - Produce rotated elliptical contours not aligned with coordinate axes")
    print("   - Positive correlation (ρ > 0): Ellipses tilted along y = x direction")
    print("   - Negative correlation (ρ < 0): Ellipses tilted along y = -x direction")
    print("   - The principal axes align with the eigenvectors of the covariance matrix")
    print("   - The lengths of these axes are proportional to the square roots of the eigenvalues")
    
    print(f"\n{'='*80}")
    
    print("\nPractical Applications:")
    print("- Visualizing multivariate probability distributions")
    print("- Understanding correlation structure in data")
    print("- Analyzing principal components and directions of maximum variance")
    print("- Designing confidence regions for statistical inference")
    print("- Implementing anomaly detection based on Mahalanobis distance")
    
    return "Covariance matrix contour explanations generated successfully!"

def explain_basic_2d_example():
    """Print detailed explanations for the basic 2D normal distribution examples."""
    print(f"\n{'='*80}")
    print(f"Example: Basic 1D and 2D Normal Distributions")
    print(f"{'='*80}")
    
    print("\nExample 1: 1D Normal Distributions with Different Variances")
    print("\nMathematical Formula:")
    print("f(x) = (1/√(2πσ²)) * exp(-x²/(2σ²))")
    
    print("\nKey Points:")
    print("- Standard normal distribution (σ² = 1): Balanced trade-off between peak height and spread")
    print("- Smaller variance (σ² = 0.5): Taller peak, narrower spread (more concentrated)")
    print("- Larger variance (σ² = 2): Lower peak, wider spread (more dispersed)")
    print("- The total area under each curve is always 1 (probability axioms)")
    print("- Standard deviation (σ) quantifies the typical distance from the mean")
    
    print("\nExample 2: 2D Standard Normal Distribution (Independent Variables)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π)) * exp(-(x² + y²)/2)")
    
    print("\nKey Points:")
    print("- Identity covariance matrix: Σ = [[1, 0], [0, 1]]")
    print("- Variables x and y are independent (zero correlation)")
    print("- Equal variances in both dimensions lead to circular contours")
    print("- Distance from origin determines probability density (radial symmetry)")
    print("- 1σ, 2σ, and 3σ circles contain approximately 39%, 86%, and 99% of the probability mass")
    
    print("\nExample 3: 2D Normal with Different Variances (Independent Variables)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√(σ₁²σ₂²))) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    
    print("\nKey Points:")
    print("- Diagonal covariance matrix: Σ = [[σ₁², 0], [0, σ₂²]]")
    print("- Variables x and y are still independent (zero correlation)")
    print("- Different variances lead to axis-aligned elliptical contours")
    print("- The semi-axes of the ellipses are proportional to σ₁ and σ₂")
    print("- The shape of the ellipse directly visualizes the relative scaling between variables")
    
    print(f"\n{'='*80}")

def explain_3d_visualization():
    """Print detailed explanations for the 3D Gaussian visualization examples."""
    print(f"\n{'='*80}")
    print(f"Example: 3D Visualization of Multivariate Gaussians")
    print(f"{'='*80}")
    
    print("\nExample 1: Standard Bivariate Normal (Identity Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π)) * exp(-(x² + y²)/2)")
    
    print("\nKey Points:")
    print("- Perfect bell-shaped surface with radial symmetry")
    print("- Peak at the mean (0,0) with probability density of 1/(2π) ≈ 0.159")
    print("- Circular contours when projected onto the x-y plane")
    print("- Symmetric decay in all directions from the peak")
    print("- The volume under the surface is exactly 1 (probability axioms)")
    
    print("\nExample 2: Bivariate Normal with Different Variances (Diagonal Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√(σ₁²σ₂²))) * exp(-1/2 * (x²/σ₁² + y²/σ₂²))")
    
    print("\nKey Points:")
    print("- Bell-shaped surface stretched along one axis and compressed along the other")
    print("- Elliptical contours when projected onto the x-y plane")
    print("- Peak height is reduced compared to standard case due to normalization")
    print("- The spread reflects the different uncertainties in each dimension")
    print("- The principal axes of the ellipses align with the coordinate axes")
    
    print("\nExample 3: Bivariate Normal with Correlation (Non-Diagonal Covariance)")
    print("\nMathematical Formula:")
    print("f(x,y) = (1/(2π√|Σ|)) * exp(-1/2 * [(x,y)ᵀ Σ⁻¹ (x,y)])")
    print("where Σ = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]")
    
    print("\nKey Points:")
    print("- Bell-shaped surface tilted according to the correlation structure")
    print("- Rotated elliptical contours when projected onto the x-y plane")
    print("- The tilt direction indicates the type of correlation (positive/negative)")
    print("- The strength of correlation affects the eccentricity of the ellipses")
    print("- This visualization shows how correlated variables cluster along a specific direction")
    
    print(f"\n{'='*80}")

def explain_eigenvalue_visualization():
    """Print detailed explanations for the eigenvalue and eigenvector visualization."""
    print(f"\n{'='*80}")
    print(f"Example: Eigenvalues, Eigenvectors, and Covariance Effects")
    print(f"{'='*80}")
    
    print("\nMathematical Background:")
    print("- Covariance matrix Σ can be decomposed as Σ = VΛV^T")
    print("- V contains eigenvectors (principal directions)")
    print("- Λ is a diagonal matrix of eigenvalues (variance along principal directions)")
    
    print("\nExample 1: No Correlation (ρ = 0)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0], [0, 1]]")
    print("- Eigenvalues: λ₁ = λ₂ = 1")
    print("- Eigenvectors align with the coordinate axes")
    print("- Circular contours indicate equal variance in all directions")
    print("- No preferred direction of variability in the data")
    
    print("\nExample 2: Weak Correlation (ρ = 0.3)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.3], [0.3, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.3, λ₂ ≈ 0.7")
    print("- Eigenvectors begin to rotate from the coordinate axes")
    print("- Slightly elliptical contours with mild rotation")
    print("- Beginning of a preferred direction of variability")
    
    print("\nExample 3: Moderate Correlation (ρ = 0.6)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.6], [0.6, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.6, λ₂ ≈ 0.4")
    print("- Eigenvectors rotate further from the coordinate axes")
    print("- More eccentric elliptical contours with significant rotation")
    print("- Clear preferred direction of variability emerges")
    
    print("\nExample 4: Strong Correlation (ρ = 0.9)")
    print("\nKey Points:")
    print("- Covariance matrix: Σ = [[1, 0.9], [0.9, 1]]")
    print("- Eigenvalues: λ₁ ≈ 1.9, λ₂ ≈ 0.1")
    print("- Eigenvectors nearly align with the y = x and y = -x directions")
    print("- Highly eccentric elliptical contours with strong rotation")
    print("- Dominant direction of variability along the first eigenvector")
    print("- Very little variability along the second eigenvector")
    
    print("\nKey Insights:")
    print("- As correlation increases, eigenvalues become more disparate")
    print("- The largest eigenvalue increases, the smallest decreases")
    print("- The orientation of eigenvectors approaches y = x (for positive correlation)")
    print("- The ellipses become increasingly elongated (higher eccentricity)")
    print("- This illustrates why PCA works: it identifies the directions of maximum variance")
    
    print(f"\n{'='*80}")

def explain_simple_covariance_real_world():
    """Print explanations for the simple real-world covariance example."""
    print(f"\n{'='*80}")
    print(f"Example: Height and Weight - A Real-World Covariance Example")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("How does natural covariance appear in the real world, and how can it be visualized using height and weight data?")
    
    print("\nKey Points:")
    print("- Height and weight are naturally correlated variables (taller people tend to weigh more)")
    print("- The data cloud forms an elliptical pattern aligned along a positive correlation direction")
    print("- The covariance matrix quantifies the strength and direction of this relationship")
    print("- Principal components (eigenvectors) show the main directions of variance in the data")
    print("- The confidence ellipse visualizes the region containing ~95% of the data (2σ)")
    
    print("\nInsights:")
    print("- The first principal component points along the direction of maximum variance")
    print("- This direction aligns with the 'growth trajectory' where both height and weight increase")
    print("- The second principal component captures variance orthogonal to the main relationship")
    print("- This direction represents variations in body type (more weight relative to height or vice versa)")
    print("- The covariance matrix and its properties reveal the underlying data structure")
    
    print("\nReal-World Applications:")
    print("- Medical research and anthropometry: establishing normal ranges and relationships")
    print("- Clothing industry: designing size systems based on correlated body measurements")
    print("- Sports science: analyzing performance metrics and their relationships")
    print("- Public health: monitoring population trends in body metrics")
    
    print(f"\n{'='*80}")

def explain_toy_data_covariance_change():
    """Print explanations for the toy data covariance change example."""
    print(f"\n{'='*80}")
    print(f"Example: How Rotation Affects Covariance Structure")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("What happens to the covariance matrix when we rotate a dataset, and why is this important?")
    
    print("\nKey Points:")
    print("- Starting with uncorrelated data (diagonal covariance matrix)")
    print("- Rotation introduces correlation between variables (non-diagonal covariance matrix)")
    print("- The covariance changes systematically with the rotation angle")
    print("- The total variance (trace of covariance matrix) remains constant under rotation")
    print("- The determinant of the covariance matrix also remains constant")
    
    print("\nMathematical Explanation:")
    print("- For a rotation matrix R and covariance matrix Σ:")
    print("- The transformed covariance matrix is Σ' = R·Σ·R^T")
    print("- For initially uncorrelated data with equal variances (Σ = σ²I):")
    print("  * At 0° rotation: Cov(x,y) = 0 (no correlation)")
    print("  * At 45° rotation: Cov(x,y) is maximized (strongest correlation)")
    print("  * At 90° rotation: Cov(x,y) returns to 0 (variables swap positions)")
    
    print("\nPractical Significance:")
    print("- Coordinate system choice affects the observed covariance structure")
    print("- Feature engineering: rotation can introduce or remove correlations")
    print("- Principal Component Analysis (PCA) exploits this by finding a rotation that diagonalizes the covariance matrix")
    print("- Understanding these transformations is crucial for data preprocessing and interpretation")
    
    print(f"\n{'='*80}")

def explain_mahalanobis_distance():
    """Print explanations for the Mahalanobis distance example."""
    print(f"\n{'='*80}")
    print(f"Example: Mahalanobis Distance vs Euclidean Distance")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("Why is Euclidean distance inadequate for correlated data, and how does Mahalanobis distance address this limitation?")
    
    print("\nKey Concepts:")
    print("- Euclidean distance treats all dimensions equally and independently")
    print("  * Represented by circles of equal distance from the mean")
    print("- Mahalanobis distance accounts for the covariance structure of the data")
    print("  * Represented by ellipses aligned with the data's natural distribution")
    print("  * Points at the same Mahalanobis distance have equal probability density under a multivariate normal model")
    
    print("\nMathematical Formula:")
    print("- Euclidean distance: d_E(x) = √[(x-μ)^T(x-μ)]")
    print("- Mahalanobis distance: d_M(x) = √[(x-μ)^T Σ^(-1) (x-μ)]")
    print("  where μ is the mean and Σ is the covariance matrix")
    
    print("\nKey Points from the Visualization:")
    print("- Points at the same Euclidean distance can have very different Mahalanobis distances:")
    print("  * Points along the major axis of correlation have smaller Mahalanobis distances")
    print("  * Points perpendicular to the correlation direction have larger Mahalanobis distances")
    print("- The Mahalanobis distance effectively 'scales' the space according to the data variance")
    
    print("\nPractical Applications:")
    print("- Anomaly detection: identifying outliers that account for correlation structure")
    print("- Classification: creating decision boundaries that respect data covariance")
    print("- Clustering: defining distance metrics that capture the natural data structure")
    print("- Quality control: monitoring multivariate processes and detecting unusual states")
    
    print(f"\n{'='*80}")

def explain_emoji_covariance():
    """Print explanations for the emoji covariance example."""
    print(f"\n{'='*80}")
    print(f"Example: Positive vs Negative Correlation (The Emoji Edition)")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("How can we intuitively understand positive and negative correlation using everyday visual metaphors?")
    
    print("\nKey Concepts Illustrated:")
    print("- Positive correlation (Happy Face 😊):")
    print("  * Variables tend to increase or decrease together")
    print("  * Data points form a pattern from bottom-left to top-right")
    print("  * The covariance ellipse is tilted along the y = x direction")
    print("  * Common in naturally related quantities (height-weight, study time-grades)")
    
    print("- Negative correlation (Sad Face 😢):")
    print("  * As one variable increases, the other tends to decrease")
    print("  * Data points form a pattern from top-left to bottom-right")
    print("  * The covariance ellipse is tilted along the y = -x direction")
    print("  * Common in trade-off relationships (speed-accuracy, price-demand)")
    
    print("\nVisual Metaphors:")
    print("- The smiley/sad faces provide an intuitive memory aid:")
    print("  * Smile curves upward ⌣ like positive correlation")
    print("  * Frown curves downward ⌢ like negative correlation")
    
    print("\nEmotional Interpretation (Just for Fun):")
    print("- Positive correlation makes data 'happy' because variables 'agree' with each other")
    print("- Negative correlation makes data 'sad' because variables 'disagree' with each other")
    print("- Zero correlation is 'emotionless' data - variables show no relationship")
    
    print("\nLearning Value:")
    print("- Memorable visualizations help anchor abstract statistical concepts")
    print("- Connecting emotional resonance to mathematical patterns enhances retention")
    print("- Visual intuition complements formal mathematical understanding")
    
    print(f"\n{'='*80}")

def explain_sketch_contour_problem():
    """Print detailed explanations for the sketch contour problem example."""
    print(f"\n{'='*80}")
    print(f"Example: Sketch Contour Lines for Bivariate Normal Distribution")
    print(f"{'='*80}")
    
    print("\nProblem Statement:")
    print("Sketch the contour lines for the probability density function of a bivariate normal distribution")
    print("with mean μ = (0,0) and covariance matrix Σ = [[1, 0], [0, 1]].")
    
    print("\nStep-by-Step Solution:")
    
    print("\nStep 1: Understand the mathematical formula")
    print("The PDF of a bivariate normal distribution is given by:")
    print("f(x,y) = (1/2π√|Σ|) * exp(-1/2 * (x,y)ᵀ Σ⁻¹ (x,y))")
    print("where Σ is the covariance matrix and |Σ| is its determinant.")
    
    print("\nStep 2: Analyze the covariance matrix")
    print("For Σ = [[1, 0], [0, 1]]:")
    print("- This is the identity matrix (variances = 1, covariance = 0)")
    print("- The variables are uncorrelated and have equal variances")
    print("- The determinant |Σ| = 1")
    print("- The inverse Σ⁻¹ = Σ (identity matrix is its own inverse)")
    
    print("\nStep 3: Identify the equation for contour lines")
    print("Contour lines connect points with equal probability density")
    print("For a specific contour value c, the points satisfy:")
    print("(x,y)ᵀ Σ⁻¹ (x,y) = -2ln(c*2π) = constant")
    print("Which simplifies to: x² + y² = constant")
    
    print("\nStep 4: Recognize that contours form circles")
    print("The equation x² + y² = constant describes a circle:")
    print("- Centered at the origin (0,0)")
    print("- With radius r = √constant")
    
    print("\nStep 5: Sketch the contours")
    print("Draw concentric circles centered at the origin with various radii.")
    print("These circles represent different probability density levels:")
    print("- Inner circles (smaller radii): higher probability density")
    print("- Outer circles (larger radii): lower probability density")
    print("- The 1σ circle has radius 1")
    print("- The 2σ circle has radius 2")
    print("- The 3σ circle has radius 3")
    
    print("\nConclusion:")
    print("For a standard bivariate normal distribution (Σ = I), the contour lines form")
    print("concentric circles centered at the mean (0,0), reflecting equal spread in all directions.")
    print("This is the simplest case of a multivariate normal distribution, where the")
    print("variables are uncorrelated and have equal variances.")
    
    print(f"\n{'='*80}")
    
    return "Sketch contour problem explanation generated successfully!"

def generate_covariance_explanations():
    """Generate detailed explanations for all covariance examples."""
    # Print explanations for all examples
    explain_covariance_contours()
    explain_basic_2d_example()
    explain_3d_visualization()
    explain_eigenvalue_visualization()
    # New simple examples
    explain_simple_covariance_real_world()
    explain_toy_data_covariance_change()
    explain_mahalanobis_distance()
    explain_emoji_covariance()
    explain_sketch_contour_problem()
    
    return "Covariance matrix explanations generated successfully!"

if __name__ == "__main__":
    generate_covariance_explanations() 