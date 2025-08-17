import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
import time
import psutil
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_40")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostFinalChallenge:
    def __init__(self):
        print("=== AdaBoost Final Challenge - Question 40 ===")
        print("Building a comprehensive AdaBoost ensemble for real-world application")
        print("=" * 80)
        
        # Problem constraints
        self.n_samples = 5000
        self.n_features = 25
        self.target_accuracy = 0.95
        self.time_constraint = 2 * 3600  # 2 hours in seconds
        self.memory_constraint = 500 * 1024 * 1024  # 500MB in bytes
        self.interpretability_required = True
        
        print(f"Constraints:")
        print(f"  - Dataset: {self.n_samples:,} samples, {self.n_features} features")
        print(f"  - Performance: {self.target_accuracy*100}% accuracy")
        print(f"  - Time: {self.time_constraint/3600:.1f} hours")
        print(f"  - Memory: {self.memory_constraint/(1024*1024):.0f}MB")
        print(f"  - Interpretability: Required")
        print("=" * 80)
        
        # Generate synthetic dataset
        self.generate_dataset()
        
        # Store results
        self.results = {}
        
    def generate_dataset(self):
        """Generate a synthetic dataset with controlled characteristics"""
        print("Generating synthetic dataset...")
        
        # Create dataset with some noise to make it challenging
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=20,  # 20 informative features
            n_redundant=3,     # 3 redundant features
            n_repeated=2,      # 2 repeated features
            n_classes=2,
            n_clusters_per_class=3,
            weights=[0.6, 0.4],  # Slightly imbalanced
            random_state=42
        )
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        
        self.X = X
        self.y = y
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Dataset generated:")
        print(f"  - Training set: {self.X_train.shape[0]:,} samples")
        print(f"  - Test set: {self.X_test.shape[0]:,} samples")
        print(f"  - Class distribution: {np.bincount(self.y_train)}")
        print("=" * 80)
        
    def analyze_weak_learners(self):
        """Analyze different types of weak learners"""
        print("Step 1: Analyzing Weak Learner Options")
        print("-" * 50)
        
        weak_learners = {
            'Decision Stump': DecisionTreeClassifier(max_depth=1, random_state=42),
            'Shallow Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
            'Medium Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Linear SVM': None,  # We'll use decision trees for interpretability
        }
        
        results = {}
        
        for name, learner in weak_learners.items():
            if learner is not None:
                print(f"\nTesting {name}:")
                
                # Measure training time
                start_time = time.time()
                learner.fit(self.X_train, self.y_train)
                train_time = time.time() - start_time
                
                # Measure memory usage
                import sys
                memory_usage = sys.getsizeof(learner)
                
                # Measure performance
                train_score = learner.score(self.X_train, self.y_train)
                test_score = learner.score(self.X_test, self.y_test)
                
                results[name] = {
                    'train_time': train_time,
                    'memory_usage': memory_usage,
                    'train_score': train_score,
                    'test_score': test_score,
                    'complexity': self.measure_complexity(learner)
                }
                
                print(f"  Training time: {train_time:.4f}s")
                print(f"  Memory usage: {memory_usage} bytes")
                print(f"  Training accuracy: {train_score:.4f}")
                print(f"  Test accuracy: {test_score:.4f}")
                print(f"  Complexity score: {results[name]['complexity']:.2f}")
        
        self.weak_learner_analysis = results
        return results
    
    def measure_complexity(self, model):
        """Measure model complexity for interpretability"""
        if hasattr(model, 'tree_'):
            # For decision trees, count total nodes
            return model.tree_.node_count
        else:
            # For other models, use a different metric
            return 1
    
    def design_adaboost_configuration(self):
        """Design the optimal AdaBoost configuration"""
        print("\nStep 2: Designing AdaBoost Configuration")
        print("-" * 50)
        
        # Based on analysis, choose shallow trees for interpretability
        base_learner = DecisionTreeClassifier(max_depth=3, random_state=42)
        
        # Estimate optimal number of iterations
        # Start with conservative estimate
        n_estimators_options = [50, 100, 200, 300, 500]
        
        print("Testing different numbers of estimators:")
        print("Estimating training time and memory usage...")
        
        config_results = {}
        
        for n_est in n_estimators_options:
            print(f"\nTesting n_estimators = {n_est}")
            
            # Estimate training time (linear scaling)
            estimated_time = n_est * self.weak_learner_analysis['Shallow Tree']['train_time']
            
            # Estimate memory usage
            estimated_memory = n_est * self.weak_learner_analysis['Shallow Tree']['memory_usage']
            
            # Estimate accuracy improvement
            # AdaBoost typically improves accuracy with more estimators up to a point
            base_accuracy = self.weak_learner_analysis['Shallow Tree']['test_score']
            estimated_accuracy = min(0.98, base_accuracy + 0.1 * np.log(n_est/50 + 1))
            
            config_results[n_est] = {
                'estimated_time': estimated_time,
                'estimated_memory': estimated_memory,
                'estimated_accuracy': estimated_accuracy,
                'feasible': (estimated_time < self.time_constraint and 
                           estimated_memory < self.memory_constraint)
            }
            
            print(f"  Estimated time: {estimated_time/60:.1f} minutes")
            print(f"  Estimated memory: {estimated_memory/(1024*1024):.1f}MB")
            print(f"  Estimated accuracy: {estimated_accuracy:.4f}")
            print(f"  Feasible: {config_results[n_est]['feasible']}")
        
        # Choose optimal configuration
        feasible_configs = {k: v for k, v in config_results.items() if v['feasible']}
        
        if feasible_configs:
            # Choose the one with highest estimated accuracy
            optimal_n = max(feasible_configs.keys(), 
                          key=lambda x: feasible_configs[x]['estimated_accuracy'])
            
            print(f"\nOptimal configuration:")
            print(f"  Base learner: Decision Tree (max_depth=3)")
            print(f"  Number of estimators: {optimal_n}")
            print(f"  Expected accuracy: {feasible_configs[optimal_n]['estimated_accuracy']:.4f}")
            print(f"  Expected time: {feasible_configs[optimal_n]['estimated_time']/60:.1f} minutes")
            print(f"  Expected memory: {feasible_configs[optimal_n]['estimated_memory']/(1024*1024):.1f}MB")
            
            self.optimal_config = {
                'base_learner': base_learner,
                'n_estimators': optimal_n
            }
        else:
            print("No feasible configuration found!")
            # Use most conservative option
            self.optimal_config = {
                'base_learner': base_learner,
                'n_estimators': 50
            }
        
        return self.optimal_config
    
    def implement_adaboost(self):
        """Implement the AdaBoost ensemble"""
        print("\nStep 3: Implementing AdaBoost Ensemble")
        print("-" * 50)
        
        config = self.optimal_config
        
        print(f"Training AdaBoost with {config['n_estimators']} estimators...")
        
        # Start timing
        start_time = time.time()
        
        # Create and train AdaBoost
        adaboost = AdaBoostClassifier(
            estimator=config['base_learner'],
            n_estimators=config['n_estimators'],
            learning_rate=1.0,
            random_state=42
        )
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        adaboost.fit(self.X_train, self.y_train)
        
        # Measure actual performance
        training_time = time.time() - start_time
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before
        
        # Evaluate performance
        train_score = adaboost.score(self.X_train, self.y_train)
        test_score = adaboost.score(self.X_test, self.y_test)
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(adaboost, self.X_train, self.y_train, cv=5)
        
        print(f"Training completed!")
        print(f"  Actual training time: {training_time/60:.1f} minutes")
        print(f"  Memory used: {memory_used/(1024*1024):.1f}MB")
        print(f"  Training accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.adaboost_model = adaboost
        self.performance_metrics = {
            'training_time': training_time,
            'memory_used': memory_used,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return adaboost
    
    def evaluate_strategy(self):
        """Evaluate the strategy to ensure 95% accuracy"""
        print("\nStep 4: Evaluation Strategy for 95% Accuracy")
        print("-" * 50)
        
        # Check if we meet the accuracy requirement
        current_accuracy = self.performance_metrics['test_score']
        
        print(f"Current test accuracy: {current_accuracy:.4f}")
        print(f"Target accuracy: {self.target_accuracy:.4f}")
        
        if current_accuracy >= self.target_accuracy:
            print("✓ Target accuracy achieved!")
        else:
            print("✗ Target accuracy not achieved. Implementing improvement strategies...")
            
            # Strategy 1: Hyperparameter tuning
            print("\nStrategy 1: Hyperparameter Tuning")
            param_grid = {
                'n_estimators': [self.optimal_config['n_estimators'], 
                                self.optimal_config['n_estimators'] + 100],
                'learning_rate': [0.8, 1.0, 1.2]
            }
            
            grid_search = GridSearchCV(
                AdaBoostClassifier(estimator=self.optimal_config['base_learner'], random_state=42),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update model with best parameters
            self.adaboost_model = grid_search.best_estimator_
            self.performance_metrics['test_score'] = self.adaboost_model.score(self.X_test, self.y_test)
            
            print(f"Updated test accuracy: {self.performance_metrics['test_score']:.4f}")
        
        # Final evaluation
        final_accuracy = self.performance_metrics['test_score']
        print(f"\nFinal evaluation:")
        print(f"  Test accuracy: {final_accuracy:.4f}")
        print(f"  Target met: {'✓' if final_accuracy >= self.target_accuracy else '✗'}")
        print(f"  Training time: {self.performance_metrics['training_time']/60:.1f} minutes")
        print(f"  Memory used: {self.performance_metrics['memory_used']/(1024*1024):.1f}MB")
        
        return final_accuracy >= self.target_accuracy
    
    def create_interpretability_analysis(self):
        """Create analysis for explaining decisions to stakeholders"""
        print("\nStep 5: Interpretability Analysis for Stakeholders")
        print("-" * 50)
        
        # Feature importance analysis
        feature_importance = self.adaboost_model.feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = 15  # Show top 15 features
        
        plt.bar(range(top_features), feature_importance[sorted_idx[:top_features]])
        plt.xlabel('Feature Rank')
        plt.ylabel('Feature Importance')
        plt.title('Top 15 Most Important Features in AdaBoost Ensemble')
        plt.xticks(range(top_features), [f'Feature {i+1}' for i in sorted_idx[:top_features]], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Learning curve analysis
        plt.figure(figsize=(12, 8))
        
        # Get staged predictions for learning curve
        train_scores = []
        test_scores = []
        n_estimators_range = range(1, min(51, self.optimal_config['n_estimators'] + 1), 5)
        
        for n_est in n_estimators_range:
            # Create partial model
            partial_model = AdaBoostClassifier(
                estimator=self.optimal_config['base_learner'],
                n_estimators=n_est,
                learning_rate=1.0,
                random_state=42
            )
            partial_model.fit(self.X_train, self.y_train)
            
            train_scores.append(partial_model.score(self.X_train, self.y_train))
            test_scores.append(partial_model.score(self.X_test, self.y_test))
        
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training Accuracy', linewidth=2)
        plt.plot(n_estimators_range, test_scores, 's-', label='Test Accuracy', linewidth=2)
        plt.axhline(y=self.target_accuracy, color='r', linestyle='--', label=f'Target Accuracy ({self.target_accuracy})')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('AdaBoost Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Decision boundary visualization (for 2D subset)
        plt.figure(figsize=(12, 8))
        
        # Use first two features for visualization
        X_2d = self.X_train[:, :2]
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Create a simple model for 2D visualization
        simple_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=20,
            random_state=42
        )
        simple_model.fit(X_2d, self.y_train)
        
        # Predict on mesh grid
        Z = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.y_train, alpha=0.6, cmap='RdYlBu')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('AdaBoost Decision Boundary (2D Visualization)')
        plt.colorbar(label='Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'decision_boundary_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create comprehensive interpretability report
        interpretability_report = {
            'top_features': sorted_idx[:10],
            'feature_importance': feature_importance[sorted_idx[:10]],
            'model_complexity': self.optimal_config['n_estimators'],
            'base_learner_type': 'Decision Tree (max_depth=3)',
            'learning_rate': 1.0
        }
        
        print("Interpretability analysis completed:")
        print(f"  Top feature: Feature {sorted_idx[0] + 1} (importance: {feature_importance[sorted_idx[0]]:.4f})")
        print(f"  Model complexity: {self.optimal_config['n_estimators']} weak learners")
        print(f"  Base learner: Decision Tree with max_depth=3")
        print(f"  Learning rate: 1.0")
        
        self.interpretability_report = interpretability_report
        return interpretability_report
    
    def generate_stakeholder_explanation(self):
        """Generate explanation for stakeholders"""
        print("\nStep 6: Stakeholder Explanation Strategy")
        print("-" * 50)
        
        explanation_strategy = {
            'simple_analogy': "AdaBoost works like a team of experts where each expert focuses on the mistakes of previous experts",
            'key_benefits': [
                "High accuracy through ensemble learning",
                "Interpretable decisions through simple base models",
                "Automatic feature selection",
                "Robust to overfitting"
            ],
            'decision_explanation': [
                "Each prediction is a weighted vote from multiple simple models",
                "Weights are based on how well each model performs",
                "Features are ranked by their importance across all models",
                "Confidence can be measured by the strength of the vote"
            ],
            'business_impact': [
                f"Achieves {self.performance_metrics['test_score']*100:.1f}% accuracy",
                f"Training time: {self.performance_metrics['training_time']/60:.1f} minutes",
                f"Memory usage: {self.performance_metrics['memory_used']/(1024*1024):.1f}MB",
                "Can explain why each prediction was made"
            ]
        }
        
        print("Stakeholder explanation strategy:")
        print(f"  Simple analogy: {explanation_strategy['simple_analogy']}")
        print("\n  Key benefits:")
        for benefit in explanation_strategy['key_benefits']:
            print(f"    - {benefit}")
        
        print("\n  Decision explanation:")
        for explanation in explanation_strategy['decision_explanation']:
            print(f"    - {explanation}")
        
        print("\n  Business impact:")
        for impact in explanation_strategy['business_impact']:
            print(f"    - {impact}")
        
        self.stakeholder_explanation = explanation_strategy
        return explanation_strategy
    
    def run_complete_analysis(self):
        """Run the complete AdaBoost analysis"""
        print("Starting comprehensive AdaBoost analysis...")
        print("=" * 80)
        
        # Step 1: Analyze weak learners
        self.analyze_weak_learners()
        
        # Step 2: Design configuration
        self.design_adaboost_configuration()
        
        # Step 3: Implement AdaBoost
        self.implement_adaboost()
        
        # Step 4: Evaluate strategy
        self.evaluate_strategy()
        
        # Step 5: Interpretability analysis
        self.create_interpretability_analysis()
        
        # Step 6: Stakeholder explanation
        self.generate_stakeholder_explanation()
        
        # Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        print(f"✓ AdaBoost configuration designed and implemented")
        print(f"✓ Target accuracy: {'Achieved' if self.performance_metrics['test_score'] >= self.target_accuracy else 'Not achieved'}")
        print(f"✓ Time constraint: {'Met' if self.performance_metrics['training_time'] <= self.time_constraint else 'Exceeded'}")
        print(f"✓ Memory constraint: {'Met' if self.performance_metrics['memory_used'] <= self.memory_constraint else 'Exceeded'}")
        print(f"✓ Interpretability: {'Provided' if self.interpretability_report else 'Not provided'}")
        
        print(f"\nFinal model performance:")
        print(f"  Test accuracy: {self.performance_metrics['test_score']:.4f}")
        print(f"  Training time: {self.performance_metrics['training_time']/60:.1f} minutes")
        print(f"  Memory used: {self.performance_metrics['memory_used']/(1024*1024):.1f}MB")
        print(f"  Number of estimators: {self.optimal_config['n_estimators']}")
        
        print(f"\nAll visualizations saved to: {save_dir}")
        
        return self.results

if __name__ == "__main__":
    # Run the complete analysis
    challenge = AdaBoostFinalChallenge()
    results = challenge.run_complete_analysis()
