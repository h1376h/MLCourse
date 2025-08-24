import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Any
import seaborn as sns
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX rendering for better mathematical expressions
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionTreeAnalyzer:
    """
    A comprehensive class to analyze decision tree splitting criteria
    and compare Gini impurity vs Entropy for feature selection.
    """
    
    def __init__(self, data: Dict[str, List[int]]):
        """
        Initialize with dataset containing feature values and class counts.
        
        Args:
            data: Dictionary with feature names as keys and lists of class counts as values
        """
        self.data = data
        self.features = list(data.keys())
        self.class_counts = data[self.features[0]]  # Use first feature as reference
        self.total_samples = sum(self.class_counts)
        
    def gini_impurity(self, class_counts: List[int]) -> float:
        """
        Calculate Gini impurity for given class distribution.
        
        Formula: Gini(p) = 1 - Σ(p_i²)
        
        Args:
            class_counts: List of counts for each class
            
        Returns:
            float: Gini impurity value
        """
        if sum(class_counts) == 0:
            return 0.0
        
        total = sum(class_counts)
        probabilities = [count / total for count in class_counts]
        gini = 1 - sum(p**2 for p in probabilities)
        return gini
    
    def entropy(self, class_counts: List[int]) -> float:
        """
        Calculate entropy for given class distribution.
        
        Formula: H(p) = -Σ(p_i * log₂(p_i))
        
        Args:
            class_counts: List of counts for each class
            
        Returns:
            float: Entropy value
        """
        if sum(class_counts) == 0:
            return 0.0
        
        total = sum(class_counts)
        probabilities = [count / total for count in class_counts]
        entropy_val = 0
        
        for p in probabilities:
            if p > 0:
                entropy_val -= p * np.log2(p)
        
        return entropy_val
    
    def information_gain(self, parent_impurity: float, splits: List[List[int]], 
                        impurity_func) -> float:
        """
        Calculate information gain for a given split.
        
        Formula: IG = I_parent - Σ(N_j/N) * I_j
        
        Args:
            parent_impurity: Impurity of parent node
            splits: List of splits, each containing class counts
            impurity_func: Function to calculate impurity (gini or entropy)
            
        Returns:
            float: Information gain value
        """
        total_samples = sum(sum(split) for split in splits)
        weighted_impurity = 0
        
        for split in splits:
            if sum(split) > 0:
                split_impurity = impurity_func(split)
                weight = sum(split) / total_samples
                weighted_impurity += weight * split_impurity
        
        information_gain = parent_impurity - weighted_impurity
        return information_gain
    
    def generate_binary_splits(self, feature_values: List[int]) -> List[Tuple[List[int], List[int]]]:
        """
        Generate all possible binary splits for a feature.
        
        Args:
            feature_values: List of feature values
            
        Returns:
            List of tuples, each containing two lists representing the split
        """
        splits = []
        n = len(feature_values)
        
        # Generate all possible binary partitions
        for i in range(1, 2**(n-1)):
            left = []
            right = []
            
            for j in range(n):
                if (i >> j) & 1:
                    left.append(feature_values[j])
                else:
                    right.append(feature_values[j])
            
            if left and right:  # Ensure both sides have values
                splits.append((left, right))
        
        return splits
    
    def calculate_detailed_impurity(self, class_counts: List[int], impurity_type: str) -> Dict[str, Any]:
        """
        Calculate detailed impurity with step-by-step breakdown.
        
        Args:
            class_counts: List of counts for each class
            impurity_type: 'gini' or 'entropy'
            
        Returns:
            Dictionary with detailed calculation steps
        """
        total = sum(class_counts)
        probabilities = [count / total for count in class_counts]
        
        if impurity_type == 'gini':
            squared_probs = [p**2 for p in probabilities]
            gini = 1 - sum(squared_probs)
            
            return {
                'type': 'gini',
                'total_samples': total,
                'class_counts': class_counts,
                'probabilities': probabilities,
                'squared_probabilities': squared_probs,
                'sum_squared_probs': sum(squared_probs),
                'impurity': gini,
                'formula': f"Gini = 1 - ({' + '.join([f'{p:.4f}²' for p in probabilities])}) = 1 - {sum(squared_probs):.4f} = {gini:.4f}"
            }
        
        elif impurity_type == 'entropy':
            log_probs = []
            entropy_terms = []
            
            for p in probabilities:
                if p > 0:
                    log_p = np.log2(p)
                    log_probs.append(log_p)
                    entropy_terms.append(-p * log_p)
                else:
                    log_probs.append(float('inf'))
                    entropy_terms.append(0)
            
            entropy = sum(entropy_terms)
            
            return {
                'type': 'entropy',
                'total_samples': total,
                'class_counts': class_counts,
                'probabilities': probabilities,
                'log_probabilities': log_probs,
                'entropy_terms': entropy_terms,
                'impurity': entropy,
                'formula': f"H = -({' + '.join([f'{p:.4f}×{log_p:.4f}' for p, log_p in zip(probabilities, log_probs) if p > 0])}) = {entropy:.4f}"
            }
    
    def analyze_feature_splits(self, feature_name: str) -> Dict[str, Any]:
        """
        Analyze all possible splits for a given feature using both Gini and Entropy.
        
        Args:
            feature_name: Name of the feature to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        feature_values = self.data[feature_name]
        splits = self.generate_binary_splits(feature_values)
        
        # Calculate baseline impurities with detailed breakdown
        baseline_gini_details = self.calculate_detailed_impurity(feature_values, 'gini')
        baseline_entropy_details = self.calculate_detailed_impurity(feature_values, 'entropy')
        
        baseline_gini = baseline_gini_details['impurity']
        baseline_entropy = baseline_entropy_details['impurity']
        
        results = {
            'feature': feature_name,
            'baseline_gini': baseline_gini,
            'baseline_entropy': baseline_entropy,
            'baseline_gini_details': baseline_gini_details,
            'baseline_entropy_details': baseline_entropy_details,
            'splits': [],
            'best_gini_split': None,
            'best_entropy_split': None,
            'max_gini_gain': -float('inf'),
            'max_entropy_gain': -float('inf')
        }
        
        for split in splits:
            left, right = split
            
            # Calculate detailed impurities for left and right groups
            left_gini_details = self.calculate_detailed_impurity(left, 'gini')
            right_gini_details = self.calculate_detailed_impurity(right, 'gini')
            left_entropy_details = self.calculate_detailed_impurity(left, 'entropy')
            right_entropy_details = self.calculate_detailed_impurity(right, 'entropy')
            
            # Calculate information gain for Gini
            gini_gain = self.information_gain(baseline_gini, [left, right], self.gini_impurity)
            
            # Calculate information gain for Entropy
            entropy_gain = self.information_gain(baseline_entropy, [left, right], self.entropy)
            
            split_result = {
                'split': split,
                'left': left,
                'right': right,
                'gini_gain': gini_gain,
                'entropy_gain': entropy_gain,
                'left_gini': left_gini_details['impurity'],
                'right_gini': right_gini_details['impurity'],
                'left_entropy': left_entropy_details['impurity'],
                'right_entropy': right_entropy_details['impurity'],
                'left_gini_details': left_gini_details,
                'right_gini_details': right_gini_details,
                'left_entropy_details': left_entropy_details,
                'right_entropy_details': right_entropy_details,
                'gini_calculation': f"IG = {baseline_gini:.4f} - ({len(left)}/{len(feature_values)}×{left_gini_details['impurity']:.4f} + {len(right)}/{len(feature_values)}×{right_gini_details['impurity']:.4f}) = {gini_gain:.4f}",
                'entropy_calculation': f"IG = {baseline_entropy:.4f} - ({len(left)}/{len(feature_values)}×{left_entropy_details['impurity']:.4f} + {len(right)}/{len(feature_values)}×{right_entropy_details['impurity']:.4f}) = {entropy_gain:.4f}"
            }
            
            results['splits'].append(split_result)
            
            # Update best splits
            if gini_gain > results['max_gini_gain']:
                results['max_gini_gain'] = gini_gain
                results['best_gini_split'] = split_result
            
            if entropy_gain > results['max_entropy_gain']:
                results['max_entropy_gain'] = entropy_gain
                results['best_entropy_split'] = split_result
        
        return results
    
    def print_detailed_analysis(self, results: Dict[str, Any]):
        """
        Print detailed step-by-step analysis of the results.
        
        Args:
            results: Analysis results from analyze_feature_splits
        """
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS FOR FEATURE: {results['feature']}")
        print(f"{'='*80}")
        
        print(f"\n1. BASELINE IMPURITIES CALCULATION:")
        print(f"   Total samples: {sum(self.data[results['feature']])}")
        print(f"   Class distribution: {self.data[results['feature']]}")
        
        # Gini baseline details
        gini_details = results['baseline_gini_details']
        print(f"\n   GINI IMPURITY CALCULATION:")
        print(f"   ├─ Probabilities: {[f'{p:.4f}' for p in gini_details['probabilities']]}")
        print(f"   ├─ Squared probabilities: {[f'{p:.4f}' for p in gini_details['squared_probabilities']]}")
        print(f"   ├─ Sum of squared probabilities: {gini_details['sum_squared_probs']:.4f}")
        print(f"   ├─ Formula: {gini_details['formula']}")
        print(f"   └─ Final Gini impurity: {results['baseline_gini']:.6f}")
        
        # Entropy baseline details
        entropy_details = results['baseline_entropy_details']
        print(f"\n   ENTROPY CALCULATION:")
        print(f"   ├─ Probabilities: {[f'{p:.4f}' for p in entropy_details['probabilities']]}")
        print(f"   ├─ Log₂(probabilities): {[f'{p:.4f}' if p != float('inf') else '∞' for p in entropy_details['log_probabilities']]}")
        print(f"   ├─ Entropy terms: {[f'{p:.4f}' for p in entropy_details['entropy_terms']]}")
        print(f"   ├─ Formula: {entropy_details['formula']}")
        print(f"   └─ Final Entropy: {results['baseline_entropy']:.6f}")
        
        print(f"\n2. BINARY SPLITS ANALYSIS:")
        print(f"   Total possible binary splits: {len(results['splits'])}")
        
        for i, split_result in enumerate(results['splits']):
            print(f"\n   {'='*60}")
            print(f"   SPLIT {i+1}: {split_result['split']}")
            print(f"   {'='*60}")
            
            print(f"   LEFT GROUP: {split_result['left']}")
            print(f"   ├─ Gini calculation: {split_result['left_gini_details']['formula']}")
            print(f"   ├─ Entropy calculation: {split_result['left_entropy_details']['formula']}")
            
            print(f"   RIGHT GROUP: {split_result['right']}")
            print(f"   ├─ Gini calculation: {split_result['right_gini_details']['formula']}")
            print(f"   ├─ Entropy calculation: {split_result['right_entropy_details']['formula']}")
            
            print(f"   INFORMATION GAIN CALCULATIONS:")
            print(f"   ├─ Gini Information Gain: {split_result['gini_calculation']}")
            print(f"   └─ Entropy Information Gain: {split_result['entropy_calculation']}")
        
        print(f"\n3. OPTIMAL SPLITS SUMMARY:")
        print(f"   Best Gini split: {results['best_gini_split']['split']}")
        print(f"   ├─ Gini Information Gain: {results['max_gini_gain']:.6f}")
        print(f"   └─ Split details: {results['best_gini_split']['left']} | {results['best_gini_split']['right']}")
        
        print(f"   Best Entropy split: {results['best_entropy_split']['split']}")
        print(f"   ├─ Entropy Information Gain: {results['max_entropy_gain']:.6f}")
        print(f"   └─ Split details: {results['best_entropy_split']['left']} | {results['best_entropy_split']['right']}")
        
        print(f"\n4. COMPARISON:")
        if results['best_gini_split']['split'] == results['best_entropy_split']['split']:
            print(f"   ✓ Both criteria selected the SAME optimal split!")
        else:
            print(f"   ✗ Different optimal splits selected by Gini vs Entropy")
        
        print(f"   Gini gain difference: {results['max_gini_gain'] - results['max_entropy_gain']:.6f}")
    
    def create_comparison_visualization(self, results: Dict[str, Any]):
        """
        Create separate visualizations for different aspects of Gini vs Entropy comparison.
        
        Args:
            results: Analysis results from analyze_feature_splits
        """
        feature_name = results["feature"]
        
        # 1. Information Gain Comparison Chart
        self._create_information_gain_chart(results, feature_name)
        
        # 2. Baseline Impurities Chart
        self._create_baseline_impurities_chart(results, feature_name)
        
        # 3. Best Split Details Charts
        self._create_best_split_charts(results, feature_name)
        
        # 4. Impurity Comparison Chart
        self._create_impurity_comparison_chart(results, feature_name)
        
        # 5. Split Heatmap
        self._create_split_heatmap(results, feature_name)
    
    def _create_information_gain_chart(self, results: Dict[str, Any], feature_name: str):
        """Create information gain comparison chart."""
        plt.figure(figsize=(14, 8))
        
        splits_labels = [f"Split {i+1}" for i in range(len(results['splits']))]
        gini_gains = [split['gini_gain'] for split in results['splits']]
        entropy_gains = [split['entropy_gain'] for split in results['splits']]
        
        x = np.arange(len(splits_labels))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, gini_gains, width, label=r'$Gini$ Information Gain', 
                color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = plt.bar(x + width/2, entropy_gains, width, label=r'$H$ Information Gain', 
                color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Split Number', fontsize=12)
        plt.ylabel('Information Gain', fontsize=12)
        plt.title(f'Information Gain Comparison for {feature_name}', fontsize=14, fontweight='bold')
        plt.xticks(x, splits_labels, rotation=45)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f'information_gain_comparison_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_baseline_impurities_chart(self, results: Dict[str, Any], feature_name: str):
        """Create baseline impurities comparison chart."""
        plt.figure(figsize=(8, 6))
        
        criteria = [r'$Gini$', r'$H$']
        baseline_values = [results['baseline_gini'], results['baseline_entropy']]
        colors = ['skyblue', 'lightcoral']
        
        bars = plt.bar(criteria, baseline_values, color=colors, alpha=0.8, 
                       edgecolor=['navy', 'darkred'], linewidth=2)
        plt.ylabel('Impurity Value', fontsize=12)
        plt.title(f'Baseline Impurities for {feature_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, baseline_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'baseline_impurities_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_best_split_charts(self, results: Dict[str, Any], feature_name: str):
        """Create best split details charts."""
        best_gini = results['best_gini_split']
        best_entropy = results['best_entropy_split']
        
        # Gini best split
        plt.figure(figsize=(8, 6))
        left_gini = best_gini['left_gini_details']
        right_gini = best_gini['right_gini_details']
        
        left_probs = left_gini['probabilities']
        right_probs = right_gini['probabilities']
        
        x_pos = np.arange(max(len(left_probs), len(right_probs)))
        width = 0.35
        
        plt.bar(x_pos[:len(left_probs)] - width/2, left_probs, width, 
                label='Left Group', color='lightgreen', alpha=0.8)
        plt.bar(x_pos[:len(right_probs)] + width/2, right_probs, width, 
                label='Right Group', color='lightblue', alpha=0.8)
        
        plt.xlabel('Class Index', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Best Gini Split: {best_gini["split"]}\nGini Gain: {best_gini["gini_gain"]:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f'best_gini_split_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Entropy best split
        plt.figure(figsize=(8, 6))
        left_entropy = best_entropy['left_entropy_details']
        right_entropy = best_entropy['right_entropy_details']
        
        left_probs = left_entropy['probabilities']
        right_probs = right_entropy['probabilities']
        
        plt.bar(x_pos[:len(left_probs)] - width/2, left_probs, width, 
                label='Left Group', color='lightgreen', alpha=0.8)
        plt.bar(x_pos[:len(right_probs)] + width/2, right_probs, width, 
                label='Right Group', color='lightblue', alpha=0.8)
        
        plt.xlabel('Class Index', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'Best Entropy Split: {best_entropy["split"]}\nEntropy Gain: {best_entropy["entropy_gain"]:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f'best_entropy_split_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_impurity_comparison_chart(self, results: Dict[str, Any], feature_name: str):
        """Create impurity comparison chart."""
        plt.figure(figsize=(10, 6))
        
        best_gini = results['best_gini_split']
        best_entropy = results['best_entropy_split']
        
        categories = ['Left Gini', 'Right Gini', 'Left Entropy', 'Right Entropy']
        values = [best_gini['left_gini'], best_gini['right_gini'], 
                 best_entropy['left_entropy'], best_entropy['right_entropy']]
        colors = ['lightgreen', 'lightblue', 'lightgreen', 'lightblue']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.8)
        plt.ylabel('Impurity Value', fontsize=12)
        plt.title(f'Impurity Comparison for Best Splits - {feature_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'impurity_comparison_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_split_heatmap(self, results: Dict[str, Any], feature_name: str):
        """Create split heatmap visualization."""
        plt.figure(figsize=(12, 6))
        
        # Create a heatmap-like visualization of all splits
        split_matrix = []
        for split_result in results['splits']:
            split_matrix.append([split_result['gini_gain'], split_result['entropy_gain']])
        
        split_matrix = np.array(split_matrix)
        
        im = plt.imshow(split_matrix.T, cmap='RdYlBu_r', aspect='auto')
        plt.xlabel('Split Number', fontsize=12)
        plt.ylabel('Criterion', fontsize=12)
        plt.title(f'Information Gain Heatmap for {feature_name}\nAll Splits Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(results['splits'])), [f'S{i+1}' for i in range(len(results['splits']))])
        plt.yticks([0, 1], [r'$Gini$', r'$H$'])
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Information Gain', fontsize=10)
        
        # Add text annotations
        for i in range(len(results['splits'])):
            for j in range(2):
                plt.text(i, j, f'{split_matrix[i, j]:.3f}', 
                        ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'split_heatmap_{feature_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_analysis_visualization(self, all_results: List[Dict[str, Any]]):
        """
        Create visualization comparing all features.
        
        Args:
            all_results: List of analysis results for all features
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Maximum Information Gain by Feature (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        features = [result['feature'] for result in all_results]
        max_gini_gains = [result['max_gini_gain'] for result in all_results]
        max_entropy_gains = [result['max_entropy_gain'] for result in all_results]
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, max_gini_gains, width, label=r'Max $Gini$ Gain', 
                color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = ax1.bar(x + width/2, max_entropy_gains, width, label=r'Max $H$ Gain', 
                color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Maximum Information Gain', fontsize=12)
        ax1.set_title('Maximum Information Gain by Feature and Criterion', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(features)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Baseline Impurities by Feature (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        baseline_gini = [result['baseline_gini'] for result in all_results]
        baseline_entropy = [result['baseline_entropy'] for result in all_results]
        
        bars1 = ax2.bar(x - width/2, baseline_gini, width, label=r'Baseline $Gini$', 
                color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        bars2 = ax2.bar(x + width/2, baseline_entropy, width, label=r'Baseline $H$', 
                color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        ax2.set_xlabel('Feature', fontsize=12)
        ax2.set_ylabel('Baseline Impurity', fontsize=12)
        ax2.set_title('Baseline Impurities by Feature and Criterion', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Performance Ranking (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Sort features by maximum information gain
        sorted_results = sorted(all_results, key=lambda x: max(x['max_gini_gain'], x['max_entropy_gain']), reverse=True)
        sorted_features = [result['feature'] for result in sorted_results]
        best_gains = [max(result['max_gini_gain'], result['max_entropy_gain']) for result in sorted_results]
        
        bars = ax3.barh(sorted_features, best_gains, color='lightgreen', alpha=0.8)
        ax3.set_xlabel('Best Information Gain', fontsize=12)
        ax3.set_title('Feature Performance Ranking', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, best_gains):
            width = bar.get_width()
            ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        # 4. Criterion Preference Analysis (bottom center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Count which criterion performs better for each feature
        gini_wins = sum(1 for result in all_results if result['max_gini_gain'] > result['max_entropy_gain'])
        entropy_wins = sum(1 for result in all_results if result['max_entropy_gain'] > result['max_gini_gain'])
        ties = sum(1 for result in all_results if abs(result['max_gini_gain'] - result['max_entropy_gain']) < 1e-10)
        
        criteria = ['Gini Wins', 'Entropy Wins', 'Ties']
        counts = [gini_wins, entropy_wins, ties]
        colors = ['skyblue', 'lightcoral', 'lightgray']
        
        bars = ax4.bar(criteria, counts, color=colors, alpha=0.8)
        ax4.set_ylabel('Number of Features', fontsize=12)
        ax4.set_title('Criterion Performance Summary', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 5. Detailed Split Analysis (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Show the distribution of information gains across all splits
        all_gini_gains = []
        all_entropy_gains = []
        
        for result in all_results:
            all_gini_gains.extend([split['gini_gain'] for split in result['splits']])
            all_entropy_gains.extend([split['entropy_gain'] for split in result['splits']])
        
        # Create histogram
        ax5.hist(all_gini_gains, bins=20, alpha=0.7, label='Gini Gains', color='skyblue', edgecolor='navy')
        ax5.hist(all_entropy_gains, bins=20, alpha=0.7, label='Entropy Gains', color='lightcoral', edgecolor='darkred')
        
        ax5.set_xlabel('Information Gain', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Distribution of Information Gains\nAcross All Features', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_splits_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    


def main():
    """
    Main function to demonstrate the decision tree analysis.
    """
    print("DECISION TREE SPLITTING CRITERIA ANALYSIS")
    print("=" * 60)
    print("Comparing Gini Impurity vs Entropy for Feature Selection")
    print("Enhanced Version with LaTeX Rendering and Separate Visualizations")
    print("=" * 60)
    
    # Define the dataset with feature values and class counts
    # This represents a real-world scenario where we have multiple features
    # and want to determine the best splitting criterion
    dataset = {
        'Feature_A': [2, 8, 5, 3, 7],  # Class counts for different values
        'Feature_B': [4, 6, 3, 9, 2],  # Class counts for different values
        'Feature_C': [1, 7, 4, 6, 8]   # Class counts for different values
    }
    
    print(f"\nDATASET OVERVIEW:")
    for feature, values in dataset.items():
        print(f"  {feature}: {values} (Total: {sum(values)})")
    
    # Create analyzer instance
    analyzer = DecisionTreeAnalyzer(dataset)
    
    # Analyze each feature
    all_results = []
    for feature in dataset.keys():
        print(f"\n{'='*80}")
        print(f"ANALYZING FEATURE: {feature}")
        print(f"{'='*80}")
        
        results = analyzer.analyze_feature_splits(feature)
        analyzer.print_detailed_analysis(results)
        all_results.append(results)
        
        # Create individual feature visualization
        analyzer.create_comparison_visualization(results)
    
    # Create comprehensive comparison visualization
    analyzer.create_feature_analysis_visualization(all_results)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nFEATURE RANKING BY INFORMATION GAIN:")
    print(f"{'Feature':<12} {'Max Gini Gain':<15} {'Max Entropy Gain':<18} {'Best Criterion':<15}")
    print(f"{'-'*60}")
    
    # Sort features by maximum information gain
    sorted_results = sorted(all_results, key=lambda x: max(x['max_gini_gain'], x['max_entropy_gain']), reverse=True)
    
    for result in sorted_results:
        best_criterion = "Gini" if result['max_gini_gain'] > result['max_entropy_gain'] else "Entropy"
        print(f"{result['feature']:<12} {result['max_gini_gain']:<15.6f} {result['max_entropy_gain']:<18.6f} {best_criterion:<15}")
    
    print(f"\nKEY INSIGHTS:")
    print(f"1. Total features analyzed: {len(all_results)}")
    print(f"2. Features with identical optimal splits: {sum(1 for r in all_results if r['best_gini_split']['split'] == r['best_entropy_split']['split'])}")
    print(f"3. Best overall feature: {sorted_results[0]['feature']}")
    print(f"4. Maximum information gain achieved: {max(sorted_results[0]['max_gini_gain'], sorted_results[0]['max_entropy_gain']):.6f}")
    
    print(f"\nVisualizations saved to: {save_dir}")
    print(f"Files generated:")
    print(f"  - information_gain_comparison_[Feature].png (information gain comparison)")
    print(f"  - baseline_impurities_[Feature].png (baseline impurity values)")
    print(f"  - best_gini_split_[Feature].png (best Gini split details)")
    print(f"  - best_entropy_split_[Feature].png (best Entropy split details)")
    print(f"  - impurity_comparison_[Feature].png (impurity comparison)")
    print(f"  - split_heatmap_[Feature].png (split performance heatmap)")
    print(f"  - feature_splits_analysis.png (comprehensive comparison)")

if __name__ == "__main__":
    main()
