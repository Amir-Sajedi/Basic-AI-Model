from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from collections import Counter


class Node:
    """
    A single node in the decision tree.

    Think of this as a single decision point in your tree - it either:
    1. Asks a question about a feature (internal node)
    2. Gives a final answer/prediction (leaf node)
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=0, impurity=0.0, information_gain=0.0):
        # For internal nodes: which feature to split on and at what value
        self.feature = feature
        self.threshold = threshold

        # Child nodes (left = True condition, right = False condition)
        self.left = left
        self.right = right

        # For leaf nodes: the final prediction
        self.value = value

        # Node statistics for analysis
        self.samples = samples
        self.impurity = impurity
        self.information_gain = information_gain

    def is_leaf(self):
        """Check if this node is a leaf (final decision) node"""
        return self.value is not None


class OptimizedDecisionTree:
    """
    An optimized decision tree classifier with multiple splitting criteria,
    pruning capabilities, and comprehensive visualization.

    Key improvements over basic implementations:
    - Efficient threshold selection using quartiles instead of all unique values
    - Post-pruning to prevent overfitting
    - Support for both entropy and Gini criteria
    - Comprehensive feature importance calculation
    - Beautiful tree visualization with detailed node information
    """

    def __init__(self, criterion='entropy', max_depth=10, min_samples_split=20,
                 min_samples_leaf=5, enable_pruning=True, ccp_alpha=0.0, feature_names=None):
        # Core parameters that control tree growth
        self.criterion = criterion  # How we measure impurity: 'entropy' or 'gini'
        self.max_depth = max_depth  # Maximum tree depth to prevent overfitting
        self.min_samples_split = min_samples_split  # Min samples needed to split a node
        self.min_samples_leaf = min_samples_leaf  # Min samples required in each leaf

        # Pruning parameters to improve generalization
        self.enable_pruning = enable_pruning
        self.ccp_alpha = ccp_alpha  # Cost complexity pruning parameter

        # Tree structure and metadata
        self.root = None
        self.feature_names = feature_names or []

        # Performance tracking
        self._node_count = 0
        self._max_depth_reached = 0

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the decision tree on the given data.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation set for pruning (optional)
            y_val: Validation targets for pruning (optional)
        """
        print(f"Training decision tree with {self.criterion} criterion...")
        self._node_count = 0
        self._max_depth_reached = 0

        # Build the tree using recursive algorithm
        self.root = self._grow_tree(X, y, depth=0)

        # Apply post-pruning if enabled and validation data is provided
        if self.enable_pruning and X_val is not None and y_val is not None:
            print("Applying post-pruning to improve generalization...")
            self.root = self._prune_tree(self.root, X_val, y_val)

        print(f"Tree built successfully! Nodes: {self._node_count}, Max depth: {self._max_depth_reached}")

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        This is the heart of the algorithm - it decides whether to:
        1. Create a leaf node (stop splitting)
        2. Find the best split and create child nodes
        """
        num_samples, num_features = X.shape
        self._node_count += 1
        self._max_depth_reached = max(self._max_depth_reached, depth)

        # Calculate current impurity (how mixed are the classes?)
        impurity = self._calculate_impurity(y)
        unique_classes, counts = np.unique(y, return_counts=True)

        # Stopping criteria - when to create a leaf node
        should_stop = (
                len(unique_classes) == 1 or  # All samples have same class (pure node)
                depth >= self.max_depth or  # Reached maximum allowed depth
                num_samples < self.min_samples_split or  # Too few samples to split
                impurity < 1e-10  # Node is already very pure
        )

        if should_stop:
            # Create leaf node with majority class
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Find the best way to split this node
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # If no good split found, create leaf node
        if best_feature is None or best_gain <= 0:
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Split the data based on the best split found
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Ensure both splits have minimum required samples
        if (np.sum(left_mask) < self.min_samples_leaf or
                np.sum(right_mask) < self.min_samples_leaf):
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Recursively build left and right subtrees
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=num_samples,
            impurity=impurity,
            information_gain=best_gain
        )

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        This uses an optimized approach: instead of testing every unique value
        as a threshold (which can be thousands for continuous features),
        we only test quartiles + min/max values.
        """
        num_samples, num_features = X.shape
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        # Test each feature as a potential split
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]

            # Smart threshold selection: use quartiles instead of all unique values
            # This dramatically improves performance while maintaining quality
            thresholds = self._get_candidate_thresholds(feature_values)

            # Test each threshold for this feature
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if split would create empty partitions
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain for this split
                gain = self._calculate_information_gain(y, y[left_mask], y[right_mask])

                # Keep track of the best split found so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _get_candidate_thresholds(self, feature_values):
        """
        Get smart candidate thresholds for a feature.

        Instead of testing every unique value (expensive), we test:
        - Quartiles (25th, 50th, 75th percentiles)
        - Min and max values
        - A few values in between

        This gives us good splits while being much faster.
        """
        if len(np.unique(feature_values)) <= 10:
            # If feature has few unique values, test them all
            return np.unique(feature_values)
        else:
            # For continuous features, use smart sampling
            percentiles = [0, 25, 50, 75, 100]
            thresholds = np.percentile(feature_values, percentiles)
            return np.unique(thresholds)

    def _calculate_impurity(self, y):
        """Calculate impurity of a set of labels"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini_index(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _entropy(self, y):
        """
        Calculate entropy - measures how mixed/uncertain the labels are.

        Entropy = 0 means all samples have same class (pure)
        Entropy is high when classes are evenly mixed

        Formula: -Σ(p_i * log2(p_i)) where p_i is proportion of class i
        """
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        # Add small epsilon to avoid log(0)
        return -np.sum(proportions * np.log2(proportions + 1e-15))

    def _gini_index(self, y):
        """
        Calculate Gini impurity - alternative to entropy.

        Gini = 0 means pure node (all same class)
        Gini approaches 0.5 as classes become more evenly mixed

        Formula: 1 - Σ(p_i²) where p_i is proportion of class i
        """
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions ** 2)

    def _calculate_information_gain(self, parent_y, left_y, right_y):
        """
        Calculate information gain from a split.

        Information Gain = Parent Impurity - Weighted Average of Child Impurities
        Higher gain = better split
        """
        if len(parent_y) == 0:
            return 0

        parent_impurity = self._calculate_impurity(parent_y)

        # Calculate weighted average of child impurities
        n_left, n_right = len(left_y), len(right_y)
        n_total = len(parent_y)

        if n_left == 0 or n_right == 0:
            return 0

        left_weight = n_left / n_total
        right_weight = n_right / n_total

        weighted_child_impurity = (left_weight * self._calculate_impurity(left_y) +
                                   right_weight * self._calculate_impurity(right_y))

        return parent_impurity - weighted_child_impurity

    def _prune_tree(self, node, X_val, y_val):
        """
        Post-pruning to improve generalization.

        This removes parts of the tree that don't improve accuracy on validation data.
        Helps prevent overfitting to training data.
        """
        if node.is_leaf():
            return node

        # Recursively prune children first
        if node.left:
            node.left = self._prune_tree(node.left, X_val, y_val)
        if node.right:
            node.right = self._prune_tree(node.right, X_val, y_val)

        # Consider pruning this node if both children are leaves
        if (node.left.is_leaf() and node.right.is_leaf()):
            # Calculate accuracy before pruning
            accuracy_before = self._calculate_accuracy(X_val, y_val)

            # Temporarily convert to leaf (prune)
            original_state = self._save_node_state(node)
            self._convert_to_leaf(node)

            # Calculate accuracy after pruning
            accuracy_after = self._calculate_accuracy(X_val, y_val)

            # Keep pruning if it doesn't hurt accuracy (within tolerance)
            if accuracy_after >= accuracy_before - self.ccp_alpha:
                return node  # Keep as leaf
            else:
                # Restore original structure
                self._restore_node_state(node, original_state)

        return node

    def _save_node_state(self, node):
        """Save current state of a node for potential restoration"""
        return {
            'left': node.left,
            'right': node.right,
            'feature': node.feature,
            'threshold': node.threshold
        }

    def _restore_node_state(self, node, state):
        """Restore a node to its previous state"""
        node.left = state['left']
        node.right = state['right']
        node.feature = state['feature']
        node.threshold = state['threshold']
        node.value = None  # No longer a leaf

    def _convert_to_leaf(self, node):
        """Convert an internal node to a leaf node"""
        # Determine majority class from children
        left_samples = node.left.samples if node.left else 0
        right_samples = node.right.samples if node.right else 0

        if left_samples >= right_samples:
            node.value = node.left.value
        else:
            node.value = node.right.value

        # Remove children
        node.left = None
        node.right = None
        node.feature = None
        node.threshold = None

    def _calculate_accuracy(self, X, y):
        """Calculate prediction accuracy on given dataset"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def predict(self, X):
        """Make predictions for multiple samples"""
        if self.root is None:
            raise ValueError("Tree has not been fitted yet!")

        return np.array([self._predict_single(sample) for sample in X])

    def _predict_single(self, sample):
        """
        Predict class for a single sample by traversing the tree.

        Start at root and follow the path based on feature values
        until we reach a leaf node.
        """
        node = self.root

        while not node.is_leaf():
            if sample[node.feature] <= node.threshold:
                node = node.left  # Go left (True condition)
            else:
                node = node.right  # Go right (False condition)

        return node.value

    def get_feature_name(self, feature_idx):
        """Get human-readable name for a feature"""
        if feature_idx < len(self.feature_names):
            return self.feature_names[feature_idx]
        return f"Feature_{feature_idx}"

    def calculate_feature_importance(self):
        """
        Calculate feature importance based on information gain.

        More important features contribute more to reducing impurity
        across the tree.
        """
        if self.root is None:
            return {}

        importances = np.zeros(len(self.feature_names))
        self._accumulate_importance(self.root, importances)

        # Normalize to sum to 1
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance

        # Return as dictionary
        return {
            self.get_feature_name(i): importance
            for i, importance in enumerate(importances)
        }

    def _accumulate_importance(self, node, importances):
        """Recursively accumulate feature importance scores"""
        if node.is_leaf():
            return

        # Add this node's contribution to feature importance
        importance_contribution = (node.information_gain * node.samples / self.root.samples)
        importances[node.feature] += importance_contribution

        # Recurse to children
        if node.left:
            self._accumulate_importance(node.left, importances)
        if node.right:
            self._accumulate_importance(node.right, importances)

    def print_tree(self, max_depth=None):
        """Print tree structure in a readable format"""
        if self.root is None:
            print("Tree has not been fitted yet!")
            return

        print(f"\nDecision Tree Structure ({self.criterion.upper()} criterion):")
        print("=" * 60)
        self._print_node(self.root, depth=0, max_depth=max_depth)

    def _print_node(self, node, depth=0, max_depth=None):
        """Recursively print node information"""
        if max_depth is not None and depth > max_depth:
            print("  " * depth + "...")
            return

        indent = "  " * depth

        if node.is_leaf():
            print(f"{indent}→ Predict: {node.value} "
                  f"(samples: {node.samples}, {self.criterion}: {node.impurity:.3f})")
        else:
            feature_name = self.get_feature_name(node.feature)
            print(f"{indent}{feature_name} <= {node.threshold:.3f} "
                  f"(samples: {node.samples}, {self.criterion}: {node.impurity:.3f}, "
                  f"gain: {node.information_gain:.3f})")

            # Print left subtree (True branch)
            if node.left:
                self._print_node(node.left, depth + 1, max_depth)

            # Print right branch condition
            print(f"{indent}{feature_name} > {node.threshold:.3f}")

            # Print right subtree (False branch)
            if node.right:
                self._print_node(node.right, depth + 1, max_depth)

    def visualize_tree(self, max_depth=3, figsize=(20, 12)):
        """
        Create a beautiful visualization of the decision tree.

        Shows feature splits, information gain, impurity, and sample counts
        for each node in an easy-to-read format.
        """
        if self.root is None:
            print("Tree has not been fitted yet!")
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Calculate positions for all nodes
        positions = {}
        self._calculate_positions(self.root, 5, 9, 4, 0, max_depth, positions)

        # Draw the tree
        self._draw_tree(ax, self.root, positions, max_depth, 0)

        plt.title(f'Decision Tree Visualization ({self.criterion.upper()} Criterion)\n'
                  f'Nodes: {self._node_count}, Max Depth: {self._max_depth_reached}',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def _calculate_positions(self, node, x, y, width, depth, max_depth, positions):
        """Calculate optimal positions for tree nodes"""
        if node is None or depth > max_depth:
            return

        positions[id(node)] = (x, y)

        if not node.is_leaf() and depth < max_depth:
            child_width = width / 2
            horizontal_spread = width / 3
            left_x = x - horizontal_spread
            right_x = x + horizontal_spread
            child_y = y - 1.5

            if node.left:
                self._calculate_positions(node.left, left_x, child_y,
                                          child_width, depth + 1, max_depth, positions)
            if node.right:
                self._calculate_positions(node.right, right_x, child_y,
                                          child_width, depth + 1, max_depth, positions)

    def _draw_tree(self, ax, node, positions, max_depth, depth):
        """Draw tree nodes and connections"""
        if node is None or depth > max_depth:
            return

        x, y = positions[id(node)]

        if node.is_leaf():
            # Draw leaf node
            color = '#90EE90' if node.value == 1 else '#FFB6C1'

            box = FancyBboxPatch((x - 0.5, y - 0.3), 1.0, 0.6,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)

            ax.text(x, y, f'Class: {node.value}\nSamples: {node.samples}\n'
                          f'{self.criterion}: {node.impurity:.3f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            # Draw internal node
            color = '#E6E6FA'

            box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                                 boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)

            feature_name = self.get_feature_name(node.feature)
            if len(feature_name) > 15:
                feature_name = feature_name[:12] + "..."

            ax.text(x, y, f'{feature_name} ≤ {node.threshold:.2f}\n'
                          f'Samples: {node.samples}\n'
                          f'{self.criterion}: {node.impurity:.3f}\n'
                          f'Gain: {node.information_gain:.3f}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

            # Draw connections to children
            if depth < max_depth:
                if node.left and id(node.left) in positions:
                    left_x, left_y = positions[id(node.left)]
                    ax.plot([x - 0.2, left_x + 0.2], [y - 0.4, left_y + 0.4], 'k-', linewidth=2)
                    ax.text((x - 0.2 + left_x + 0.2) / 2, (y - 0.4 + left_y + 0.4) / 2 + 0.1,
                            'True', ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))

                if node.right and id(node.right) in positions:
                    right_x, right_y = positions[id(node.right)]
                    ax.plot([x + 0.2, right_x - 0.2], [y - 0.4, right_y + 0.4], 'k-', linewidth=2)
                    ax.text((x + 0.2 + right_x - 0.2) / 2, (y - 0.4 + right_y + 0.4) / 2 + 0.1,
                            'False', ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.8))

        # Recursively draw children
        if not node.is_leaf() and depth < max_depth:
            if node.left:
                self._draw_tree(ax, node.left, positions, max_depth, depth + 1)
            if node.right:
                self._draw_tree(ax, node.right, positions, max_depth, depth + 1)


def preprocess_data(X, y):
    """
    Clean and prepare data for machine learning.

    This function handles:
    - Converting categorical variables to numbers
    - Filling missing values
    - Encoding target labels
    """
    print("Preprocessing data...")

    X_processed = X.copy()
    label_encoders = {}

    # Convert categorical features to numeric
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le

    # Fill missing values with column means
    X_processed = X_processed.fillna(X_processed.mean())

    # Process target variable
    y_processed = y.iloc[:, 0] if hasattr(y, 'iloc') else y

    # Encode target if categorical
    if y_processed.dtype == 'object':
        target_encoder = LabelEncoder()
        y_processed = target_encoder.fit_transform(y_processed.astype(str))
        label_encoders['target'] = target_encoder

    print(f"Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    return X_processed.values, y_processed, label_encoders


def main():
    """
    Main function to demonstrate the optimized decision tree.
    """
    print("🌳 Optimized Decision Tree Classifier Demo")
    print("=" * 50)

    # Load and preprocess data
    print("Loading Bank Marketing dataset...")
    bank_marketing = fetch_ucirepo(id=222)
    X, y = bank_marketing.data.features, bank_marketing.data.targets

    X_processed, y_processed, encoders = preprocess_data(X, y)
    feature_names = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )

    # Further split training data for validation (used in pruning)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Compare different criteria
    criteria = ['entropy', 'gini']
    results = {}

    for criterion in criteria:
        print(f"\n🔍 Training with {criterion.upper()} criterion")
        print("-" * 40)

        # Train tree with pruning
        tree = OptimizedDecisionTree(
            criterion=criterion,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            enable_pruning=True,
            ccp_alpha=0.001,  # Small tolerance for accuracy loss during pruning
            feature_names=feature_names
        )

        tree.fit(X_train_fit, y_train_fit, X_val, y_val)

        # Evaluate performance
        train_predictions = tree.predict(X_train_fit)
        test_predictions = tree.predict(X_test)

        train_accuracy = np.mean(train_predictions == y_train_fit)
        test_accuracy = np.mean(test_predictions == y_test)

        results[criterion] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'tree': tree
        }

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")

        # Show tree structure (limited depth for readability)
        tree.print_tree(max_depth=3)

        # Visualize tree
        tree.visualize_tree(max_depth=3, figsize=(18, 12))

        # Show feature importance
        importance = tree.calculate_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\nTop 10 Most Important Features ({criterion}):")
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("=" * 60)

    for criterion in criteria:
        result = results[criterion]
        print(f"{criterion.upper():>8}: Train={result['train_accuracy']:.4f}, "
              f"Test={result['test_accuracy']:.4f}, "
              f"Generalization={result['test_accuracy']:.4f}")

    # Determine best model
    best_criterion = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\n🥇 Best performing model: {best_criterion.upper()} "
          f"(Test Accuracy: {results[best_criterion]['test_accuracy']:.4f})")


if __name__ == "__main__":
    main()