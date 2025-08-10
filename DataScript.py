from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Handle matplotlib backend issues
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.ioff()


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=0, impurity=0.0, information_gain=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
        self.information_gain = information_gain

    def is_leaf(self):
        return self.value is not None


class OptimizedDecisionTree:
    def __init__(self, criterion='entropy', max_depth=10, min_samples_split=20,
                 min_samples_leaf=5, enable_pruning=True, ccp_alpha=0.01,
                 feature_names=None, max_features='sqrt'):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.enable_pruning = enable_pruning
        self.ccp_alpha = ccp_alpha
        self.feature_names = feature_names or []
        self.max_features = max_features

        self.root = None
        self._node_count = 0
        self._max_depth_reached = 0

    def _get_n_features_to_sample(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, (int, float)):
            return max(1, int(self.max_features * n_features) if isinstance(self.max_features, float)
            else min(self.max_features, n_features))
        return n_features

    def fit(self, X, y, X_val=None, y_val=None):
        print(f"Training optimized decision tree with {self.criterion} criterion...")

        self._node_count = 0
        self._max_depth_reached = 0

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        self.root = self._grow_tree(X, y, depth=0)

        if self.enable_pruning and X_val is not None and y_val is not None:
            print("Applying cost complexity pruning...")
            self.root = self._prune_tree(self.root, X_val, y_val)

        print(f"Tree completed: {self._node_count} nodes, max depth: {self._max_depth_reached}")
        return self

    def _grow_tree(self, X, y, depth=0, indices=None):
        if indices is None:
            indices = np.arange(len(y))

        num_samples = len(indices)
        self._node_count += 1
        self._max_depth_reached = max(self._max_depth_reached, depth)

        y_subset = y[indices]
        impurity = self._calculate_impurity(y_subset)
        unique_classes, counts = np.unique(y_subset, return_counts=True)

        # Stopping criteria
        if (len(unique_classes) == 1 or depth >= self.max_depth or
                num_samples < self.min_samples_split or impurity < 1e-7):
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, indices)

        if best_feature is None or best_gain <= 1e-7:
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Split data
        feature_values = X[indices, best_feature]
        left_mask = feature_values <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            majority_class = unique_classes[np.argmax(counts)]
            return Node(value=majority_class, samples=num_samples, impurity=impurity)

        # Build children
        left_child = self._grow_tree(X, y, depth + 1, left_indices)
        right_child = self._grow_tree(X, y, depth + 1, right_indices)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child,
                    right=right_child, samples=num_samples, impurity=impurity,
                    information_gain=best_gain)

    def _find_best_split(self, X, y, indices):
        n_samples, n_features = X.shape
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        # Feature sampling
        n_features_to_sample = self._get_n_features_to_sample(n_features)
        feature_candidates = np.random.choice(n_features, n_features_to_sample, replace=False)

        for feature_idx in feature_candidates:
            feature_values = X[indices, feature_idx]
            thresholds = self._get_thresholds(feature_values, y[indices])

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
                    continue

                gain = self._calculate_information_gain(
                    y[indices], y[indices[left_mask]], y[indices[~left_mask]]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _get_thresholds(self, feature_values, y_subset):
        unique_values = np.unique(feature_values)

        if len(unique_values) <= 20:
            return unique_values[:-1]

        # Use percentiles and class-aware quantiles
        thresholds = list(np.percentile(feature_values, [10, 25, 33, 50, 67, 75, 90]))

        for class_val in np.unique(y_subset):
            class_mask = y_subset == class_val
            if np.sum(class_mask) > 1:
                class_values = feature_values[class_mask]
                thresholds.extend(np.percentile(class_values, [25, 50, 75]))

        return np.unique(thresholds)

    def _calculate_impurity(self, y):
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini_index(y)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-15))

    def _gini_index(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions ** 2)

    def _calculate_information_gain(self, parent_y, left_y, right_y):
        if len(parent_y) == 0:
            return 0

        parent_impurity = self._calculate_impurity(parent_y)
        n_left, n_right, n_total = len(left_y), len(right_y), len(parent_y)

        if n_left == 0 or n_right == 0:
            return 0

        weighted_child_impurity = (n_left / n_total * self._calculate_impurity(left_y) +
                                   n_right / n_total * self._calculate_impurity(right_y))

        return parent_impurity - weighted_child_impurity

    def _prune_tree(self, node, X_val, y_val):
        if node.is_leaf():
            return node

        # Recursively prune children
        if node.left:
            node.left = self._prune_tree(node.left, X_val, y_val)
        if node.right:
            node.right = self._prune_tree(node.right, X_val, y_val)

        # Consider pruning this node if both children are leaves
        if node.left.is_leaf() and node.right.is_leaf():
            accuracy_before = np.mean(self.predict(X_val) == y_val)

            # Save current state and convert to leaf
            original_left, original_right = node.left, node.right
            node.value = node.left.value if node.left.samples >= node.right.samples else node.right.value
            node.left = node.right = None

            accuracy_after = np.mean(self.predict(X_val) == y_val)

            # Keep pruning if within tolerance
            if accuracy_after >= accuracy_before - self.ccp_alpha:
                return node
            else:
                # Restore original structure
                node.left, node.right = original_left, original_right
                node.value = None

        return node

    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree has not been fitted yet!")

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return np.array([self._predict_single(sample) for sample in X])

    def _predict_single(self, sample):
        node = self.root
        while not node.is_leaf():
            node = node.left if sample[node.feature] <= node.threshold else node.right
        return node.value

    def get_feature_name(self, feature_idx):
        return (self.feature_names[feature_idx] if feature_idx < len(self.feature_names)
                else f"Feature_{feature_idx}")

    def calculate_feature_importance(self):
        if self.root is None:
            return {}

        n_features = len(self.feature_names) or self._infer_n_features()
        importances = np.zeros(n_features)
        self._accumulate_importance(self.root, importances, self.root.samples)

        total_importance = np.sum(importances)
        if total_importance > 0:
            importances /= total_importance

        return {self.get_feature_name(i): importance for i, importance in enumerate(importances)}

    def _infer_n_features(self):
        def find_max_feature(node):
            if node is None or node.is_leaf():
                return 0
            return max(node.feature, find_max_feature(node.left), find_max_feature(node.right))

        return find_max_feature(self.root) + 1

    def _accumulate_importance(self, node, importances, total_samples):
        if node.is_leaf():
            return

        importances[node.feature] += (node.information_gain * node.samples) / total_samples

        if node.left:
            self._accumulate_importance(node.left, importances, total_samples)
        if node.right:
            self._accumulate_importance(node.right, importances, total_samples)

    def print_tree(self, max_depth=None):
        if self.root is None:
            print("Tree has not been fitted yet!")
            return

        print(f"\nDecision Tree Structure ({self.criterion.upper()} criterion):")
        print("=" * 60)
        self._print_node(self.root, depth=0, max_depth=max_depth)

    def _print_node(self, node, depth=0, max_depth=None):
        if max_depth is not None and depth > max_depth:
            print("  " * depth + "...")
            return

        indent = "  " * depth

        if node.is_leaf():
            print(f"{indent}-> Predict: {node.value} "
                  f"(samples: {node.samples}, {self.criterion}: {node.impurity:.3f})")
        else:
            feature_name = self.get_feature_name(node.feature)
            print(f"{indent}{feature_name} <= {node.threshold:.3f} "
                  f"(samples: {node.samples}, {self.criterion}: {node.impurity:.3f}, "
                  f"gain: {node.information_gain:.3f})")

            if node.left:
                self._print_node(node.left, depth + 1, max_depth)

            print(f"{indent}{feature_name} > {node.threshold:.3f}")

            if node.right:
                self._print_node(node.right, depth + 1, max_depth)

    def visualize_tree(self, max_depth=3, figsize=(18, 12)):
        if self.root is None:
            print("Tree has not been fitted yet!")
            return

        try:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            positions = {}
            self._calculate_positions(self.root, 5, 9, 4, 0, max_depth, positions)
            self._draw_tree(ax, self.root, positions, max_depth, 0)

            plt.title(f'Optimized Decision Tree ({self.criterion.upper()} Criterion)\n'
                      f'Nodes: {self._node_count}, Max Depth: {self._max_depth_reached}',
                      fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as e:
            print(f"Visualization failed: {e}")
            self._draw_simple_tree(self.root, "", True, 0, max_depth)

    def _calculate_positions(self, node, x, y, width, depth, max_depth, positions):
        if node is None or depth > max_depth:
            return

        positions[id(node)] = (x, y)

        if not node.is_leaf() and depth < max_depth:
            horizontal_spread = width / 3
            child_y = y - 1.5

            if node.left:
                self._calculate_positions(node.left, x - horizontal_spread, child_y,
                                          width / 2, depth + 1, max_depth, positions)
            if node.right:
                self._calculate_positions(node.right, x + horizontal_spread, child_y,
                                          width / 2, depth + 1, max_depth, positions)

    def _draw_tree(self, ax, node, positions, max_depth, depth):
        if node is None or depth > max_depth:
            return

        x, y = positions[id(node)]

        if node.is_leaf():
            color = '#90EE90' if node.value == 1 else '#FFB6C1'
            box = FancyBboxPatch((x - 0.5, y - 0.3), 1.0, 0.6, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, f'Class: {node.value}\nSamples: {node.samples}\n'
                          f'{self.criterion}: {node.impurity:.3f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8, boxstyle="round,pad=0.02",
                                 facecolor='#E6E6FA', edgecolor='black', linewidth=2)
            ax.add_patch(box)

            feature_name = self.get_feature_name(node.feature)
            if len(feature_name) > 15:
                feature_name = feature_name[:12] + "..."

            ax.text(x, y, f'{feature_name} <= {node.threshold:.2f}\n'
                          f'Samples: {node.samples}\n{self.criterion}: {node.impurity:.3f}\n'
                          f'Gain: {node.information_gain:.3f}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

            # Draw connections
            if depth < max_depth:
                for child, label, color in [(node.left, 'True', 'lightgreen'),
                                            (node.right, 'False', 'lightcoral')]:
                    if child and id(child) in positions:
                        child_x, child_y = positions[id(child)]
                        offset = -0.2 if child == node.left else 0.2
                        ax.plot([x + offset, child_x - offset], [y - 0.4, child_y + 0.4],
                                'k-', linewidth=2)
                        ax.text((x + offset + child_x - offset) / 2,
                                (y - 0.4 + child_y + 0.4) / 2 + 0.1,
                                label, ha='center', va='center', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

        # Draw children
        if not node.is_leaf() and depth < max_depth:
            if node.left:
                self._draw_tree(ax, node.left, positions, max_depth, depth + 1)
            if node.right:
                self._draw_tree(ax, node.right, positions, max_depth, depth + 1)

    def _draw_simple_tree(self, node, prefix, is_last, depth, max_depth):
        if depth > max_depth or node is None:
            return

        connector = "└── " if is_last else "├── "

        if node.is_leaf():
            print(f"{prefix}{connector}LEAF: Class={node.value}, Samples={node.samples}")
        else:
            feature_name = self.get_feature_name(node.feature)
            print(f"{prefix}{connector}{feature_name} <= {node.threshold:.3f}")

            child_prefix = prefix + ("    " if is_last else "│   ")

            if node.left and depth < max_depth:
                print(f"{child_prefix}├── TRUE:")
                self._draw_simple_tree(node.left, child_prefix + "│   ", False, depth + 1, max_depth)

            if node.right and depth < max_depth:
                print(f"{child_prefix}└── FALSE:")
                self._draw_simple_tree(node.right, child_prefix + "    ", True, depth + 1, max_depth)


def preprocess_data(X, y):
    print("Preprocessing data...")
    X_processed = X.copy()
    label_encoders = {}

    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le

    X_processed = X_processed.fillna(X_processed.mean())
    y_processed = y.iloc[:, 0] if hasattr(y, 'iloc') else y

    if y_processed.dtype == 'object':
        target_encoder = LabelEncoder()
        y_processed = target_encoder.fit_transform(y_processed.astype(str))
        label_encoders['target'] = target_encoder

    print(f"Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    return X_processed.values, y_processed, label_encoders


def main():
    print("Optimized Decision Tree Classifier Demo")
    print("=" * 50)

    # Load data
    bank_marketing = fetch_ucirepo(id=222)
    X, y = bank_marketing.data.features, bank_marketing.data.targets
    X_processed, y_processed, encoders = preprocess_data(X, y)
    feature_names = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Compare criteria
    results = {}
    for criterion in ['entropy', 'gini']:
        print(f"\nTraining with {criterion.upper()} criterion")
        print("-" * 40)

        tree = OptimizedDecisionTree(
            criterion=criterion, max_depth=10, min_samples_split=15,
            min_samples_leaf=8, enable_pruning=True, ccp_alpha=0.005,
            feature_names=feature_names, max_features='sqrt'
        )

        tree.fit(X_train_fit, y_train_fit, X_val, y_val)

        train_accuracy = np.mean(tree.predict(X_train_fit) == y_train_fit)
        test_accuracy = np.mean(tree.predict(X_test) == y_test)

        results[criterion] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Generalization Gap: {train_accuracy - test_accuracy:.4f}")

        tree.print_tree(max_depth=3)
        tree.visualize_tree(max_depth=3)

        # Feature importance
        importance = tree.calculate_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 Features ({criterion}):")
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for criterion in ['entropy', 'gini']:
        result = results[criterion]
        print(f"{criterion.upper():>8}: Train={result['train_accuracy']:.4f}, "
              f"Test={result['test_accuracy']:.4f}")

    best_criterion = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest performing model: {best_criterion.upper()} "
          f"(Test Accuracy: {results[best_criterion]['test_accuracy']:.4f})")


if __name__ == "__main__":
    main()