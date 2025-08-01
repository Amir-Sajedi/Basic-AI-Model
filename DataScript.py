from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from collections import Counter

# Fetch the dataset
online_news_popularity = fetch_ucirepo(id=332)
data = online_news_popularity.data.features
target = online_news_popularity.data.targets

# Combine features and target into a single DataFrame for preprocessing
data = pd.DataFrame(data, columns=online_news_popularity.data.feature_names)
data['shares'] = target

# Clean column names
data.columns = data.columns.str.strip()

# Print available columns for debugging
print("Available columns in the DataFrame:", data.columns.tolist())

# Convert regression to classification problem (popular vs not popular)
# Using median as threshold
median_shares = data['shares'].median()
data['popularity'] = (data['shares'] > median_shares).astype(int)

# Quartiling the n_tokens_content
data['n_tokens_content_quartile'] = pd.qcut(data['n_tokens_content'],
                                            q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                            duplicates='drop')

# Features to drop (from previous analysis)
features_to_drop = [
    'url', 'timedelta', 'n_tokens_content',
    'kw_min_min', 'kw_max_min', 'kw_avg_min',
    'abs_title_subjectivity', 'abs_title_sentiment_polarity',
    'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04',
    'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
    'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',
    'weekday_is_sunday'
]

# Filter out columns that don't exist in the DataFrame
features_to_drop = [col for col in features_to_drop if col in data.columns]
print("Columns to drop:", features_to_drop)

try:
    # Drop features and target
    feat = data.drop(features_to_drop + ['shares', 'popularity'], axis=1)
    target = data['popularity'].values  # Use binary classification target

    # Encode categorical feature n_tokens_content_quartile
    le = LabelEncoder()
    feat['n_tokens_content_quartile'] = le.fit_transform(feat['n_tokens_content_quartile'])

    # Convert features to numpy array
    X = feat.values
    y = target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split training data for pruning validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )


    # Enhanced Decision Tree Node
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None,
                     value=None, samples=None, impurity=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.samples = samples
            self.impurity = impurity


    # Enhanced Decision Tree Classifier with multiple splitting criteria
    class DecisionTreeClassifier:
        def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5,
                     criterion='information_gain', pruning=True):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.criterion = criterion
            self.pruning = pruning
            self.root = None
            self.feature_names = feat.columns

        def entropy(self, y):
            """Calculate entropy of a dataset"""
            if len(y) == 0:
                return 0
            proportions = np.bincount(y) / len(y)
            proportions = proportions[proportions > 0]  # Remove zeros
            return -np.sum(proportions * np.log2(proportions))

        def gini_index(self, y):
            """Calculate Gini index of a dataset"""
            if len(y) == 0:
                return 0
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)

        def information_gain(self, y, y_left, y_right):
            """Calculate information gain"""
            parent_entropy = self.entropy(y)
            n = len(y)
            n_left, n_right = len(y_left), len(y_right)

            if n_left == 0 or n_right == 0:
                return 0

            child_entropy = (n_left / n) * self.entropy(y_left) + (n_right / n) * self.entropy(y_right)
            return parent_entropy - child_entropy

        def gini_gain(self, y, y_left, y_right):
            """Calculate Gini gain"""
            parent_gini = self.gini_index(y)
            n = len(y)
            n_left, n_right = len(y_left), len(y_right)

            if n_left == 0 or n_right == 0:
                return 0

            child_gini = (n_left / n) * self.gini_index(y_left) + (n_right / n) * self.gini_index(y_right)
            return parent_gini - child_gini

        def best_split(self, X, y):
            """Find the best split using the specified criterion"""
            best_feature, best_threshold, best_gain = None, None, -float('inf')
            n_samples, n_features = X.shape

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_indices = X[:, feature] <= threshold
                    right_indices = X[:, feature] > threshold

                    if (sum(left_indices) < self.min_samples_leaf or
                            sum(right_indices) < self.min_samples_leaf):
                        continue

                    y_left, y_right = y[left_indices], y[right_indices]

                    if self.criterion == 'information_gain':
                        gain = self.information_gain(y, y_left, y_right)
                    elif self.criterion == 'gini':
                        gain = self.gini_gain(y, y_left, y_right)
                    else:
                        gain = self.information_gain(y, y_left, y_right)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

            return best_feature, best_threshold, best_gain

        def fit(self, X, y, X_val=None, y_val=None):
            """Fit the decision tree"""
            self.root = self._grow_tree(X, y, depth=0)

            if self.pruning and X_val is not None and y_val is not None:
                self._prune_tree(X_val, y_val)

        def _grow_tree(self, X, y, depth):
            """Recursive algorithm to grow the tree"""
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))

            # Calculate impurity based on criterion
            if self.criterion == 'gini':
                impurity = self.gini_index(y)
            else:
                impurity = self.entropy(y)

            # Stopping criteria
            if (depth >= self.max_depth or
                    n_samples < self.min_samples_split or
                    n_classes == 1 or
                    impurity < 1e-10):
                # Create leaf node
                most_common_class = Counter(y).most_common(1)[0][0]
                return Node(value=most_common_class, samples=n_samples, impurity=impurity)

            # Find best split
            best_feature, best_threshold, best_gain = self.best_split(X, y)

            if best_feature is None or best_gain <= 0:
                most_common_class = Counter(y).most_common(1)[0][0]
                return Node(value=most_common_class, samples=n_samples, impurity=impurity)

            # Split the data
            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold

            # Recursively build left and right subtrees
            left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
            right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

            return Node(best_feature, best_threshold, left, right,
                        samples=n_samples, impurity=impurity)

        def _prune_tree(self, X_val, y_val):
            """Post-pruning using validation set"""

            def prune_node(node):
                if node.left is None and node.right is None:
                    return node

                # Recursively prune children
                if node.left:
                    node.left = prune_node(node.left)
                if node.right:
                    node.right = prune_node(node.right)

                # Calculate accuracy before and after pruning
                accuracy_before = self._calculate_accuracy(X_val, y_val)

                # Temporarily prune this node
                original_left, original_right = node.left, node.right
                original_feature, original_threshold = node.feature, node.threshold

                # Convert to leaf node
                node.left, node.right = None, None
                node.feature, node.threshold = None, None
                # Use majority class from validation predictions
                val_predictions = self.predict(X_val)
                node.value = Counter(val_predictions).most_common(1)[0][0]

                accuracy_after = self._calculate_accuracy(X_val, y_val)

                # Keep pruning if it improves or maintains accuracy
                if accuracy_after >= accuracy_before:
                    return node
                else:
                    # Restore original node
                    node.left, node.right = original_left, original_right
                    node.feature, node.threshold = original_feature, original_threshold
                    node.value = None
                    return node

            self.root = prune_node(self.root)

        def _calculate_accuracy(self, X, y):
            """Calculate accuracy on given dataset"""
            predictions = self.predict(X)
            return np.mean(predictions == y)

        def predict(self, X):
            """Make predictions for input data"""
            return np.array([self._predict_single(x, self.root) for x in X])

        def _predict_single(self, x, node):
            """Predict single sample"""
            if node.value is not None:
                return node.value

            if x[node.feature] <= node.threshold:
                return self._predict_single(x, node.left)
            return self._predict_single(x, node.right)

        def print_tree(self, node=None, depth=0):
            """Print tree structure"""
            if node is None:
                node = self.root

            if node.value is not None:
                print("  " * depth + f"Predict: {node.value} (samples: {node.samples}, impurity: {node.impurity:.3f})")
                return

            feature_name = self.feature_names[node.feature]
            print(
                "  " * depth + f"{feature_name} <= {node.threshold:.3f} (samples: {node.samples}, impurity: {node.impurity:.3f})")

            if node.left:
                self.print_tree(node.left, depth + 1)
            if node.right:
                self.print_tree(node.right, depth + 1)


    # Train and evaluate different criteria
    criteria = ['information_gain', 'gini']

    for criterion in criteria:
        print(f"\n=== Training with {criterion.upper()} criterion ===")

        # Train with pruning
        tree_pruned = DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                                             min_samples_leaf=10, criterion=criterion,
                                             pruning=True)
        tree_pruned.fit(X_train_fit, y_train_fit, X_val, y_val)

        # Train without pruning
        tree_unpruned = DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                                               min_samples_leaf=10, criterion=criterion,
                                               pruning=False)
        tree_unpruned.fit(X_train_fit, y_train_fit)

        # Evaluate both trees
        y_pred_pruned = tree_pruned.predict(X_test)
        y_pred_unpruned = tree_unpruned.predict(X_test)

        accuracy_pruned = np.mean(y_pred_pruned == y_test)
        accuracy_unpruned = np.mean(y_pred_unpruned == y_test)

        print(f"Accuracy (with pruning): {accuracy_pruned:.4f}")
        print(f"Accuracy (without pruning): {accuracy_unpruned:.4f}")

        # Print tree structure (only for information gain to save space)
        if criterion == 'information_gain':
            print("\nTree structure (pruned):")
            tree_pruned.print_tree()


    # Visualization function (updated for classification)
    def plot_tree(node, tree, depth=0, pos_x=0, pos_y=0, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))

        if node.value is not None:
            label = f"Class: {node.value}\nSamples: {node.samples}"
            box = FancyBboxPatch((pos_x - 0.5, pos_y - 0.15), 1.0, 0.3,
                                 boxstyle="round,pad=0.1", edgecolor="black",
                                 facecolor="lightgreen", mutation_scale=0.2)
            ax.add_patch(box)
            ax.text(pos_x, pos_y, label, ha="center", va="center", fontsize=8)
            return ax

        feature_name = tree.feature_names[node.feature]
        label = f"{feature_name} <= {node.threshold:.2f}\nSamples: {node.samples}\nImpurity: {node.impurity:.3f}"
        box = FancyBboxPatch((pos_x - 0.6, pos_y - 0.2), 1.2, 0.4,
                             boxstyle="round,pad=0.1", edgecolor="black",
                             facecolor="lightblue", mutation_scale=0.2)
        ax.add_patch(box)
        ax.text(pos_x, pos_y, label, ha="center", va="center", fontsize=7)

        if node.left:
            left_x, left_y = pos_x - 1.5 / (2 ** (depth * 0.5)), pos_y - 1.5
            ax.plot([pos_x, left_x], [pos_y - 0.2, left_y + 0.2], 'k-')
            ax = plot_tree(node.left, tree, depth + 1, left_x, left_y, ax)
        if node.right:
            right_x, right_y = pos_x + 1.5 / (2 ** (depth * 0.5)), pos_y - 1.5
            ax.plot([pos_x, right_x], [pos_y - 0.2, right_y + 0.2], 'k-')
            ax = plot_tree(node.right, tree, depth + 1, right_x, right_y, ax)

        return ax


    # Plot the tree (using information gain with pruning)
    final_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=20,
                                        min_samples_leaf=10, criterion='information_gain',
                                        pruning=True)
    final_tree.fit(X_train_fit, y_train_fit, X_val, y_val)

    ax = plot_tree(final_tree.root, final_tree)
    plt.title("Decision Tree for Online News Popularity\n(Information Gain with Pruning)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Feature importance analysis
    def calculate_feature_importance(tree):
        importances = np.zeros(len(tree.feature_names))

        def traverse(node):
            if node.value is not None:
                return

            # Calculate weighted impurity decrease
            left_samples = node.left.samples if node.left else 0
            right_samples = node.right.samples if node.right else 0
            total_samples = node.samples

            if total_samples > 0:
                left_impurity = node.left.impurity if node.left else 0
                right_impurity = node.right.impurity if node.right else 0

                weighted_impurity = (left_samples * left_impurity +
                                     right_samples * right_impurity) / total_samples
                importance = node.impurity - weighted_impurity
                importances[node.feature] += importance * (total_samples / tree.root.samples)

            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)

        traverse(tree.root)
        return importances / np.sum(importances) if np.sum(importances) > 0 else importances


    # Calculate and display feature importance
    importance_scores = calculate_feature_importance(final_tree)
    feature_importance_df = pd.DataFrame({
        'feature': final_tree.feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance_df.head(10))

except KeyError as error:
    print(f"Error: {error}. Please check the column names in the dataset.")