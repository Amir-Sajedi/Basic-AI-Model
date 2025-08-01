from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

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
    feat = data.drop(features_to_drop + ['shares'], axis=1)
    target = data['shares'].values  # Convert to numpy array

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


    # Custom Decision Tree Regressor
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value


    class DecisionTreeRegressor:
        def __init__(self, max_depth=3, min_samples_split=2):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.root = None
            self.feature_names = feat.columns

        def fit(self, X, y):
            self.root = self._grow_tree(X, y, depth=0)

        def _grow_tree(self, X, y, depth):
            n_samples, n_features = X.shape
            if (depth >= self.max_depth or
                    n_samples < self.min_samples_split or
                    np.var(y) < 1e-10):
                return Node(value=np.mean(y))

            best_feature, best_threshold, best_variance_reduction = None, None, -float('inf')
            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_indices = X[:, feature] <= threshold
                    right_indices = X[:, feature] > threshold
                    if sum(left_indices) < 1 or sum(right_indices) < 1:
                        continue
                    variance_reduction = self._variance_reduction(y, y[left_indices], y[right_indices])
                    if variance_reduction > best_variance_reduction:
                        best_variance_reduction = variance_reduction
                        best_feature = feature
                        best_threshold = threshold

            if best_feature is None:
                return Node(value=np.mean(y))

            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold
            left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
            right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
            return Node(best_feature, best_threshold, left, right)

        def _variance_reduction(self, y, y_left, y_right):
            parent_var = np.var(y)
            n = len(y)
            n_left, n_right = len(y_left), len(y_right)
            if n_left == 0 or n_right == 0:
                return 0
            child_var = (n_left / n) * np.var(y_left) + (n_right / n) * np.var(y_right)
            return parent_var - child_var

        def predict(self, X):
            return np.array([self._predict_single(x, self.root) for x in X])

        def _predict_single(self, x, node):
            if node.value is not None:
                return node.value
            if x[node.feature] <= node.threshold:
                return self._predict_single(x, node.left)
            return self._predict_single(x, node.right)


    # Train and evaluate
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    # Compute MSE
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.2f}")


    # Visualization
    def plot_tree(node, depth=0, pos_x=0, pos_y=0, parent_x=0, parent_y=0, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        if node.value is not None:
            label = f"Value: {node.value:.2f}"
            box = FancyBboxPatch((pos_x - 0.4, pos_y - 0.1), 0.8, 0.2, boxstyle="round,pad=0.1",
                                 edgecolor="black", facecolor="lightgreen", mutation_scale=0.2)
            ax.add_patch(box)
            ax.text(pos_x, pos_y, label, ha="center", va="center")
            return ax

        feature_name = tree.feature_names[node.feature]
        label = f"{feature_name} <= {node.threshold:.2f}"
        box = FancyBboxPatch((pos_x - 0.4, pos_y - 0.1), 0.8, 0.2, boxstyle="round,pad=0.1",
                             edgecolor="black", facecolor="lightblue", mutation_scale=0.2)
        ax.add_patch(box)
        ax.text(pos_x, pos_y, label, ha="center", va="center")

        if node.left:
            left_x, left_y = pos_x - 1 / (2 ** depth), pos_y - 1
            ax.plot([pos_x, left_x], [pos_y - 0.1, left_y + 0.1], 'k-')
            ax = plot_tree(node.left, depth + 1, left_x, left_y, pos_x, pos_y, ax)
        if node.right:
            right_x, right_y = pos_x + 1 / (2 ** depth), pos_y - 1
            ax.plot([pos_x, right_x], [pos_y - 0.1, right_y + 0.1], 'k-')
            ax = plot_tree(node.right, depth + 1, right_x, right_y, pos_x, pos_y, ax)

        return ax


    # Plot the tree
    ax = plot_tree(tree.root)
    plt.title("Decision Tree for Online News Popularity")
    plt.axis('off')
    plt.show()

except KeyError as error:
    print(f"Error: {error}. Please check the column names in the dataset.")