# Optimized Decision Tree Classifier

## Overview
This project implements an **Optimized Decision Tree Classifier** in Python, designed for efficient classification tasks with support for both **Gini** and **Entropy** criteria. The implementation includes advanced features such as cost complexity pruning, feature sampling, and visualization capabilities. The classifier is demonstrated using the [Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI Machine Learning Repository.

## Features
- **Custom Decision Tree Implementation**: Supports Gini index and Entropy as splitting criteria.
- **Feature Sampling**: Implements `sqrt` and other strategies for feature selection at each split.
- **Cost Complexity Pruning**: Reduces overfitting by pruning the tree based on a validation set and a cost complexity parameter (`ccp_alpha`).
- **Data Preprocessing**: Handles categorical variables using `LabelEncoder` and missing values by mean imputation.
- **Visualization**: Generates visual representations of the decision tree using Matplotlib, with a fallback to text-based tree printing.
- **Feature Importance**: Calculates and displays the importance of each feature based on information gain.
- **Configurable Parameters**:
  - `criterion`: 'gini' or 'entropy'
  - `max_depth`: Maximum depth of the tree
  - `min_samples_split`: Minimum samples required to split a node
  - `min_samples_leaf`: Minimum samples required at a leaf node
  - `enable_pruning`: Toggle for pruning
  - `ccp_alpha`: Cost complexity pruning parameter
  - `max_features`: Number of features to consider at each split (`sqrt`, `log2`, or a number/proportion)

## Requirements
To run the code, ensure you have the following Python packages installed:
- `ucimlrepo`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

Install them using:
```bash
pip install ucimlrepo scikit-learn numpy pandas matplotlib
