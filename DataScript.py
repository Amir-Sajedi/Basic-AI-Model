import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("OnlineNewsPopularity.csv")

data.columns = data.columns.str.strip()

# Quartiling the data
# We add `duplicates='drop'` to handle cases where bin edges are not unique.
data['n_tokens_content_quartile'] = pd.qcut(data['n_tokens_content'],
                                            q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                            duplicates='drop')

# DATA_SPLIT 20-80 way (hold out validation)
try:
    # Define features and target
    # 2. UPDATE THE FEATURE SET
    # We add the original 'n_tokens_content' to the drop list to avoid data redundancy
    feat = data.drop(['url', 'timedelta', 'shares', 'n_tokens_content'], axis=1)
    target = data['shares']

    X_train, X_test, y_train, y_test = train_test_split(
        feat,
        target,
        test_size=0.2,
        random_state=42
    )

    print("Data successfully binned and split!")
    # You can check the new feature in the training set



except KeyError as error:
    print(f"Error: {error}. The data set is too dirty, remove some of the bad features")