import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("OnlineNewsPopularity.csv")  # WE GOT A CLEAN DATA!


data.columns = data.columns.str.strip() #Cleaing the empty spaces


# DATA_SPLIT 20-80 way
try:
    # Define features and target
    feat = data.drop(['url', 'timedelta', 'shares'], axis=1)  # Drop non-feature columns
    target = data['shares']  # Target variable

  
    X_train, X_test, y_train, y_test = train_test_split(
        feat, 
        target, 
        test_size=0.2,  
        random_state=42  
    )


except KeyError as error:
    print(f"Error: {error}. The data set is too dirty, remove some of the bad features")