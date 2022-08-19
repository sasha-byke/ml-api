import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# load data
df = pd.read_csv('data/pima-indians-diabetes.csv')

# split data into X and y
# X contains all columns but the last one, last column is our label y
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

filename = "models/model_diabetes.pkl"

# save
pickle.dump(model, open(filename, "wb"))