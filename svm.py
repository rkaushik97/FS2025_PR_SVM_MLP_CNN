from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
import pandas as pd

df = pd.read_csv("data/train.csv")
print(df.shape)
X_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.001]
}

grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=2)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)