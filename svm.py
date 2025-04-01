from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X, y


def find_best_params(data_path, param_grid = None, train_size = 0.2):
    X_train, y_train = prepare_data(data_path= data_path)
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size = train_size, stratify=y_train, random_state=42)
    if param_grid == None:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.01, 0.001]
        }
    grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=2)
    grid.fit(X_tune, y_tune)
    print("Best parameters:", grid.best_params_)


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.show()

#find_best_params("data/train.csv")
#Best parameters: {'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'}

X_train, y_train = prepare_data("data/train.csv")
X_test, y_test = prepare_data("data/test.csv")


#X_smaller, _, y_smaller, _ = train_test_split(X_train, y_train, train_size = 0, stratify=y_train, random_state=42)
#model = SVC(C= 0.1, kernel = 'linear')
#model.fit(X_train, y_train)
#print("finished_training")
#joblib.dump(model, "svm_model.joblib")
# Accuracy: 0.9148, could do better