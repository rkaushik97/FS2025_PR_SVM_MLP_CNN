from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib

tuning_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.93)),
    ('svm', SVC(kernel='rbf'))
])

model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.93)),
    ('svm', SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced'))
])

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
            'svm__C': [8, 9, 10, 11, 12],
            'svm__gamma': [0.0008, 0.0009, 0.001, 0.0011, 0.0012]
        }
    grid = GridSearchCV(tuning_pipeline, param_grid, cv=3, verbose=2)
    grid.fit(X_tune, y_tune)
    print("Best parameters:", grid.best_params_)

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.show()

X_train, y_train = prepare_data("data/train.csv")
X_test, y_test = prepare_data("data/test.csv")
X_smaller, _, y_smaller, _ = train_test_split(X_train, y_train, train_size = 0.4, stratify=y_train, random_state=42)

#FIND BEST HYPERPARAMETERS:
#Èfind_best_params("data/train.csv")
#Best parameters: {'svm__C': 8, 'svm__gamma': 0.0012}
#Best parameters: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

# #TEST MODEL:
print("Training")
model.fit(X_train, y_train)
# #STORE MODEL:
#joblib.dump(model, "svm_model.joblib")
test_model(model, X_test=X_test, y_test=y_test)
# #SVM: Improve Accuracy: 0.9745 -> 0.9756