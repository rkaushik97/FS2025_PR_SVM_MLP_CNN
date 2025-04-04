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

scaler = StandardScaler()
pca = PCA(n_components=0.95)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Or a fixed number like 50 or 100
    ('svm', SVC(kernel='rbf'))  # You can try 'linear' or tune kernel
])


def prepare_data(data_path):
    df = pd.read_csv(data_path)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X, y

def find_best_params(data_path, param_grid = None, train_size = 0.2):
    X_train, y_train = prepare_data(data_path= data_path)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)

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
#Best parameters: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

X_train, y_train = prepare_data("data/train.csv")
X_test, y_test = prepare_data("data/test.csv")
svm_optim = SVC(C= 10, gamma= 0.001, kernel= 'rbf')


X_smaller, _, y_smaller, _ = train_test_split(X_train, y_train, train_size = 0.2, stratify=y_train, random_state=42)



X_smaller = pca.fit_transform(scaler.fit_transform(X_smaller))
X_test = pca.transform(scaler.transform(X_test))
model = SVC(C= 10, kernel = 'rbf', gamma= 0.001)

print("_training")
model.fit(X_smaller, y_smaller)
print("finished_training")
joblib.dump(model, "svm_model.joblib")
test_model(model, X_test=X_test, y_test=y_test)
#SVM: Improve Accuracy: 0.9148 -> 0.9553