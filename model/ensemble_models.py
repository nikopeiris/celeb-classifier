import json
import joblib
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.model_selection import train_test_split
# from cuml.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
import numpy as np
import cupy as cp


x, y, celebrity_names = joblib.load('data_bundle.joblib')
x = x.astype('float32')
y = np.array(y).astype('int32')
print(f"Loaded data with {len(x)} samples and {len(celebrity_names)} classes.")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr_cpu = joblib.load('saved_model_logistic_regression_sklearn.joblib')
svm_cpu = joblib.load('saved_model_svm.joblib')
linear_svc_cpu = joblib.load('saved_model_linear_svc_search.joblib')
stacked_clf = joblib.load('stacked_classifier.joblib')
voting_clf = joblib.load('voting_classifier.joblib')
estimators = [
    ('logistic_regression', lr_cpu),
    ('svm', svm_cpu),
]
print(f"Voting Classifier Accuracy: {voting_clf.score(x_test, y_test):.4f}")
print(f"Stacking Classifier Accuracy: {stacked_clf.score(x_test, y_test):.4f}")
print(f"Logistic Regression Accuracy: {lr_cpu.score(x_test, y_test):.4f}")
print(f"SVM Accuracy: {svm_cpu.score(x_test, y_test):.4f}")
print(f"Linear SVC Accuracy: {linear_svc_cpu.score(x_test, y_test):.4f}")