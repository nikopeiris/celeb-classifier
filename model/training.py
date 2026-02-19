# import cuml
# cuml.set_global_output_type('numpy')
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
# from cuml.svm import SVC, LinearSVC
# from cuml.ensemble import RandomForestClassifier
# from cuml.linear_model import LogisticRegression
# from cuml.preprocessing import StandardScaler
# from cuml.decomposition import PCA
# from cuml.pipeline import make_pipeline
# from cuml.model_selection import GridSearchCV, train_test_split
import joblib
import pandas as pd
import numpy as np
import json

x, y, celebrity_names = joblib.load('data_bundle.joblib')
x = x.astype('float32')
y = np.array(y).astype('int32')
print(f"Loaded data with {len(x)} samples and {len(celebrity_names)} classes.")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model_params = {
    'random_forest': {
        'model': RandomForestClassifier(n_bins = 64, n_streams = 4),
        "params": {
          "randomforestclassifier__n_estimators": [100, 150, 200, 500],
          "randomforestclassifier__split_criterion": ['gini', 'entropy'],
        },
        'n_jobs': 1,
    },
    'logistic_regression' : {
        'model': LogisticRegression(max_iter=20000, class_weight='balanced', penalty='elasticnet'),
        'params': {
            'logisticregression__C': [4, 5, 6, 8, 10],
            'logisticregression__tol': [1e-6, 1e-7],
            'logisticregression__fit_intercept': [True],
            'logisticregression__solver': ['qn'],
            'logisticregression__l1_ratio': [0.01, 0.05, 0.1, 0.2],
        },
        'n_jobs': 1,
    },
    'svm': {
        'model': SVC(probability=True, class_weight='balanced', cache_size=2000),
        'params' : {
            'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5],
            'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'svc__gamma': [1e-3, 1e-4, 1e-5, 'scale', 'auto'],
        },
        'n_jobs': 4, 
    },
    'linear_svc_search' : {
    'model': LinearSVC(class_weight='balanced', max_iter=20000, dual="auto"),
    'params': {
        'linearsvc__C': [0.01, 0.05, 0.1, 0.5, 1],
        'linearsvc__loss': ['hinge', 'squared_hinge'],
        'linearsvc__intercept_scaling': [1, 5, 10],
        'linearsvc__tol': [1e-3, 1e-4],
    },
      'n_jobs': 15
    }
}

scores = []
best_estimators = {}
best_model = None
best_score = 0

print("Starting model training and hyperparameter tuning...")
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(),mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False, n_jobs=mp['n_jobs'], pre_dispatch='2*n_jobs', verbose=3)
    clf.fit(x_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    print(f"Completed training for {algo} with best score: {clf.best_score_}")  
    if best_model is None or clf.best_score_ > best_score:
        best_model = algo
        best_score = clf.best_score_
    with open(f'training_data.json', 'r') as f:
        data = json.load(f)
    if data[algo]["best_score"] < clf.best_score_:
        data[algo] = {
            "best_score": clf.best_score_,
            "best_params": clf.best_params_
        }
        with open(f'training_data.json', 'w') as f:
            json.dump(data, f, indent=4)
        joblib.dump(best_estimators[algo], f'saved_model_{algo}.joblib') 
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)

print("Best algorithm is :", best_model)
best_model = best_estimators[best_model]
print("Test score: ", best_model.score(x_test, y_test))