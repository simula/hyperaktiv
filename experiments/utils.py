from collections import defaultdict
from copy import deepcopy

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def calculate_classification_metrics(y_preds, y_trues):
    y_preds = np.round(y_preds)
    y_trues = np.round(y_trues)
    return {
        "ACC": metrics.accuracy_score(y_trues, y_preds),
        "PREC": metrics.precision_score(y_trues, y_preds, average="binary"),
        "REC": metrics.recall_score(y_trues, y_preds, average="binary"),
        "F1": metrics.f1_score(y_trues, y_preds, average="binary"),
        "MCC": metrics.matthews_corrcoef(y_trues, y_preds)
    }

def k_fold_model_evaluation( model_func, model_parameters, x_train, y_train, x_test, y_test,
     fit_parameters={}, n_splits=10, shuffle=True, random_state=0 ):
     
    valid_eval = defaultdict(list)
    test_eval = defaultdict(list)

    k_fold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state)

    models = [ ]

    for train_index, fold_index in k_fold.split(np.zeros(len(x_train)), y_train.ravel()):
        
        model = model_func(**model_parameters)
        
        x_fold_train, x_fold_test = x_train.iloc[train_index, :], x_train.iloc[fold_index, :]
        y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[fold_index]
        
        try:
            model = model.fit( x_fold_train, y_fold_train, eval_set=[(x_fold_test, y_fold_test)], eval_metric="aucpr", **fit_parameters )
        except:
            model = model.fit( x_fold_train, y_fold_train, **fit_parameters )

        models.append( deepcopy(model) )

        y_pred = model.predict_proba( x_fold_test )[:, 1]
        
        for metric, value in calculate_classification_metrics(list(y_pred), list(y_fold_test)).items():
            valid_eval[metric].append(value)

    for model in models:
        y_test_preds = model.predict_proba( x_test )[:, 1]
        for metric, value in calculate_classification_metrics(y_test_preds, y_test).items():
            test_eval[metric].append(value)

    return valid_eval, test_eval