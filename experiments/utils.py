from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from copy import deepcopy

from collections import defaultdict
from datetime import datetime

import csv
import glob
import os

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from tsfresh import extract_features as ts_extract_features

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

def extract_features(data_dir, fc_parameters, day_index=None):

    def read_activity_file(filepath, patient_id):
        data = [ ]
        with open(filepath) as f:
            csv_reader = csv.reader(f, delimiter=";")
            next(csv_reader)
            for line in csv_reader:
                data.append([ datetime.strptime(line[0], "%m-%d-%Y %H:%M"), int(line[1].split(" ")[0])])
        data = pd.DataFrame(data, columns=["TIME", "ACC"])
        data["ID"] = patient_id
        data["TIME"] = pd.to_datetime(data['TIME'], format="%m-%d-%Y %H:%M")
        if day_index is not None:
            gb = data.groupby(pd.Grouper(key="TIME", freq='D'))
            day_data = gb.get_group((list(gb.groups)[day_index]))
            return day_data
        return data

    patient_records = []

    for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
        patient_id = int(os.path.splitext(os.path.basename(filepath))[0].split("_")[-1])
        data = read_activity_file(filepath, patient_id)
        patient_records.append(data)

    patient_records = pd.concat(patient_records)

    dataX = ts_extract_features(patient_records, default_fc_parameters=fc_parameters,
        column_id="ID", column_value="ACC", column_sort="TIME", n_jobs=0, show_warnings=False)
    dataX['ID'] = dataX.index
    return dataX

def find_most_important_features(x_data, y_data, n_features):
    model = LogisticRegression()
    model.fit(x_data, y_data)
    most_important_features = pd.DataFrame(data={ 'Attribute': x_data.columns, 'Importance': model.coef_[0] })
    most_important_features = most_important_features.sort_values(by='Importance', ascending=False)[:n_features]
    return list(most_important_features.Attribute.values)

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