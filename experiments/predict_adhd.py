from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import re
from sklearn import preprocessing as pp
from datetime import datetime

from tsfresh.feature_selection.selection import select_features

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from utils import find_most_important_features
from utils import k_fold_model_evaluation
from utils import extract_features

_RANDOM_SEED = 0
_NUMBER_OF_FOLDS = 10
_NUMBER_OF_FEATURES = 50
_TEST_RATIO = .20

_PATH_TO_ACTIVITY_DATA = "C:\\Users\\Steven\\github\\hyperaktiv\\data\\activity_data"
_PATH_TO_FEATURES = "C:\\Users\\Steven\\github\\hyperaktiv\\data\\features.csv"
_PATH_TO_GT = "C:\\Users\\Steven\\github\\hyperaktiv\\data\\patient_data.csv"

TS_FEATURE_PARAMETERS = {
    "mean": None,
    "count_below": [{"t": 0}],
    "standard_deviation": None
}

_PARAMS_LORGREG = {
    "penalty": "l2", "C": 1.0, "class_weight": "balanced",
    "random_state": 0, "solver": "liblinear", "n_jobs": 1
}

_PARAMS_RFC = {
    "n_estimators": 1000,
    "max_features": "auto", "max_depth": None,
    "min_samples_split": 2, "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_leaf_nodes": None, "bootstrap": True,
    "oob_score": False, "n_jobs": -1, "random_state": 0,
    "class_weight": "balanced"
}

_PARAMS_XGB = {
    "n_estimators": 1000, "random_state": _RANDOM_SEED, "verbosity": 0,
    'objective':'binary:logistic', "learning_rate": .0001
}

_PARAMS_LIGHTGB = {
    "n_estimators": 1000, "random_state": _RANDOM_SEED, "verbosity": 0,
    "objective": "binary", "learning_rate": .0001
}

if __name__ == "__main__":

    # dataX = extract_features(_PATH_TO_ACTIVITY_DATA, TS_FEATURE_PARAMETERS).sort_values(by="ID")
    dataX = pd.read_csv(_PATH_TO_FEATURES, sep=";").sort_values(by="ID")
    dataY = pd.read_csv(_PATH_TO_GT, sep=";").sort_values(by="ID")

    dataX = dataX.fillna(0)

    # Remove JSON symbols from headers
    dataX = dataX.rename(columns = lambda x:re.sub('"', '', x))
    dataX = dataX.rename(columns = lambda x:re.sub(',', '', x))
    dataY = dataY.rename(columns = lambda x:re.sub('"', '', x))
    dataY = dataY.rename(columns = lambda x:re.sub(',', '', x))
        
    # Match X and Y data
    dataY = dataY[dataY["ID"].isin(dataX["ID"])]
    dataX = dataX[dataX["ID"].isin(dataY["ID"])]
    
    dataX = dataX.sort_values(by="ID")
    dataY = dataY.sort_values(by="ID")

    dataY = dataY.set_index("ID")
    dataX = dataX.set_index("ID")

    # Set ground truth
    dataY = dataY["ADHD"].copy()

    # Find relevant features using tsfresh
    # dataX = select_features(dataX, dataY)
    
    scaler = pp.StandardScaler(copy=True)
    dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        dataX,
        dataY,
        test_size=_TEST_RATIO,
        random_state=_RANDOM_SEED,
        stratify=dataY)

    # Find most relevant features using linear regression
    if _NUMBER_OF_FEATURES is not None:

        important_features = find_most_important_features(X_TRAIN, Y_TRAIN, _NUMBER_OF_FEATURES)

        X_TRAIN = X_TRAIN[ important_features ]
        X_TEST = X_TEST[ important_features ]
        
    metric_names = ["ACC", "PREC", "REC", "F1", "MCC"]

    stratified_train_eval, stratified_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "stratified", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    most_frequent_train_eval, most_frequent_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "most_frequent", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    prior_train_eval, prior_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "prior", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    minor_train_eval, minor_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 1 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    major_train_eval, major_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    random_train_eval, random_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "uniform", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)

    logreg_train_eval, logreg_test_eval = k_fold_model_evaluation(LogisticRegression, _PARAMS_LORGREG,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    rfc_train_eval, rfc_test_eval = k_fold_model_evaluation(RandomForestClassifier, _PARAMS_RFC,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    xgb_train_eval, xgb_test_eval = k_fold_model_evaluation(xgb.XGBClassifier, _PARAMS_XGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)
    gbm_train_eval, gbm_test_eval = k_fold_model_evaluation(lgb.LGBMClassifier, _PARAMS_LIGHTGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=_NUMBER_OF_FOLDS, random_state=_RANDOM_SEED)

    with open("adhd_prediction.txt", "w") as f:
        f.write("***CROSS-VALIDATION PERFORMANCE***\n")
        f.write("MODEL\t" + "\t ".join(metric_names) + "\n")
        f.write("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("RFC\t" + "\t ".join([ "%.2f" % (np.mean(rfc_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("XGB\t" + "\t ".join([ "%.2f" % (np.mean(xgb_train_eval[name])) for name in metric_names ]) + "\n")
        f.write("GBM\t" + "\t ".join([ "%.2f" % (np.mean(gbm_train_eval[name])) for name in metric_names ]) + "\n")

        f.write("\n")

        f.write("***TEST PERFORMANCE***\n")
        f.write("MODEL\t" + "\t ".join(metric_names) + "\n")
        f.write("Rand\t" + "\t ".join([ "%.2f" % (np.mean(random_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("Strat\t" + "\t ".join([ "%.2f" % (np.mean(stratified_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("Minor\t" + "\t ".join([ "%.2f" % (np.mean(minor_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("Major\t" + "\t ".join([ "%.2f" % (np.mean(major_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("Prior\t" + "\t ".join([ "%.2f" % (np.mean(prior_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("LogReg\t" + "\t ".join([ "%.2f" % (np.mean(logreg_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("RFC\t" + "\t ".join([ "%.2f" % (np.mean(rfc_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("XGB\t" + "\t ".join([ "%.2f" % (np.mean(xgb_test_eval[name])) for name in metric_names ]) + "\n")
        f.write("GBM\t" + "\t ".join([ "%.2f" % (np.mean(gbm_test_eval[name])) for name in metric_names ]) + "\n")