from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd

import re
import argparse

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_selection.selection import select_features

from utils import k_fold_model_evaluation

parser = argparse.ArgumentParser(description="Script that runs the baseline experiments.")

parser.add_argument("-x", "--x-file-path", type=str, required=True)
parser.add_argument("-y", "--y-file-path", type=str, required=True)
parser.add_argument("-o", "--output-file-path", type=str, default="results.txt")
parser.add_argument("-k", "--k-folds", type=int, default=10)
parser.add_argument("-t", "--test-ratio", type=float, default=0.20)
parser.add_argument("-s", "--random-seed", type=int, default=0)

if __name__ == "__main__":
    
    args = parser.parse_args()

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
        "random_state": args.random_seed, "verbosity": 0,
        'objective':'binary:logistic'
    }

    _PARAMS_LIGHTGB = {
        "random_state": args.random_seed, "verbosity": 0,
        "objective": "binary"
    }

    dataX = pd.read_csv(args.x_file_path, sep=";").sort_values(by="ID")
    dataY = pd.read_csv(args.y_file_path, sep=";").sort_values(by="ID")

    dataX = dataX.fillna(0)

    # Remove JSON symbols from headers
    dataX = dataX.rename(columns = lambda x:re.sub('"', '', x))
    dataX = dataX.rename(columns = lambda x:re.sub(',', '', x))
    dataY = dataY.rename(columns = lambda x:re.sub('"', '', x))
    dataY = dataY.rename(columns = lambda x:re.sub(',', '', x))
        
    # Match X and Y data
    dataY = dataY[dataY["ID"].isin(dataX["ID"])]
    dataX = dataX[dataX["ID"].isin(dataY["ID"])]

    dataY = dataY.set_index("ID")
    dataX = dataX.set_index("ID")

    # Set ground truth
    dataY = dataY["ADHD"].copy()

    # Find relevant features using tsfresh
    dataX = select_features(dataX, dataY)
    
    scaler = StandardScaler(copy=True)
    dataX.loc[:, dataX.columns] = scaler.fit_transform(dataX[dataX.columns])
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        dataX,
        dataY,
        test_size=args.test_ratio,
        random_state=args.random_seed,
        stratify=dataY)
        
    metric_names = ["ACC", "PREC", "REC", "F1", "MCC"]

    stratified_train_eval, stratified_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "stratified", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    most_frequent_train_eval, most_frequent_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "most_frequent", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    prior_train_eval, prior_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "prior", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    minor_train_eval, minor_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 1 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    major_train_eval, major_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "constant", "random_state": 0, "constant": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    random_train_eval, random_test_eval = k_fold_model_evaluation(DummyClassifier, { "strategy": "uniform", "random_state": 0 },
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)

    logreg_train_eval, logreg_test_eval = k_fold_model_evaluation(LogisticRegression, _PARAMS_LORGREG,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    rfc_train_eval, rfc_test_eval = k_fold_model_evaluation(RandomForestClassifier, _PARAMS_RFC,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    xgb_train_eval, xgb_test_eval = k_fold_model_evaluation(XGBClassifier, _PARAMS_XGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)
    gbm_train_eval, gbm_test_eval = k_fold_model_evaluation(LGBMClassifier, _PARAMS_LIGHTGB,
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, n_splits=args.k_folds, random_state=args.random_seed)

    with open(args.output_file_path, "w") as f:
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