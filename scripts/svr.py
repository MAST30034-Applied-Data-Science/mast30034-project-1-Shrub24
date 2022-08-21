import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
import numpy as np
import altair as alt
from sklearn.exceptions import ConvergenceWarning
import warnings

TRAIN_DATA_PATH = {"Dropoffs": "data/curated/train/Airport Dropoffs_2021_train", "Pickups": "data/curated/train/Airport Pickups_2021_train"}
TEST_DATA_PATH = {"Dropoffs": "data/curated/test/Airport Dropoffs_2022_test", "Pickups": "data/curated/test/Airport Pickups_2022_test"}
SELECTED_FEATURE_TYPES = {"Dropoffs": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"],
                             "Pickups": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"]}
HOUR_INTERVALS = range(1, 6)
CATEGORICAL_FEATURES = ["Day", "Facility", ]

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
def fit_gs(X_train, y_train):
        pipeline = make_pipeline(StandardScaler(), SVR())

        params = {"svr__epsilon": [1, 10],
                'svr__C':[100, 1000]}

        cv = KFold(shuffle=True)
        gs = GridSearchCV(pipeline, params, scoring="r2", cv=cv)

        gs.fit(X_train, y_train)
        return gs

if __name__ == "__main__":
    for dataset in ["Dropoffs", "Pickups"]:
        train = pd.read_parquet(TRAIN_DATA_PATH[dataset])
        y_train = train["count"]
        feature_list = ["Day", "Departures For Metric Computation", "Arrivals For Metric Computation", "Facility"]
        for interval in HOUR_INTERVALS:
            for feature_type in SELECTED_FEATURE_TYPES[dataset]:
                feature_list.append(feature_type + str(interval))

        X_train = pd.get_dummies(train[feature_list], columns=CATEGORICAL_FEATURES)

        gs = fit_gs(X_train, y_train)

        print("-" * 10)
        print(f"Airport {dataset}:")
        print(f"Training Best R2 Score {gs.best_score_}:")
        print(f"Training Grid Search Best Params {gs.best_params_}:")
        print("-" * 10)

        test = pd.read_parquet(TEST_DATA_PATH[dataset])
        y_test = test["count"]
        X_test = pd.get_dummies(test[feature_list], columns=CATEGORICAL_FEATURES)
        pred = gs.best_estimator_.predict(X_test)
        print(f"Test R2 Score {r2_score(y_test, pred)}:")
        print(f"Test Mean Absolute Error {mean_absolute_error(y_test, pred)}:")
        print("-" * 10)




