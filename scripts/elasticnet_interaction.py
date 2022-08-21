import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
import numpy as np
import altair as alt
from sklearn.exceptions import ConvergenceWarning
import warnings
import datetime

TRAIN_DATA_PATH = {"Dropoffs": "data/curated/train/Airport Dropoffs_2021_train", "Pickups": "data/curated/train/Airport Pickups_2021_train"}
TEST_DATA_PATH = {"Dropoffs": "data/curated/test/Airport Dropoffs_2022_test", "Pickups": "data/curated/test/Airport Pickups_2022_test"}
SELECTED_FEATURE_TYPES = {"Dropoffs": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"],
                             "Pickups": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"]}
HOUR_INTERVALS = range(1, 6)
CATEGORICAL_FEATURES = ["Day", "Facility", ]
AIRPORT_CODE_MAP = {132: "JFK", 1: "EWR", 138: "LGA"}

plt.style.use('fivethirtyeight')


# Fit our grid search hyperparameters and our fully trained resultant model
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
def fit_gs(X_train, y_train):
        pipeline = make_pipeline(PolynomialFeatures(interaction_only=True, include_bias = False), StandardScaler(), ElasticNet())

        params =  {"elasticnet__alpha": [.1, 1, 10],
                      "elasticnet__l1_ratio": [.1, .5, .7, .9, 1]}
        cv = KFold(shuffle=True)
        gs = GridSearchCV(pipeline, params, scoring="r2", cv=cv)

        gs.fit(X_train, y_train)
        return gs

if __name__ == "__main__":
    # Load dataset and obtain X and y dfs with correct features
    for dataset in ["Dropoffs", "Pickups"]:
        train = pd.read_parquet(TRAIN_DATA_PATH[dataset])
        y_train = train["count"]
        feature_list = ["Day", "Departures For Metric Computation", "Arrivals For Metric Computation", "Facility"]
        for interval in HOUR_INTERVALS:
            for feature_type in SELECTED_FEATURE_TYPES[dataset]:
                feature_list.append(feature_type + str(interval))
        # Create dummy variables for categorical features
        X_train = pd.get_dummies(train[feature_list], columns=CATEGORICAL_FEATURES)

        gs = fit_gs(X_train, y_train)


        # Training results
        print("-" * 10)
        print(f"Airport {dataset}:")
        print(f"Training Best R2 Score {gs.best_score_}:")
        print(f"Training Grid Search Best Params {gs.best_params_}:")
        print("-" * 10)

        # Testing results and evaluation metrics
        test = pd.read_parquet(TEST_DATA_PATH[dataset])
        y_test = test["count"]
        X_test = pd.get_dummies(test[feature_list], columns=CATEGORICAL_FEATURES)
        pred = gs.best_estimator_.predict(X_test)
        print(f"Test R2 Score {r2_score(y_test, pred)}:")
        print(f"Test Mean Absolute Error {mean_absolute_error(y_test, pred)}:")
        print("-" * 10)

        coefs = gs.best_estimator_.get_params()["elasticnet"].coef_
        print(f"Used {sum((coef != 0 for coef in coefs))}/{len(coefs)} features")

        # Save grid search results
        pd.DataFrame(gs.cv_results_).to_csv("plots/grid_search_interaction_elastic_" + dataset)

        # Generate a plot of predicted vs test labels by airport
        test["datetime"] = pd.to_datetime(test["Date"].astype(str) + test["Hour"].astype(str), format='%Y-%m-%d%H')

        plt.rcParams["figure.figsize"] = (20,10)

        test = pd.get_dummies(test, columns=CATEGORICAL_FEATURES)
        fig, ax = plt.subplots()
        legend = []

        for i, airport in enumerate(AIRPORT_CODE_MAP.values()):
            curr = test[test[f"Facility_{airport}"] == 1].sort_values("datetime")
            preds = gs.best_estimator_.predict(curr[gs.feature_names_in_])
            p = plt.scatter(curr["datetime"], curr["count"])
            plt.plot(curr["datetime"], preds, color = p.get_facecolor()[0] )
            legend.extend([f"{airport} Actual", f"{airport} Predicted"])
        plt.title(f"{dataset} Test Labels vs Predicted by Facility")
        plt.xlim((datetime.date(2022, 2, 1), datetime.date(2022, 3, 1)))
        plt.legend(legend, loc="upper right")
        plt.xlabel("Datetime")
        plt.ylabel(f"{dataset} Count")


        plt.show()













