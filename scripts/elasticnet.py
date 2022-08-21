import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
import numpy as np
import altair as alt
import datetime


TRAIN_DATA_PATH = {"Dropoffs": "data/curated/train/Airport Dropoffs_2021_train", "Pickups": "data/curated/train/Airport Pickups_2021_train"}
TEST_DATA_PATH = {"Dropoffs": "data/curated/test/Airport Dropoffs_2022_test", "Pickups": "data/curated/test/Airport Pickups_2022_test"}
SELECTED_FEATURE_TYPES = {"Dropoffs": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"],
                             "Pickups": ["Departures For Metric Computation+", "Arrivals For Metric Computation-"]}
HOUR_INTERVALS = range(1, 6)
CATEGORICAL_FEATURES = ["Day", "Facility", ]
AIRPORT_CODE_MAP = {132: "JFK", 1: "EWR", 138: "LGA"}

plt.style.use('fivethirtyeight')

if __name__ == "__main__":
    for dataset in ["Dropoffs", "Pickups"]:
        # Load data, split into train and test columns, subset relevant feature columns as determined in report
        train = pd.read_parquet(TRAIN_DATA_PATH[dataset])
        y_train = train["count"]
        feature_list = ["Day", "Departures For Metric Computation", "Arrivals For Metric Computation", "Facility"]
        for interval in HOUR_INTERVALS:
            for feature_type in SELECTED_FEATURE_TYPES[dataset]:
                feature_list.append(feature_type + str(interval))

        # Create dummy variables for categorical features
        X_train = pd.get_dummies(train[feature_list], columns=CATEGORICAL_FEATURES)

        pipeline = make_pipeline(StandardScaler(), ElasticNet())

        # fit our pipeline with grid search on the below parameters
        params =  {"elasticnet__alpha": [.1, 1, 10],
                      "elasticnet__l1_ratio": [.1, .5, .7, .9, 1]}
        cv = KFold(shuffle=True)
        gs = GridSearchCV(pipeline, params, scoring="r2", cv=cv)
        gs.fit(X_train, y_train)


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

        # Save grid search results

        # pd.DataFrame(gs.cv_results_).to_csv("grid_search_elastic_" + dataset)

        test["datetime"] = pd.to_datetime(test["Date"].astype(str) + test["Hour"].astype(str), format='%Y-%m-%d%H')


        # Generate a plot of predicted vs test labels by airport

        plt.rcParams["figure.figsize"] = (20,10)
        # colours = ["orange", "lightblue", "purple"]

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








