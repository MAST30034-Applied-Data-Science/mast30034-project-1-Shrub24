import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from itertools import chain

# Constants need to be modified to produce relevant train/test data
FHVHV_DATA_PATHS = {"train": 'data/raw/fhvhv', "test":'data/raw/fhvhv_TEST'}
FLIGHT_DATA_PATHS = {"train": "data/raw but i dont want this to be gitignored/APM_Report_Formatted_2021.csv", "test":"data/raw but i dont want this to be gitignored/APM_Report_Formatted_2022_TEST.csv"}
OUT_PATH = {"train": "data/curated/train/", "test": "data/curated/test/"}
OUT_SUFFIX = {"train": "_2021_train", "test": "_2022_test"}
AIRPORT_CODE_MAP = {132: "JFK", 1: "EWR", 138: "LGA"}
HOUR_INTERVALS = range(1, 6)

# Outer wrapper function for processing and joining a dataset
def init_dataset(flight_path, fhvhv_path, spark):
    flight_df = pd.read_csv(flight_path)

    sdf = spark.read.parquet(fhvhv_path)

    flight_code_mapper = F.create_map([F.lit(x) for x in chain(*AIRPORT_CODE_MAP.items())])
    flight_df["Date"] = pd.to_datetime(flight_df["Date"])

    sdf = sdf.filter(F.col("base_passenger_fare") > 0)
    return_dfs = {}
    for column, name in [["PULocationID", "Airport Pickups"], ["DOLocationID", "Airport Dropoffs"]]:
        airport_sdf = generate_airport_column(sdf, column, flight_code_mapper)
        airport_sdf, cols = preprocess_columns(airport_sdf)
        cols_flat =  [item for sublist in cols for item in sublist]
        airport_df = airport_sdf.toPandas().dropna()
        airport_df = airport_df.astype({"Hour": "int32", "Date": 'datetime64[ns]', **{item: "int32" if item.endswith("Hour") else 'datetime64[ns]' for item in cols_flat}})
        feature_list = get_feature_list(cols)
        return_dfs[name] = join_data(airport_df, flight_df, cols, feature_list)
    return return_dfs

# obtain a list of columns to keep
def get_feature_list(cols):
    feature_list = ["Day", "Departures For Metric Computation", "Arrivals For Metric Computation", "Hour", "Facility", "count", "Date"]
    for col_pair in cols:
        for item in col_pair:
            feature_list.append(f"Departures For Metric Computation{item[:2]}")
            feature_list.append(f"Arrivals For Metric Computation{item[:2]}")
    return feature_list

# performance an initial join as well as iterative joins to obtain temporally shifted columns
def join_data(taxis, flights, cols, feature_list):
    joined_df = taxis.merge(flights, left_on=("Hour", "Date", "Airport"), right_on=("Hour", "Date", "Facility"), how="inner")
    for col_pair in cols:
        joined_df = joined_df.merge(flights, left_on=(*col_pair, "Airport"), right_on=("Hour", "Date", "Facility"), how="inner", suffixes=(None, col_pair[0][:2]))
    joined_df = joined_df[feature_list]
    joined_df = joined_df.loc[:,~joined_df.columns.duplicated()]        
    return joined_df

# generate time columns to join on for temporally shifted data
def preprocess_temporal_columns(sdf):
    cols = []
    for interval in HOUR_INTERVALS:
        cols += [[f"+{interval} Hour", f"+{interval} Date"], [f"-{interval} Hour", f"-{interval} Date"]]
        sdf = sdf.withColumn(f"+{interval} Hour", F.hour(F.col("request_datetime") + F.expr(f'INTERVAL {interval} HOURS')))
        sdf = sdf.withColumn(f"+{interval} Date", F.to_date(F.col("request_datetime") + F.expr(f'INTERVAL {interval} HOURS')))

        sdf = sdf.withColumn(f"-{interval} Hour", F.hour(F.col("request_datetime") + F.expr(f'INTERVAL -{interval} HOURS')))
        sdf = sdf.withColumn(f"-{interval} Date", F.to_date(F.col("request_datetime") + F.expr(f'INTERVAL -{interval} HOURS')))


    sdf = sdf.withColumn("Day", F.dayofweek(F.col("request_datetime")))
    sdf = sdf.withColumn("Hour", F.hour(F.col("request_datetime")))
    sdf = sdf.withColumn("Date", F.to_date(F.col("request_datetime")))
    return sdf, cols

# preprocess columns to obtain aggregated count label
def preprocess_columns(sdf):
    sdf, cols = preprocess_temporal_columns(sdf)
    cols_flat =  [item for sublist in cols for item in sublist]
    return sdf.groupby(*("Day", "Hour", "Date", "Airport",  *cols_flat)).count(), cols


# Map location id column to an airport code column
def generate_airport_column(df, column, mapper):
    airport_df = df.filter(F.col(column).isin(list(AIRPORT_CODE_MAP.keys())))
    airport_df = airport_df.withColumn("Airport", mapper[F.col(column)])
    return airport_df


if __name__ == "__main__":
    # Create a spark session (which will run spark jobs)
    spark = (
        SparkSession.builder.appName("MAST30034 Ass1 Preprocessing")
        .config("spark.sql.repl.eagerEval.enabled", False) 
        .config("spark.sql.parquet.cacheMetadata", "true")
        .config("spark.sql.session.timeZone", "Etc/UTC")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    for dataset in ["train", "test"]:
        # process and output columns
        for name, df in init_dataset(FLIGHT_DATA_PATHS[dataset], FHVHV_DATA_PATHS[dataset], spark).items():
            df.to_parquet(OUT_PATH[dataset] + name + OUT_SUFFIX[dataset])