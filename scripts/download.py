from urllib.request import urlretrieve

#download the data
if __name__ == "__main__":
    URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_"
    YEAR = '2022'
    MONTHS = range(2, 5)
    output_relative_dir = 'data/raw/fhvhv_TEST'

    for month in MONTHS:
        month = str(month).zfill(2) 
        print(f"Begin month {month}")
        
        # generate url
        url = f'{URL_TEMPLATE}{YEAR}-{month}.parquet'
        # generate output location and filename
        output_dir = f"{output_relative_dir}/{YEAR}-{month}.parquet"
        # download
        urlretrieve(url, output_dir) 
        
        print(f"Completed month {month}")

    YEAR = '2021'
    MONTHS = range(1, 8)
    output_relative_dir = 'data/raw/fhvhv'

    for month in MONTHS:
        month = str(month).zfill(2) 
        print(f"Begin month {month}")
        
        # generate url
        url = f'{URL_TEMPLATE}{YEAR}-{month}.parquet'
        # generate output location and filename
        output_dir = f"{output_relative_dir}/{YEAR}-{month}.parquet"
        # download
        urlretrieve(url, output_dir) 
        
        print(f"Completed month {month}")