# MAST30034 Project 1 README.md
- Name: Saurabh Jhanjee
- Student ID: 1171081

## README

**Research Goal:** My research goal is to investigate the relationship between airport flight traffic and high volume for high vehicle trip counts to and from airport tlc zones, and establish whether airport flight traffic is a driver of demand in these areas.

**Timeline:** The timeline for the research area is 2021 - 2022.

To run the pipeline, please visit the `scripts` directory ("notebooks" directory for .ipynb files) and run the files in order:
1. `download.py`: This downloads the raw test/train data into the `data/raw` directory.
2. `preprocess.py`: This preprocesses and aggregates the data into a usable format in the 'data/curated' directory
3. `graphs and stuff.ipynb`: This notebook is used to conduct analysis on the curated data to determine useful features, as well as a primitive version of the model.
4. `elasticnet.py`: The script is used to train and test the base ols elasticnet model, outputting metrics, grid search results and plots.
5. `elasticnet.py`: The script is used to train and test the interaction ols elasticnet model, outputting metrics, grid search results and plots.

