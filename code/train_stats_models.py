##########################################################
# Overview
#
# A script to train four classic stats models.
# Use nixtlaEnv conda environment.
#
# Output:
#   A dataframe containing predictions per model per dataset.
##########################################################

import os
from functools import partial

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape
from plotnine import (
    aes,
    coord_flip,
    geom_boxplot,
    ggplot,
    ggsave,
    labs,
    scale_y_continuous,
)
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoTBATS,
    AutoTheta,
    SeasonalNaive,
)

os.getcwd()

progress_folder_path = os.path.join(os.getcwd(), "data/progress_data")
if not os.path.exists(progress_folder_path):
    os.makedirs(progress_folder_path)


##########################
# Load data
##########################
def load_data(dataset):

    trainFile = os.path.join(os.getcwd(), f"data/{dataset}_train.csv")
    testFile = os.path.join(os.getcwd(), f"data/{dataset}_test.csv")

    train = pd.read_csv(trainFile)
    train["ds"] = pd.to_datetime(train["ds"]).dt.normalize()
    train = train.sort_values(by=["unique_id", "ds"])

    test = pd.read_csv(testFile)
    test["ds"] = pd.to_datetime(test["ds"]).dt.normalize()
    test = test.sort_values(by=["unique_id", "ds"])

    return train, test


##########################
# Train models
##########################
def predict_stats_models(train, test, dataset):
    if dataset == "other":
        sp = 8
        freq = "D"
        h = 8
    elif dataset == "monthly":
        sp = 12
        freq = "ME"
        h = 18
    elif dataset == "quarterly":
        sp = 4
        freq = "QE"
        h = 8
    elif dataset == "yearly":
        sp = 1
        freq = "YE"
        h = 6

    models = [
        SeasonalNaive(season_length=sp),
        AutoARIMA(season_length=sp),
        AutoETS(season_length=sp),
        AutoCES(season_length=sp),
        AutoTheta(season_length=sp),
        AutoMFLES(test_size=h, season_length=sp),
        AutoTBATS(season_length=sp),
    ]
    # models = [SeasonalNaive(season_length=sp)]
    sf = StatsForecast(models=models, freq=freq, n_jobs=8)
    sf.fit(df=train)

    predDF = sf.predict(h=h).reset_index()

    predDF = test.merge(right=predDF, on=["unique_id", "ds"], how="inner").drop(
        ["index"], axis=1
    )

    predDF["data"] = dataset
    predDF = predDF.melt(
        id_vars=["data", "unique_id", "ds", "y"], var_name="model", value_name="y_hat"
    )
    predDF = predDF[["data", "model", "unique_id", "ds", "y", "y_hat"]]

    # save results dataset by dataset
    fn = "stats_" + dataset + ".csv"
    fn = os.path.join("progress_data", fn)
    fn = os.path.join("data", fn)
    fn = os.path.join(os.getcwd(), fn)
    predDF.to_csv(fn, index=False)

    return predDF


def wrapper(dataset):
    print("datatset: " + dataset + ".")
    train, test = load_data(dataset)
    predDF = predict_stats_models(train, test, dataset)
    return predDF


##########################
# train models
##########################
datasets = ["other", "monthly", "quarterly", "yearly"]
modelPredictions = list(map(wrapper, datasets))
modelPredictions = pd.concat(modelPredictions)


def check_rows(dataset, modelPredictions):
    modelPredictions = modelPredictions.loc[modelPredictions.data == dataset]
    _, test = load_data(dataset)
    B = test.shape[0] * 7 == modelPredictions.shape[0]
    return B


temp = partial(check_rows, modelPredictions=modelPredictions)
all(list(map(temp, datasets)))


fn = os.path.join("data", "predictionsStatsDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions.to_csv(
    fn,
    index=False,
)


fn = os.path.join("data", "predictionsStatsDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions = pd.read_csv(fn)


################################################
# Summarize
################################################
def calc_smape(df, Y="y", YHAT="y_hat"):
    out = smape(df[Y], df[YHAT])
    return out


metricDF = modelPredictions.groupby(
    ["data", "model", "unique_id"], as_index=False
).apply(calc_smape, include_groups=False)
metricDF.columns.values[3] = "smape"
metricDF = metricDF.sort_values(by=["data", "model", "unique_id"])

graph = (
    ggplot(metricDF, aes(x="data", y="smape", fill="model"))
    + geom_boxplot(alpha=0.40)
    + labs(
        title="Model Performance",
        x="Data",
        y="Symmetric Mean Absolute Percentage Error",
    )
    + scale_y_continuous(breaks=np.arange(0, 2.1, 0.2))
    + coord_flip(ylim=[0, 2])
)
ggsave(
    graph,
    os.path.join(os.getcwd(), "graphs/stats_models.png"),
    width=10,
    height=10,
)
graph
