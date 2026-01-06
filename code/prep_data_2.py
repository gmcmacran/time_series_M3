##########################################################
# Overview
#
# A script to take raw exploration data and make it ready for
# nixtla's neuralforecast.
#
#
# Output:
# Two cleaned csvs per time unit in data folder.
##########################################################

import os

import numpy as np
import pandas as pd
from datasetsforecast.m3 import M3

folder = os.path.join(os.getcwd(), "data")
if not os.path.exists(folder):
    os.makedirs(folder)


def prep_data(dataset):
    print(dataset)

    if dataset == "other":
        train, _, _ = M3.load(directory=os.path.join(folder), group="Other")
        correct_series_count = 174
        h = 8

    elif dataset == "monthly":
        train, _, _ = M3.load(directory=os.path.join(folder), group="Monthly")
        correct_series_count = 1428
        h = 18

    elif dataset == "quarterly":
        train, _, _ = M3.load(directory=os.path.join(folder), group="Quarterly")
        correct_series_count = 756
        h = 8

    elif dataset == "yearly":
        train, _, _ = M3.load(directory=os.path.join(folder), group="Yearly")
        correct_series_count = 645
        h = 6

    test = train.sort_values(by="ds").groupby("unique_id").tail(h)
    train = train.loc[~train.index.isin(test.index)]

    # confirm there is the right amount of series.
    if train.unique_id.nunique() != correct_series_count:
        print("Error: Incorrect number of series after shaping.")
    if test.unique_id.nunique() != correct_series_count:
        print("Error: Incorrect number of series after shaping.")

    train_file = os.path.join(folder, f"{dataset}_train.csv")
    test_file = os.path.join(folder, f"{dataset}_test.csv")

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)


datasets = ["other", "monthly", "quarterly", "yearly"]
list(map(prep_data, datasets))
