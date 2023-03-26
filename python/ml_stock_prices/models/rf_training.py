import sys
import os
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.models.models import RandomForest
from src.definitions import DATA_ROOT
from src.models.model_utils import get_data_from_csv, plot_result
from src.models.model_utils import output_to_csv

RANDOM_STATE = 100
PERCENT_TEST = 0.20
LOOKBACK = 30
LOOKBACK_ROOF = 24
DATA_POINTS = 3000
TRAIN_SCALER = MinMaxScaler(feature_range=(0, 1))
TEST_SCALER = MinMaxScaler(feature_range=(0, 1))

def split_data_as_np(scaled_data: 'np.array', number_of_features: int):
    """
    Split the dataset into train and test sets

    :param scaled_data: data of included features as dataframe
    :return: x-values in training data, y-values in training data, x-values in test data, y-values in test data
    (all tensors)
    """
    data_raw = scaled_data.to_numpy()  # convert to numpy array
    data = []

    for index in range(len(data_raw) - LOOKBACK):
        data.append(data_raw[index: index + LOOKBACK])

    data = np.array(data)
    test_set_size = int(np.round(PERCENT_TEST * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :LOOKBACK_ROOF, :]
    y_train = data[:train_set_size, -1, 0]

    x_test = data[train_set_size:, :LOOKBACK_ROOF]
    y_test = data[train_set_size:, -1, 0]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
    y_train = y_train.reshape((y_train.shape[0], 1))

    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], 1))

    x_train = TRAIN_SCALER.fit_transform(x_train)
    y_train = TRAIN_SCALER.fit_transform(y_train)

    x_test = TEST_SCALER.fit_transform(x_test)
    y_test = TEST_SCALER.fit_transform(y_test)

    return x_train, y_train, x_test, y_test


def main(

        filename="AAPL_2010-10-01_2020-10-01_2020-10-29.csv",
        features_to_include=["Close", "Open", "High", "Low", "Volume"],
        save_plots=True,
        save=False
):
    features_to_include = np.array(features_to_include)

    model = RandomForest(RANDOM_STATE)

    if len(sys.argv) == 1:
        labeled_data, index = get_data_from_csv(os.path.join(DATA_ROOT, filename), features_to_include, DATA_POINTS)
    else:
        labeled_data, index = get_data_from_csv(str(sys.argv[1]), features_to_include, DATA_POINTS)
    x_train, y_train, x_test, y_test = split_data_as_np(labeled_data, 2)

    start = time.time()
    model.train(x_train, y_train)
    exec_time = time.time() - start
    y_pred = model.predict(x_test)

    acc = round(model.accuracy(x_test, y_test), 8)
    print("RF, final accuracy:", acc)
    plot_result("RF", y_pred, y_test, index.tail(len(y_test)), filename, acc, save=save_plots)
    output_to_csv(TEST_SCALER, "RF", filename, y_pred, y_test)
    return "RF", RANDOM_STATE, LOOKBACK, LOOKBACK_ROOF, acc, exec_time, PERCENT_TEST


if __name__ == "__main__":
    main()
