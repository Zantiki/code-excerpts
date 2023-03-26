import torch
import sys
import os
import time
import numpy as np
import pandas as pd

from alive_progress import alive_bar
from src.models.models import CNN
from sklearn.preprocessing import MinMaxScaler
from src.definitions import DATA_ROOT
from src.models.model_utils import plot_loss, plot_result, get_data_from_csv, save_model
from src.models.model_utils import output_to_csv

PERCENT_TEST = 0.2
LOOKBACK = 30
LOOKBACK_ROOF = 24
NUM_EPOCHS = 1000
LR = 0.001
DATA_POINTS = 3000

TRAIN_SCALER = MinMaxScaler(feature_range=(0, 1))
TEST_SCALER = MinMaxScaler(feature_range=(0, 1))


def split_data_as_tensors(scaled_data: 'np.array'):
    """
    Split the dataset into train and test sets

    :param scaled_data: data of included features as dataframe
    :return: x-values in training data, y-values in training data, x-values in test data, y-values in test data
    (all tensors)
    """
    data_raw = scaled_data.to_numpy()  # convert to numpy array
    data = []

    for index in range(len(data_raw) - LOOKBACK):
        temp_data = data_raw[index: index + LOOKBACK]
        data.append(temp_data)

    data = np.array(data)
    test_set_size = int(np.round(PERCENT_TEST * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :LOOKBACK_ROOF, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)

    x_test = data[train_set_size:, :LOOKBACK_ROOF]
    y_test = data[train_set_size:, -1, 0].reshape(-1, 1)


    x_train = TRAIN_SCALER.fit_transform(x_train.reshape(len(x_train), LOOKBACK_ROOF * x_train.shape[2])).reshape(
        len(x_train), LOOKBACK_ROOF, x_train.shape[2])
    y_train = TRAIN_SCALER.fit_transform(y_train)

    x_test = TEST_SCALER.fit_transform(x_test.reshape(len(x_test), LOOKBACK_ROOF * x_test.shape[2])).reshape(
        len(x_test), LOOKBACK_ROOF, x_test.shape[2])
    y_test = TEST_SCALER.fit_transform(y_test)

    return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(x_test).float(),\
           torch.from_numpy(y_test).float()


def train(model: 'torch.nn.Module', x_train: 'torch.tensor', y_train: 'torch.tensor'):
    """
    Train the supplied model with input x on data y

    :param model: untrained model-object
    :param x_train: x-values of training data (tensor)
    :param y_train: y-values of training data (tensor)
    :return: execution time, loss history
    """
    print("Training CNN")
    batches = 10
    x_train_batches = torch.split(x_train, batches)
    y_train_batches = torch.split(y_train, batches)

    optimizer = torch.optim.Adam(model.parameters(), LR)
    hist = np.zeros(NUM_EPOCHS)
    start = time.time()
    with alive_bar(NUM_EPOCHS) as bar:
        for t in range(NUM_EPOCHS):
            for batch in range(len(x_train_batches)):
                loss = model.loss(x_train_batches[batch], y_train_batches[batch])
                hist[t] = loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            bar()

    exec_time = time.time() - start
    return exec_time, hist


def main(

        filename="SPY_2010-10-01_2020-10-01_2020-10-29.csv",
        features_to_include=["Close", "Open", "High", "Low", "Volume"],
        save_plots=True,
        save=False,
        loss=True
):
    features_to_include = np.array(features_to_include)
    model = CNN(len(features_to_include), LOOKBACK_ROOF).float()

    if len(sys.argv) == 1:
        data, index = get_data_from_csv(os.path.join(DATA_ROOT, filename), features_to_include, DATA_POINTS)
    else:
        data, index = get_data_from_csv(str(sys.argv[1]), features_to_include, DATA_POINTS)

    x_train, y_train, x_test, y_test = split_data_as_tensors(data)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1], 1)

    exec_time, hist = train(model, x_train, y_train)

    x = x_test.detach()
    y = y_test.detach()
    y_pred = []

    for val in x:
        val = val.reshape(1, len(features_to_include), LOOKBACK_ROOF, 1)
        y_pred.append([max(model.f(val).detach().numpy()[0])])
    accuracy = round(model.accuracy(torch.tensor(y_pred), y_test), 8)

    plot_result("CNN", y_pred, y_test, index.tail(len(y_test)), filename, accuracy, save_plots)
    print("CNN final accuracy,", accuracy)

    if loss:
        plot_loss("CNN", hist, filename, accuracy, save_plots)

    if save:
        save_model("CNN", model, filename, accuracy)

    output_to_csv(TEST_SCALER, "CNN", filename, y_pred, y_test)
    return "CNN", NUM_EPOCHS, LR, LOOKBACK, LOOKBACK_ROOF, accuracy, exec_time, PERCENT_TEST


if __name__ == "__main__":
    main()
