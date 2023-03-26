import time
import torch
import sys
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.optim import Adam
from src.models.models import GRU
from src.definitions import DATA_ROOT
from src.models.model_utils import plot_loss, plot_result, get_data_from_csv, save_model
from alive_progress import alive_bar
from src.models.model_utils import output_to_csv

INPUT_DIM = 3
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_DIM = 1
NUM_EPOCHS = 1000
LR = 0.001
PERCENT_TEST = 0.2
DATA_POINTS = 3000
LOOKBACK = 30
LOOKBACK_ROOF = 24

TRAIN_SCALER = MinMaxScaler(feature_range=(0, 1))
TEST_SCALER = MinMaxScaler(feature_range=(0, 1))


def split_data_as_tensors(scaled_data: 'np.array'):
    """
    Split the dataset into train and test sets

    :param scaled_data: data of included features as dataframe
    :return: x-values in training data, y-values in training data, x-values in test data, y-values in test data
    (all tensors)
    """
    data_raw = scaled_data.to_numpy()
    data = []

    for index in range(len(data_raw) - LOOKBACK):
        data.append(data_raw[index: index + LOOKBACK])

    data = np.array(data)
    test_set_size = int(np.round(PERCENT_TEST * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :LOOKBACK_ROOF, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)

    x_test = data[train_set_size:, :LOOKBACK_ROOF]
    y_test = data[train_set_size:, -1, 0].reshape(-1, 1)

    x_train = TRAIN_SCALER.fit_transform(x_train.reshape(len(x_train), (LOOKBACK_ROOF)*x_train.shape[2])).reshape(len(x_train), LOOKBACK_ROOF, x_train.shape[2])
    y_train = TRAIN_SCALER.fit_transform(y_train)

    x_test = TEST_SCALER.fit_transform(x_test.reshape(len(x_test), (LOOKBACK_ROOF)*x_test.shape[2])).reshape(len(x_test), LOOKBACK_ROOF, x_test.shape[2])
    y_test = TEST_SCALER.fit_transform(y_test)
    return torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test)


def train(model: 'torch.nn.Module', x: 'torch.tensor', y: 'torch.tensor'):
    """
    Train the supplied model with input x on data y

    :param model: The untrained model-object
    :param x: x-values of training data (tensor)
    :param y: y-values of training data (tensor)
    :return: execution time, loss history
    """
    print("Training GRU")
    criterion = MSELoss(reduction='mean')
    optimiser = Adam(model.parameters(), lr=LR)

    hist = np.zeros(NUM_EPOCHS)
    start_time = time.time()
    with alive_bar(NUM_EPOCHS) as bar:
        for t in range(NUM_EPOCHS):
            y_pred = model(x)
            loss = criterion(y_pred.double(), y)
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            bar()

    training_time = time.time() - start_time
    return training_time, hist


def main(

        filename="SPY_2010-10-01_2020-10-01_2020-10-29.csv",
        features_to_include=["Close", "Open", "High", "Low", "Volume"],
        save=False,
        save_plots=True,
        loss=True
):
    features_to_include = np.array(features_to_include)

    model = GRU(input_dim=len(features_to_include), hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                num_layers=NUM_LAYERS).double()

    if len(sys.argv) == 1:
        data, index = get_data_from_csv(os.path.join(DATA_ROOT, filename), features_to_include, DATA_POINTS)
    else:
        data, index = get_data_from_csv(str(sys.argv[1]), features_to_include, DATA_POINTS)
    x_train, y_train, x_test, y_test = split_data_as_tensors(data)

    exec_time, hist = train(model, x_train.double(), y_train.double())
    x = x_test
    y_pred = model(x).detach()
    x = x_test.detach()
    y = y_test.detach()

    acc = round(model.accuracy(x, y), 8)

    plot_result("GRU", y_pred, y_test, index.tail(len(y_test)), filename, acc, save=save_plots)

    if loss:
        plot_loss("GRU", hist, filename, acc, save=save_plots)

    if save:
        save_model("GRU", model, filename, acc)

    print("GRU, final accuracy:", acc)
    output_to_csv(TEST_SCALER, "GRU", filename, y_pred.numpy(), y_test.numpy())
    return "GRU", NUM_EPOCHS, LR, LOOKBACK, LOOKBACK_ROOF, acc, exec_time, PERCENT_TEST


if __name__ == "__main__":
    main()
