import torch
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from src.definitions import DOCUMENTATION_ROOT, DATA_ROOT
from os import path


def get_documentation_root_subdir(subdir: str, filename: str) -> 'os.path':

    """
    Given a specified sub-directory and a filename, will greate the subdirectory if it does not exist.

    :param subdir: the name of the wanted subdirectory
    :param filename: the name of the filename you want under the subdirectory.
    :return: the path to the file within the subdirectory
    """

    sub_dir_path = path.join(DOCUMENTATION_ROOT, subdir)
    if not path.isdir(sub_dir_path):
        os.mkdir(sub_dir_path)

    return path.join(DOCUMENTATION_ROOT, subdir+"/"+filename)


def output_to_csv(test_scaler: 'sklearn.preprocessing.minmax_scale', model: str, filename: str,
                  predicted_output: 'np.array', test_data: 'np.array'):
    """
    Will write rescaled predicted and test data to a csv for later profit-evaluation

    :param test_scaler: the scaler to turn normalized prices back to USD.
    :param model: The name of the model where the data was predicted
    :param filename: the filename of the dataset used for testing/training
    :param predicted_output: the predicted normalized data as a np.array
    :param test_data: the predicted normalized data as a np.array
    :return: None
    """

    predicted_output = test_scaler.inverse_transform(predicted_output)
    test_data = test_scaler.inverse_transform(test_data)
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)

    today = datetime.today().strftime('%Y-%m-%d')
    file = filename.split('_')[0] + '_' + model + '_' + 'output' '_' + today
    data_path = os.path.join(DATA_ROOT, 'stored_outputs/')
    pred_df = pd.DataFrame(zip(predicted_output.reshape(-1), test_data.reshape(-1)))
    complete_filename = data_path + file

    if not os.path.exists(complete_filename):
        pred_df.to_csv(complete_filename, index=False)


def get_data_from_csv(filename: str, included_features: 'np.array', data_points: int):
    """
    Get wanted data from a given csv

    :param filename: The CSV to read the files from
    :param included_features: What features you want included in the dataset
    :param data_points: Number of datapoints you want in the set, will take x most recent ones
    :return: (selected_attribute, time_index)
    """
    data = pd.read_csv(filename)
    close_price = data[included_features].tail(data_points)
    return close_price, pd.to_datetime(data["Date"])


def plot_result(model_type, y_pred: 'np.array', y: 'np.array', times: 'np.datetime_data', filename: str,
                accuracy: float, save=False):
    """
    Plot and show accuracy on trained model

    :param model_type: name of the model where the values were predicted
    :param y_pred: predicted output
    :param y: expected output
    :param times: the time-index of the test-data
    :param filename: the dataset used
    :param accuracy: the accuracy of the model
    :param save: Whether or not to save the plot under documentation/plots
    :return: None
    """
    stockname = filename.split("_")[0]
    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(model_type + ' on ' + stockname + ", accuracy: " + str(accuracy))
    plt.ylabel('Normalized price')

    plt.plot(times, y, label='Actual', c='blue')
    plt.plot(times, y_pred, label='TrainedPoints', c='green')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        save = True
    if save:
        img_name = "{}_{}_{}.png".format(model_type, stockname, accuracy)
        img_path = get_documentation_root_subdir("plots", img_name)
        fig.savefig(img_path)


def plot_loss(model_type: str, loss_list: 'np.array', filename: str, accuracy: float, save=False):

    """
    Plot the loss of a given model

    :param model_type: the type of model trained
    :param loss_list: the losses at the different epochs
    :param filename: the filname of the dataset used for training
    :param accuracy: final accuracy of the model
    :param save: save the model under documentation/plots
    :return: None
    """

    stockname = filename.split("_")[0]
    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(model_type + ' on ' + stockname + ", accuracy: " + str(accuracy))
    plt.ylabel('Loss')

    plt.plot(range(len(loss_list)), loss_list, label='loss', c='green')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        save = True
    if save:
        img_name = "{}_LOSS_{}_{}.png".format(model_type, stockname, accuracy)
        img_path = get_documentation_root_subdir("plots", img_name)
        fig.savefig(img_path)


def save_model(model_type: str, model: 'torch.nn.Module', data_filename: str, accuracy: float):
    """
    Save a trained model

    :param model_type: the type of model
    :param model: the model object
    :param data_filename: the filename of the dataset trained on
    :param accuracy: the final accuracy of the model
    :return: None
    """
    instrument = data_filename.split("_")[0]
    acc_string = str(accuracy).split(".")[1]
    model_filename = "{}_model_{}_{}.bin".format(model_type, instrument, acc_string)
    torch.save(model.state_dict(), get_documentation_root_subdir("generated_models", model_filename))
