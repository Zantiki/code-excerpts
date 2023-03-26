import latextable
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from texttable import Texttable
from os import path
from src.definitions import DOCUMENTATION_ROOT
from src.models.model_utils import get_documentation_root_subdir


def get_results_json() -> dict:
    """
    Get the data inside the results.json

    :return: data, a the results json as a dict
    """
    results_file = path.join(DOCUMENTATION_ROOT, "tests/results.json")
    data = None
    with open(results_file, "r") as file:
        data = json.load(file)
    return data


def figure_string(description, label, filename):
    base_string = "\\begin{figure}[H]" + \
                    "\n  \centering" + \
                    "\n  \includegraphics[width=0.4\\textwidth]{%s}" % filename + \
                    "\n  \caption[%s]{%s}"  % (description, description) + \
                    "\n  \label{fig:%s}" % label + \
                    "\n\end{figure}"

    return base_string


def process_data(raw_data: dict) -> list:
    """
    Generate tables and plots based on the results.json file.

    :param raw_data: the results.json in dict-format
    :return: a list of the generated latex-tables/figures as strings
    """

    gru_data_frame = pd.DataFrame(raw_data["GRU"])
    rf_data_frame = pd.DataFrame(raw_data["RF"])
    cnn_data_frame = pd.DataFrame(raw_data["CNN"])

    default_lookback = 30
    default_lookback_roof = 24
    default_datapoints = 3000
    default_features = "Close_Open_High_Low_Volume"

    model_accuracy_table_string = accuracy_v_models([gru_data_frame, rf_data_frame, cnn_data_frame],
                                                    ["GRU", "RF", "CNN"], default_lookback, default_lookback_roof,
                                                    default_features, default_datapoints)

    acc_lookback_string = accuracy_v_lookback([gru_data_frame, rf_data_frame, cnn_data_frame],
                                              ["GRU", "RF", "CNN"], default_features, default_datapoints)

    acc_lookback_roof_string = accuracy_v_lookback_roof([gru_data_frame, rf_data_frame, cnn_data_frame],
                                                        ["GRU", "RF", "CNN"], default_features, default_lookback,
                                                        default_datapoints)
    acc_epoch_string = accuracy_v_epochs([gru_data_frame, cnn_data_frame],
                                         ["GRU", "CNN"], default_lookback, default_lookback_roof, default_features,
                                         default_datapoints)
    acc_feature_string = accuracy_v_included_features([gru_data_frame, rf_data_frame, cnn_data_frame],
                                                      ["GRU", "RF", "CNN"], default_lookback, default_lookback_roof,
                                                      default_datapoints)
    acc_datapoint_string = accuracy_v_data_points([gru_data_frame, rf_data_frame, cnn_data_frame],
                                                  ["GRU", "RF", "CNN"], default_lookback, default_lookback_roof,
                                                  default_features)
    states_string = accuracy_v_rf_states(rf_data_frame, default_lookback, default_lookback_roof, default_features,
                                         default_datapoints)

    return [model_accuracy_table_string, acc_lookback_roof_string, acc_lookback_string, acc_epoch_string,
            acc_feature_string, acc_datapoint_string, states_string]


def accuracy_v_lookback(data_tables: list, order: list, features: str, data_points: int) -> str:
    """
    Generate a mpl-plot and return in the form of a latex-figure-string using the generated plot.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param features: The feature-string used when querying the dataframes
    :param data_points: The number of datapoints used when querying the dataframes
    :return: Latex figure of the generated plot as a string
    """

    max_epochs = data_tables[0]["epochs"].max()
    max_states = 100
    result_table = {
        "GRU": {
            "x": [],
            "y": []
        },
        "CNN": {
            "x": [],
            "y": []
        },
        "RF": {
            "x": [],
            "y": []
        }
    }

    for table, table_type in zip(data_tables, order):
        table = table.sort_values("lb")
        if table_type != "RF":
            max_epochs_table = table[(table.epochs == max_epochs) & (table.features == features)
                                     & (table.data_points == data_points)]
        else:
            max_states = table["rs"].max()
            max_epochs_table = table[(table.rs == max_states) & (table.features == features)
                                     & (table.data_points == data_points)]
        for lookback in max_epochs_table.lb.unique():
            lbr = lookback - 1
            lookahead_table = max_epochs_table[(max_epochs_table.lb == lookback) & (max_epochs_table.lbr == lbr)]
            average_accuracy = lookahead_table["acc"].mean()
            result_table[table_type]["x"].append(lookback)
            result_table[table_type]["y"].append(average_accuracy)

    description_string = "Accuracy vs lookback looking one day ahead" \
        .format(max_epochs, max_states)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")

    plt.plot(result_table["GRU"]["x"], result_table["GRU"]["y"], label='GRU', c='blue')
    plt.plot(result_table["CNN"]["x"], result_table["CNN"]["y"], label='CNN', c='green')
    plt.plot(result_table["RF"]["x"], result_table["RF"]["y"], label='RF', c='red')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass
    img_name = "models_v_lookback.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)

    return figure_string(description_string, "acc-lookback", img_name)


def accuracy_v_lookback_roof(data_tables: list, order: list, features: str, lookback: int, data_points: int) -> str:
    """
    Generate a mpl-plot of the lookback_roof and return in the form of a latex-figure-string using the generated plot.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param features: The feature-string used when querying the dataframes
    :param lookback: The lookback used when querying the dataframes
    :param data_points: The number of datapoints used when querying the dataframes
    :return: Latex figure of the generated plot as a string
    """
    max_epochs = data_tables[0]["epochs"].max()
    max_states = 100
    result_table = {
        "GRU": {
            "x": [],
            "y": []
        },
        "CNN": {
            "x": [],
            "y": []
        },
        "RF": {
            "x": [],
            "y": []
        }
    }

    for table, table_type in zip(data_tables, order):
        table = table.sort_values("lbr")
        if table_type != "RF":
            max_epochs_table = table[(table.epochs == max_epochs) & (table.features == features)
                                     & (table.lb == lookback) & (table.data_points == data_points)]
        else:
            max_states = table["rs"].max()
            max_epochs_table = table[(table.rs == max_states) & (table.features == features) & (table.lb == lookback)
                                     & (table.data_points == data_points)]
        for lookahead in max_epochs_table.lbr.unique():
            lookahead_table = max_epochs_table[max_epochs_table.lbr == lookahead]
            average_accuracy = lookahead_table["acc"].mean()
            result_table[table_type]["x"].append(lookback - lookahead)
            result_table[table_type]["y"].append(average_accuracy)

    description_string = "Accuracy vs look-ahead with {} epochs and {} states for RF and lookback  of {}"\
        .format(max_epochs, max_states, lookback)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")


    plt.plot(result_table["GRU"]["x"], result_table["GRU"]["y"], label='GRU', c='blue')
    plt.plot(result_table["CNN"]["x"], result_table["CNN"]["y"], label='CNN', c='green')
    plt.plot(result_table["RF"]["x"], result_table["RF"]["y"], label='RF', c='red')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass
    img_name = "models_v_lookback_roof.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)

    return figure_string(description_string, "acc-lookback", img_name)


def accuracy_v_rf_states(table: 'pd.DataFrame', lookback: int, lookback_roof: int, features: str, data_points: int) \
        -> str:
    """
    Generate a mpl-plot for RF based on number of states and return in the form of a
    latex-figure-string using the generated plot.

    :param table: The RF-Dataframe
    :param lookback: the lookback used when querying the dataframe
    :param lookback_roof: the lookback_roof used when querying the dataframe
    :param features: The feature-string used when querying the dataframe
    :param data_points: The number of datapoints used when querying the dataframe
    :return: Latex figure of the generated plot as a string
    """
    x = []
    y = []
    table = table.sort_values("rs")
    for number_of_states in table.rs.unique():
        state_table = table[(table.rs == number_of_states) & (table.features == features) & (table.lb == lookback)
              & (table.lbr == lookback_roof) & (table.data_points == data_points)]
        x.append(number_of_states)
        y.append(state_table["acc"].mean())

    description_string = "Accuracy v. number of states for RF looking {} days ahead with lookback {} " \
        .format(lookback - lookback_roof, lookback)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")

    plt.plot(x, y, label='RF', c='red')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass
    img_name = "accuracy_v_random_states.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)
    return figure_string(description_string, "accuracy-datapoints", img_name)


def accuracy_v_models(data_tables: list, order: list, lookback: int, lookback_roof: int, features: str,
                      data_points: int) -> str:
    """
    Generate a latex table comparing accuracy and runtime of each model.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param lookback: The lookback used when querying the dataframes
    :param lookback_roof: The lookback_roof used when querying the dataframes
    :param features: The feature-string used when querying the dataframes
    :param data_points: The number of datapoints used when querying the dataframes
    :return: Latex table of the comparison
    """

    max_epochs = data_tables[0]["epochs"].max()
    max_states = 100
    latex_string = ""
    result_table = [["model", "avg r2 score", "avg runtime sec"]]

    text_table = Texttable()
    text_table.set_cols_align(["l", "r", "c"])
    text_table.set_cols_valign(["t", "m", "b"])

    for table, table_type in zip(data_tables, order):
        if table_type != "RF":
            max_epochs_table = table[(table.epochs == max_epochs) & (table.features == features) & (table.lb == lookback)
                                     & (table.lbr == lookback_roof) & (table.data_points == data_points)]
        else:
            max_states = table["rs"].max()
            max_epochs_table = table[(table.rs == max_states) & (table.features == features) & (table.lb == lookback)
                                     & (table.lbr == lookback_roof) & (table.data_points == data_points)]
        average_accuracy = max_epochs_table["acc"].mean()
        average_rt = max_epochs_table["exec_time"].mean()
        result_table.append([table_type, average_accuracy, average_rt])

    text_table.add_rows(result_table)
    description_string = "Model comparison with {} epochs and {} states for RF, looking {} days ahead with lookback {} " \
        .format(max_epochs, max_states, lookback-lookback_roof, lookback)
    print(description_string)
    print(text_table.draw() + "\n")

    return latextable.draw_latex(text_table, caption=description_string) + "\n"


def accuracy_v_epochs(data_tables: list, order: list, lookback: int, lookback_roof: int, features: str,
                      data_points: int) -> str:
    """
    Generate a mpl-plot comparing the accuracy against number of epochs for CNN and GRU. Retruns a latex-string
    as a figure using the plot.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param lookback: The lookback used when querying the dataframes
    :param lookback_roof: The lookback_roof used when querying the dataframes
    :param features: The feature-string used when querying the dataframes
    :param data_points: The number of datapoints used when querying the dataframes
    :return: Latex figure of the generated plot as a string
    """

    result_table = {
        "GRU": {
            "x":[],
            "y": []
        },
        "CNN": {
            "x": [],
            "y": []
        }
    }

    for table, table_type in zip(data_tables, order):
        table = table.sort_values("epochs")
        for epoch in table.epochs.unique():
            epoch_table = table[(table.epochs == epoch) & (table.lb == lookback) & (table.lbr == lookback_roof) &
                                (table.features == features) & (table.data_points == data_points)]
            average_accuracy = epoch_table["acc"].mean()
            result_table[table_type]["x"].append(epoch)
            result_table[table_type]["y"].append(average_accuracy)

    description_string = "Accuracy for GRU and CNN v epochs looking {} days ahead with lookback {} " \
        .format(lookback - lookback_roof, lookback)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")

    plt.plot(result_table["GRU"]["x"], result_table["GRU"]["y"], label='GRU', c='blue')
    plt.plot(result_table["CNN"]["x"], result_table["CNN"]["y"], label='CNN', c='green')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass
    img_name = "accuracy_v_epochs.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)
    return figure_string(description_string, "accuracy-epochs", img_name)


def accuracy_v_data_points(data_tables: list, order: list, lookback: int, lookback_roof: int, features: str):
    """
    Generate a mpl-plot comparing the accuracy against the size of the total dataset for CNN and GRU.
    Returns a latex-figure string using the plot.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param lookback: The lookback used when querying the dataframes
    :param lookback_roof: The lookback_roof used when querying the dataframes
    :param features: The feature-string used when querying the dataframes
    :return: Latex figure of the generated plot as a string
    """

    max_epochs = data_tables[0]["epochs"].max()
    max_states = 100
    result_table = {
        "GRU": {
            "x": [],
            "y": []
        },
        "CNN": {
            "x": [],
            "y": []
        },
        "RF": {
            "x": [],
            "y": []
        }

    }

    for table, table_type in zip(data_tables, order):
        table = table.sort_values("data_points")

        if table_type != "RF":
            table = table[(table.epochs == max_epochs) & (table.features == features) & (table.lb == lookback)
                                     & (table.lbr == lookback_roof)]
        else:
            max_states = table["rs"].max()
            table = table[(table.rs == max_states)& (table.lb == lookback) & (table.lbr == lookback_roof)]

        for points in table.data_points.unique():
            epoch_table = table[(table.data_points == points) & (table.lb == lookback) & (table.lbr == lookback_roof) & (
                        table.features == features) ]
            average_accuracy = epoch_table["acc"].mean()
            result_table[table_type]["x"].append(points)
            result_table[table_type]["y"].append(average_accuracy)

    description_string = "Accuracy v. size of dataset looking {} days ahead with lookback {} " \
        .format(lookback - lookback_roof, lookback)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")

    plt.plot(result_table["GRU"]["x"], result_table["GRU"]["y"], label='GRU', c='blue')
    plt.plot(result_table["CNN"]["x"], result_table["CNN"]["y"], label='CNN', c='green')
    plt.plot(result_table["RF"]["x"], result_table["RF"]["y"], label='RF', c='red')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass
    img_name = "accuracy_v_datapoints.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)
    return figure_string(description_string, "accuracy-datapoints", img_name)


def accuracy_v_included_features(data_tables: list, order: list, lookback: int, lookback_roof: int, data_points: int):
    """
    Generate a mpl-plot comparing the accuracy against the types of included fetures for CNN and GRU.
    Retruns a latex-string as a figure using the plot.

    :param data_tables: The different pd.DataFrames based on each model-result
    :param order: A list of string describing the type model at each index in the data_tables parameter
    :param lookback: The lookback used when querying the dataframes
    :param lookback_roof: The lookback_roof used when querying the dataframes
    :param data_points: The number of datapoints used when querying the dataframes
    :return: Latex figure of the generated plot as a string
    """

    max_epochs = data_tables[0]["epochs"].max()
    max_states = 10
    result_table = {
        "GRU": {
            "x": [],
            "y": []
        },
        "CNN": {
            "x": [],
            "y": []
        },
        "RF": {
            "x": [],
            "y": []
        }
    }

    for table, table_type in zip(data_tables, order):
        if table_type != "RF":
            max_epochs_table = table[table.epochs == max_epochs]
        else:
            max_states = table["rs"].max()
            table = table[table.rs == max_states]

        for feature in table.features.unique():
            feature_table = table[(table.features == feature) & (table.lb == lookback) &
                                             (table.lbr == lookback_roof) & (table.data_points == data_points)]
            average_accuracy = feature_table["acc"].mean()
            new_feature_string = "_".join([list(feat)[0] for feat in feature.split("_")])
            result_table[table_type]["x"].append(new_feature_string)
            result_table[table_type]["y"].append(average_accuracy)

    description_string = "Accuracy vs type of input features with {} epochs and {} states for RF".format(max_epochs, max_states)

    fig = plt.figure(figsize=(8, 4))
    plt.xlabel(description_string)
    plt.ylabel("accuracy")

    plt.scatter(result_table["GRU"]["x"], result_table["GRU"]["y"], label='GRU', c='blue')
    plt.scatter(result_table["CNN"]["x"], result_table["CNN"]["y"], label='CNN', c='green')
    plt.scatter(result_table["RF"]["x"], result_table["RF"]["y"], label='RF', c='red')
    plt.legend()
    try:
        # plt.show()
        pass
    except Exception:
        pass

    img_name = "models_v_features.png"
    img_name = os.path.join(get_documentation_root_subdir("latex", img_name))
    fig.savefig(img_name)
    return figure_string(description_string, "accuracy-features", img_name)


if __name__ == "__main__":

    """
    Read and generate the latex figure/plots based on the data in the results.json and finally append the figures to a
    .tex file under documentation.
    """

    raw_data = get_results_json()
    if not raw_data:
        raise FileNotFoundError("Could not read json file")
    result_list = process_data(raw_data)
    # output_file = "result_tables_{}.tex".format(int(time.time()))
    output_file = "result_tables_{}.tex".format("dev")
    output_file = os.path.join(get_documentation_root_subdir("latex", output_file))
    with open(output_file, "w") as file:
        file.write("\\usepackage{graphicx}\n")
        file.write("\\begin{document}")
        for tex_string in result_list:
            file.write(tex_string)
            file.write("\n\n")
        file.write("\\end{document}")
    print("You can find results under {}/{}".format(os.getcwd(), output_file))