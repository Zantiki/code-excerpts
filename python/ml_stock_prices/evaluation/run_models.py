import os
import shutil
import json
import latextable
from os import listdir
from os.path import join, isfile
from datetime import datetime
from texttable import Texttable
from src.models.model_utils import get_documentation_root_subdir
from src.models import gru_training, cnn_training, rf_training
from src.evaluation import profit_evaluation
from src.definitions import DOCUMENTATION_ROOT, CONTENT_ROOT, DATA_ROOT


def purge_old():

    """
    Remove the entire documentation directory

    :return: None
    """

    test_root = os.path.join(DOCUMENTATION_ROOT, "tests")
    plot_root = os.path.join(DOCUMENTATION_ROOT, "plots")

    if os.path.isdir(test_root):
        shutil.rmtree(test_root)

    if os.path.isdir(plot_root):
        shutil.rmtree(plot_root)


def optimal_profit():

    """
    Run all models on the specified datasets, calculate the profits, run RF on the all of datasets and finally
    generate tables based on the profit-results and save it with under documentation/latex.

    :return: None
    """

    datasets = [
        "AAPL_2010-10-01_2020-10-01_2020-10-29.csv",
        "JNJ_2010-10-01_2020-10-01_2020-10-29.csv",
        "SPY_2010-10-01_2020-10-01_2020-10-29.csv"
    ]
    models = ["GRU", "CNN", "RF"]
    latex_strings = []

    # main(save_result=False)

    for filename in datasets:
        description = "results for " + filename.split("_")[0]
        text_table = Texttable()
        text_table.set_cols_align(["l", "r", "c", "l"])
        text_table.set_cols_valign(["t", "m", "b", "t"])
        results = [["model type", "model profits", "sma profits", "percent difference"]]
        for model in models:
            today = datetime.today().strftime('%Y-%m-%d')
            file = filename.split('_')[0] + '_' + model + '_' + 'output' '_' + today
            model_profit, sma_profit = profit_evaluation.main(file)
            result = [model, model_profit, sma_profit, (round((model_profit - sma_profit) / abs(sma_profit), 4)*100)]
            results.append(result)
        text_table.add_rows(results)
        print(description)
        print(text_table.draw() + "\n")
        latex_strings.append(latextable.draw_latex(text_table, caption=description) + "\n")
    config_name = os.path.join(CONTENT_ROOT, "src/evaluation/tests.json")
    config = {}
    with open(config_name, "r") as config_file:
        config = json.load(config_file)
    text_table = Texttable()
    text_table.set_cols_align(["l", "r", "c", "l"])
    text_table.set_cols_valign(["t", "m", "b", "t"])
    description = "RF results"
    cumulative_model = 0
    for filename in [f for f in listdir(DATA_ROOT) if isfile(join(DATA_ROOT, f))]:
        results = [["Stock", "model profits", "sma profits", "percent difference"]]
        rf_training.main(features_to_include=config["features"], filename=filename, save=config["save_models"],
                         save_plots=config["save_plots"])

        today = datetime.today().strftime('%Y-%m-%d')
        file = filename.split('_')[0] + '_' + "RF" + '_' + 'output' '_' + today
        model_profit, sma_profit = profit_evaluation.main(file)
        if "VIX" not in filename and "SPY" not in filename and "JNJ" not in filename and "AAPL" not in filename:
            cumulative_model += (round((model_profit - sma_profit) / abs(sma_profit), 4) * 100)

        result = [filename.split('_')[0], model_profit, sma_profit, (round((model_profit - sma_profit) / abs(sma_profit), 4) * 100)]
        results.append(result)
        text_table.add_rows(results)
    print(description)
    print(text_table.draw() + "\n")
    latex_strings.append(latextable.draw_latex(text_table, caption=description) + "\n")
    print(cumulative_model)

    output_file = "result_profit_{}.tex".format("dev")
    output_file = os.path.join(get_documentation_root_subdir("latex", output_file))
    with open(output_file, "w") as file:
        file.write("\\begin{document}")
        for tex_string in latex_strings:
            file.write(tex_string)
            file.write("\n\n")
        file.write("\\end{document}")
    print("You can find results under {}".format(output_file))


def run_multiple():
    """
    Purge the old results and run all of the tests required for a full results.json and profit-evaluation.

    :return: None
    """


    # these must be equally long
    multi_test_config = {

       "features": [["Close", "Open", "High", "Low", "Volume"], ["Close", "Volume", "Low"],
                     ["Close", "Open", "High", "Low"], ["Close", "Dividends", "Stock Splits"], ["Close"]],
        "epochs": [50, 100, 500, 1000, 2000],
        "lookback": [30, 5, 20, 2, 10],
        "lookback_roof": [24, 10, 16, 29, 5],
        "data_points": [100, 300, 500, 700, 3000]

    }
    config_name = os.path.join(CONTENT_ROOT, "src/evaluation/tests.json")
    standard_conf = {}
    with open(config_name, "r") as config_file:
        standard_conf = json.load(config_file)

    print("--- Running multiple tests ---")
    test_amount = len(multi_test_config) * 5 * len(standard_conf["test_stocks"]) * 3
    print("\033[93mWARNING: running tests in multiple mode, this will purge plots, results.json and run {} different tests."
          .format(test_amount))
    y_n = input("Continue? (y/n) \033[0m")

    if y_n != "y":
        y_n = input("Run with standard config? (y/n) ")
        if y_n != "y":
            exit()
        else:
            main()
            exit()
    purge_old()
    for parameter in multi_test_config:
        print("\n-- Running {} tests --".format(parameter))
        for i in range(len(multi_test_config[parameter])):
            if parameter == "lookback":
                standard_conf["lookback_roof"] = multi_test_config["lookback"][i] - 1

            print("-- Running {} tests with arg {} --".format(parameter, multi_test_config[parameter][i]))
            standard_conf[parameter] = multi_test_config[parameter][i]
            main(standard_conf)
            # Reset config
            with open(config_name, "r") as config_file:
                standard_conf = json.load(config_file)
    state_test()
    optimal_profit()


def state_test(config=None):

    """
    Train and calculate accuracy of a wide range of random-states for random forest. Will append results to existing
    results.json

    :param config: specifiable config-dict, will default to standard if not specified.
    :return: None
    """

    test_root = os.path.join(DOCUMENTATION_ROOT, "tests")
    results_file = os.path.join(test_root, "results.json")
    random_states = [1, 5, 10, 50, 100]
    result_dict = {
        "RF": [],
    }

    with open(results_file, "r") as file:
        results = json.load(file)

    if not config:
        config_name = os.path.join(CONTENT_ROOT, "src/evaluation/tests.json")
        with open(config_name, "r") as config_file:
            config = json.load(config_file)

    data_points = config["data_points"]
    lookback = config["lookback"]
    lookback_roof = config["lookback_roof"]
    rf_training.DATA_POINTS = data_points
    rf_training.LOOKBACK = lookback
    rf_training.LOOKBACK_ROOF = lookback_roof

    for number_of_states in random_states:
        for filename in config["test_stocks"]:
            rf_training.RANDOM_STATE = number_of_states
            model_type, random_state, lb, lbr, acc, exec_time, percent_data = \
                rf_training.main(features_to_include=config["features"], filename=filename, save=config["save_models"],
                                 save_plots=config["save_plots"])
            results[model_type].append({
                "rs": random_state,
                "lb": lb,
                "lbr": lbr,
                "acc": acc,
                "exec_time": exec_time,
                "data_set": filename,
                "percent_data": percent_data,
                "data_points": data_points,
                "features": "_".join(config["features"])
            })

    json_string = json.dumps(results, indent=4)
    with open(results_file, "w") as file:
        file.write(json_string)


def main(config=None, save_result=True):
    """
    Run a single round of tests across all models with paramaters specified in config. If config is not used as function
    parameter, the function will default to config in tests.json.

    :param config: specifiable config-dict, will default to standard if not specified.
    :param save_result: specify if you want results saved to results.json or not.
    :return: None
    """

    test_root = os.path.join(DOCUMENTATION_ROOT, "tests")
    results_file = os.path.join(test_root, "results.json")
    if not os.path.isdir(DOCUMENTATION_ROOT):
        os.mkdir(DOCUMENTATION_ROOT)

    if not os.path.isdir(test_root):
        os.mkdir(test_root)

        result_dict = {
            "GRU": [],
            "RF": [],
            "CNN": []
        }
        json_string = json.dumps(result_dict, indent=4)
        with open(results_file, "w") as file:
            file.write(json_string)

    results = {}
    with open(results_file, "r") as file:
        results = json.load(file)

    if not config:
        config_name = os.path.join(CONTENT_ROOT, "src/evaluation/tests.json")
        with open(config_name, "r") as config_file:
            config = json.load(config_file)

    global_epochs = config["epochs"]
    data_points = config["data_points"]
    lookback = config["lookback"]
    lookback_roof = config["lookback_roof"]

    gru_training.DATA_POINTS = data_points
    cnn_training.DATA_POINTS = data_points
    rf_training.DATA_POINTS = data_points

    gru_training.LOOKBACK = lookback
    cnn_training.LOOKBACK = lookback
    rf_training.LOOKBACK = lookback

    gru_training.LOOKBACK_ROOF = lookback_roof
    cnn_training.LOOKBACK_ROOF = lookback_roof
    rf_training.LOOKBACK_ROOF = lookback_roof

    gru_training.NUM_EPOCHS = global_epochs
    cnn_training.NUM_EPOCHS = global_epochs


    for i in range(config["repeat"]):
        for filename in config["test_stocks"]:
            model_type, epochs, lr, lb, lbr, acc, exec_time, percent_data = \
                gru_training.main(features_to_include=config["features"], filename=filename, save=config["save_models"],
                                  save_plots=config["save_plots"])
            results[model_type].append({
                "epochs": epochs,
                "lr": lr,
                "lb": lb,
                "lbr": lbr,
                "acc": acc,
                "exec_time": exec_time,
                "data_set": filename,
                "percent_data": percent_data,
                "data_points": data_points,
                "features": "_".join(config["features"])
            })

            model_type, epochs, lr, lb, lbr, acc, exec_time, percent_data = \
                cnn_training.main(features_to_include=config["features"], filename=filename, save=config["save_models"],
                                  save_plots=config["save_plots"])
            results[model_type].append({
                "epochs": epochs,
                "lr": lr,
                "lb": lb,
                "lbr": lbr,
                "acc": acc,
                "exec_time": exec_time,
                "data_set": filename,
                "percent_data": percent_data,
                "data_points": data_points,
                "features": "_".join(config["features"])
            })
            model_type, random_state, lb, lbr, acc, exec_time, percent_data = \
                rf_training.main(features_to_include=config["features"], filename=filename, save=config["save_models"],
                                 save_plots=config["save_plots"])
            results[model_type].append({
                "rs": random_state,
                "lb": lb,
                "lbr": lbr,
                "acc": acc,
                "exec_time": exec_time,
                "data_set": filename,
                "percent_data": percent_data,
                "data_points": data_points,
                "features": "_".join(config["features"])
            })

        if save_result:
            json_string = json.dumps(results, indent=4)
            with open(results_file, "w") as file:
                file.write(json_string)


if __name__ == "__main__":
    """
    Will run all of the tests. Can take up to 30 hours, depending on parameters defined in tests.json.
    """
    run_multiple()





