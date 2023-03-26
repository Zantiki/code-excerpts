import pandas as pd
import numpy as np
from src.definitions import DATA_ROOT
import os

BATCH_SIZE = 6
BATCH_SIZE_TEST = 20

best_buy_days = []
best_sell_days = []
diffs = []


def get_predicted(file: str):

    """
    Get the different types of batched data necessary to complete best_transactions and falt data to compute SMA. Global
    parameters
    BATCH_SIZE_TEST determine size of SMA-window, while BATCH_SIZE determine size of predicted best_transaction window.
    BATCH_SIZE should be the same size as lookahead for most realistic results.

    :param file: path to (output)-csv-file containing columns corresponding to predicted data and real data for
    given stock
    :return: batched_pred, flat_pred, batched_test_sma, batched_test_pred, flat_test
    """

    data = pd.read_csv(file)
    flat_pred = data['0'].to_numpy()
    flat_pred = np.array(flat_pred)

    flat_test = data['1'].to_numpy()
    flat_test = np.array(flat_test)

    batched_pred = []
    batched_test_sma = []
    batched_test_pred = []
    for i in range(0, len(flat_pred), BATCH_SIZE):
        batched_pred.append(flat_pred[i:i + BATCH_SIZE])
        batched_test_pred.append(flat_test[i:i + BATCH_SIZE])

    for i in range(0, len(flat_pred), BATCH_SIZE_TEST):
        batched_test_sma.append(flat_test[i:i + BATCH_SIZE_TEST])
    return batched_pred, flat_pred, batched_test_sma, batched_test_pred, flat_test


def find_best_transactions(price_list_batch: list):
    """
    Finds the two days in each time period (BATCH_SIZE) in batched price list, which gives biggest difference in profit.
    First day is best day to buy, and second day is best day to but (buy day < sell day). Global lists best_buy_days
    and best_sell_days are updated with corresponding values on same index.
    (suboptimal as time-complexity is o(n²))

    :param price_list_batch: list with length = BATCH_SIZE. Corresponds to a period (e.g. 5-day week) in
    which to buy once and sell once
    :return: None
    """
    buy_day = 0
    sell_day = 0
    diff = 0

    # o(n²)-algorithm for determining largest sell - buy diff on predicted data
    for buy in range(len(price_list_batch)):
        buy_price = price_list_batch[buy]
        for sale in range(buy + 1, len(price_list_batch)):
            if price_list_batch[sale] - buy_price > diff:
                buy_day = buy
                sell_day = sale
                diff = price_list_batch[sale] - price_list_batch[buy]
    best_buy_days.append(buy_day)
    best_sell_days.append(sell_day)
    diffs.append(diff)


def best_transactions_on_test_data(buy: list, sell: list, batched_test_data: list):
    """
    Takes best_buy_days and best_sell_days (result from method find_best_transactions), and buy and sell on
    batched_test_data according to these

    :param buy: list of optimal days to buy, from method find_best_transactions
    :param sell: list of optimal days to sell, from method find_best_transactions
    :param batched_test_data: 2d-list where elements has length = BATCH_SIZE. Corresponds to periods (e.g. 5-day week)
    :return: cummulative profit from each transaction
    """
    difference = 0
    for i in range(len(batched_test_data)):
        # print(batched_test_data[i][sell[i]], "-", batched_test_data[i][buy[i]])
        difference += (batched_test_data[i][sell[i]] - batched_test_data[i][buy[i]])
    return difference


def batch_eval(price_list: list):
    """
    Run method find_best_transactions on each batch in 2d-list

    :param price_list: 2d-list with length of elements = BATCH_SIZE, each element is one period (e.g. 5-day week).
    :return: None
    """
    for batch in price_list:
        find_best_transactions(batch)


def simple_moving_average(predicted_data: list) -> list:
    """
    Calculate a simple mean averages, with window size = BATC_SIZE_TEST

    :param predicted_data: 1d-list of stock prices
    :return: list of averages over periods = BATCH_SIZE_TEST (global variable)
    """
    averages = []
    window_size = BATCH_SIZE_TEST
    for i in range(len(predicted_data)):
        averages.append(sum(predicted_data[i:i+(window_size-1)])/len(predicted_data[i:i+(window_size-1)]))
    return averages


def SMA_buy(sma: list, batched_data: list) -> tuple:
    """
    Use simple mean average to determine transaction (buy, hold, sell) the day after a window (BATCH_SIZE_TEST), on a
    2d-list of bached data

    :param sma: List of averages from a Simple Mean Average
    :param batched_data: 2d-list
    :return: cummulative profit, list of profit per period
    """
    diff = 0
    diffs = []
    bought = False
    for i in range(1, len(batched_data)):
        if (batched_data[i][0] > sma[i-1]*1.03) and (bought is False):
            diff -= batched_data[i][0]
            bought = True
        elif (batched_data[i][0] < sma[i-1]*0.97) and (bought is True):
            diff += batched_data[i][0]
            bought = False
    if bought is True:
        diff += batched_data[len(batched_data)-1][0]
        diffs.append(diff)
    return diff, diffs


def main(filename: str):
    """
    Driver program for predicted profits and SMA profits

    :param filename: the name of the output-csvs
    :return y_pred, d: predicted profits and SMA profits
    """
    file = os.path.join(DATA_ROOT, 'stored_outputs/' + filename)
    batched_pred_data, flat_pred_data, batched_test_data_sma, batched_test_data_pred, flat_test_data = get_predicted(file)
    print(len(batched_pred_data[0]), len(batched_test_data_pred[0]))

    batch_eval(batched_pred_data)
    # batch_eval(batched_test_data_pred)
    print(best_buy_days)
    print(best_sell_days)
    y_pred = best_transactions_on_test_data(best_buy_days, best_sell_days, batched_test_data_pred)
    SMA_test_list = simple_moving_average(flat_test_data)
    d, d_arr = SMA_buy(SMA_test_list, batched_test_data_sma)
    print("Model profits:", y_pred, "v. SMA profits", d)
    return y_pred, d


if __name__ == '__main__':
    main('SPY_GRU_output_2020-11-19')
