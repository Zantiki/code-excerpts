import yfinance as yf
import json
import os
from datetime import datetime
from src.definitions import DATA_ROOT, CONTENT_ROOT
import pandas as pd

today = datetime.today().strftime('%Y-%m-%d')

def read_json():

    """
    Read and parse the data-generation paramters from the data.json file.
    These paramaters are primary used to add/remove features when fetching from yfinance.

    :return: tickers, start, end, period, interval, auto_adjust, prepost, time_specific, threads
    """

    period = ""
    start = ""
    end = ""
    data_json = os.path.join(CONTENT_ROOT, "src/data/data.json")
    with open(data_json) as f:
        json_data = json.load(f)

        time_specific = json_data["specify_period"]

        print(json_data)
        tickers = " ".join([i for i in json_data["tickers"]]).strip()
        if time_specific:
            start = json_data["spec_period"][0]["start_date"]
            end = json_data["spec_period"][1]["end_date"]
        else:
            period = json_data["time_period"]

        interval = json_data["time_frequency"]

        auto_adjust = json_data["auto_adjust"]

        prepost = json_data["prepost"]

        threads = json_data["threads"]
    return tickers, start, end, period, interval, auto_adjust, prepost, time_specific, threads


tickers, start, end, period, interval, auto_adjust, prepost, time_specific, threads = read_json()


# date: YYYY-MM-DD
def use_history():
    """
    Fetch data from yfinance using the specified paramaterers in data.json and generate the data_csvs directory.

    :return: None
    """
    for ticker in tickers.split(" "):
        curr_ticker = yf.Ticker(ticker)

        if time_specific:
            data = curr_ticker.history(period=period, interval=interval, start=start, end=end, prepost=prepost, auto_adjust=auto_adjust,
                                       actions=True)
            filename = ticker + "_" + start + "_" + end + "_" + today + ".csv"
        else:
            data = curr_ticker.history(period=period, interval=interval, prepost=prepost, auto_adjust=auto_adjust,
                                       actions=True)
            filename = ticker + "_" + period + "_" + today + ".csv"

        try:
            data_file = os.path.join(DATA_ROOT, filename)
            data.to_csv(data_file)
        except FileExistsError:
            print("This file already exists.")


if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    use_history()
