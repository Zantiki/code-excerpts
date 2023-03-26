import os
import re
import pandas as pd
from src.definitions import DATA_ROOT


def purge_csvs():
    """
    Delete the entire data_csvs directory, used when regenerating the data-set
    """

    if os.path.exists(DATA_ROOT):
        os.rmdir(DATA_ROOT)
    else:
        os.mkdir(DATA_ROOT)


def add_stock_variance(csv: str):
    """
    Calculate, append and save a csv file with the calculated variance. The new file has the same name as the
    one in the argument, with _var at the end.

    :param csv: Original stock-csv path
    :return: None
    """
    df = pd.read_csv(csv)
    variance = [round(((s_high - s_low) / s_open) * 100, 2) for
                s_low, s_high, s_open in zip(df["Low"], df["High"], df["Open"])]
    df["Variance"] = variance
    df = df.set_index("Date")
    new_csv_path = csv.split(".")[0] + "_var.csv"
    df.to_csv(new_csv_path)


def append_features_from_csv(csv_to_append: str, csv_original: str):
    """
    Given a time-indexed csv of equal length of a specified stock csv, will append the first csv to the stock file.

    :param csv_to_append: path to the data-csv to append
    :param csv_original: stock-csv path to append to.
    :return: None
    """
    joined_root = "../../data_csvs/exploration"

    if not os.path.exists(joined_root):
        os.mkdir(joined_root)
    original_name = csv_original.split("/")[-1].split(".csv")[0]
    to_append_name = csv_to_append.split("/")[-1].split(".csv")[0]

    df_to_append = pd.read_csv(csv_to_append).drop("Date")
    df_original = pd.read_csv(csv_original)

    renamed_features = {}
    for column in df_to_append.columns:
        renamed_features[column] = column + "_" + to_append_name

    df_to_append = df_to_append.rename(columns=renamed_features)
    df_joined = df_original.join(df_to_append, lsuffix='_caller', rsuffix='_other')
    df_joined = df_joined.set_index("Date")
    df_joined.to_csv("%s/%s_%s_joined.csv" % (joined_root, original_name, to_append_name))


def join_stock_csvs(csv1: str, csv2: str):
    """
    Will join two time-indexed stock csvs into one.

    :param csv1: the first stock-csv path
    :param csv2: the second stock-csv path
    :return: None
    """
    joined_root = os.path.join(DATA_ROOT, "exploration")
    if not os.path.exists(joined_root):
        os.mkdir(joined_root)

    csv_name1 = csv1.split("/")[-1]
    csv_name2 = csv2.split("/")[-1]
    ticker1 = csv_name1.split("_")[0]
    ticker2 = csv_name2.split("_")[0]

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df2 = df2.drop(columns="Date")
    renamed1 = {}
    renamed2 = {}
    for column in df1.columns:
        if column == "Date":
            continue
        renamed1[column] = column + "_" + ticker1

    for column in df2.columns:
        renamed2[column] = column + "_" + ticker2

    df1 = df1.rename(columns=renamed1)
    df2 = df2.rename(columns=renamed2)

    df_joined = df1.join(df2, lsuffix='_caller', rsuffix='_other')
    df_joined = df_joined.set_index("Date")

    df_joined.to_csv("%s/%s_%s_joined.csv" % (joined_root, ticker1, ticker2))


if __name__ == "__main__":
    """
    A simple main method that automatically resolves a varied set of stock-data to produce a diverse range of 
    joined data to use for feature-exploration.
    """
    amt= ""
    spy = ""
    vix = ""
    aapl = ""

    data_dir = str(DATA_ROOT)

    data_files = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT) if os.path.isfile(os.path.join(DATA_ROOT, f))]

    for file in data_files:

        if re.match(".*AAPL.*", file):
            aapl = file
        if re.match(".*AMT.*", file):
            amt = file
        if re.match(".*SPY.*", file):
            spy = file
        if re.match(".*VIX.*", file):
            vix = file

    join_stock_csvs(spy, amt)
    join_stock_csvs(amt, aapl)
    join_stock_csvs(vix, spy)
    add_stock_variance(aapl)