from mysql import connector  # type: ignore
import time
import datetime
from typing import Dict
from src.data.json_utils import get_key_from_json_file
from src.common.definitions import AUTH_CONFIG


class Database:
    # TODO: Consistent field-naming in database

    INSERT_TRADE = "INSERT INTO tradeTables (stockId, state, name, realEntry, realExit, createdAt, updatedAt) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    UPDATE_TRADE = "UPDATE tradeTables SET state = %s, realEntry = %s, realExit = %s, updatedAt = %s WHERE tradeId = %s"

    INSERT_STOCK = "INSERT INTO stockTables (ticker, recent_value, last_price_update, currency, volume, createdAt, updatedAt) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    UPDATE_STOCK = "UPDATE stockTables SET recent_value = %s, last_price_update = %s, volume = %s, updatedAt = %s WHERE ticker = %s"
    GET_STOCK_ID = "SELECT stockId FROM stockTables WHERE ticker = %s"

    def __init__(self):
        self.con = None
        self.cursor = None
        self.db_name = get_key_from_json_file(AUTH_CONFIG, "databaseName")
        self.db_user = get_key_from_json_file(AUTH_CONFIG, "databaseUser")
        self.db_password = get_key_from_json_file(AUTH_CONFIG, "databasePassword")
        self.db_host = get_key_from_json_file(AUTH_CONFIG, "databaseURL")

    def __enter__(self):
        self.con =  connector.connect(user=self.db_user, password=self.db_password, host=self.db_host, database=self.db_name)
        self.cursor = self.con.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor_reset()
        self.con.close()
        self.cursor = None
        self.con = None

    def cursor_reset(self):
        self.cursor.fetchall()
        self.con.commit()

    @staticmethod
    def get_timestamp():
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    def update_stock(self, stock_observer):
        timestamp = self.get_timestamp()
        self.cursor.execute(self.UPDATE_STOCK, [stock_observer.price, timestamp, 1000, timestamp, stock_observer.ticker])

    def add_stock_if_not_exists(self, stock_observer_instance):
        print(f"Adding {stock_observer_instance.ticker}")
        timestamp = self.get_timestamp()
        stock_id = self.get_id_from_stock_ticker(stock_observer_instance.ticker)
        based = [
                stock_observer_instance.ticker,
                stock_observer_instance.price,
                timestamp,
                "USD",
                1000,
                timestamp,
                timestamp
            ]
        if stock_id == 0:
            print(f"{self.INSERT_STOCK}\n{based}")
            self.cursor.execute(self.INSERT_STOCK, [
                stock_observer_instance.ticker,
                stock_observer_instance.price,
                timestamp,
                "USD",
                1000,
                timestamp,
                timestamp
            ])
            print(f"{stock_observer_instance.ticker}added with {self.cursor.lastrowid}")
            stock_id = self.cursor.lastrowid
        return stock_id

    def get_id_from_stock_ticker(self, ticker):
        self.cursor.execute(self.GET_STOCK_ID, [ticker])
        stock_id = self.cursor.lastrowid
        self.cursor_reset()
        return stock_id

    def update_trade(self, position):
        timestamp = self.get_timestamp()
        self.cursor.execute(self.UPDATE_TRADE, [
            position.current_strategy.state,
            position.get_entry(),
            position.get_exit(),
            timestamp,
            position.trade_id
        ])

    def add_trade(self, position):
        stock_id = self.add_stock_if_not_exists(position.stock)
        timestamp = self.get_timestamp()
        self.cursor.execute(self.INSERT_TRADE, [
            stock_id,
            position.current_strategy.state,
            position.current_strategy.get_name(),
            position.get_entry(),
            position.get_exit(),
            timestamp,
            timestamp
        ])
        print(f"Added {position.current_strategy.get_name()} to database")
        trade_id = self.cursor.lastrowid
        position.stock.set_stock_id(stock_id)
        return trade_id

class DBController:

    _trade_ids_to_stock_ids: Dict[int, int] = {}
    db = Database()

    @staticmethod
    def get_or_create_trade(position):
        with DBController.db as con:
            if not position.trade_id:
                return con.add_trade(position)
            else:
                return position.trade_id

    @staticmethod
    def add_trade(position):
        stock_id = DBController.get_or_create_stock(position.stock)
        position.stock.set_stock_id(stock_id)

        trade_id = DBController.get_or_create_trade(position)
        position.set_trade_id(trade_id)
        DBController._trade_ids_to_stock_ids[trade_id] = position
        return trade_id

    @staticmethod
    def get_or_create_stock(stock_observer):
        with DBController.db as con:
            return con.add_stock_if_not_exists(stock_observer)

    @staticmethod
    def update_trades():
        with DBController.db as con:
            for trade_id in DBController._trade_ids_to_stock_ids:
                position = DBController._trade_ids_to_stock_ids[trade_id]
                con.update_trade(position)
                con.update_stock(position.stock)


