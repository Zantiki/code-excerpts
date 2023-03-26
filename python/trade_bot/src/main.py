
from concurrent.futures import ThreadPoolExecutor
from src.trading.position import Position
from src.data.database import DBController
from src.trading.trade_selector import TradeSelector
from src.trading.results import print_return_table
from src.runtime.daemon_threads import InternalClock, TimeFrame
from src.data.json_utils import get_key_from_json_file
from src.common.definitions import CONFIG
from src.common.enums import DataGranularity
from src.backtesting.runtime_test import test

def trade_wrapper(position):
    try:
        position.do_strategy()
        position.controller.remove_instance(position.stock.ticker)
    except Exception as e:
        return e
    return position


def setup_config():
    concurrent_threads = get_key_from_json_file(CONFIG, "concurrent_threads")
    enable_db = get_key_from_json_file(CONFIG, "sync_db")
    enable_testing = get_key_from_json_file(CONFIG, "enable_testing")
    test_parameters = None if not enable_testing else get_key_from_json_file(CONFIG, "testing_params")
    data_granularity = DataGranularity.from_str(get_key_from_json_file(CONFIG, "data_granularity"))
    update_wait = get_key_from_json_file(CONFIG, "update_wait")
    return concurrent_threads, enable_db, test_parameters, data_granularity, update_wait


def trade():

    concurrent_threads, enable_db, test_parameters, data_granularity, update_wait = setup_config()
    time_frame = None
    if not test_parameters:
        selector = TradeSelector()
        tickers_to_trade = selector.get_tickers_to_trade()
        trades_to_execute = selector.get_trades_position_size_by_risk(tickers_to_trade)
    else:
        trades_to_execute, time_frame = test(testing_params=test_parameters, simple=True)

    clock = InternalClock(time_frame=time_frame,
                          db_sync=enable_db,
                          step_size=data_granularity,
                          realtime_wait=update_wait)

    with ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
        result_threads = {
            # ticker: return
        }
        positions = []
        for ticker in trades_to_execute:
            abs_position_size = trades_to_execute[ticker]["cash"]
            starting_strategy = trades_to_execute[ticker]["strategy"]
            strategy_args = trades_to_execute[ticker]["strategy_args"]
            required_api_type = trades_to_execute[ticker]["api"]

            api_instance = required_api_type()
            position = Position(
                    ticker,
                    starting_strategy,
                    abs_position_size,
                    strategy_args,
                    required_api_type
                )
            if enable_db:
                DBController.add_trade(position)
            clock.register_api(api_instance)

            positions.append(positions)

        clock.start()
        for position in positions:
            result_threads[position.stock.ticker] = executor.submit(trade_wrapper, position)

        results = {}
        for ticker in result_threads:
            position = result_threads[ticker].result()
            results[ticker] = position
            print(results[ticker])
            print(f"Collected Result from {ticker}, state is {position.current_strategy.state}")

        print("Results are collected")
        clock.join()
        print_return_table(results, selector.spendable_money)


if __name__ == '__main__':
    trade()
