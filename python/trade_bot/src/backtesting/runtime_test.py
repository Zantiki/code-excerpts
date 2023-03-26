from src.data.api_controllers import TestController, AlpacaApiController
from src.data.api_implementations import AlpacaTestApi, TestApi
from src.trading.trade_selector import TradeSelector
from src.trading.strategies.strategy_implementations import Strategy
from src.runtime.daemon_threads import TimeFrame


def test(testing_params, simple = False):
    # "testing_params": {
    #     "positions": [
    #         {
    #             "ticker": "AAPL",
    #             "percentage": 33.00,
    #             "strategy": "simple_enter_exit"
    #         }
    #
    #     ],
    #     "start_period": "",
    #     "end_period": ""
    # }
    # Todo: schema for test-parameters
    tradeable_tests = {}
    time_frame = TimeFrame(upper=testing_params["start_period"], lower=testing_params["end_period"])
    api_type = TestController if simple else AlpacaApiController
    total_cash = testing_params["start_value"]
    selector = TradeSelector()
    for position in testing_params["positions"]:
        ticker = position["ticker"]
        strat = Strategy.get_type_from_str(position["strategy"])
        size_of_position = position["percentage"]
        tradeable_tests[ticker] = {
            "api": api_type,
            "strategy": strat,
            "cash": total_cash * size_of_position,
            "strategy_args": selector.get_strategy_arguments(strat, ticker)
        }

    return tradeable_tests, time_frame

