from src.data.api_controllers import TestController
from src.trading.strategies.strategy_implementations import SimpleEnterAndExitLong


class TradeSelector:


    def __init__(self):
       self.spendable_money = 10000.00

    def get_strategy_arguments(self, strategy_type, ticker):
        """
        Some magic involving ml and whatnot
        """
        return {
                    "exit_gain": 1.1,
                    "entry_gain": 0.9
                }

    def get_trades_position_size_by_risk(self, tickers_to_trade):
        ticker_strategy_cash = {
            # Ticker: {strategy, cash}
        }
        for ticker in tickers_to_trade:
            ticker_strategy_cash[ticker] = {
                "api": TestController,
                "strategy": SimpleEnterAndExitLong,
                "cash": self.spendable_money / len(tickers_to_trade),
                "strategy_args": self.get_strategy_arguments(SimpleEnterAndExitLong, ticker)
            }
        return ticker_strategy_cash

    def get_tickers_to_trade(self):
        return ["AAPL", "GOOGL", "AMT"]