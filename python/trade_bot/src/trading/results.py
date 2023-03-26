from typing import Dict

from src.data.api_controllers import StockObserver
from src.trading.position import Position


def get_value_of_exit(position: Position) -> float:
    return position.get_shares_owned() * position.get_exit()


def get_value_of_enter(position: Position) -> float:
    return position.get_shares_owned() * position.get_entry()


def get_return_of_position(position: Position) -> float:
    return get_value_of_exit(position) - get_value_of_enter(position)


def get_return_percentage_from_position(position: Position) -> float:
    return get_value_of_exit(position) / get_value_of_enter(position)


def print_return_table(position_dict: Dict[str, Position], initial_total_value: float):
    print("\n------- TRADE RESULTS ---------")
    total_return = 0
    for ticker in position_dict:
        position = position_dict[ticker]
        percentage = get_return_percentage_from_position(position)
        abs_return = get_return_of_position(position)
        print(f"{ticker} gave a return of {percentage}%,"
              f" {abs_return} {position.currency}")
        total_return += abs_return

    print(f"\nTotal return percentage is {total_return / initial_total_value}% earned {total_return}")
