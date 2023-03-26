from typing import Dict
from threading import Thread
from time import sleep
from datetime import datetime, timedelta

from src.common.enums import DataGranularity
from src.data.database import DBController
from src.data.api_implementations import AbstractApi


class TimeFrame:

    def __init__(self, upper, lower):
        self.upper_timestamp = upper
        self.lower_timestamp = lower


class InternalClock(Thread):

    _realtime_wait = 0
    _steps = 0
    _kill = False
    _registered_apis: Dict[str, AbstractApi] = {}

    def __init__(self, time_frame, db_sync=True, realtime_wait=5, step_size=DataGranularity(60)):
        self._realtime_wait = realtime_wait
        self.start_time = time_frame.upper_timestamp
        self.end_time = time_frame.lower_timestamp
        self.step_size = step_size
        self.db_sync = db_sync
        super().__init__()

    @staticmethod
    def register_api(api_instance):

        if api_instance.__name__ not in InternalClock._registered_apis:
            InternalClock._registered_apis[api_instance.__name__] = api_instance

    @staticmethod
    def signal_update(timestamp=None):
        for api_instance in InternalClock._registered_apis:
            api_instance.update_prices(timestamp)

    @staticmethod
    def signal_shutdown():
        for api_instance in InternalClock._registered_apis:
            api_instance.shutdown_stock_observers()

    @staticmethod
    def step(step_size_value):
        InternalClock._steps += step_size_value

    def get_timestamp(self):
        return self.start_time + timedelta(seconds=self._steps)

    def run(self) -> None:
        while not InternalClock._kill and (self.end_time and self.get_timestamp() < self.end_time):
            if self.db_sync:
                DBController.update_trades()

            # To run in real time
            if self._realtime_wait:
                sleep(self._realtime_wait)
                InternalClock.step(self._realtime_wait)
                InternalClock.signal_update()
            else:
                # To simulate time passing
                InternalClock.step(self.step_size.value)
                InternalClock.signal_update(self.get_timestamp())

            InternalClock._kill = (self.end_time and self.get_timestamp() > self.end_time)




