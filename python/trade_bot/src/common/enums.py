from enum import Enum

class EnumFromStrMixin:

    @classmethod
    def from_str(cls, string: str):
        for enum_type in cls:   # type: ignore
            if enum_type.name.lower() == string:
                return enum_type
        raise NotImplementedError(f"No enum matching {string}")

class DataGranularity(EnumFromStrMixin, Enum):
    SECONDS = 1
    MINUTES = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800

    def __str__(self) -> str:
        # Satisfy various api-formats
        return self.name.lower()


class PositionState(Enum):
    OPEN = "OPEN"
    WAIT_BUY = "WAIT_BUY"
    WAIT_SELL = "WAIT_SELL"
    CLOSED = "CLOSED"