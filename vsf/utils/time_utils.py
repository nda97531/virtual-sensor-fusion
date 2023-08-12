import time
from datetime import datetime, timezone


def datetime_2_timestamp(dt: datetime, tz: int) -> int:
    """
    Convert a datetime object to timestamp

    Args:
        dt: datetime object
        tz: timezone of the datetime object

    Returns:
        timestamp in millisecond
    """
    dt = dt.replace(tzinfo=timezone.utc)
    timestamp = dt.timestamp() - tz * 3600
    return round(timestamp * 1000)


def str_2_timestamp(str_time: str, tz: int, str_format: str = '%Y/%m/%d %H:%M:%S') -> int:
    """
    Convert datetime as string to timestamp (msec)

    Args:
        str_time: string of datetime value
        tz: timezone of string value
        str_format: datetime string format, default is 'yyyy/mm/dd HH:MM:SS'

    Returns:
        timestamp in millisecond
    """
    dt = datetime.strptime(str_time, str_format)
    ts = datetime_2_timestamp(dt, tz)
    return ts


def timestamp_2_datetime(timestamp: int, tz: int) -> datetime:
    """
    Convert timestamp (millisecond) to a datetime object

    Args:
        timestamp: timestamp in millisecond
        tz: timezone to convert

    Returns:
        a datetime object
    """
    return datetime.utcfromtimestamp(timestamp / 1000 + tz * 3600)


def timestamp_2_str(timestamp: int, tz: int, str_format: str = '%Y/%m/%d %H:%M:%S') -> str:
    """
    Convert timestamp (msec) to string datetime

    Args:
        timestamp: timestamp in millisecond
        tz: timezone of the output string datetime
        str_format: format of the output string, default is 'yyyy/mm/dd HH:MM:SS'

    Returns:
        string datetime
    """
    dt = timestamp_2_datetime(timestamp, tz)
    str_time = dt.strftime(str_format)
    return str_time


class TimeThis:
    def __init__(self, op_name: str = 'operation', printer=print, **kwargs):
        """
        Measure running time of a block of code

        Args:
            op_name: just some text to print
            printer: print function
            **kwargs: keyword args for the print function
        """
        self.op_name = op_name
        self.printer = printer
        self.printer_kwargs = kwargs

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.printer(f'Elapsed time for {self.op_name}: {time.time() - self.start_time}(s)', **self.printer_kwargs)
