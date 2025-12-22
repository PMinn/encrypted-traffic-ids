import datetime


def timestamp_to_unix(timestamp: int | float) -> float:
    """
    將時間轉換為 Unix 時間戳
    Args:
        timestamp (int): 時間
    Returns:
        float: Unix 時間戳
    """
    timestamp_datetime = datetime.datetime.fromtimestamp(int(timestamp))
    unix_timestamp = (
        timestamp_datetime - datetime.datetime(1970, 1, 1)
    ).total_seconds()
    return unix_timestamp
