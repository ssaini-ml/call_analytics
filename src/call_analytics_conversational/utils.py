from datetime import date, timedelta
from typing import List, Tuple


def day_month_list(start_date: date, end_date: date) -> List[Tuple[int, int]]:
    """
    Generates a list of tuples representing the day and month for each date
    in the range from start_date to end_date (inclusive).

    Parameters:
    -----------
    start_date : date
        The starting date of the range.
    end_date : date
        The ending date of the range.

    Returns:
    --------
    list of tuples
        A list of (day, month) tuples corresponding to each date in the range.

    Example:
    --------
    >>> from datetime import date
    >>> day_month_list(date(2023, 12, 29), date(2024, 1, 2))
    [(29, 12), (30, 12), (31, 12), (1, 1), (2, 1)]
    """
    current_date = start_date
    date_list = []
    while current_date <= end_date:
        date_list.append((current_date.day, current_date.month))
        current_date += timedelta(days=1)
    return date_list
