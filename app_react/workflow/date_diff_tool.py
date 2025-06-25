import os
from typing import TypedDict, List, Optional, Annotated
import operator
from pydantic import BaseModel
from langchain_core.tools import tool
from functools import partial
from typing import List



from datetime import datetime
from langchain.tools import tool

@tool
def calculate_date_diff(
    start_date: str,
    end_date: str
) -> int:
    """
    Calculates the number of days between two dates.
    Dates must be in formats like 'YYYY-MM-DD' or 'MM/DD/YYYY'.
    Returns a positive integer if end_date > start_date.
    """
    print(f"📆 Date Diff Tool Called:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date:   {end_date}")
    def parse(date_str):
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%-m/%-d/%Y", "%#m/%#d/%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized date format: {date_str}")

    start = parse(start_date)
    end = parse(end_date)

    if not start or not end:
        raise ValueError("Both start_date and end_date must be valid")
    print(f"   ➤ Days Between: {(end - start).days}")
    return (end - start).days
