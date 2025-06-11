from datetime import datetime
import re

def create_date_from_string(data):
    """Creates a datetime object from a string, robust to filenames with extra tokens like 'RunX'."""
    try:
        # Split and filter out non-date tokens (like 'RunX')
        data_list = [token for token in str(data).split() if not re.match(r"Run\\d+", token)]
        # Find year (4-digit number)
        year = next(int(token) for token in data_list if token.isdigit() and len(token) == 4)
        # Find month (3-letter abbreviation)
        month_str = next(token for token in data_list if len(token) == 3 and token.isalpha())
        # Find day (number, possibly with comma)
        day = next(int(token.strip(",")) for token in data_list if token.strip(",").isdigit() and len(token.strip(",")) <= 2)
        month = datetime.strptime(month_str, "%b").month
        date_obj = datetime(year, month, day).date()
        return date_obj
    except Exception as e:
        print(f"Error creating date from list: {e}")
        return None
