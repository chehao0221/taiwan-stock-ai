import pandas_market_calendars as mcal
from datetime import datetime

def is_market_open(market="US"):
    today = datetime.now().strftime('%Y-%m-%d')

    if market == "US":
        cal = mcal.get_calendar('NYSE')
    elif market == "TW":
        cal = mcal.get_calendar('XTAI')
    else:
        raise ValueError("market 必須是 'US' 或 'TW'")

    schedule = cal.schedule(start_date=today, end_date=today)
    return not schedule.empty
