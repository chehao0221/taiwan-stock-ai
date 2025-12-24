import pandas_market_calendars as mcal
from datetime import datetime
from zoneinfo import ZoneInfo

def _get_today(market: str) -> str:
    """
    回傳該市場「當地時區」的今天日期（YYYY-MM-DD）
    """
    if market == "TW":
        tz = ZoneInfo("Asia/Taipei")
        cal = "XTAI"
    elif market == "US":
        tz = ZoneInfo("US/Eastern")
        cal = "NYSE"
    else:
        raise ValueError("market 必須是 'TW' 或 'US'")

    today = datetime.now(tz).strftime("%Y-%m-%d")
    return today, cal

def is_market_open(market: str) -> bool:
    """
    判斷指定市場今天是否為交易日（自動處理週末與官方假日）
    """
    today, cal_name = _get_today(market)
    cal = mcal.get_calendar(cal_name)
    schedule = cal.schedule(start_date=today, end_date=today)
    return not schedule.empty
