import datetime as dt
def prorated(amount, start, end, period_end):
    """Charge for [start, end) proportionally to the billing period [period_start, period_end) of one calendar month."""
    period_start = period_end.replace(day=1)
    days = (period_end - period_start).days
    used = (end - start).days
    return round(amount * used / 30, 2)      # bug: hardcoded 30
