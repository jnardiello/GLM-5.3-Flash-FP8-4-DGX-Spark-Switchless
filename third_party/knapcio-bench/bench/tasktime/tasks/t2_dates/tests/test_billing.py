import datetime as dt
from billing import prorated
def test_feb():
    assert prorated(28.0, dt.date(2026,2,1), dt.date(2026,2,15), dt.date(2026,3,1)) == 14.0
def test_full_month():
    assert prorated(31.0, dt.date(2026,1,1), dt.date(2026,2,1), dt.date(2026,2,1)) == 31.0
