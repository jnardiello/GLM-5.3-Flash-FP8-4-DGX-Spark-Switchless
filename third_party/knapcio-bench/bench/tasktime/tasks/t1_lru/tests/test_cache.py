from cache import LRU
def test_evicts_least_recent():
    c=LRU(2); c.put('a',1); c.put('b',2); c.get('a'); c.put('c',3)
    assert c.get('b') is None and c.get('a')==1 and c.get('c')==3
def test_update_keeps_size():
    c=LRU(2); c.put('a',1); c.put('a',2); c.put('b',3); c.put('c',4)
    assert c.get('a') is None and c.get('c')==4
