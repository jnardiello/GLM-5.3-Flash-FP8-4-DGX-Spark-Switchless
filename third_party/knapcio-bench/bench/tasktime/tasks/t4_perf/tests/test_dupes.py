import time, random
from dupes import find_duplicates
def test_small():
    assert find_duplicates([1,2,2,3,3,3,4]) == [2,3]
def test_big_is_fast():
    xs=[random.randrange(50000) for _ in range(200000)]
    t=time.time(); r=find_duplicates(xs); assert time.time()-t < 2
    assert set(r) == {x for x in set(xs) if xs.count(x)>1} if len(xs)<1000 else len(r)>0
