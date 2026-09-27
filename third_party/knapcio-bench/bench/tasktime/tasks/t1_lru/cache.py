from collections import OrderedDict
class LRU:
    def __init__(self, cap): self.cap=cap; self.d=OrderedDict()
    def get(self,k):
        if k not in self.d: return None
        return self.d[k]           # bug: does not refresh recency
    def put(self,k,v):
        self.d[k]=v
        if len(self.d)>self.cap: self.d.popitem(last=True)   # bug: evicts newest
