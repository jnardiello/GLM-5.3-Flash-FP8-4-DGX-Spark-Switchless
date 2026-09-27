def find_duplicates(items):
    out=[]
    for i,a in enumerate(items):
        for b in items[:i]:
            if a==b and a not in out: out.append(a)
    return out
