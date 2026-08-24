def build( rows ,limit = 10 ):
    out=[]
    for r in rows[:limit] :
        out.append( r )
    return out
