import numpy as np

def build(vertices):
    leaf = []
    v_len = len(vertices)
    def tree_construction(tl, tr, v):
        print(v)
        if(tl == tr):
            leaf.append(v)
            return
        
        tm = (tl+tr)>>1
        tree_construction(tm+1,tr,2*v+1)
        tree_construction(tl,tm,2*v)

    tree_construction(1,v_len+1,1)
    return leaf

def next_2_power(size_vertex):
    if(size_vertex&(size_vertex-1) == 0):
        return size_vertex
    else:
        count = 0
        while(size_vertex):
            size_vertex>>=1
            count += 1

        return 1<<(count)


V = np.arange(start=1,stop=5, step=1, dtype=int)
leafs = build(V)
print(leafs)


