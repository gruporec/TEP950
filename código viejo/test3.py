import pandas as pd
import itertools

x0=[]
for x in range(5):
    x0.append([])
    for y in range(5):
        x0[x].append([])
        for z in range(5):
            x0[x][y].append(0)

with open('x0.txt', 'w') as f:
    f.write(str(x0))