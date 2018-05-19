#!/usr/bin/python3

import numpy as np
import math
import itertools

VAR = 16
N = 2**VAR

def xor(o1, o2):
    return (o1 and not o2) or (not o1 and o2)

def f(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    v5 = x[5]
    v6 = x[6]
    v7 = x[7]
    v8 = x[8]
    v9 = x[9]
    v10 = x[10]
    v11 = x[11]
    v12 = x[12]
    v13 = x[13]
    v14 = x[14]
    v15 = x[15]
    return (((xor(v10, v11 or v12) or not v0) or xor(v1, v2)) and (xor((v3 and v4), (not v5 or v6)))) and (((not v7) and xor(v8, not v9)) or (xor(not v13, v14 or v15)))
    # return ((not(v0) or xor(v1, v2)) and (xor((v3 and v4), (not v5 or v6)))) and (((not v7) and xor(v8, not v9)) or (xor(not v1, v0 or v7)))
#    return x[0] and x[1] or not x[2] or x[3] and not x[4]


table = np.asarray(list(itertools.product([0, 1], repeat=VAR)))

print(N)
print(VAR)

for i in range(N):
    for j in range(VAR):
        print("{} ".format(table[i,j]), end='')
    print(f(table[i])*1, end='')
    print()


