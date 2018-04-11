#!/usr/bin/python3

import numpy as np

N = 5000
DIM = 5

xx = np.random.random((N, DIM))

print(N)
print(DIM)

for i in range(N):
       for j in range(DIM):
              print("{} ".format(xx[i,j]), end='')
       print()


def f(var):
    return var[1]


for row in xx:
    print(f(row))

