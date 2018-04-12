#!/usr/bin/python3

import numpy as np
import math

N = 5000
DIM = 5

xx = np.random.random((N, DIM))

print(N)
print(DIM)

for i in range(N):
       for j in range(DIM):
              print("{} ".format(xx[i,j]), end='')
       print()


def f(x):
    return x[1] * x[2] + x[4]
#    return (x[0]*x[1])+((x[2]/x[3])-x[4])
#    return ((math.sin(x[0]) + (x[1]*x[2])) * ((x[3]/0.777) - (math.cos(x[4]))))


for row in xx:
    print(f(row))

