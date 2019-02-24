#!/usr/bin/python3

import numpy as np
import math

N = 10000
DIM = 5
SCALE = 5

xx = np.random.random((N, DIM)) * SCALE

print(N)
print(DIM)

for i in range(N):
       for j in range(DIM):
              print("{} ".format(xx[i,j]), end='')
       print()


def f(x):
    return ((math.sin(x[0]) + (x[1]*x[2])) * ((x[3]/0.777) - (math.cos(x[4]))))
#    return x[1] * x[2] + x[4]
#    return (x[0]*x[1])+((x[2]/x[3])-x[4])


for row in xx:
    print(f(row))

