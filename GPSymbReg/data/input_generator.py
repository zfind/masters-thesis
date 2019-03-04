#!/usr/bin/python3

import numpy as np
import math


N = 10000
DIM = 5
SCALE = 5


def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def sin(x):
    return math.sin(x)

def cos(x):
    return math.cos(x)

def f(x):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    return mul(add(sin(x0), mul(x1, x2)), sub(div(x3, 0.777), cos(x4)))


if __name__ == "__main__":

    xx = np.random.random((N, DIM)) * SCALE

    print(N)
    print(DIM)

    for i in range(N):
        for j in range(DIM):
            print("{} ".format(xx[i,j]), end='')
        print(f(xx[i]), end='')
        print()
