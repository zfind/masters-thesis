#!/usr/bin/python3


import numpy as np
import math
import itertools
import random


VAR = 16
N = 2**VAR


def lxor(o1, o2):
    return (o1 and not o2) or (not o1 and o2)

def land(o1, o2):
    return o1 and o2

def lor(o1, o2):
    return o2 or o2

def lnot(o1):
    return not o1


def f5(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    return lor( land(v0, v1),
                lxor( lor(v2, v3),
                        lnot(v4)))


def f6(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    v5 = x[5]
    return lor( lxor( land(v0, v1),
                        lnot(v2)),
                land( lor(v3, v4),
                        v5))


def f7(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    v5 = x[5]
    v6 = x[6]
    return lor( lxor( land(v0,v1),
                        lnot(v2)),
                land( lor(v3, v4),
                        lxor(v5, v6)))


def f8(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    v5 = x[5]
    v6 = x[6]
    v7 = x[7]
    return lor( lxor( land( v0, v1),
                        lor( v2, v3)),
                land( lor(v4, v5),
                        lxor(v6, v7)))


def f9(x):
    v0 = x[0]
    v1 = x[1]
    v2 = x[2]
    v3 = x[3]
    v4 = x[4]
    v5 = x[5]
    v6 = x[6]
    v7 = x[7]
    v8 = x[8]
    return lor( lxor( land( lor(v0,v1),
                            v2),
                        lor(v3, v4)),
                land( lor(v5, v6),
                        lxor(v7, v8)))


def f10(x):
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
    return lor( lxor( land( lor(v0, v1),
                            lor(v2, v3)),
                        lor(v4,v5)),
                land( lor(v6, v7),
                        lxor(v8, v9)))


def f11(x):
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
    return lor(lxor( land( lor(v0, v1),
                            lor(v2, v3)),
                    lor( lxor(v4, v5),
                            v6)),
                land( lor(v7, v8),
                        lxor(v9, v10)))


def f12(x):
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
    return lor( lxor( land( lor( v0, v1),
                            lor(v2, v3)),
                        lor(lxor(v4, v5),
                            lor(v6, v7))),
                land( lor( v8, v9),
                        lxor(v10, v11)))


def f13(x):
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
    return lor( lxor( land( lor(v0, v1),
                            lor(v2, v3)),
                        lor(lxor(v4, v5),
                            lor(v6, v7))),
                land( lor( lxor(v8, v9),
                            v10),
                        lxor( v11, v12)))


def f14(x):
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
    return lor( lxor( land( lor(v0, v1),
                            lor(v2, v3)),
                        lor( lxor(v4, v5),
                            lor( v6, v7))),
                land( lor( lxor( v8, v9),
                            lor(v10, v11)),
                        lxor(v12, v13)))


def f_random(x):
    return bool(random.getrandbits(1))


if __name__ == "__main__":

    table = np.asarray(list(itertools.product([0, 1], repeat=VAR)))

    print(N)
    print(VAR)

    f = f_random;

    for i in range(N):
        for j in range(VAR):
            print("{} ".format(table[i,j]), end='')
        print(f(table[i])*1, end='')
        print()
