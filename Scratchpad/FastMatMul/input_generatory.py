#!/usr/bin/python3

import numpy as np
import math
import sys

DIMX = int(sys.argv[1])
DIMY = int(sys.argv[2])
DIMZ = int(sys.argv[3])
SCALE = 5

A = np.random.random((DIMX, DIMY)) * SCALE
fA = open('matA.in','w')
fA.write(str(DIMX)+ "\n")
fA.write(str(DIMY) + "\n")
for i in range(DIMX):
    for j in range(DIMY):
        fA.write("{} ".format(A[i,j]))
    fA.write("\n")
fA.close()

B = np.random.random((DIMY, DIMZ)) * SCALE
fB = open('matB.in','w')
fB.write(str(DIMY)+ "\n")
fB.write(str(DIMZ)+ "\n")
for i in range(DIMY):
    for j in range(DIMZ):
        fB.write("{} ".format(B[i,j]))
    fB.write("\n")
fB.close()

C = np.dot(A, B)
fC = open('matC.in','w')
fC.write(str(DIMX)+ "\n")
fC.write(str(DIMZ)+ "\n")
for i in range(DIMX):
    for j in range(DIMZ):
        fC.write("{} ".format(C[i,j]))
    fC.write("\n")

