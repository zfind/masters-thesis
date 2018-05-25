//
// Created by zac on 04.01.18..
//

#ifndef NET_H
#define NET_H

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Common.h"
#include "Dataset.h"
#include "Matrix.h"

using namespace std;


#define BLOCK_SIZE 16


extern "C"
__global__ void mulMatrixKernel(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC);

extern "C"
__device__ inline double GetElement(double *matrix, int XX, int YY, int x, int y);

extern "C"
__device__ inline void SetElement(double *matrix, int XX, int YY, int x, int y, double val);


class Net {
private:

    vector<int> layers;

    double *h_output;
    double *h_new_output;

    double *d_output;
    double *d_new_output;

    double *d_datasetInput;
    double *d_datasetOutput;

public:

    Net(vector<int> layers, Dataset &dataset);

    ~Net();

    double evaluate(double weights[], Dataset &dataset);

    double evaluateGPU(double *weights, Dataset &dataset);

    int getWeightsCount();

private:

    double sigmoid(double x);

    void mulMatrix(Matrix mA, int rA, int cA, Matrix mB, int rB, int cB, Matrix mC, int rC, int cC);

    void mulMatrix(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC);

    inline double GetElement(double *matrix, int XX, int YY, int x, int y);

    inline void SetElement(double *matrix, int XX, int YY, int x, int y, double val);

};


#endif //NET_H
