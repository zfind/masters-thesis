//
// Created by zac on 04.01.18..
//

#ifndef NEURALNET_NET_H
#define NEURALNET_NET_H

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

__global__ void mulMatrixKernel(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC);


class Net {
private:
    vector<int> layers;
    double *h_output;
    double *h_new_output;

    double *d_datasetInput, *d_datasetOutput;
    double *d_output;
    double *d_new_output;



public:

    Net(vector<int> layers, Dataset &dataset);

    double evaluate(double weights[], Dataset& dataset);

    double evaluateParallel(double weights[], Dataset & dataset);

    int getWeightsCount();

    double sigmoid(double x);

    void mulMatrix(Matrix mA, int rA, int cA, Matrix mB, int rB, int cB, Matrix mC, int rC, int cC);

    void mulMatrix(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC);

    ~Net();

};


#endif //NEURALNET_NET_H
