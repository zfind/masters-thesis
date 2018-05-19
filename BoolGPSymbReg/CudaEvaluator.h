//
// Created by zac on 01.05.18..
//

#ifndef BOOLGPSYMBREG_CUDAEVALUATOR_H
#define BOOLGPSYMBREG_CUDAEVALUATOR_H

#include <iostream>
#include <vector>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <chrono>
#include "Constants.h"

using namespace std;

class CudaEvaluator {
public:
    CudaEvaluator(uint NUM_SAMPLES, uint INPUT_DIMENSION, uint MAX_PROG_SIZE,
                  vector<vector<bool>> &datasetInput, vector<bool> &datasetOutput);

    ~CudaEvaluator();

    uint h_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                      vector<BOOL_TYPE> &result);

    uint d_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                      vector<BOOL_TYPE> &result);

private:

    BOOL_TYPE h_evaluateIndividual(char *postfixMem, uint PROG_SIZE, uint MEM_SIZE,
                                std::vector<BOOL_TYPE> &input);

private:
    int NUM_SAMPLES;
    int INPUT_DIMENSION;
    int MAX_PROG_SIZE;

    std::vector<std::vector<BOOL_TYPE>> datasetInput;
    std::vector<BOOL_TYPE> datasetOutput;

    uint *d_program;
    BOOL_TYPE *d_datasetInput;
    BOOL_TYPE *d_datasetOutput;
    BOOL_TYPE *d_resultOutput;
    uint *d_resultFitness;
};

__global__ void d_evaluateIndividual(uint *d_program,
                                     BOOL_TYPE *d_datasetInput,
                                     BOOL_TYPE *d_datasetOutput,
                                     BOOL_TYPE *d_resultOutput,
                                     uint *d_resultFitness,
                                     int N, int DIM, int PROG_SIZE);


#endif //BOOLGPSYMBREG_CUDAEVALUATOR_H
