#ifndef GPSYMBREG_CUDAEVALUATOR_H
#define GPSYMBREG_CUDAEVALUATOR_H

#include <iostream>
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

using namespace std;




class CudaEvaluator {
public:
    CudaEvaluator(uint NUM_SAMPLES, uint INPUT_DIMENSION, uint MAX_PROG_SIZE,
                  vector<vector<double>> &datasetInput, vector<double> &datasetOutput);

    ~CudaEvaluator();

    double h_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                      vector<double> &result);

    double d_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                      vector<double> &result);

    char *postfixMemPinned;

private:

    double h_evaluateIndividual(char *postfixMem, uint PROG_SIZE, uint MEM_SIZE,
                                std::vector<double> &input);

private:
    int NUM_SAMPLES;
    int INPUT_DIMENSION;
    int MAX_PROG_SIZE;

    std::vector<std::vector<double>> datasetInput;
    std::vector<double> datasetOutput;

    uint *d_program;
    double *d_programConst;
    double *d_datasetInput;
    double *d_datasetOutput;
    double *d_globalStack;
    double *d_resultOutput;
    double *d_resultFitness;
};


__global__ void d_evaluateIndividual(uint *d_program,
                                     double *d_programConst,
                                     double *d_datasetInput,
                                     double *d_datasetOutput,
                                     double *d_resultOutput,
                                     double *d_globalStack,
                                     double *d_resultFitness,
                                     int N, int DIM, int PROG_SIZE);


#endif //GPSYMBREG_CUDAEVALUATOR_H
