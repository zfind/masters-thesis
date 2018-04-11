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
    CudaEvaluator(int N, int DIM, int MAX_PROG_SIZE, vector<vector<double>> &input);

    ~CudaEvaluator();

    void evaluate(vector<uint> &program, vector<double> &programConst,
                  vector<vector<double>> &input,
                  vector<double> &result);

private:
    int N;
    int DIM;
    int MAX_PROG_SIZE;

    uint *d_program;
    double *d_programConst;
    double *d_input;
    double *d_output;
    double *d_stack;
};

__global__ void evaluateParallel(uint *d_program,
                                 double *d_programConstant,
                                 double *d_input,
                                 double *d_output,
                                 double *d_stack,
                                 int N, int DIM, int prog_size);


#endif //GPSYMBREG_CUDAEVALUATOR_H
