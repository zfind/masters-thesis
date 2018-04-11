//
// Created by zac on 11.04.18..
//

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


#define VAR_X0   0x00000000
#define VAR_X1   0x00000001
#define VAR_X2   0x00000002
#define VAR_X3   0x00000003
#define VAR_X4   0x00000004

#define CONST   0x0000000FF

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3

#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define COS 0xFFFFFFF6

#define ERR 0xFFFFFFFF

class CudaEvaluator {
public:
    CudaEvaluator(int N, int DIM, int MAX_PROG_SIZE, vector<vector<double>> &input);
    ~CudaEvaluator();
    void evaluate(vector<uint> &program, vector<double> &programConst,
                  vector<vector<double>> &input,
                  vector<double> &result);

private:
    int N; int DIM; int MAX_PROG_SIZE;

    uint* d_program;
    double* d_programConst;
    double* d_input;
    double* d_output;
    double* d_stack;
};

__global__ void evaluateParallel(uint *d_program,
                                 double *d_programConstant,
                                 double *d_input,
                                 double *d_output,
                                 double *d_stack,
                                 int N, int DIM, int prog_size);


#endif //GPSYMBREG_CUDAEVALUATOR_H
