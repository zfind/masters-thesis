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
    CudaEvaluator(int N, int DIM, int MAX_PROG_SIZE,
                  vector<vector<double>> &input, vector<double> &datasetOutput);

    ~CudaEvaluator();

    double h_evaluate(std::vector<uint> &program, std::vector<double> &programConst,
                      std::vector<vector<double>> &input, vector<double> &real,
                      std::vector<double> &result);

    double h_evaluateNew(char* postfixMem, uint PROG_SIZE, uint MEM_SIZE, std::vector<double> &result);

    double d_evaluate(char* postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                      vector<double> &result);

    void evaluate(std::vector<uint> &solution, std::vector<double> &solutionConstants);

    char *postfixMemPinned;

private:

    double h_evaluateIndividual(std::vector<uint> &solution, std::vector<double> &solutionConst,
                                std::vector<double> &input,
                                int validLength);

    double h_evaluateIndividualNew(char* postfixMem, uint PROG_SIZE, uint MEM_SIZE,
                                   std::vector<double> &input);

private:
    int N;
    int DIM;
    int MAX_PROG_SIZE;

    std::vector<std::vector<double>> datasetInput;
    std::vector<double> datasetOutput;

    uint *d_program;
    double *d_programConst;
    double *d_input;
    double *d_output;
    double *d_stack;
    double *d_real;
    double *d_answer;
};

__global__ void d_evaluateIndividual(uint *d_program,
                                     double *d_programConstant,
                                     double *d_input,
                                     double *d_output,
                                     double *d_stack,
                                     int N, int DIM, int prog_size);


__global__ void d_evaluateIndividualNew(uint *d_program,
                                        double *d_programConstant,
                                        double *d_input,
                                        double *d_output,
                                        double *d_stack,
                                        double *d_real,
                                        int N, int DIM, int prog_size, double *d_answer);

#endif //GPSYMBREG_CUDAEVALUATOR_H
