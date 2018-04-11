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
                  vector<vector<double>> &input, vector<double>& datasetOutput);

    ~CudaEvaluator();

    double d_evaluate(vector<uint> &program, vector<double> &programConst,
                    vector<vector<double>> &input, vector<double> &real,
                    vector<double> &result);

    void evaluate(std::vector<uint> &solution, std::vector<double> &solutionConstants);


private:
//    void convertToPostfix(IndividualP individual, std::vector<uint> &solution, std::vector<double> &solutionConstants);

    double h_evaluateDataset(std::vector<uint> &program, std::vector<double> &programConst,
                           std::vector<vector<double>> &input, vector<double>& real,
                           std::vector<double> &result);

    double h_evaluatePoint(std::vector<uint> &solution, std::vector<double> &solutionConst, std::vector<double> &input,
                           int validLength);

//    void printSolution(std::vector<uint> &solution, std::vector<double> &solutionConst);

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
};

__global__ void evaluateParallel(uint *d_program,
                                 double *d_programConstant,
                                 double *d_input,
                                 double *d_output,
                                 double *d_stack,
                                 int N, int DIM, int prog_size);


#endif //GPSYMBREG_CUDAEVALUATOR_H
