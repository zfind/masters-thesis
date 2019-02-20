//
// Created by zac on 19.02.19..
//

#include "CUPostfixEvalOp.h"
#include "Constants.h"
#include <memory>
#include <chrono>
#include <iostream>
#include <stack>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "Dataset.h"
#include "Utils.h"
using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);



extern "C"
__global__ void d_evaluateIndividualKernel(uint *d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
                                           double *d_datasetInput, double *d_datasetOutput,
                                           double *d_resultOutput, double *d_resultFitness,
                                           int N_SAMPLES, int SAMPLE_DIMENSION);


// called only once, before the evolution  generates training data
bool CUPostfixEvalOp::initialize(StateP state) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>("data/input.txt");

    conversionTime = 0L;
    gpuTime = 0L;

    int N_SAMPLES = dataset->size();
    int SAMPLE_DIMENSION = dataset->dim();

    cudaMalloc((void **) &d_program, BUFFER_SIZE);
    cudaMalloc((void **) &d_datasetInput, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double));
    cudaMalloc((void **) &d_resultOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_datasetOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_resultFitness, sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[N_SAMPLES * SAMPLE_DIMENSION];
    double *p_input = h_input;
    for (int i = 0; i < N_SAMPLES; i++) {
        const std::vector<double> &inputVector = dataset->getSampleInput(i);
        std::copy(inputVector.cbegin(), inputVector.cend(), p_input);
        p_input += SAMPLE_DIMENSION;
    }

    cudaMemcpy(d_datasetInput, h_input, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, &dataset->getOutputVector()[0], N_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_input;

    return true;
}

CUPostfixEvalOp::~CUPostfixEvalOp() {
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_resultOutput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultFitness);

    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "GPU time:\t" << gpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
}

FitnessP CUPostfixEvalOp::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROGRAM_SIZE;
    Utils::convertToPostfix(individual, programBuffer, PROGRAM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<double> d_result;
    double d_fitness = d_evaluate(programBuffer, PROGRAM_SIZE, d_result);


    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(d_fitness);


    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;

    return fitness;
}



