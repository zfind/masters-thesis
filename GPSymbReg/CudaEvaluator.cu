#include <cmath>
#include <stack>
#include <limits>
#include <iostream>
#include <vector>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <chrono>

using namespace std;

#include "CudaEvaluator.h"


#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


CudaEvaluator::CudaEvaluator(const uint N_SAMPLES, const uint SAMPLE_DIMENSION, const uint MAX_PROGRAM_NODES,
                             const vector<vector<double>> &datasetInput, const vector<double> &datasetOutput) :
        N_SAMPLES(N_SAMPLES), SAMPLE_DIMENSION(SAMPLE_DIMENSION), MAX_PROGRAM_NODES(MAX_PROGRAM_NODES),
        datasetInput(datasetInput),
        datasetOutput(datasetOutput) {

    size_t BUFFER_MAX_PROGRAM_SIZE = (int) ((MAX_PROGRAM_NODES * sizeof(uint) + sizeof(double) - 1)
                                            / sizeof(double))
                                     * sizeof(double);
    size_t BUFFER_MAX_CONSTANTS_SIZE = MAX_PROGRAM_NODES * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_MAX_PROGRAM_SIZE + BUFFER_MAX_CONSTANTS_SIZE;

    cudaMalloc((void **) &d_program, BUFFER_SIZE);
    cudaMalloc((void **) &d_datasetInput, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double));
    cudaMalloc((void **) &d_resultOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_datasetOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_resultFitness, sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[N_SAMPLES * SAMPLE_DIMENSION];
    double *p_input = h_input;
    for (int i = 0; i < N_SAMPLES; i++) {
        std::copy(datasetInput[i].begin(), datasetInput[i].end(), p_input);
        p_input += SAMPLE_DIMENSION;
    }

    cudaMemcpy(d_datasetInput, h_input, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, &datasetOutput[0], N_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_input;
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_resultOutput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultFitness);
}

double CudaEvaluator::h_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result) {
    result.resize(N_SAMPLES, 0.);

    double fitness = 0.;
    for (int i = 0; i < N_SAMPLES; i++) {
        result[i] = h_evaluateIndividual(buffer, PROGRAM_SIZE, datasetInput[i]);
        fitness += fabs(datasetOutput[i] - result[i]);
    }

    return fitness;
}

double CudaEvaluator::h_evaluateIndividual(char *buffer, uint PROGRAM_SIZE, std::vector<double> &input) {

    uint *program = reinterpret_cast<uint *>(buffer);

    size_t BUFFER_PROGRAM_SIZE = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    double *programConstants = reinterpret_cast<double *>(buffer + BUFFER_PROGRAM_SIZE);


    double stack[PROGRAM_SIZE];

    int SP = 0;
    double o1, o2, tmp;

    for (int i = 0; i < PROGRAM_SIZE; i++) {

        if (program[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (program[i]) {
                case ADD:
                    tmp = o1 + o2;
                    break;
                case SUB:
                    tmp = o1 - o2;
                    break;
                case MUL:
                    tmp = o1 * o2;
                    break;
                case DIV:
                    tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (program[i]) {
                case SQR:
                    tmp = (o1 >= 0.) ? sqrt(o1) : 1.;
                    break;
                case SIN:
                    tmp = sin(o1);
                    break;
                case COS:
                    tmp = cos(o1);
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] == CONST) {
            tmp = *programConstants;
            programConstants++;

        } else if (program[i] >= VAR && program[i] < CONST) {
            uint code = program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            CPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];
    return result;
}


double CudaEvaluator::d_evaluate(char *buffer, uint PROGRAM_SIZE, vector<double> &result) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    cudaMemcpy(d_program, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);

    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((N_SAMPLES + block.x - 1) / block.x, 1);
    size_t SHARED_MEM_SIZE = PROGRAM_SIZE * sizeof(uint);

    d_evaluateIndividualKernel<<<grid, block, SHARED_MEM_SIZE>>>(
            d_program, PROGRAM_SIZE, BUFFER_PROGRAM_SIZE,
            d_datasetInput, d_datasetOutput,
            d_resultOutput, d_resultFitness,
            N_SAMPLES, SAMPLE_DIMENSION
    );


//    result.resize(N_SAMPLES, 0.);
//    cudaMemcpy(&result[0], d_resultOutput, N_SAMPLES * sizeof(double), cudaMemcpyDeviceToHost);

    double fitness = 0.;
    cudaMemcpy(&fitness, d_resultFitness, sizeof(double), cudaMemcpyDeviceToHost);

    return fitness;
}


__global__ void d_evaluateIndividualKernel(uint *d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
                                           double *d_datasetInput, double *d_datasetOutput,
                                           double *d_resultOutput, double *d_resultFitness,
                                           int N_SAMPLES, int SAMPLE_DIMENSION) {

    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) *d_resultFitness = 0.;

    extern __shared__ uint shared_programCache[];
    for (uint idx = threadIdx.x; idx < PROGRAM_SIZE; idx += THREADS_IN_BLOCK) {
        shared_programCache[idx] = d_program[idx];
    }

    __syncthreads();

    if (tid >= N_SAMPLES) return;


    // in registers or local memory, faster than global
    double stack[MAX_STACK_SIZE];

    double *inputSample = d_datasetInput + tid * SAMPLE_DIMENSION;

    double *d_programConst = (double *) ((char *) (d_program) + BUFFER_PROGRAM_SIZE);

    int SP = 0;
    double o1, o2, tmp;
    uint code, idx;

    for (int i = 0; i < PROGRAM_SIZE; i++) {

        if (shared_programCache[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (shared_programCache[i]) {
                case ADD:
                    tmp = o1 + o2;
                    break;
                case SUB:
                    tmp = o1 - o2;
                    break;
                case MUL:
                    tmp = o1 * o2;
                    break;
                case DIV:
                    tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;
                    break;
                default:
                    GPU_EVALUATE_ERROR
            }

        } else if (shared_programCache[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (shared_programCache[i]) {
                case SQR:
                    tmp = (o1 >= 0.) ? sqrt(o1) : 1.;
                    break;
                case SIN:
                    tmp = sin(o1);
                    break;
                case COS:
                    tmp = cos(o1);
                    break;
                default:
                    GPU_EVALUATE_ERROR
            }

        } else if (shared_programCache[i] == CONST) {
            tmp = *d_programConst;
            d_programConst++;

        } else if (shared_programCache[i] >= VAR && shared_programCache[i] < CONST) {
            code = shared_programCache[i];
            idx = code - VAR;
            tmp = inputSample[idx];

        } else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    d_resultOutput[tid] = result;

    result = fabs(d_datasetOutput[tid] - result);

    atomicAdd(d_resultFitness, result);
}
