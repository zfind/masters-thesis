#include <cmath>
#include <stack>
#include <chrono>
#include <limits>

#include "CudaEvaluator.h"
#include "Constants.h"


#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


CudaEvaluator::CudaEvaluator(uint NUM_SAMPLES, uint INPUT_DIMENSION, uint MAX_PROG_SIZE,
                             vector<vector<double>> &datasetInput, vector<double> &datasetOutput) :
        NUM_SAMPLES(NUM_SAMPLES), INPUT_DIMENSION(INPUT_DIMENSION), MAX_PROG_SIZE(MAX_PROG_SIZE), datasetInput(datasetInput),
        datasetOutput(datasetOutput) {
    cudaMalloc((void **) &d_program, MAX_PROG_SIZE * sizeof(uint));
    cudaMalloc((void **) &d_programConst, MAX_PROG_SIZE * sizeof(double));
    cudaMalloc((void **) &d_datasetInput, NUM_SAMPLES * INPUT_DIMENSION * sizeof(double));
    cudaMalloc((void **) &d_resultOutput, NUM_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_globalStack, NUM_SAMPLES * MAX_PROG_SIZE * sizeof(double));
    cudaMalloc((void **) &d_datasetOutput, NUM_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_resultFitness, sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[NUM_SAMPLES * INPUT_DIMENSION];
    double *p_input = h_input;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        copy(datasetInput[i].begin(), datasetInput[i].end(), p_input);
        p_input += INPUT_DIMENSION;
    }

    cudaMemcpy(d_datasetInput, h_input, NUM_SAMPLES * INPUT_DIMENSION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, &datasetOutput[0], NUM_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);

    cudaMallocHost((void **) &postfixMemPinned, MAX_PROG_SIZE * (sizeof(uint) + sizeof(double)));

    delete[] h_input;
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_programConst);
    cudaFree(d_datasetInput);
    cudaFree(d_resultOutput);
    //cudaFree(d_globalStack);
    //cudaFree(d_datasetOutput);
    //cudaFree(d_resultFitness);
}

double CudaEvaluator::d_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                                 vector<double> &result) {

    cudaMemcpy(d_program, postfixMem, PROG_SIZE * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_programConst, postfixMem + PROG_SIZE * sizeof(uint), CONST_SIZE * sizeof(double),
               cudaMemcpyHostToDevice);

    uint THREADS_IN_BLOCK = 128;
    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((NUM_SAMPLES + block.x - 1) / block.x, 1);
//    size_t shared_size = THREADS_IN_BLOCK * PROG_SIZE * sizeof(double);

    d_evaluateIndividual <<<grid, block>>>(d_program, d_programConst,
            d_datasetInput, d_datasetOutput, d_resultOutput, d_globalStack, d_resultFitness,
            NUM_SAMPLES, INPUT_DIMENSION, PROG_SIZE);


//    result.resize(NUM_SAMPLES, 0.);
//    cudaMemcpy(&result[0], d_resultOutput, NUM_SAMPLES * sizeof(double), cudaMemcpyDeviceToHost);

    double fitness = 0.;
    cudaMemcpy(&fitness, d_resultFitness, sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < NUM_SAMPLES; i++) {
//        fitness += fabs(datasetOutput[i] - result[i]);
//    }

    return fitness;
}


__global__ void d_evaluateIndividual(uint *d_program,
                                     double *d_programConst,
                                     double *d_datasetInput,
                                     double *d_datasetOutput,
                                     double *d_resultOutput,
                                     double *d_globalStack,
                                     double *d_resultFitness,
                                     int N, int DIM, int PROG_SIZE) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) *d_resultFitness = 0.;

    __syncthreads();

    if (tid >= N) return;


    // in global memory, slow
    double *stack = d_globalStack + tid * PROG_SIZE;
    // in local, faster
    // double stack[50];

    //  stack in low latency shared memory
    //extern __shared__ double stackChunk[];
    //double *stack = stackChunk + threadIdx.x * PROG_SIZE;

    double *inputSample = d_datasetInput + tid * DIM;

    int SP = 0;
    double o1, o2, tmp;
    uint code, idx;

    for (int i = 0; i < PROG_SIZE; i++) {

        if (d_program[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (d_program[i]) {
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

        } else if (d_program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (d_program[i]) {
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

        } else if (d_program[i] == CONST) {
            tmp = *d_programConst;
            d_programConst++;

        } else if (d_program[i] >= VAR && d_program[i] < CONST) {
            code = d_program[i];
            idx = code - VAR;
            tmp = inputSample[idx];

        } else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    d_resultOutput[tid] = result;

    // slow reduction
    /*
    __syncthreads();
    if (tid == 0) {
        result=0.;
        #pragma unroll
        for (uint i = 0; i < N; i++) {
            result += fabs(d_datasetOutput[i] - d_resultOutput[i]);
        }
        *d_resultFitness = result;
    } */

    result = fabs(d_datasetOutput[tid] - result);

    atomicAdd(d_resultFitness, result);
}


double CudaEvaluator::h_evaluateIndividual(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                                           std::vector<double> &input) {

    uint *program = (uint *) postfixMem;
    double *programConst = (double *) &program[PROG_SIZE];

    double stack[PROG_SIZE];

    int SP = 0;
    double o1, o2, tmp;

    for (int i = 0; i < PROG_SIZE; i++) {

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
            tmp = *programConst;
            programConst++;

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


double CudaEvaluator::h_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE, std::vector<double> &result) {
    result.resize(NUM_SAMPLES, 0.);

    double fitness = 0.;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        result[i] = h_evaluateIndividual(postfixMem, PROG_SIZE, CONST_SIZE, datasetInput[i]);
        fitness += fabs(datasetOutput[i] - result[i]);
    }

    return fitness;
}
