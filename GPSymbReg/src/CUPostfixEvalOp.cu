#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Constants.h"
#include "CUPostfixEvalOp.h"


#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


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


double CUPostfixEvalOp::d_evaluate(char *buffer, uint PROGRAM_SIZE, vector<double> &result) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    cudaMemcpy(d_program, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);

    int N_SAMPLES = dataset->size();
    int SAMPLE_DIMENSION = dataset->dim();

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
