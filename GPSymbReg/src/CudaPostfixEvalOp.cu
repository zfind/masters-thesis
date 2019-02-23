#include "CudaPostfixEvalOp.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Constants.h"


#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);

extern "C"
__global__ void d_evaluateIndividualKernel(gp_code_t *d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
                                           gp_val_t *d_datasetInput, gp_val_t *d_datasetOutput,
                                           gp_val_t *d_resultOutput, gp_fitness_t *d_resultFitness,
                                           int N_SAMPLES, int SAMPLE_DIMENSION) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) *d_resultFitness = 0.;

    extern __shared__ gp_code_t shared_programCache[];
    for (int idx = threadIdx.x; idx < PROGRAM_SIZE; idx += THREADS_IN_BLOCK) {
        shared_programCache[idx] = d_program[idx];
    }

    __syncthreads();

    if (tid >= N_SAMPLES) return;


    // in registers or local memory, faster than global
    gp_val_t stack[MAX_STACK_SIZE];

    gp_val_t *inputSample = d_datasetInput + tid * SAMPLE_DIMENSION;

    gp_val_t *d_programConst = (gp_val_t *) ((char *) (d_program) + BUFFER_PROGRAM_SIZE);

    int SP = 0;
    gp_val_t o1, o2, tmp;
    gp_code_t code;
    int idx;

    for (int i = 0; i < PROGRAM_SIZE; ++i) {

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

    gp_val_t result = stack[--SP];

    d_resultOutput[tid] = result;

    result = fabs(d_datasetOutput[tid] - result);

    atomicAdd(d_resultFitness, result);
}


gp_fitness_t CudaPostfixEvalOp::d_evaluate(char *buffer, int PROGRAM_SIZE, vector<gp_val_t> &result) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((PROGRAM_SIZE * sizeof(gp_code_t) + sizeof(gp_val_t) - 1)
                                        / sizeof(gp_val_t))
                                 * sizeof(gp_val_t);
    size_t BUFFER_CONSTANTS_SIZE = PROGRAM_SIZE * sizeof(gp_val_t);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    cudaMemcpy(d_program, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);

    int N_SAMPLES = dataset->size();
    int SAMPLE_DIMENSION = dataset->dim();

    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((N_SAMPLES + block.x - 1) / block.x, 1);
    size_t SHARED_MEM_SIZE = PROGRAM_SIZE * sizeof(gp_code_t);

    d_evaluateIndividualKernel<<<grid, block, SHARED_MEM_SIZE>>>(
            d_program, PROGRAM_SIZE, BUFFER_PROGRAM_SIZE,
                    d_datasetInput, d_datasetOutput,
                    d_resultOutput, d_resultFitness,
                    N_SAMPLES, SAMPLE_DIMENSION
    );


//    result.resize(N_SAMPLES, 0.);
//    cudaMemcpy(&result[0], d_resultOutput, N_SAMPLES * sizeof(gp_val_t), cudaMemcpyDeviceToHost);

    gp_fitness_t fitness = 0.;
    cudaMemcpy(&fitness, d_resultFitness, sizeof(gp_fitness_t), cudaMemcpyDeviceToHost);

    return fitness;
}
