#include "CudaPostfixEvalOp.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "PostfixEvalOpUtils.h"

#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);

extern "C"
__global__ void d_evaluateIndividual(gp_code_t* d_program,
        gp_val_t* d_datasetInput, gp_val_t* d_datasetOutput,
        gp_val_t* d_resultOutput, gp_fitness_t* d_resultFitness,
        int N, int DIM, int PROG_SIZE)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) *d_resultFitness = 0;

    extern __shared__ gp_code_t shared_programCache[];
    for (int idx = threadIdx.x; idx < PROG_SIZE; idx += THREADS_IN_BLOCK) {
        shared_programCache[idx] = d_program[idx];
    }

    __syncthreads();

    if (tid >= N) return;

    // in local memory, faster than global
    bool stack[MAX_STACK_SIZE];

    gp_val_t* inputSample = d_datasetInput + tid * DIM;

    int SP = 0;
    bool o1, o2, tmp;
    gp_code_t code;
    int idx;

    for (int i = 0; i < PROG_SIZE; ++i) {

        if (shared_programCache[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (shared_programCache[i]) {
            case AND:
                tmp = o1 && o2;
                break;
            case OR:
                tmp = o1 || o2;
                break;
            case XOR:
                tmp = (o1 && !o2) || (!o1 && o2);
                break;
            case XNOR:
                tmp = (!(o1 && !o2) || (!o1 && o2));
                break;
            case NAND:
                tmp = (!o1) || (!o2);
                break;
            case NOR:
                tmp = (!o1) && (!o2);
                break;
            default:
                GPU_EVALUATE_ERROR
            }

        }
        else if (shared_programCache[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (shared_programCache[i]) {
            case NOT:
                tmp = !o1;
                break;
            default:
                GPU_EVALUATE_ERROR
            }

        }
        else if (shared_programCache[i] >= VAR && shared_programCache[i] < CONST) {
            code = shared_programCache[i];
            idx = code - VAR;
            tmp = inputSample[idx];

        }
        else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    gp_val_t result = stack[--SP] ? '1' : '0';

    //    d_resultOutput[tid] = result;

    if (result != d_datasetOutput[tid]) {
        atomicAdd(d_resultFitness, 1);
    }

}

gp_fitness_t CudaPostfixEvalOp::d_evaluate(char* postfixMem, int programSize, vector<gp_val_t>& result)
{
    cudaMemcpy(d_program, postfixMem, programSize * sizeof(gp_code_t), cudaMemcpyHostToDevice);

    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();

    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((NUM_SAMPLES + block.x - 1) / block.x, 1);
    size_t SHARED_MEM_SIZE = programSize * sizeof(gp_code_t);

    d_evaluateIndividual<<< grid, block, SHARED_MEM_SIZE >>>(d_program,
            d_datasetInput, d_datasetOutput, d_resultOutput, d_resultFitness,
            NUM_SAMPLES, INPUT_DIMENSION, programSize);

    // uncomment in kernel if needed
    //    result.resize(NUM_SAMPLES, 0);
    //    cudaMemcpy(&result[0], d_resultOutput, NUM_SAMPLES * sizeof(gp_val_t), cudaMemcpyDeviceToHost);

    gp_fitness_t fitness = 0;
    cudaMemcpy(&fitness, d_resultFitness, sizeof(gp_fitness_t), cudaMemcpyDeviceToHost);

    return fitness;
}
