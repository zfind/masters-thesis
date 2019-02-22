#include "CudaEvaluator.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Constants.h"

#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);

extern "C"
__global__ void d_evaluateIndividual(uint *d_program,
                                     BOOL_TYPE *d_datasetInput, BOOL_TYPE *d_datasetOutput,
                                     BOOL_TYPE *d_resultOutput, uint *d_resultFitness,
                                     int N, int DIM, int PROG_SIZE) {

    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) *d_resultFitness = 0;

    extern __shared__ uint shared_programCache[];
    for (uint idx = threadIdx.x; idx < PROG_SIZE; idx += THREADS_IN_BLOCK) {
        shared_programCache[idx] = d_program[idx];
    }

    __syncthreads();

    if (tid >= N) return;


    // in global memory, slow
    //    double *stack = d_globalStack + tid * PROG_SIZE;
    // in local, faster
    BOOL_TYPE stack[MAX_STACK_SIZE];

    //  stack in low latency shared memory
    //    extern __shared__ double stackChunk[];
    //    double *stack = stackChunk + threadIdx.x * PROG_SIZE;

    BOOL_TYPE *inputSample = d_datasetInput + tid * DIM;

    int SP = 0;
    BOOL_TYPE o1, o2, tmp;
    uint code, idx;

    for (int i = 0; i < PROG_SIZE; i++) {

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

        } else if (shared_programCache[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (shared_programCache[i]) {
                case NOT:
                    tmp = !o1;
                    break;
                default:
                    GPU_EVALUATE_ERROR
            }

    //        } else if (shared_programCache[i] == CONST) {
    //            tmp = *d_programConst;
    //            d_programConst++;

        } else if (shared_programCache[i] >= VAR && shared_programCache[i] < CONST) {
            code = shared_programCache[i];
            idx = code - VAR;
            tmp = inputSample[idx];

        } else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    BOOL_TYPE result = stack[--SP];

    //    d_resultOutput[tid] = result;

    if (result != d_datasetOutput[tid]) {
        atomicAdd(d_resultFitness, 1);
    }

}


uint CudaEvaluator::d_evaluate(char *postfixMem, uint PROG_SIZE, vector<BOOL_TYPE> &result) {
    cudaMemcpy(d_program, postfixMem, PROG_SIZE * sizeof(uint), cudaMemcpyHostToDevice);

    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();

    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((NUM_SAMPLES + block.x - 1) / block.x, 1);
    size_t SHARED_MEM_SIZE = PROG_SIZE * sizeof(uint);

    d_evaluateIndividual << < grid, block, SHARED_MEM_SIZE >> > (d_program,
            d_datasetInput, d_datasetOutput, d_resultOutput, d_resultFitness,
            NUM_SAMPLES, INPUT_DIMENSION, PROG_SIZE);


    // ukljuci u kernelu!!!
    //    result.resize(NUM_SAMPLES, 0);
    //    cudaMemcpy(&result[0], d_resultOutput, NUM_SAMPLES * sizeof(BOOL_TYPE), cudaMemcpyDeviceToHost);

    uint fitness = 0;
    cudaMemcpy(&fitness, d_resultFitness, sizeof(uint), cudaMemcpyDeviceToHost);

    //    for (uint i = 0; i < NUM_SAMPLES; i++) {
    //        if (datasetOutput[i] != result[i]) {
    //            fitness++;
    //        }
    //    }

    return fitness;
}
