//
// Created by zac on 01.05.18..
//

#include "CudaEvaluator.h"

#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


CudaEvaluator::CudaEvaluator(uint NUM_SAMPLES, uint INPUT_DIMENSION, uint MAX_PROG_SIZE,
                             vector<vector<bool>> &datasetInput, vector<bool> &datasetOutput) :
        NUM_SAMPLES(NUM_SAMPLES), INPUT_DIMENSION(INPUT_DIMENSION), MAX_PROG_SIZE(MAX_PROG_SIZE) {
    cudaMalloc((void **) &d_program, MAX_PROG_SIZE * sizeof(uint));
    cudaMalloc((void **) &d_datasetInput, NUM_SAMPLES * INPUT_DIMENSION * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_resultOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_datasetOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_resultFitness, sizeof(uint));

    this->datasetInput.resize(NUM_SAMPLES);
    this->datasetOutput.resize(NUM_SAMPLES);

    //  copy input matrix to 1D array
    BOOL_TYPE *h_input = new BOOL_TYPE[NUM_SAMPLES * INPUT_DIMENSION];
    BOOL_TYPE *h_output = new BOOL_TYPE[NUM_SAMPLES];
    for (uint y = 0; y < NUM_SAMPLES; y++) {
        this->datasetInput[y].resize(INPUT_DIMENSION);
        for (uint x = 0; x < INPUT_DIMENSION; x++) {
            h_input[y * INPUT_DIMENSION + x] = (BOOL_TYPE) datasetInput[y][x];
            this->datasetInput[y][x] = (BOOL_TYPE) datasetInput[y][x];
        }
        h_output[y] = (BOOL_TYPE) datasetOutput[y];
        this->datasetOutput[y] = (BOOL_TYPE) datasetOutput[y];
    }

    cudaMemcpy(d_datasetInput, h_input, NUM_SAMPLES * INPUT_DIMENSION * sizeof(BOOL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, h_output, NUM_SAMPLES * sizeof(BOOL_TYPE), cudaMemcpyHostToDevice);

    delete[] h_input;
    delete[] h_output;
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultOutput);
    cudaFree(d_resultFitness);
}

uint CudaEvaluator::h_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE, vector<BOOL_TYPE> &result) {
    result.resize(NUM_SAMPLES, 0);

    uint fitness = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        result[i] = h_evaluateIndividual(postfixMem, PROG_SIZE, CONST_SIZE, datasetInput[i]);
        if (result[i] != datasetOutput[i]) {
            fitness++;
        }
    }

    return fitness;
}

BOOL_TYPE CudaEvaluator::h_evaluateIndividual(char *postfixMem, uint PROG_SIZE, uint MEM_SIZE, std::vector<BOOL_TYPE> &input) {
    uint *program = (uint *) postfixMem;

    BOOL_TYPE stack[PROG_SIZE];

    int SP = 0;
    BOOL_TYPE o1, o2, tmp;

    for (int i = 0; i < PROG_SIZE; i++) {

        if (program[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (program[i]) {
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
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (program[i]) {
                case NOT:
                    tmp = !o1;
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= VAR && program[i] < CONST) {
            uint code = program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            CPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    BOOL_TYPE result = stack[--SP];
    return result;
}

uint CudaEvaluator::d_evaluate(char *postfixMem, uint PROG_SIZE, uint CONST_SIZE, vector<BOOL_TYPE> &result) {
    cudaMemcpy(d_program, postfixMem, PROG_SIZE * sizeof(uint), cudaMemcpyHostToDevice);

    dim3 block(THREADS_IN_BLOCK, 1);
    dim3 grid((NUM_SAMPLES + block.x - 1) / block.x, 1);
    size_t SHARED_MEM_SIZE = PROG_SIZE * sizeof(uint);

    d_evaluateIndividual <<<grid, block, SHARED_MEM_SIZE>>>(d_program,
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
