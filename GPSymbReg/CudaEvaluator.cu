#include <cmath>
//#include <ecf/ECF.h>
#include <stack>
#include <chrono>
#include <limits>

#include "CudaEvaluator.h"
#include "Constants.h"


#define EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_output[tid] = NAN; return;} while(0);




CudaEvaluator::CudaEvaluator(int N, int DIM, int MAX_PROG_SIZE, vector<vector<double>> &input, vector<double> &output) :
        N(N), DIM(DIM), MAX_PROG_SIZE(MAX_PROG_SIZE), datasetInput(input), datasetOutput(output) {
    cudaMalloc((void **) &d_program, MAX_PROG_SIZE * sizeof(uint));
    cudaMalloc((void **) &d_programConst, MAX_PROG_SIZE * sizeof(double));
    cudaMalloc((void **) &d_input, N * DIM * sizeof(double));
    cudaMalloc((void **) &d_output, N * sizeof(double));
    cudaMalloc((void **) &d_stack, N * MAX_PROG_SIZE * sizeof(double));
    cudaMalloc((void **) &d_real, N * sizeof(double));
    cudaMalloc((void **) &d_answer, sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[N * DIM];
    double *p_input = h_input;
    for (int i = 0; i < N; i++) {
        copy(input[i].begin(), input[i].end(), p_input);
        p_input += DIM;
    }

    cudaMemcpy(d_input, h_input, N * DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_real, &datasetOutput[0], N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMallocHost((void**)&postfixMemPinned, MAX_PROG_SIZE*(sizeof(uint) + sizeof(double)));

    delete[] h_input;


    cerr << "INICIJALIZIRAM EVALUATOR" << endl;
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_programConst);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_stack);
}

double CudaEvaluator::d_evaluate(char* postfixMem, uint PROG_SIZE, uint CONST_SIZE,
                                 vector<double> &result) {


//    int PROG_SIZE = program.size();

    cudaMemcpy(d_program, postfixMem, PROG_SIZE * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_programConst, postfixMem + PROG_SIZE * sizeof(uint), CONST_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(128, 1);
    dim3 grid((N + block.x - 1) / block.x, 1);
    size_t shared_size = block.x * PROG_SIZE * sizeof(double);

//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

//    double fitness;

    d_evaluateIndividualNew<<<grid, block, shared_size>>>(d_program, d_programConst,
            d_input, d_output, d_stack, d_real,
            N, DIM, PROG_SIZE, d_answer);
//    cudaDeviceSynchronize();

//    cudaMemcpy(&fitness, d_answer, sizeof(double), cudaMemcpyDeviceToHost);

    result.resize(N, 0.);
//    double *h_output = new double[N];
    cudaMemcpy(&result[0], d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cout << "GPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    double fitness = 0.;
    for (int i = 0; i < N; i++) {
        fitness += fabs(datasetOutput[i] - result[i]);
    }

    return fitness;
}



__global__ void d_evaluateIndividualNew(uint *d_program,
                                     double *d_programConstant,
                                     double *d_input,
                                     double *d_output,
                                     double *d_stack,
                                        double *d_real,
                                     int N, int DIM, int prog_size, double *d_answer) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

//    double *stack = d_stack + tid * prog_size;

//    double stack[50];

    extern __shared__ double stackChunk[];
    double *stack = stackChunk + threadIdx.x * prog_size;

    double *input = d_input + tid * DIM;


    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < prog_size; i++) {
        if (d_program[i] >= ARR_2) {
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


        } else if (d_program[i] >= ARR_1) {
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
            tmp = *d_programConstant;
            d_programConstant++;

        } else if (d_program[i] >= VAR && d_program[i] < CONST) {
            uint code = d_program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    d_output[tid] = result;

//    __syncthreads();
//
//    if (tid == 0) {
//        result=0.;
//        #pragma unroll
//        for (uint i = 0; i < N; i++) {
//            result += fabs(d_real[i] - d_output[i]);
//        }
//        *d_answer = result;
//    }
}

__global__ void d_evaluateIndividual(uint *d_program,
                                     double *d_programConstant,
                                     double *d_input,
                                     double *d_output,
                                     double *d_stack,
                                     int N, int DIM, int prog_size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N) return;

//    double *stack = d_stack + tid * prog_size;

//    double stack[50];

    extern __shared__ double stackChunk[];
    double *stack = stackChunk + threadIdx.x * prog_size;

    double *input = d_input + tid * DIM;


    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < prog_size; i++) {
        if (d_program[i] >= ARR_2) {
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


        } else if (d_program[i] >= ARR_1) {
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
            tmp = d_programConstant[i];

        } else if (d_program[i] >= VAR && d_program[i] < CONST) {
            uint code = d_program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            GPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    d_output[tid] = result;
}


double CudaEvaluator::h_evaluateIndividual(std::vector<uint> &solution, std::vector<double> &solutionConst,
                                           std::vector<double> &input, int validLength) {
    double stack[validLength];
    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < validLength; i++) {
        if (solution[i] >= ARR_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (solution[i]) {
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
                    EVALUATE_ERROR
            }


        } else if (solution[i] >= ARR_1) {
            o1 = stack[--SP];

            switch (solution[i]) {
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
                    EVALUATE_ERROR
            }


        } else if (solution[i] == CONST) {
            tmp = solutionConst[i];

        } else if (solution[i] >= VAR && solution[i] < CONST) {
            uint code = solution[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    return result;
}

double CudaEvaluator::h_evaluateIndividualNew(char* postfixMem, uint PROG_SIZE, uint MEM_SIZE,
                                              std::vector<double> &input) {

    uint* program = (uint*) postfixMem;
    double* programConst = (double*) &program[PROG_SIZE];

    double stack[PROG_SIZE];
    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < PROG_SIZE; i++) {
        if (program[i] >= ARR_2) {
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
                    EVALUATE_ERROR
            }


        } else if (program[i] >= ARR_1) {
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
                    EVALUATE_ERROR
            }


        } else if (program[i] == CONST) {
            tmp = *programConst;
            programConst++;

        } else if (program[i] >= VAR && program[i] < CONST) {
            uint code = program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];

    return result;
}


double CudaEvaluator::h_evaluate(std::vector<uint> &program, std::vector<double> &programConst,
                                 std::vector<vector<double>> &input, vector<double> &real,
                                 std::vector<double> &result) {
//    int N = input.size();
    result.resize(N, 0.);

    double fitness = 0.;
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        result[i] = h_evaluateIndividual(program, programConst, input[i], program.size());
        fitness += fabs(real[i] - result[i]);
    }
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cerr << "CPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    return fitness;
}

double CudaEvaluator::h_evaluateNew(char* postfixMem, uint PROG_SIZE, uint MEM_SIZE, std::vector<double> &result) {
//    int N = input.size();
    result.resize(N, 0.);

    double fitness = 0.;
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        result[i] = h_evaluateIndividualNew(postfixMem, PROG_SIZE, MEM_SIZE, datasetInput[i]);
        fitness += fabs(datasetOutput[i] - result[i]);
    }
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cerr << "CPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    return fitness;
}

/*
void CudaEvaluator::evaluate(vector<uint> &postfix, vector<double> &postfixConstants) {

    // evaluiraj na cpu
    vector<double> h_result;
    double h_fitness = h_evaluate(postfix, postfixConstants, datasetInput, datasetOutput, h_result);

    // evaluiraj na gpu
    vector<double> d_result;
    double d_fitness = d_evaluate(postfix, postfixConstants, datasetInput, datasetOutput, d_result);

    // provjeri jesu li jednaki
//    for (int i = 0; i < h_result.size(); i++) {
//        if (fabs(h_result[i] - d_result[i]) > 1E-10) {     // std::numeric_limits<double>::epsilon()
//            cerr << "FAIL\t" << "host:\t" << h_result[i] << "\tdev:\t" << d_result[i] << endl;
//        }
//    }

    cerr << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << endl;

}
 */