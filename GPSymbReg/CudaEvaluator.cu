#include <cmath>
//#include <ecf/ECF.h>
#include <stack>
#include <chrono>
#include <limits>

#include "CudaEvaluator.h"
#include "Constants.h"


CudaEvaluator::CudaEvaluator(int N, int DIM, int MAX_PROG_SIZE, vector<vector<double>> &input, vector<double>& output) :
        N(N), DIM(DIM), MAX_PROG_SIZE(MAX_PROG_SIZE), datasetInput(input), datasetOutput(output) {
    cudaMalloc((void **) &d_program, MAX_PROG_SIZE * sizeof(uint));
    cudaMalloc((void **) &d_programConst, MAX_PROG_SIZE * sizeof(double));
    cudaMalloc((void **) &d_input, N * DIM * sizeof(double));
    cudaMalloc((void **) &d_output, N * sizeof(double));
    cudaMalloc((void **) &d_stack, N * MAX_PROG_SIZE * sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[N * DIM];
    double *p_input = h_input;
    for (int i = 0; i < N; i++) {
        copy(input[i].begin(), input[i].end(), p_input);
        p_input += DIM;
    }

    cudaMemcpy(d_input, h_input, N * DIM * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_input;

}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_programConst);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_stack);
}

double CudaEvaluator::d_evaluate(vector<uint> &program, vector<double> &programConst,
                               vector<vector<double>> &input, vector<double>& real,
                               vector<double> &result) {


    int PROG_SIZE = program.size();

    cudaMemcpy(d_program, &program[0], program.size() * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_programConst, &programConst[0], program.size() * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimGridN(N, 1);
    dim3 dimBlock(1, 1, 1);

//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    evaluateParallel << < dimGridN, dimBlock >> > (d_program, d_programConst,
            d_input, d_output, d_stack,
            N, DIM, program.size());
    cudaDeviceSynchronize();

    result.resize(N, 0.);
//    double *h_output = new double[N];
    cudaMemcpy(&result[0], d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cout << "GPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    double fitness = 0.;
    for (int i = 0; i < N; i++) {
        fitness += fabs(real[i] - result[i]);
    }

    return fitness;
}

__global__ void evaluateParallel(uint *d_program,
                                 double *d_programConstant,
                                 double *d_input,
                                 double *d_output,
                                 double *d_stack,
                                 int N, int DIM, int prog_size) {
    int tid = blockIdx.x;

    double *stack = d_stack + tid * prog_size;

    double *input = d_input + tid * DIM;


    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < prog_size; i++) {
        switch (d_program[i]) {
            case ADD:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 + o2;

                stack[SP++] = tmp;
                break;
            case SUB:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 - o2;

                stack[SP++] = tmp;
                break;
            case MUL:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 * o2;

                stack[SP++] = tmp;
                break;
            case DIV:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1.;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            case COS:
                o1 = stack[--SP];

                tmp = cos(o1);

                stack[SP++] = tmp;
                break;
            case VAR_X0:
                tmp = input[0];

                stack[SP++] = tmp;
                break;
            case VAR_X1:
                tmp = input[1];

                stack[SP++] = tmp;
                break;
            case VAR_X2:
                tmp = input[2];

                stack[SP++] = tmp;
                break;
            case VAR_X3:
                tmp = input[3];

                stack[SP++] = tmp;
                break;
            case VAR_X4:
                tmp = input[4];

                stack[SP++] = tmp;
                break;
            case CONST:
                tmp = d_programConstant[i];

                stack[SP++] = tmp;
                break;
            case ERR:
            default:
                d_output[tid] = -1;
                return;
        }
    }

    double result = stack[--SP];

    d_output[tid] = result;
}


double CudaEvaluator::h_evaluatePoint(std::vector<uint> &solution, std::vector<double> &solutionConst,
                                      std::vector<double> &input, int validLength) {

    double *stack = new double[validLength];
    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < validLength; i++) {
        switch (solution[i]) {
            case ADD:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 + o2;

                stack[SP++] = tmp;
                break;
            case SUB:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 - o2;

                stack[SP++] = tmp;
                break;
            case MUL:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 * o2;

                stack[SP++] = tmp;
                break;
            case DIV:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1.;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            case COS:
                o1 = stack[--SP];

                tmp = cos(o1);

                stack[SP++] = tmp;
                break;
            case VAR_X0:
                tmp = input[0];

                stack[SP++] = tmp;
                break;
            case VAR_X1:
                tmp = input[1];

                stack[SP++] = tmp;
                break;
            case VAR_X2:
                tmp = input[2];

                stack[SP++] = tmp;
                break;
            case VAR_X3:
                tmp = input[3];

                stack[SP++] = tmp;
                break;
            case VAR_X4:
                tmp = input[4];

                stack[SP++] = tmp;
                break;
            case CONST:
                tmp = solutionConst[i];

                stack[SP++] = tmp;
                break;
            case ERR:
            default:
                cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl;
                return -1.;
        }
    }

    double result = stack[--SP];

    delete[] stack;

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
        result[i] = h_evaluatePoint(program, programConst, input[i], program.size());
        fitness += fabs(real[i] - result[i]);
    }
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cerr << "CPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    return fitness;
}


void CudaEvaluator::evaluate(vector<uint> &postfix, vector<double> &postfixConstants) {

    // evaluiraj na cpu
    vector<double> h_result;
    double h_fitness = h_evaluate(postfix, postfixConstants, datasetInput, datasetOutput, h_result);

    // evaluiraj na gpu
    vector<double> d_result;
    double d_fitness = d_evaluate(postfix, postfixConstants, datasetInput,datasetOutput, d_result);

    // provjeri jesu li jednaki
//    for (int i = 0; i < h_result.size(); i++) {
//        if (fabs(h_result[i] - d_result[i]) > 1E-10) {     // std::numeric_limits<double>::epsilon()
//            cerr << "FAIL\t" << "host:\t" << h_result[i] << "\tdev:\t" << d_result[i] << endl;
//        }
//    }

    cerr << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << endl;

}