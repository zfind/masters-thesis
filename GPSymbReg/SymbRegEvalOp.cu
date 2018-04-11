#include <iostream>
#include <vector>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <chrono>
using namespace std;

#define VAR_X0   0x00000000
#define VAR_X1   0x00000001
#define VAR_X2   0x00000002
#define VAR_X3   0x00000003
#define VAR_X4   0x00000004

#define CONST   0x0000000FF

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3

#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define COS 0xFFFFFFF6

#define ERR 0xFFFFFFFF

__global__ void myKernel(double *d_stack, int size) {
    int tid = blockIdx.x;

    double *t_stack = d_stack + tid * size;

    for (int i = 0; i < size; i++) {
        t_stack[i] = (double) i;
    }
}

extern "C"
void stackDraft() {
    int N = 5;
    int size = 10;

    double *d_stack;
    cudaMalloc((void **) &d_stack, N * size * sizeof(double));

    dim3 dimGridN(N, 1);
    dim3 dimBlock(1, 1, 1);

    myKernel << < dimGridN, dimBlock >> > (d_stack, size);
    cudaDeviceSynchronize();

    double *h_stack = new double[N * size];
    cudaMemcpy(h_stack, d_stack, N * size * sizeof(double), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N * size; i++) {
        cout << h_stack[i] << endl;
    }

    delete[] h_stack;
    cudaFree(d_stack);
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

//    for (int i=0; i<prog_size; i++) {
//        t_stack[i] = (double) i;
//    }

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
//                cerr<< "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl;
                d_output[tid] = -1;
                return;
        }
    }

//    cerr << "SP:\t" << SP << endl;
    double result = stack[--SP];

    d_output[tid] = result;

}

extern "C"
void evaluateDevice(vector<uint> &program, vector<double> &programConst, vector<vector<double>> &input,
                    vector<double> &result) {
    int N = input.size();
    int DIM = input[0].size();
    int PROG_SIZE = program.size();

    //  copy input matrix to 1D array
    double *h_input = new double[N * DIM];
    double *p_input = h_input;
    for (int i = 0; i < N; i++) {
        copy(input[i].begin(), input[i].end(), p_input);
        p_input += DIM;
    }

    uint *d_program;
    cudaMalloc((void **) &d_program, PROG_SIZE * sizeof(uint));
    cudaMemcpy(d_program, &program[0], program.size() * sizeof(uint), cudaMemcpyHostToDevice);

    double *d_programConst;
    cudaMalloc((void **) &d_programConst, PROG_SIZE * sizeof(double));
    cudaMemcpy(d_programConst, &programConst[0], program.size() * sizeof(double), cudaMemcpyHostToDevice);


    double *d_input;
    cudaMalloc((void **) &d_input, N * DIM * sizeof(double));
    cudaMemcpy(d_input, h_input, N * DIM * sizeof(double), cudaMemcpyHostToDevice);

    double *d_output;
    cudaMalloc((void **) &d_output, N * sizeof(double));

    double *d_stack;
    cudaMalloc((void **) &d_stack, N * PROG_SIZE * sizeof(double));

    dim3 dimGridN(N, 1);
    dim3 dimBlock(1, 1, 1);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    evaluateParallel<<<dimGridN, dimBlock>>>(d_program, d_programConst, d_input, d_output, d_stack, N, DIM, program.size());
    cudaDeviceSynchronize();

    double *h_output = new double[N];
    cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "GPU Time difference [us] = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

    result.resize(N, 0.);
    for (int i = 0; i < N; i++) {
        result[i] = h_output[i];
//        cout << h_output[i] << endl;
    }
    delete[] h_output;

    cudaFree(d_stack);

    delete[] h_input;
    cudaFree(d_programConst);
    cudaFree(d_program);
    cudaFree(d_input);
    cudaFree(d_output);
}
