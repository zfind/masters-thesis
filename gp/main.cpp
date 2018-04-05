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

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3
#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define ERR 0xFFFFFFFF

#define OPERAND 0
#define UNARY   1
#define BINARY  2

typedef unsigned int uint;


uint generateBinaryOperator() {
    uint element;
    int choice = rand() % 4;
    if (choice == 0) {
        return ADD;
    } else if (choice == 1) {
        return SUB;
    } else if (choice == 2) {
        return MUL;
    } else if (choice == 3) {
        return DIV;
    }
    return ERR;
}

uint generateUnaryOperator() {
    uint element;
    int choice = rand() % 2;
    if (choice) {
        return SQR;
    } else {
        return SIN;
    }
}

uint generateOperand() {
    uint element;
    uint choice = rand() % 4;
    return choice;
}

vector<uint> generateIndividual(int chromosomeLength) {
    int count = 0;
    int stackcount = 0;
    stack<uint> st;
    vector<uint> solution;

    uint element = ERR;

    while (count < chromosomeLength) {
        if (stackcount < 1) {
            // generate a random operand, push it to the stack, increment stackcount
            uint operand = generateOperand();
            element = operand;
            st.push(operand);
            stackcount++;
        } else if (stackcount == 1) {
            int x = rand() % 2;
            if (x) {
                uint operand = generateOperand();
                element = operand;
                st.push(operand);
                stackcount++;
            } else {
                uint unary_op = generateUnaryOperator();
                element = unary_op;
                st.pop();
                stackcount--;
                st.push(unary_op);
                stackcount++;
            }
        } else if (stackcount > 1) {
            int y = rand() % 2;
            if (y) {
                uint operand = generateOperand();
                element = operand;
                st.push(operand);
                stackcount++;
            } else {
                int x = rand() % 2;
                if (x) {
                    uint unary_op = generateUnaryOperator();
                    element = unary_op;
                    st.pop();
                    stackcount--;
                    st.push(unary_op);
                    stackcount++;
                } else {
                    uint binary_op = generateBinaryOperator();
                    element = binary_op;
                    st.pop();
                    st.pop();
                    stackcount -= 2;
                    st.push(binary_op);
                    stackcount++;
                }
            }
        }
        solution.push_back(element);
        count++;
    }
    return solution;
}

void printSolution(vector<uint> &solution) {
    for (uint i : solution) {
        switch (i) {
            case ADD:
                cout << "ADD" << endl;
                break;
            case SUB:
                cout << "SUB" << endl;
                break;
            case MUL:
                cout << "MUL" << endl;
                break;
            case DIV:
                cout << "DIV" << endl;
                break;
            case SQR:
                cout << "SQT" << endl;
                break;
            case SIN:
                cout << "SIN" << endl;
                break;
            default:
                cout << i << endl;
                break;
        }
    }
}

int getArity(uint i) {
    switch (i) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
            return BINARY;
        case SQR:
        case SIN:
            return UNARY;
        default:
            return OPERAND;
    }
}

int getValidLength(vector<uint> &solution) {
    int length = solution.size();
    int count = 0, validpos = 0;
    int validLength = 0;
    for (int i = 0; i < length; i++) {
        int arity = getArity(solution[i]);
        if (arity == OPERAND) {
            count++;
        } else if (arity == UNARY) {
            count = count - (UNARY - 1);
        } else if (arity == BINARY) {
            count = count - (BINARY - 1);
        }

        if (count == 0) {
            break;
        }
        if (count == 1) {
            validpos = i;
        }
    }
    validLength = validpos + 1;
    return validLength;
}


double evaluate(vector<uint> &solution, vector<double> &input) {
    int validLength = getValidLength(solution);

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

                tmp = (abs(o2) > 10E-9) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            case ERR:
                return -1.;
            default:
                tmp = input[solution[i]];

                stack[SP++] = tmp;
                break;
        }
    }

//    cerr << "SP:\t" << SP << endl;
    double result = stack[--SP];

    delete[] stack;

    return result;
}


void test1() {
    vector<uint> individual = generateIndividual(10);

    printSolution(individual);

    cout << "valid length:\t" << getValidLength(individual) << endl;
}


void test2() {
    vector<uint> test = {0, 1, MUL, 1, 0, SIN, DIV, ADD};
    vector<double> input = {3., 5.};

    cout << "valid:\t" << getValidLength(test) << endl;
    double eval = evaluate(test, input);
    cout << "eval:\t" << eval << "\ttrue:\t50.4308" << endl;
}


void test3() {
    vector<double> input = {0.1, 5.1, -2.3, 7.7, -1.33};

    vector<uint> s1 = {0, 1, SUB, 2, ADD, 4, 3, 4, ADD, DIV, MUL};
    double o1 = evaluate(s1, input);
    cout << "eval:\t" << o1 << "\treal:\t" << 1.5241758 << endl;

    vector<uint> s2 = {0, SIN, 0, 2, MUL, SUB, 1, SQR, 3, DIV, MUL, 4, SIN, 2, SQR, DIV, ADD};
    double o2 = evaluate(s2, input);
    cout << "eval:\t" << o2 << "\treal:\t" << -0.8744121795 << endl;

    vector<uint> s3 = {0, SIN, 1, SQR, MUL, 1, 0, SUB, 2, 3, ADD, DIV, ADD, 4, SIN, 3, ADD, 4, SQR, 1, SIN, 3, SQR, ADD,
                       DIV, MUL, SUB};
    double o3 = evaluate(s3, input);
    cout << "eval:\t" << o3 << "\treal:\t" << -2.48766 << endl;
}


void loadInput(string filename, vector<vector<double>> &matrix) {
    ifstream in(filename);

    if (!in) {
        cerr << "Cannot open file.\n";
        exit(-1);
    }

    int N, DIM;
    in >> N;
    in >> DIM;

    vector<double> initRow;
    initRow.resize(DIM, 0.);
    matrix.resize(N, initRow);

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < DIM; x++) {
            in >> matrix[y][x];
        }
    }

    in.close();
}


void evaluateHost(vector<uint> &program, vector<vector<double>> &input, vector<double> &result) {
    int N = input.size();
    result.resize(N, 0.);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        result[i] = evaluate(program, input[i]);
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference [us] = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
}


__global__ void
evaluateParallel(uint *d_program, double *d_input, double *d_output, double *d_stack, int N, int DIM, int prog_size) {
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

                tmp = (abs(o2) > 10E-9) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            default:
                tmp = input[d_program[i]];

                stack[SP++] = tmp;
                break;
        }
    }

//    cerr << "SP:\t" << SP << endl;
    double result = stack[--SP];

    d_output[tid] = result;

}


void evaluateDevice(vector<uint> &program, vector<vector<double>> &input, vector<double> &result) {
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

    evaluateParallel<<<dimGridN, dimBlock>>>(d_program, d_input, d_output, d_stack, N, DIM, program.size());
    cudaDeviceSynchronize();

    double *h_output = new double[N];
    cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference [us] = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

//    for (int i = 0; i < N; i++) {
//        cout << h_output[i] << endl;
//    }
    delete[] h_output;

    cudaFree(d_stack);

    delete[] h_input;
    cudaFree(d_program);
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void myKernel(double *d_stack, int size) {
    int tid = blockIdx.x;

    double *t_stack = d_stack + tid * size;

    for (int i = 0; i < size; i++) {
        t_stack[i] = (double) i;
    }
}

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

void test4() {
    vector<vector<double>> matrix;
    loadInput("/home/zac/Projekti/masters-thesis/gp/big.txt", matrix);

    vector<double> result;
    vector<uint> program = {0, SIN, 1, SQR, MUL, 1, 0, SUB, 2, 3, ADD, DIV, ADD, 4, SIN, 3, ADD, 4, SQR, 1, SIN, 3, SQR,
                            ADD, DIV, MUL, SUB};
    evaluateHost(program, matrix, result);

//    for (int i = 0; i < result.size(); i++) {
//        cout << result[i] << endl;
//    }

    evaluateDevice(program, matrix, result);

    return;
}

void test5() {
    stackDraft();
}


int main() {
    srand((unsigned) time(nullptr));

    test4();

    return 0;
}