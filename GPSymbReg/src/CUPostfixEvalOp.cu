//
// Created by zac on 19.02.19..
//

#include "CUPostfixEvalOp.h"
#include "Constants.h"
#include <memory>
#include <chrono>
#include <iostream>
#include <stack>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "Dataset.h"



using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


// called only once, before the evolution  generates training data
bool CUPostfixEvalOp::initialize(StateP state) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>("data/input.txt");

    conversionTime = 0L;
    gpuTime = 0L;

    int N_SAMPLES = dataset->size();
    int SAMPLE_DIMENSION = dataset->dim();

    cudaMalloc((void **) &d_program, BUFFER_SIZE);
    cudaMalloc((void **) &d_datasetInput, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double));
    cudaMalloc((void **) &d_resultOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_datasetOutput, N_SAMPLES * sizeof(double));
    cudaMalloc((void **) &d_resultFitness, sizeof(double));

    //  copy input matrix to 1D array
    double *h_input = new double[N_SAMPLES * SAMPLE_DIMENSION];
    double *p_input = h_input;
    for (int i = 0; i < N_SAMPLES; i++) {
        const std::vector<double> &inputVector = dataset->getSampleInput(i);
        std::copy(inputVector.cbegin(), inputVector.cend(), p_input);
        p_input += SAMPLE_DIMENSION;
    }

    cudaMemcpy(d_datasetInput, h_input, N_SAMPLES * SAMPLE_DIMENSION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, &dataset->getOutputVector()[0], N_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_input;

    return true;
}

CUPostfixEvalOp::~CUPostfixEvalOp() {
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_resultOutput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultFitness);

    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "GPU time:\t" << gpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
}

FitnessP CUPostfixEvalOp::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROGRAM_SIZE;
    convertToPostfix(individual, programBuffer, PROGRAM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<double> d_result;
    double d_fitness = d_evaluate(programBuffer, PROGRAM_SIZE, d_result);


    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(d_fitness);


    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;

    return fitness;
}

void CUPostfixEvalOp::convertToPostfix(IndividualP individual, char *buffer, uint &PROGRAM_SIZE) {
    DBG(cerr << "=====================================================" << endl;)

    DBG(
            uint nTrees = (uint) individual->size();
            if (nTrees != 1) {
                cerr << "more than one tree in genotype" << endl;
            }
    )

    TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(0));

    PROGRAM_SIZE = (uint) pTree->size();

    //  prefix print
    DBG(
            for (int i = 0; i < PROGRAM_SIZE; i++) {
                string primName = (*pTree)[i]->primitive_->getName();
                cerr << primName << " ";
            }
            cerr << endl;
    )

    //  convert to postfix
    stack<vector<int>> st;
    for (int i = PROGRAM_SIZE - 1; i >= 0; i--) {
        int arity = (*pTree)[i]->primitive_->getNumberOfArguments();
        if (arity == 2) {
            vector<int> op1 = st.top();
            st.pop();
            vector<int> op2 = st.top();
            st.pop();
            op1.insert(op1.end(), op2.begin(), op2.end());
            op1.push_back(i);
            st.push(op1);
        } else if (arity == 1) {
            vector<int> op1 = st.top();
            st.pop();
            op1.push_back(i);
            st.push(op1);
        } else {
            vector<int> tmp;
            tmp.push_back(i);
            st.push(tmp);
        }
    }
    vector<int> result = st.top();


    //  postfix ispis
    DBG(
            for (int i = 0; i < result.size(); i++) {
                string pName = (*pTree)[result[i]]->primitive_->getName();
                cerr << pName << " ";
            }
            cerr << endl;
    )


    DBG(cerr << "Velicina:\t" << length << endl;)

    uint *program = reinterpret_cast<uint *>( buffer);

    size_t CONSTANTS_OFFSET = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
    double *programConstants = reinterpret_cast<double *>(buffer + CONSTANTS_OFFSET);


    for (int i : result) {
        string pName = (*pTree)[i]->primitive_->getName();
        if (pName[0] == '+') {
            *program = ADD;
            program++;
        } else if (pName[0] == '-') {
            *program = SUB;
            program++;
        } else if (pName[0] == '*') {
            *program = MUL;
            program++;
        } else if (pName[0] == '/') {
            *program = DIV;
            program++;
        } else if (pName[0] == 's') {
            *program = SIN;
            program++;
        } else if (pName[0] == 'c') {
            *program = COS;
            program++;
        } else if (pName[0] == 'X') {
            string xx = pName.substr(1);
            uint idx = VAR + (uint) stoi(xx);
            *program = idx;
            program++;
        } else if (pName == "1") {
            *program = CONST;
            program++;
            *programConstants = 1.;
            programConstants++;
        } else if (pName[0] == 'D' && pName[1] == '_') {
            *program = CONST;
            program++;
            double value;
            (*pTree)[i]->primitive_->getValue(&value);
            *programConstants = value;
            programConstants++;
        } else {
            cerr << pName << endl;
        }
    }

    // DBG(printSolution(tmp, tmpd);)

    DBG(cerr << "*******************************************************" << endl;)
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
