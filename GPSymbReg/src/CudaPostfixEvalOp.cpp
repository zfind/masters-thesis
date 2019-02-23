#include "CudaPostfixEvalOp.h"

#include <stack>
#include <cuda_runtime_api.h>

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);

extern "C"
__global__ void d_evaluateIndividualKernel(gp_code_t* d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
        gp_val_t* d_datasetInput, gp_val_t* d_datasetOutput,
        gp_val_t* d_resultOutput, gp_fitness_t* d_resultFitness,
        int N_SAMPLES, int SAMPLE_DIMENSION);

// called only once, before the evolution  generates training data
bool CudaPostfixEvalOp::initialize(StateP state)
{
    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(gp_code_t) + sizeof(gp_val_t) - 1)
            / sizeof(gp_val_t))
            * sizeof(gp_val_t);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(gp_val_t);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>("data/input.txt");

    int N_SAMPLES = dataset->size();
    int SAMPLE_DIMENSION = dataset->dim();

    cudaMalloc((void**) &d_program, BUFFER_SIZE);
    cudaMalloc((void**) &d_datasetInput, N_SAMPLES * SAMPLE_DIMENSION * sizeof(gp_val_t));
    cudaMalloc((void**) &d_resultOutput, N_SAMPLES * sizeof(gp_val_t));
    cudaMalloc((void**) &d_datasetOutput, N_SAMPLES * sizeof(gp_val_t));
    cudaMalloc((void**) &d_resultFitness, sizeof(gp_fitness_t));

    //  copy input matrix to 1D array
    gp_val_t* h_input = new gp_val_t[N_SAMPLES * SAMPLE_DIMENSION];
    gp_val_t* p_input = h_input;
    for (int i = 0; i < N_SAMPLES; ++i) {
        const std::vector<gp_val_t>& inputVector = dataset->getSampleInput(i);
        std::copy(inputVector.cbegin(), inputVector.cend(), p_input);
        p_input += SAMPLE_DIMENSION;
    }

    cudaMemcpy(d_datasetInput, h_input, N_SAMPLES * SAMPLE_DIMENSION * sizeof(gp_val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, &dataset->getOutputVector()[0], N_SAMPLES * sizeof(gp_val_t), cudaMemcpyHostToDevice);

    delete[] h_input;

    return true;
}

CudaPostfixEvalOp::~CudaPostfixEvalOp()
{
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_resultOutput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultFitness);

    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "GPU time:\t" << gpuTimer.get() << endl;
    cerr << "Conversion time: " << conversionTimer.get() << endl;
}

FitnessP CudaPostfixEvalOp::evaluate(IndividualP individual)
{
    gpuTimer.start();

    //  convert to postfix
    conversionTimer.start();
    int programSize;
    PostfixEvalOpUtils::ConvertToPostfix(individual, programBuffer, programSize);
    conversionTimer.pause();

    // evaluate on GPU
    vector<gp_val_t> d_result;  // TODO move to d_evaluate() if needed
    gp_fitness_t d_fitness = d_evaluate(programBuffer, programSize, d_result);

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(d_fitness);

    gpuTimer.pause();

    return fitness;
}



