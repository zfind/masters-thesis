#include "CudaEvaluator.h"

#include "Constants.h"
#include <chrono>
#include <stack>
#include <cuda_runtime_api.h>
#include "Utils.h"

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);


extern "C"
__global__ void d_evaluateIndividual(uint *d_program,
                                     BOOL_TYPE *d_datasetInput,
                                     BOOL_TYPE *d_datasetOutput,
                                     BOOL_TYPE *d_resultOutput,
                                     uint *d_resultFitness,
                                     int N, int DIM, int PROG_SIZE);


bool CudaEvaluator::initialize(StateP state) {

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));
    programBuffer = new char[BUFFER_SIZE];
    dataset = std::make_unique<Dataset>("data/input.txt");

    conversionTime = 0L;
    gpuTime = 0L;

    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();


    cudaMalloc((void **) &d_program, MAX_PROGRAM_SIZE * sizeof(uint));
    cudaMalloc((void **) &d_datasetInput, NUM_SAMPLES * INPUT_DIMENSION * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_resultOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_datasetOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void **) &d_resultFitness, sizeof(uint));


    //  copy input matrix to 1D array
    BOOL_TYPE *h_input = new BOOL_TYPE[NUM_SAMPLES * INPUT_DIMENSION];
    BOOL_TYPE *h_output = new BOOL_TYPE[NUM_SAMPLES];
    for (uint y = 0; y < NUM_SAMPLES; y++) {
        for (uint x = 0; x < INPUT_DIMENSION; x++) {
            h_input[y * INPUT_DIMENSION + x] = (BOOL_TYPE) dataset->getSampleInput(y)[x];
        }
        h_output[y] = (BOOL_TYPE) dataset->getSampleOutput(y);
    }

    cudaMemcpy(d_datasetInput, h_input, NUM_SAMPLES * INPUT_DIMENSION * sizeof(BOOL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, h_output, NUM_SAMPLES * sizeof(BOOL_TYPE), cudaMemcpyHostToDevice);

    delete[] h_input;
    delete[] h_output;

    return true;
}

CudaEvaluator::~CudaEvaluator() {
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultOutput);
    cudaFree(d_resultFitness);

    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "GPU time:\t" << gpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
}

FitnessP CudaEvaluator::evaluate(IndividualP individual) {

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROGRAM_SIZE;
    Utils::convertToPostfixNew(individual, programBuffer, PROGRAM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<BOOL_TYPE> d_result;
    uint d_fitness = d_evaluate(programBuffer, PROGRAM_SIZE, d_result);


    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(d_fitness);


    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;

    return fitness;
}
