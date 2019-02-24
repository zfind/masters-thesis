#include "CudaPostfixEvalOp.h"

#include <stack>
#include <cuda_runtime_api.h>

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);
#define GPU_EVALUATE_ERROR do {d_resultOutput[tid] = NAN; return;} while(0);

extern "C"
__global__ void d_evaluateIndividual(gp_code_t* d_program,
        gp_val_t* d_datasetInput,
        gp_val_t* d_datasetOutput,
        gp_val_t* d_resultOutput,
        gp_fitness_t* d_resultFitness,
        int N, int DIM, int PROG_SIZE);

void CudaPostfixEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP) (new std::string), ECF::STRING);
}

bool CudaPostfixEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState](int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    if (!state->getRegistry()->isModified("dataset.filename"))
        return false;

    voidP pEntry = state->getRegistry()->getEntry("dataset.filename");
    std::string datasetFilename = *(static_cast<std::string*>(pEntry.get()));

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_unique<Dataset>(datasetFilename);

    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();

    cudaMalloc((void**) &d_program, MAX_PROGRAM_SIZE * sizeof(uint));
    cudaMalloc((void**) &d_datasetInput, NUM_SAMPLES * INPUT_DIMENSION * sizeof(BOOL_TYPE));
    cudaMalloc((void**) &d_resultOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void**) &d_datasetOutput, NUM_SAMPLES * sizeof(BOOL_TYPE));
    cudaMalloc((void**) &d_resultFitness, sizeof(uint));


    //  copy input matrix to 1D array
    gp_val_t* h_input = new gp_val_t[NUM_SAMPLES * INPUT_DIMENSION];
    gp_val_t* h_output = new gp_val_t[NUM_SAMPLES];
    for (int y = 0; y < NUM_SAMPLES; ++y) {
        for (int x = 0; x < INPUT_DIMENSION; ++x) {
            h_input[y * INPUT_DIMENSION + x] = (gp_val_t) dataset->getSampleInput(y)[x];
        }
        h_output[y] = (gp_val_t) dataset->getSampleOutput(y);
    }

    cudaMemcpy(d_datasetInput, h_input, NUM_SAMPLES * INPUT_DIMENSION * sizeof(gp_val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_datasetOutput, h_output, NUM_SAMPLES * sizeof(gp_val_t), cudaMemcpyHostToDevice);

    delete[] h_input;
    delete[] h_output;

    return true;
}

CudaPostfixEvalOp::~CudaPostfixEvalOp()
{
    cudaFree(d_program);
    cudaFree(d_datasetInput);
    cudaFree(d_datasetOutput);
    cudaFree(d_resultOutput);
    cudaFree(d_resultFitness);

    delete programBuffer;

    std::stringstream ss;
    ss.precision(7);
    ss << "===== STATS [us] =====" << endl;
    ss << "GPU time:\t" << gpuTimer.get() << endl;
    ss << "Conversion time: " << conversionTimer.get() << endl;
    LOG(1, ss.str());
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
