#include "CudaPostfixEvalOp.h"

#include <stack>
#include <cuda_runtime_api.h>

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

    dataset = std::make_unique<Dataset>(datasetFilename);

    size_t BUFFER_SIZE = MAX_PROGRAM_SIZE * sizeof(gp_code_t);
    programBuffer = new char[BUFFER_SIZE];

    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();

    cudaMalloc((void**) &d_program, BUFFER_SIZE);
    cudaMalloc((void**) &d_datasetInput, NUM_SAMPLES * INPUT_DIMENSION * sizeof(gp_val_t));
    cudaMalloc((void**) &d_resultOutput, NUM_SAMPLES * sizeof(gp_val_t));
    cudaMalloc((void**) &d_datasetOutput, NUM_SAMPLES * sizeof(gp_val_t));
    cudaMalloc((void**) &d_resultFitness, sizeof(gp_fitness_t));

    //  copy input matrix to 1D array
    gp_val_t* h_input = new gp_val_t[NUM_SAMPLES * INPUT_DIMENSION];
    gp_val_t* h_output = new gp_val_t[NUM_SAMPLES];
    for (int y = 0; y < NUM_SAMPLES; ++y) {
        for (int x = 0; x < INPUT_DIMENSION; ++x) {
            h_input[y * INPUT_DIMENSION + x] = dataset->getSampleInput(y)[x];
        }
        h_output[y] = dataset->getSampleOutput(y);
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

    conversionTimer.start();
    int programSize;
    PostfixEvalOpUtils::ConvertToPostfix(individual, programBuffer, programSize);
    conversionTimer.pause();

    vector<gp_val_t> d_result;  // TODO move to d_evaluate() if needed
    gp_fitness_t d_fitness = d_evaluate(programBuffer, programSize, d_result);

    FitnessP fitness(new FitnessMin);
    fitness->setValue(d_fitness);

    gpuTimer.pause();

    return fitness;
}
