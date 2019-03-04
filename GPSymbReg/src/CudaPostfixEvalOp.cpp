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

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(gp_code_t) + sizeof(gp_val_t) - 1)
            / sizeof(gp_val_t))
            * sizeof(gp_val_t);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(gp_val_t);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>(datasetFilename);

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
        auto inputVector = dataset->getSampleInput(i);
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



