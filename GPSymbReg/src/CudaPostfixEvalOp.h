#pragma once

#include <ECF/ECF.h>
#include "Dataset.h"
#include "Timer.h"
#include "PostfixEvalOpUtils.h"

class CudaPostfixEvalOp : public EvaluateOp {
public:
    ~CudaPostfixEvalOp() override;

    void registerParameters(StateP state) override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    gp_fitness_t d_evaluate(char* buffer, int programSize, std::vector<gp_val_t>& result);

private:
    std::shared_ptr<Dataset> dataset;

    char* programBuffer;

    gp_code_t* d_program;
    gp_val_t* d_datasetInput;
    gp_val_t* d_datasetOutput;
    gp_val_t* d_resultOutput;
    gp_fitness_t* d_resultFitness;

    Timer conversionTimer, gpuTimer;

    std::function<void(int, std::string)> LOG;
};
