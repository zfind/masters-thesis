#pragma once

#include "ECF/ECF.h"
#include "Dataset.h"
#include "Timer.h"
#include "PostfixEvalOpUtils.h"

class CpuPostfixEvalOp : public EvaluateOp {
public:
    ~CpuPostfixEvalOp() override;

    void registerParameters(StateP state) override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    gp_fitness_t h_evaluate(char* buffer, int programSize, std::vector<gp_val_t>& result);

    gp_val_t h_evaluateIndividual(char* buffer, int programSize, const std::vector<gp_val_t>& input);

private:
    std::shared_ptr<Dataset> dataset;

    std::vector<std::vector<gp_val_t>> datasetInput;
    std::vector<gp_val_t> datasetOutput;

    char* programBuffer;

    Timer conversionTimer, cpuTimer;

    std::function<void(int, std::string)> LOG;
};
