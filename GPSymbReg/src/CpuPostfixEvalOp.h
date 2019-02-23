#pragma once

#include "ECF/ECF.h"
#include "Dataset.h"
#include "Timer.h"

class CpuPostfixEvalOp : public EvaluateOp {
public:
    ~CpuPostfixEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    double h_evaluate(char* buffer, uint PROGRAM_SIZE, std::vector<double>& result);

    double h_evaluateIndividual(char* buffer, uint PROGRAM_SIZE, const std::vector<double>& input);

private:
    std::shared_ptr<Dataset> dataset;

    char* programBuffer;

    Timer conversionTimer, cpuTimer;
};
