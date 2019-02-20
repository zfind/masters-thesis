#pragma once

#include <ECF/ECF.h>
#include "Dataset.h"


class CUPostfixEvalOp : public EvaluateOp {
public:
    ~CUPostfixEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    double d_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result);

private:
    std::shared_ptr<Dataset> dataset;

    char *programBuffer;

    uint *d_program;
    double *d_datasetInput;
    double *d_datasetOutput;
    double *d_resultOutput;
    double *d_resultFitness;

    long conversionTime, gpuTime;
};
