#pragma once

#include "ECF/ECF.h"
#include "Dataset.h"
#include "Constants.h"


class PostfixEvaluator : public EvaluateOp {
public:
    ~PostfixEvaluator() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;


private:
    uint h_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<BOOL_TYPE> &result);

    BOOL_TYPE h_evaluateIndividual(char *buffer, uint PROGRAM_SIZE, const std::vector<BOOL_TYPE> &input);


private:
    std::shared_ptr<Dataset> dataset;

    std::vector<std::vector<BOOL_TYPE>> datasetInput;
    std::vector<BOOL_TYPE> datasetOutput;

    char *programBuffer;

    long conversionTime, cpuTime;
};
