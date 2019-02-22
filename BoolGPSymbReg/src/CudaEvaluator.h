#pragma once


#include <ECF/ECF.h>
#include "Dataset.h"
#include "Constants.h"

class CudaEvaluator : public EvaluateOp {
public:
    ~CudaEvaluator() override;

    bool initialize(StateP state) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    uint d_evaluate(char *postfixMem, uint PROG_SIZE, vector<BOOL_TYPE> &result);

private:
    std::shared_ptr<Dataset> dataset;

    char *programBuffer;

    uint *d_program;
    BOOL_TYPE *d_datasetInput;
    BOOL_TYPE *d_datasetOutput;
    BOOL_TYPE *d_resultOutput;
    uint *d_resultFitness;

    long conversionTime, gpuTime;
};
