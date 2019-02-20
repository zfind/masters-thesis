//
// Created by zac on 19.02.19..
//

#ifndef GPSYMBREG_CUPOSTFIXEVALOP_H
#define GPSYMBREG_CUPOSTFIXEVALOP_H

#include <ECF/ECF.h>
#include <host_defines.h>
#include "Dataset.h"

class CUPostfixEvalOp : public EvaluateOp {
public:
    ~CUPostfixEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

    double d_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result);


private:
    std::shared_ptr<Dataset> dataset;

    char *programBuffer;

    void convertToPostfix(IndividualP individual, char *buffer, uint &PROGRAM_SIZE);

    uint *d_program;
    double *d_datasetInput;
    double *d_datasetOutput;
    double *d_resultOutput;
    double *d_resultFitness;

    long conversionTime, gpuTime;
};


extern "C"
__global__ void d_evaluateIndividualKernel(uint *d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
                                           double *d_datasetInput, double *d_datasetOutput,
                                           double *d_resultOutput, double *d_resultFitness,
                                           int N_SAMPLES, int SAMPLE_DIMENSION);


#endif //GPSYMBREG_CUPOSTFIXEVALOP_H
