#ifndef GPSYMBREG_CUDAEVALUATOR_H
#define GPSYMBREG_CUDAEVALUATOR_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Constants.h"

class CudaEvaluator {
public:
    CudaEvaluator(const uint N_SAMPLES, const uint SAMPLE_DIMENSION, const uint MAX_PROGRAM_NODES,
                  const std::vector<std::vector<double>> &datasetInput, const std::vector<double> &datasetOutput);

    ~CudaEvaluator();

    double h_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result);

    double d_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result);

private:

    double h_evaluateIndividual(char *buffer, uint PROGRAM_SIZE, std::vector<double> &input);

private:
    int N_SAMPLES;
    int SAMPLE_DIMENSION;
    int MAX_PROGRAM_NODES;

    std::vector<std::vector<double>> datasetInput;
    std::vector<double> datasetOutput;

    uint *d_program;
    double *d_datasetInput;
    double *d_datasetOutput;
    double *d_resultOutput;
    double *d_resultFitness;
};


extern "C"
__global__ void d_evaluateIndividualKernel(uint *d_program, int PROGRAM_SIZE, size_t BUFFER_PROGRAM_SIZE,
                                           double *d_datasetInput, double *d_datasetOutput,
                                           double *d_resultOutput, double *d_resultFitness,
                                           int N_SAMPLES, int SAMPLE_DIMENSION);


#endif //GPSYMBREG_CUDAEVALUATOR_H
