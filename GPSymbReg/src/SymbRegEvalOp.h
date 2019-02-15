#ifndef SymbRegEvalOp_h
#define SymbRegEvalOp_h

#include <vector>

#include "CudaEvaluator.h"


class SymbRegEvalOp : public EvaluateOp {
public:
    FitnessP evaluate(IndividualP individual);

    bool initialize(StateP);

    ~SymbRegEvalOp();

private:
    uint N_SAMPLES;
    std::vector<std::vector<double>> datasetInput;
    std::vector<double> codomain;

    void loadFromFile(std::string filename, std::vector<std::vector<double>> &matrix, std::vector<double> &output);

    char *programBuffer;

    void convertToPostfix(IndividualP individual, char *buffer, uint &PROGRAM_SIZE);

    CudaEvaluator *evaluator;

    long conversionTime, ecfTime, cpuTime, gpuTime;
};


#endif // SymbRegEvalOp_h
