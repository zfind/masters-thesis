//
// Created by zac on 01.05.18..
//

#ifndef SymbRegEvalOp_h
#define SymbRegEvalOp_h

#include <cfloat>
#include <ecf/ECF.h>
#include "Primitives.cpp"
#include "CudaEvaluator.h"


class SymbRegEvalOp : public EvaluateOp {
public:
    FitnessP evaluate(IndividualP individual) override;

    bool initialize(StateP) override;

    ~SymbRegEvalOp() override;

private:
    uint NUM_SAMPLES;
    std::vector<std::vector<bool>> datasetInput;
    std::vector<std::vector<bool>> domain;
    std::vector<bool> codomain;
    void loadFromFile(std::string filename, std::vector<std::vector<bool>> &matrix, std::vector<bool> &output);

    char *postfixBuffer;
    void convertToPostfixNew(IndividualP individual, char *postfixMem, uint &PROG_SIZE, uint &MEM_SIZE);

    CudaEvaluator *evaluator;

    long conversionTime, ecfTime, cpuTime, gpuTime;
};

typedef boost::shared_ptr<SymbRegEvalOp> SymbRegEvalOpP;



#endif // SymbRegEvalOp_h
