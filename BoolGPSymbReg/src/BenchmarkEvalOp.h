#pragma once


#include <ECF/ECF.h>
#include "SymbRegEvalOp.h"
#include "CpuPostfixEvalOp.h"
#include "CudaPostfixEvalOp.h"

class BenchmarkEvalOp : public EvaluateOp {
public:
    ~BenchmarkEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;


private:
    std::unique_ptr<SymbRegEvalOp> simpleEvaluator;
    std::unique_ptr<CpuPostfixEvalOp> postfixEvalOp;
    std::unique_ptr<CudaPostfixEvalOp> cudaEvalOp;

    long ecfTime, cpuTime, gpuTime;
};

typedef boost::shared_ptr<BenchmarkEvalOp> BenchmarkOpP;
