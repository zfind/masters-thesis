#pragma once


#include <ECF/ECF.h>
#include "SymbRegEvalOp.h"
#include "PostfixEvaluator.h"
#include "CudaEvaluator.h"

class BenchmarkOp : public EvaluateOp {
public:
    ~BenchmarkOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;


private:
    std::unique_ptr<SymbRegEvalOp> simpleEvaluator;
    std::unique_ptr<PostfixEvaluator> postfixEvalOp;
    std::unique_ptr<CudaEvaluator> cudaEvalOp;

    long ecfTime, cpuTime, gpuTime;
};

typedef boost::shared_ptr<BenchmarkOp> BenchmarkOpP;
