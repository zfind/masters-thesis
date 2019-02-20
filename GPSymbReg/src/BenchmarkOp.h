#pragma once

#include <ECF/ECF.h>
#include "SimpleEvaluator.h"
#include "PostfixEvaluator.h"
#include "CUPostfixEvalOp.h"


class BenchmarkOp : public EvaluateOp {
public:
    ~BenchmarkOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    std::unique_ptr<SimpleEvaluator> symbRegEvalOp;
    std::unique_ptr<PostfixEvaluator> postfixEvalOp;
    std::unique_ptr<CUPostfixEvalOp> cuPostfixEvalOp;

    long ecfTime, cpuTime, gpuTime;
};

