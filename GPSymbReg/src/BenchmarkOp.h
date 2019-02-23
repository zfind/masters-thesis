#pragma once

#include <ECF/ECF.h>
#include "SymbRegEvalOp.h"
#include "CpuPostfixEvalOp.h"
#include "CudaPostfixEvalOp.h"


class BenchmarkOp : public EvaluateOp {
public:
    ~BenchmarkOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    std::unique_ptr<SymbRegEvalOp> symbRegEvalOp;
    std::unique_ptr<CpuPostfixEvalOp> postfixEvalOp;
    std::unique_ptr<CudaPostfixEvalOp> cuPostfixEvalOp;

    long ecfTime, cpuTime, gpuTime;
};

