#pragma once

#include <ECF/ECF.h>
#include "SymbRegEvalOp.h"
#include "CpuPostfixEvalOp.h"
#include "CudaPostfixEvalOp.h"
#include "Timer.h"

class BenchmarkEvalOp : public EvaluateOp {
public:
    ~BenchmarkEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    std::unique_ptr<SymbRegEvalOp> symbRegEvalOp;
    std::unique_ptr<CpuPostfixEvalOp> cpuPostfixEvalOp;
    std::unique_ptr<CudaPostfixEvalOp> cudaPostfixEvalOp;

    Timer ecfTimer, cpuTimer, gpuTimer;
};

