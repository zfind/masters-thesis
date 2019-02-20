//
// Created by zac on 19.02.19..
//

#ifndef GPSYMBREG_BENCHMARKOP_H
#define GPSYMBREG_BENCHMARKOP_H

#include <vector>
#include <ECF/ECF.h>
#include "Dataset.h"

class SimpleEvaluator;
class PostfixEvaluator;
class CUPostfixEvalOp;


class BenchmarkOp : public EvaluateOp{
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


#endif //GPSYMBREG_BENCHMARKOP_H
