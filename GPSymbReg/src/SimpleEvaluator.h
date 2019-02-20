//
// Created by zac on 18.02.19..
//

#ifndef GPSYMBREG_SIMPLEEVALUATOR_H
#define GPSYMBREG_SIMPLEEVALUATOR_H

#include <vector>
#include <ECF/ECF.h>
#include "Dataset.h"


class SimpleEvaluator : public EvaluateOp{
public:
    ~SimpleEvaluator() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    std::shared_ptr<Dataset> dataset;

    long ecfTime;
};


#endif //GPSYMBREG_SIMPLEEVALUATOR_H
