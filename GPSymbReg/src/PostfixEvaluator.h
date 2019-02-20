//
// Created by zac on 18.02.19..
//

#ifndef GPSYMBREG_POSTFIXEVALUATOR_H
#define GPSYMBREG_POSTFIXEVALUATOR_H

#include "ECF/ECF.h"
#include "Dataset.h"


class PostfixEvaluator : public EvaluateOp {
public:
    ~PostfixEvaluator() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

    double h_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result);

    double h_evaluateIndividual(char *buffer, uint PROGRAM_SIZE, const std::vector<double> &input);


private:
    std::shared_ptr<Dataset> dataset;

    char *programBuffer;

    void convertToPostfix(IndividualP individual, char *buffer, uint &PROGRAM_SIZE);

    long conversionTime, cpuTime;
};


#endif //GPSYMBREG_POSTFIXEVALUATOR_H
