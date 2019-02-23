#pragma  once

#include <vector>
#include <ECF/ECF.h>
#include "Dataset.h"
#include "Timer.h"

class SymbRegEvalOp : public EvaluateOp {
public:
    ~SymbRegEvalOp() override;

    bool initialize(StateP) override;

    FitnessP evaluate(IndividualP individual) override;

private:
    std::shared_ptr<Dataset> dataset;

    Timer ecfTimer;
};
