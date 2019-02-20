#include <memory>
#include <ECF/ECF.h>
#include "Dataset.h"
#include "SimpleEvaluator.h"
#include "PostfixEvaluator.h"
#include "CUPostfixEvalOp.h"
#include "BenchmarkOp.h"


int main(int argc, char **argv) {
    StateP state(new State);

    auto benchmarkOp = std::make_unique<BenchmarkOp>();
    state->setEvalOp(benchmarkOp.get());

    state->initialize(argc, argv);
    state->run();


    return 0;
}
