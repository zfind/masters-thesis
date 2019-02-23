#include <memory>
#include <ECF/ECF.h>
#include "BenchmarkEvalOp.h"

int main(int argc, char** argv)
{
    StateP state(new State);

    auto benchmarkOp = std::make_unique<BenchmarkEvalOp>();
    state->setEvalOp(benchmarkOp.get());

    state->initialize(argc, argv);
    state->run();

    return 0;
}
