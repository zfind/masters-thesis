#include <iostream>
#include "CudaEvaluator.h"
#include "ClonAlg.h"

using namespace std;

int main() {
    Dataset dataset;

    vector<int> layers{4, 5, 3, 3};
    Net net{layers, dataset};

    CudaEvaluator evaluator{net, dataset};

    ClonAlg alg{40, 0.001, 1000, net.getWeightsCount(), evaluator};

//    Solution &solution = alg.run();

    Solution &solution = alg.runParallel();

    return 0;
}
