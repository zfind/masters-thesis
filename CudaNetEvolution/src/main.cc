#include <iostream>
#include "Net.hpp"
#include "ClonAlg.hpp"

using namespace std;

int main() {
    Dataset dataset;

    vector<int> layers{4, 5, 3, 3};
    Net net(layers, dataset);

    ClonAlg alg(40, 0.001, 1000, net.getWeightsCount(), net, dataset);

//    Solution &solution = alg.run();

    Solution &solution = alg.runParallel();

    return 0;
}
