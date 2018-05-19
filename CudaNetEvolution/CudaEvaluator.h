//
// Created by zac on 19.05.18..
//

#ifndef CUDANETEVOLUTION_CUDAEVALUATOR_H
#define CUDANETEVOLUTION_CUDAEVALUATOR_H

#include "Net.h"
#include "Dataset.h"

class CudaEvaluator {
private:
    Net &net;
    Dataset &dataset;

    double *d_newPopulation;
//    int populationSize;
//    int DIM;

public:
    CudaEvaluator(Net &net, Dataset &dataset) :
            net(net), dataset(dataset) {
        //    populationSize = size;
//    DIM = dimensions;
        d_newPopulation = nullptr;
//    cudaMalloc((void **) &d_newPopulation, populationSize * DIM * sizeof(double));

    }

    double evaluate(double weights[]);

    void evaluateNewParallel(double *newPopulation, int size, int dimensions,
                             vector<SolutionFitness> &newPopulationFitnessMap);

};


#endif //CUDANETEVOLUTION_CUDAEVALUATOR_H
