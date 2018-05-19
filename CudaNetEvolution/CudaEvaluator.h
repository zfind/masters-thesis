//
// Created by zac on 19.05.18..
//

#ifndef CUDAEVALUATOR_H
#define CUDAEVALUATOR_H

#include "Net.h"
#include "Dataset.h"

class CudaEvaluator {
private:
    Net &net;
    Dataset &dataset;

    double *d_newPopulation;

public:
    CudaEvaluator(Net &net, Dataset &dataset);

    ~CudaEvaluator();

    double evaluateIndividual(double weights[]);

    void evaluatePopulation(double *newPopulation, int size, int dimensions,
                            vector<SolutionFitness> &newPopulationFitnessMap);

};


#endif //CUDAEVALUATOR_H
