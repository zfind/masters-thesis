//
// Created by zac on 19.05.18..
//

#include "CudaEvaluator.h"

double CudaEvaluator::evaluate(double *weights) {
    double fitness = net.evaluate(weights, dataset);
    return fitness;
}

void CudaEvaluator::evaluateNewParallel(double* newPopulation, int size, int dimensions, vector<SolutionFitness>& newPopulationFitnessMap) {
    if (d_newPopulation == nullptr) {
        cudaMalloc((void **) &d_newPopulation, size * dimensions * sizeof(double));
    }
    cudaMemcpy(d_newPopulation, newPopulation, (size) * dimensions * sizeof(double),
               cudaMemcpyHostToDevice);
    newPopulationFitnessMap.clear();

    for (int i = 0; i < size; i++) {
        double fitness = net.evaluateParallel(&d_newPopulation[i * dimensions], dataset);
        newPopulationFitnessMap.push_back(SolutionFitness(fitness, i));
    }
}
