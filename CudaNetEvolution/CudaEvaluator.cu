//
// Created by zac on 19.05.18..
//

#include "CudaEvaluator.h"

CudaEvaluator::CudaEvaluator(Net &net, Dataset &dataset) :
        net(net), dataset(dataset) {
    d_newPopulation = nullptr;
}

CudaEvaluator::~CudaEvaluator() {
    if (d_newPopulation != nullptr) {
        cudaFree(d_newPopulation);
    }
}

double CudaEvaluator::evaluateIndividual(double *weights) {
    double fitness = net.evaluate(weights, dataset);
    return fitness;
}

void CudaEvaluator::evaluatePopulation(double *newPopulation, int size, int dimensions,
                                       vector<SolutionFitness> &newPopulationFitnessMap) {
    if (d_newPopulation == nullptr) {
        cudaMalloc((void **) &d_newPopulation, size * dimensions * sizeof(double));
    }
    cudaMemcpy(d_newPopulation,
               newPopulation, (size) * dimensions * sizeof(double),
               cudaMemcpyHostToDevice);

    newPopulationFitnessMap.clear();

    for (int i = 0; i < size; i++) {
        double fitness = net.evaluateGPU(&d_newPopulation[i * dimensions], dataset);
        newPopulationFitnessMap.push_back(SolutionFitness(fitness, i));
    }
}
