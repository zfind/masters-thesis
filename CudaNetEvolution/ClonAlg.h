//
// Created by zac on 06.01.18..
//

#ifndef CLONALG_H
#define CLONALG_H

#include <algorithm>
#include <vector>
#include <random>
#include <array>
#include <cstring>
#include <iostream>

#include "Common.h"
#include "CudaEvaluator.h"


using namespace std;


class ClonAlg {
private:
    std::random_device rd;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniformRealDistribution;
    std::normal_distribution<double> normalDistribution;

    const double WEIGHT_LOWER_BOUND = -1.;
    const double WEIGHT_UPPER_BOUND = 1.;
    const double BETA = 1.;
    const int D = 10;

    int maxIter;
    int populationSize;
    double minimalFitness;
    int dimensions;

    CudaEvaluator evaluator;

    vector<Solution> population;

    double *newPopulation;
    int newPopulationSize;
    vector<SolutionFitness> newPopulationFitnessMap;

    double *d_newPopulation;

public:

    ClonAlg(int populationSize, double minimalFitness, int maxIters, int dimensions, CudaEvaluator &evaluator);

    ~ClonAlg();

    Solution &run();

    Solution &runParallel();

private:

    void newClonedMutated();

    void newCreated();

    void selectNewSolutions(int beginIdx, int endIdx);

    void evaluateNew();

    void evaluateNewParallel();

    Solution &pickBest();

    void printPopulation();

    void printPopulation(vector<Solution> &population);


};


#endif //CLONALG_H
