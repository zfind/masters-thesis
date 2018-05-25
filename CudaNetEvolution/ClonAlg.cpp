//
// Created by zac on 06.01.18..
//


#include "ClonAlg.h"


ClonAlg::ClonAlg(int populationSize, double minimalFitness, int maxIters, int dimensions, CudaEvaluator &evaluator) :
        populationSize(populationSize),
        minimalFitness(minimalFitness),
        maxIter(maxIters),
        dimensions(dimensions),
        rng(0),
        uniformRealDistribution(uniform_real_distribution<double>(0., 1.)),
        normalDistribution(normal_distribution<double>(0., 1.)),
        evaluator(evaluator) {


    int maxClones = 0;
    for (int i = 1; i <= populationSize; i++) {
        maxClones += (int) ((BETA * populationSize) / (double) i);
    }
    newPopulationSize = maxClones;
    newPopulation = (double *) malloc((newPopulationSize + D) * dimensions * sizeof(double));
    vector<pair<double, int>> newPopulationMap;

//    cudaMalloc((void **) &d_newPopulation, (newPopulationSize + D) * dimensions * sizeof(double));

    for (int i = 0; i < populationSize; i++) {
        vector<double> weights(dimensions);
        for (int j = 0; j < dimensions; j++) {
            weights[j] = uniformRealDistribution(rng) * (WEIGHT_UPPER_BOUND - WEIGHT_LOWER_BOUND) + WEIGHT_LOWER_BOUND;
        }
        double fitness = evaluator.evaluateIndividual(&weights[0]);
        population.push_back(Solution(fitness, weights));
    }

//    printPopulation();
//    population = create(populationSize);
}

ClonAlg::~ClonAlg() {
    free(newPopulation);
    cudaFree(d_newPopulation);
}

Solution &ClonAlg::run() {

    for (int i = 0; i < maxIter; i++) {
        newClonedMutated();
        newCreated();
        evaluateNew();

        population.clear();
        selectNewSolutions(0, newPopulationSize);
        selectNewSolutions(newPopulationSize, newPopulationSize + D);

        Solution &best = pickBest();
        if (best.first <= minimalFitness) {
            break;
        }
        if (i % 10 == 0) {
            cout << i << ":\t" << best.first << endl;
        }
    }

    Solution &best = pickBest();
    cout << "Rjesenje:\t" << best.first << endl;

    return best;
}

Solution &ClonAlg::runParallel() {

    for (int i = 0; i < maxIter; i++) {
        newClonedMutated();
        newCreated();
        evaluateNewParallel();

        population.clear();
        selectNewSolutions(0, newPopulationSize);
        selectNewSolutions(newPopulationSize, newPopulationSize + D);

        Solution &best = pickBest();
        if (best.first <= minimalFitness) {
            break;
        }
        if (i % 10 == 0) {
            cout << i << ":\t" << best.first << endl;
        }
    }

    Solution &best = pickBest();
    cout << "Rjesenje:\t" << best.first << endl;

    return best;
}

void ClonAlg::newClonedMutated() {
    sort(population.begin(), population.end());

    int currentIdx = 0;

    for (int i = 1; i <= populationSize; i++) {
        Solution &origin = population[i - 1];
        int numOfClones = (int) ((BETA * populationSize) / (double) i);
        for (int j = 0; j < numOfClones; j++) {
            memcpy(&newPopulation[currentIdx * dimensions],
                   &origin.second[0],
                   dimensions * sizeof(double));

            int mutationCount = (int) ((currentIdx / (double) newPopulationSize) * dimensions);
            for (int k = 0; k < mutationCount; k++) {
                int idx = (int) (uniformRealDistribution(rng) * (double) dimensions);
                newPopulation[currentIdx * dimensions + idx] += normalDistribution(rng);
            }

            currentIdx++;
        }

    }

}

void ClonAlg::newCreated() {
    for (int i = newPopulationSize; i < newPopulationSize + D; i++) {
        for (int j = 0; j < dimensions; j++) {
            newPopulation[i * dimensions + j] = uniformRealDistribution(rng)
                                                * (WEIGHT_UPPER_BOUND - WEIGHT_LOWER_BOUND)
                                                + WEIGHT_LOWER_BOUND;
        }
    }
}

void ClonAlg::selectNewSolutions(int beginIdx, int endIdx) {
    sort(newPopulationFitnessMap.begin() + beginIdx, newPopulationFitnessMap.begin() + endIdx);
    for (int i = beginIdx; i < endIdx; i++) {
        SolutionFitness &origin = newPopulationFitnessMap[i];
        vector<double> weights(dimensions);
        memcpy(&weights[0],
               &newPopulation[origin.second * dimensions],
               dimensions * sizeof(double));
        double fitness = origin.first;
        population.push_back(Solution(fitness, weights));
    }
}

void ClonAlg::evaluateNew() {
    newPopulationFitnessMap.clear();
    for (int i = 0; i < newPopulationSize + D; i++) {
        double fitness = evaluator.evaluateIndividual(&newPopulation[i * dimensions]);
        newPopulationFitnessMap.push_back(SolutionFitness(fitness, i));
    }
}

void ClonAlg::evaluateNewParallel() {
    evaluator.evaluatePopulation(newPopulation, newPopulationSize + D, dimensions, newPopulationFitnessMap);
}

Solution &ClonAlg::pickBest() {
    double bestFitness = 1.E7;
    Solution *best;
    for (int i = 0; i < populationSize; i++) {
        if (population[i].first <= bestFitness) {
            best = &population[i];
            bestFitness = best->first;
        }
    }
    return *best;
}

void ClonAlg::printPopulation() {
    printf("----------\n");
    for (auto &i : population) {
        printf("\t%f\t%f %f %f\n", i.first, i.second[0], i.second[1], i.second[2]);
    }
    printf("----------\n");
}

void ClonAlg::printPopulation(vector<Solution> &population) {
    printf("----------\n");
    for (auto &i : population) {
        printf("\t%f\t%f %f %f\n", i.first, i.second[0], i.second[1], i.second[2]);
    }
    printf("----------\n");
}
