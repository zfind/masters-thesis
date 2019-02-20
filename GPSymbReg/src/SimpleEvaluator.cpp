//
// Created by zac on 18.02.19..
//

#include <chrono>
#include "SimpleEvaluator.h"

// called only once, before the evolution  generates training data
bool SimpleEvaluator::initialize(StateP state) {

    dataset = std::make_shared<Dataset>("data/input.txt");

    ecfTime = 0L;

    return true;
}


FitnessP SimpleEvaluator::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  legacy ECF evaluate

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();

    begin = std::chrono::steady_clock::now();
    double value = 0;
    for (uint i = 0; i < dataset->size(); i++) {
        // for each test data instance, the x value (domain) must be set
        // tree->setTerminalValue("X", &domain[i]);
        auto inputVector = dataset->getSampleInput(i);
        tree->setTerminalValue("X0", &inputVector[0]);
        tree->setTerminalValue("X1", &inputVector[1]);
        tree->setTerminalValue("X2", &inputVector[2]);
        tree->setTerminalValue("X3", &inputVector[3]);
        tree->setTerminalValue("X4", &inputVector[4]);
        // get the y value of the current tree
        double result;
        tree->execute(&result);
        // add the difference
        value += fabs(dataset->getSampleOutput(i) - result);
    }
    fitness->setValue(value);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    ecfTime += diff;

    return fitness;
}

SimpleEvaluator::~SimpleEvaluator() {
    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "ECF time:\t" << ecfTime << endl;
}
