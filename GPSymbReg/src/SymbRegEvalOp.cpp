#include "SymbRegEvalOp.h"

// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state)
{
    dataset = std::make_shared<Dataset>("data/input.txt");

    return true;
}

FitnessP SymbRegEvalOp::evaluate(IndividualP individual)
{
    ecfTimer.start();

    //  legacy ECF evaluate

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree* tree = (Tree::Tree*) individual->getGenotype().get();

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

    ecfTimer.pause();

    return fitness;
}

SymbRegEvalOp::~SymbRegEvalOp()
{
    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "ECF time:\t" << ecfTimer.get() << endl;
}
