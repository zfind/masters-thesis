#include "SymbRegEvalOp.h"

void SymbRegEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP) (new std::string), ECF::STRING);
}

// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState] (int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    // check if the parameters are stated (used) in the conf. file
    // if not, we return false so the initialization fails
    if (!state->getRegistry()->isModified("dataset.filename"))
        return false;

    voidP pEntry = state->getRegistry()->getEntry("dataset.filename"); // get parameter value
    std::string datasetFilename = *(static_cast<std::string*>(pEntry.get())); // convert from voidP to user defined type

    dataset = std::make_shared<Dataset>(datasetFilename);

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
    std::stringstream ss;
    ss.precision(7);
    ss << "===== STATS [us] =====" << endl;
    ss << "ECF time:\t" << ecfTimer.get() << endl;
    LOG(1, ss.str());
}
