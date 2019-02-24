#include "SymbRegEvalOp.h"

void SymbRegEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP) (new std::string), ECF::STRING);
}

// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState](int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    // check if the parameters are stated (used) in the conf. file
    // if not, we return false so the initialization fails
    if (!state->getRegistry()->isModified("dataset.filename"))
        return false;

    voidP pEntry = state->getRegistry()->getEntry("dataset.filename"); // get parameter value
    std::string datasetFilename = *(static_cast<std::string*>(pEntry.get())); // convert from voidP to user defined type

    dataset = std::make_shared<Dataset>(datasetFilename);

    Tree::Tree* tree = (Tree::Tree*) state->getGenotypes().at(0).get();
    // zadaj vrijednosti obicnim varijablama (ne mijenjaju se tijekom evolucije!)

    //    for (uint i = 0; i < NUM_SAMPLES; i++) {
    // for each test data instance, the x value (domain) must be set
    // tree->setTerminalValue("X", &domain[i]);
    tree->setTerminalValue("v0", &dataset->getSampleInputVector(0));
    tree->setTerminalValue("v1", &dataset->getSampleInputVector(1));
    tree->setTerminalValue("v2", &dataset->getSampleInputVector(2));
    tree->setTerminalValue("v3", &dataset->getSampleInputVector(3));
    tree->setTerminalValue("v4", &dataset->getSampleInputVector(4));
    tree->setTerminalValue("v5", &dataset->getSampleInputVector(5));
    tree->setTerminalValue("v6", &dataset->getSampleInputVector(6));
    tree->setTerminalValue("v7", &dataset->getSampleInputVector(7));
    tree->setTerminalValue("v8", &dataset->getSampleInputVector(8));
    tree->setTerminalValue("v9", &dataset->getSampleInputVector(9));
    tree->setTerminalValue("v10", &dataset->getSampleInputVector(10));
    tree->setTerminalValue("v11", &dataset->getSampleInputVector(11));
    tree->setTerminalValue("v12", &dataset->getSampleInputVector(12));
    tree->setTerminalValue("v13", &dataset->getSampleInputVector(13));
    tree->setTerminalValue("v14", &dataset->getSampleInputVector(14));
    tree->setTerminalValue("v15", &dataset->getSampleInputVector(15));

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

    uint value = 0;
    int NUM_SAMPLES = dataset->size();

    // get the y value of the current tree
    vector<bool> result;
    result.resize(NUM_SAMPLES, 0);
    tree->execute(&result);

    for (uint i = 0; i < NUM_SAMPLES; i++) {
        // add the difference
        if (dataset->getSampleOutput(i) != result[i]) {
            value++;
        }
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
