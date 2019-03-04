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

    Dataset dataset(datasetFilename);

    for (int x = 0; x < dataset.dim(); x++) {
        vector<bool> v;
        v.resize(dataset.size(), 0);
        for (int y = 0; y < dataset.size(); y++) {
            v[y] = dataset.getSampleInput(y)[x];
        }
        domain.push_back(v);
    }

    for (int y=0; y < dataset.size(); ++y) {
        codomain.push_back(dataset.getSampleOutput(y) != '0');
    }

    Tree::Tree* tree = (Tree::Tree*) state->getGenotypes().at(0).get();
    // zadaj vrijednosti obicnim varijablama (ne mijenjaju se tijekom evolucije!)

    //    for (uint i = 0; i < NUM_SAMPLES; i++) {
    // for each test data instance, the x value (domain) must be set
    // tree->setTerminalValue("X", &domain[i]);
    tree->setTerminalValue("v0", &domain[0]);
    tree->setTerminalValue("v1", &domain[1]);
    tree->setTerminalValue("v2", &domain[2]);
    tree->setTerminalValue("v3", &domain[3]);
    tree->setTerminalValue("v4", &domain[4]);
    tree->setTerminalValue("v5", &domain[5]);
    tree->setTerminalValue("v6", &domain[6]);
    tree->setTerminalValue("v7", &domain[7]);
    tree->setTerminalValue("v8", &domain[8]);
    tree->setTerminalValue("v9", &domain[9]);
    tree->setTerminalValue("v10", &domain[10]);
    tree->setTerminalValue("v11", &domain[11]);
    tree->setTerminalValue("v12", &domain[12]);
    tree->setTerminalValue("v13", &domain[13]);
    tree->setTerminalValue("v14", &domain[14]);
    tree->setTerminalValue("v15", &domain[15]);

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
    int NUM_SAMPLES = domain[0].size();

    // get the y value of the current tree
    vector<bool> result;
    result.resize(NUM_SAMPLES, false);
    tree->execute(&result);

    for (uint i = 0; i < NUM_SAMPLES; i++) {
        // add the difference
        if (codomain[i] != result[i]) {
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
