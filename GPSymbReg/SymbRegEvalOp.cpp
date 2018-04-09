#include <cmath>
#include <ecf/ECF.h>
#include <stack>
#include "SymbRegEvalOp.h"


// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state) {
    nSamples = 10;
    double x = -10;
    for (uint i = 0; i < nSamples; i++) {
        domain.push_back(x);
        codomain.push_back(x + sin(x));
        x += 2;
    }
    return true;
}


FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

    uint nTreeSize, nTree;
    uint nTrees = (uint) individual->size();
    for (nTree = 0; nTree < nTrees; nTree++) {
        TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(nTree));
        nTreeSize = (uint) pTree->size();

        //  prefix ispis
        for (int i = 0; i < nTreeSize; i++) {
            string primName = (*pTree)[i]->primitive_->getName();
            cerr << primName << " ";
            assert(j < TOTAL_NODES);
        }
        cerr << endl;

        //  pretvori u postfix
        stack<vector<int>> st;
        int length = nTreeSize;
        for (int i = length - 1; i >= 0; i--) {
            int arity = (*pTree)[i]->primitive_->getNumberOfArguments();
            if (arity == 2) {
                vector<int> op2 = st.top();
                st.pop();
                vector<int> op1 = st.top();
                st.pop();
                op1.insert(op1.end(), op2.begin(), op2.end());
                op1.push_back(i);
                st.push(op1);
            } else if (arity == 1) {
                vector<int> op1 = st.top();
                st.pop();
                op1.push_back(i);
                st.push(op1);
            } else {
                vector<int> tmp;
                tmp.push_back(i);
                st.push(tmp);
            }
        }

        //  postfix ispis
        vector<int> result = st.top();
        for (int i = 0; i < result.size(); i++) {
            string pName = (*pTree)[result[i]]->primitive_->getName();
            cerr << pName << " ";
        }
        cerr << endl;
    }


    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();
    // (you can also use boost smart pointers:)
    //TreeP tree = boost::static_pointer_cast<Tree::Tree> (individual->getGenotype());

    double value = 0;
    for (uint i = 0; i < nSamples; i++) {
        // for each test data instance, the x value (domain) must be set
        tree->setTerminalValue("X", &domain[i]);
        // get the y value of the current tree
        double result;
        tree->execute(&result);
        // add the difference
        value += fabs(codomain[i] - result);
    }
    fitness->setValue(value);

    return fitness;
}
