#include <cmath>
#include <ecf/ECF.h>
#include <stack>
#include <chrono>
#include <limits>

#include "SymbRegEvalOp.h"
#include "Constants.h"


#define DBG(x)


void SymbRegEvalOp::printSolution(std::vector<uint> &solution, std::vector<double> &solutionConst) {
    for (uint i = 0; i < solution.size(); i++) {
        switch (solution[i]) {
            case ADD:
                cerr << "+ ";
                break;
            case SUB:
                cerr << "- ";
                break;
            case MUL:
                cerr << "* ";
                break;
            case DIV:
                cerr << "/ ";
                break;
            case SQR:
                cerr << "sqrt ";
                break;
            case SIN:
                cerr << "sin ";
                break;
            case COS:
                cerr << "cos ";
                break;
            case VAR_X0:
                cerr << "X ";
                break;
            case VAR_X1:
                cerr << "X1 ";
                break;
            case VAR_X2:
                cerr << "X2 ";
                break;
            case VAR_X3:
                cerr << "X3 ";
                break;
            case VAR_X4:
                cerr << "X4 ";
                break;
            case CONST:
                cerr << "D_";
                cerr << solutionConst[i] << " ";
                break;
            default:
                cerr << " ERR ";
                break;
        }
    }
    cerr << endl;
}




void SymbRegEvalOp::convertToPostfix(IndividualP individual, std::vector<uint> &solution,
                                     std::vector<double> &solutionConstants) {
    DBG(cerr << "=====================================================" << endl;)

    uint nTreeSize, nTree;
    uint nTrees = (uint) individual->size();
    for (nTree = 0; nTree < nTrees; nTree++) {
        TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(nTree));
        nTreeSize = (uint) pTree->size();

        //  prefix ispis
        for (int i = 0; i < nTreeSize; i++) {
            string primName = (*pTree)[i]->primitive_->getName();
            DBG(cerr << primName << " ";)
            assert(j < TOTAL_NODES);
        }
        DBG(cerr << endl;)

        //  pretvori u postfix
        stack<vector<int>> st;
        int length = nTreeSize;
        for (int i = length - 1; i >= 0; i--) {
            int arity = (*pTree)[i]->primitive_->getNumberOfArguments();
            if (arity == 2) {
                vector<int> op1 = st.top();
                st.pop();
                vector<int> op2 = st.top();
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

        vector<int> result = st.top();

        //  postfix ispis
        DBG(for (int i = 0; i < result.size(); i++) {
            string pName = (*pTree)[result[i]]->primitive_->getName();
            cerr << pName << " ";
        }
        cerr << endl;)


        vector<uint> tmp;
        vector<double> tmpd(result.size());
        for (int i = 0; i < result.size(); i++) {
            string pName = (*pTree)[result[i]]->primitive_->getName();
            if (pName == "+") {
                tmp.push_back(ADD);
            } else if (pName == "-") {
                tmp.push_back(SUB);
            } else if (pName == "*") {
                tmp.push_back(MUL);
            } else if (pName == "/") {
                tmp.push_back(DIV);
            } else if (pName == "sin") {
                tmp.push_back(SIN);
            } else if (pName == "cos") {
                tmp.push_back(COS);
            } else if (pName == "X") {
                tmp.push_back(VAR_X0);
            } else if (pName == "X1") {
                tmp.push_back(VAR_X1);
            } else if (pName == "X2") {
                tmp.push_back(VAR_X2);
            } else if (pName == "X3") {
                tmp.push_back(VAR_X3);
            } else if (pName == "X4") {
                tmp.push_back(VAR_X4);
            } else if (pName == "1") {
                tmp.push_back(CONST);
                tmpd[i] = 1.;
            } else if (pName[0] == 'D' && pName[1] == '_') {
                tmp.push_back(CONST);
                double value;
                (*pTree)[result[i]]->primitive_->getValue(&value);
                tmpd[i] = value;
            } else {
                cerr << pName << endl;
            }
        }

        DBG(printSolution(tmp, tmpd);)
        solution = tmp;
        solutionConstants = tmpd;
    }

    DBG(cerr << "*******************************************************" << endl;)

}



// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state) {
    nSamples = 10;
    double x = -10;
    for (uint i = 0; i < nSamples; i++) {
        vector<double> tmp;
        tmp.push_back(x);
        datasetInput.push_back(tmp);
        domain.push_back(x);
        codomain.push_back(x + sin(x));
        x += 2;
    }


    evaluator = new CudaEvaluator(nSamples, 1, 100, datasetInput, codomain);

    return true;
}




FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

    //  preciznost ispisa decimalnih brojeva
    cerr.precision(std::numeric_limits<double>::max_digits10);

    //  pretvori u postfix
    vector<uint> postfix;
    vector<double> postfixConstants;
    convertToPostfix(individual, postfix, postfixConstants);

    //  evaluiraj
    evaluator->evaluate(postfix, postfixConstants);


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

    cerr << "real:\t" << value << endl;

    return fitness;
}

SymbRegEvalOp::~SymbRegEvalOp() {
    delete evaluator;
}


