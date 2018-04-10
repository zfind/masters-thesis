#include <cmath>
#include <ecf/ECF.h>
#include <stack>
#include "SymbRegEvalOp.h"
#include <chrono>

#define VAR_X0   0x00000000
#define VAR_X1   0x00000001
#define VAR_X2   0x00000002
#define VAR_X3   0x00000003
#define VAR_X4   0x00000004

#define CONST   0x0000000FF

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3

#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define COS 0xFFFFFFF6

#define ERR 0xFFFFFFFF

#define OPERAND 0
#define UNARY   1
#define BINARY  2

void printSolution(vector<uint> &solution, vector<double> &solutionConst) {
    for (uint i = 0; i < solution.size(); i++) {
//        cerr << i << ": ";
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


double evaluate1(vector<uint> &solution, vector<double> &solutionConst, vector<double> &input, int validLength) {
//    int validLength = getValidLength(solution);

    double *stack = new double[validLength];
    int SP = 0;

    double o1, o2, tmp;

    for (int i = 0; i < validLength; i++) {
        switch (solution[i]) {
            case ADD:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 + o2;

                stack[SP++] = tmp;
                break;
            case SUB:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 - o2;

                stack[SP++] = tmp;
                break;
            case MUL:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = o1 * o2;

                stack[SP++] = tmp;
                break;
            case DIV:
                o2 = stack[--SP];
                o1 = stack[--SP];

                tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1.;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            case COS:
                o1 = stack[--SP];

                tmp = cos(o1);

                stack[SP++] = tmp;
                break;
            case VAR_X0:
                tmp = input[0];

                stack[SP++] = tmp;
//                cerr << "st:\t" << stack[SP-1] << endl;
                break;
            case VAR_X1:
                tmp = input[1];

                stack[SP++] = tmp;
                break;
            case VAR_X2:
                tmp = input[2];

                stack[SP++] = tmp;
                break;
            case VAR_X3:
                tmp = input[3];

                stack[SP++] = tmp;
                break;
            case VAR_X4:
                tmp = input[4];

                stack[SP++] = tmp;
                break;
            case CONST:
                tmp = solutionConst[i];

                stack[SP++] = tmp;
//                cerr << "st:\t" << stack[SP-1] << endl;
                break;
            case ERR:
            default:
                cerr<< "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl;
                return -1.;
        }
    }

//    cerr << "SP:\t" << SP << endl;
    double result = stack[--SP];

    delete[] stack;

//    cerr << result << endl;
    return result;
}


void evaluateHost(vector<uint> &program, vector<double> &programConst, vector<vector<double>> &input, vector<double> &result) {
    int N = input.size();
    result.resize(N, 0.);

//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        result[i] = evaluate1(program, programConst, input[i], program.size());
    }
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    std::cerr << "CPU Time difference [us] = "
//              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
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

    for (int i = 0; i < nSamples; i++) {
        cerr << i << ":\t" << datasetInput[i][0] << endl;
    }
    return true;
}


void convertToPostfix(IndividualP individual, vector<uint> &solution, vector<double> &solutionConstants) {
    cerr << "=====================================================" << endl;

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

        //  postfix ispis
        vector<int> result = st.top();
        for (int i = 0; i < result.size(); i++) {
            string pName = (*pTree)[result[i]]->primitive_->getName();
            cerr << pName << " ";
        }
        cerr << endl;


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

        printSolution(tmp, tmpd);
        solution = tmp;
        solutionConstants = tmpd;
    }

    cerr << "*******************************************************" << endl;

}


FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

    vector<uint> postfix;
    vector<double> postfixConstants;
    convertToPostfix(individual, postfix, postfixConstants);

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();
    // (you can also use boost smart pointers:)
    //TreeP tree = boost::static_pointer_cast<Tree::Tree> (individual->getGenotype());

    vector<double> resvec;
    evaluateHost(postfix, postfixConstants, datasetInput, resvec);

    double value = 0;
    for (uint i = 0; i < nSamples; i++) {
        // for each test data instance, the x value (domain) must be set
        tree->setTerminalValue("X", &domain[i]);
        // get the y value of the current tree
        double result;
        tree->execute(&result);
        double myresult;
//        myresult = evaluate1(postfix, postfixConstants, datasetInput[i], postfix.size());
        // add the difference
        value += fabs(codomain[i] - result);
        cerr << "real:\t" << codomain[i] << "\tcurr:\t" << result << "\thost:\t" << resvec[i] << endl;
    }
    fitness->setValue(value);

    return fitness;
}
