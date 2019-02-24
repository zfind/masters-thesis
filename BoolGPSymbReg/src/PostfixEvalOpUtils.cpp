#include "Utils.h"

#include <stack>
#include "Constants.h"

using namespace std;

#define DBG(x)

void Utils::convertToPostfixNew(IndividualP individual, char *postfixMem, uint &PROG_SIZE) {
    DBG(cerr << "=====================================================" << endl;)

    uint nTreeSize, nTree;
    uint nTrees = (uint) individual->size();
    for (nTree = 0; nTree < nTrees; nTree++) {
        TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(nTree));
        nTreeSize = (uint) pTree->size();

        //  prefix print
        DBG(
                for (int i = 0; i < nTreeSize; i++) {
                    string primName = (*pTree)[i]->primitive_->getName();
                    cerr << primName << " ";
                }
                cerr << endl;)

        //  convert to postfix
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
        DBG(
                for (int i = 0; i < result.size(); i++) {
                    string pName = (*pTree)[result[i]]->primitive_->getName();
                    cerr << pName << " ";
                }
                cerr << endl;)


        DBG(cerr << "Velicina:\t" << length << endl;)

        PROG_SIZE = length;
        uint *program = (uint *) postfixMem;

        for (int i : result) {
            string pName = (*pTree)[i]->primitive_->getName();
            if (pName == "AND") {
                *program = AND;
                program++;
            } else if (pName == "OR") {
                *program = OR;
                program++;
            } else if (pName == "NOT") {
                *program = NOT;
                program++;
            } else if (pName == "XOR") {
                *program = XOR;
                program++;
            } else if (pName == "XNOR") {
                *program = XNOR;
                program++;
            } else if (pName == "NAND") {
                *program = NAND;
                program++;
            } else if (pName == "NOR") {
                *program = NOR;
                program++;
            } else if (pName[0] == 'v') {
                string xx = pName.substr(1);
                uint idx = VAR + (uint) stoi(xx);
                *program = idx;
                program++;
//            } else if (pName == "1") {
//                *program = CONST;
//                program++;
//                *programConstants = 1.;
//                programConstants++;
//                CONST_SIZE++;
//            } else if (pName[0] == 'D' && pName[1] == '_') {
//                *program = CONST;
//                program++;
//                double value;
//                (*pTree)[i]->primitive_->getValue(&value);
//                *programConstants = value;
//                programConstants++;
//                CONST_SIZE++;
            } else {
                cerr << pName << endl;
            }
        }

        // DBG(printSolution(tmp, tmpd);)
    }

    DBG(cerr << "*******************************************************" << endl;)
}
