#include "Utils.h"

#include <stack>
#include "Constants.h"

using namespace std;

#define DBG(x)

void Utils::ConvertToPostfix(IndividualP individual, char* programBuffer, int& programSize)
{
    DBG(cerr << "=====================================================" << endl;)

    DBG(
            uint nTrees = (uint) individual->size();
            if (nTrees != 1) {
                cerr << "more than one tree in genotype" << endl;
            }
    )

    TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(0));

    programSize = (uint) pTree->size();

    //  prefix print
    DBG(
            for (int i = 0; i < programSize; i++) {
                string primName = (*pTree)[i]->primitive_->getName();
                cerr << primName << " ";
            }
            cerr << endl;
    )

    //  convert to postfix
    stack<vector<int>> st;
    for (int i = programSize - 1; i >= 0; i--) {
        int arity = (*pTree)[i]->primitive_->getNumberOfArguments();
        if (arity == 2) {
            vector<int> op1 = st.top();
            st.pop();
            vector<int> op2 = st.top();
            st.pop();
            op1.insert(op1.end(), op2.begin(), op2.end());
            op1.push_back(i);
            st.push(op1);
        }
        else if (arity == 1) {
            vector<int> op1 = st.top();
            st.pop();
            op1.push_back(i);
            st.push(op1);
        }
        else {
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
            cerr << endl;
    )


    DBG(cerr << "Velicina:\t" << length << endl;)

    uint* program = reinterpret_cast<uint*>( programBuffer);

    size_t CONSTANTS_OFFSET =
            (int) ((programSize * sizeof(uint) + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
    double* programConstants = reinterpret_cast<double*>(programBuffer + CONSTANTS_OFFSET);

    for (int i : result) {
        string pName = (*pTree)[i]->primitive_->getName();
        if (pName[0] == '+') {
            *program = ADD;
            program++;
        }
        else if (pName[0] == '-') {
            *program = SUB;
            program++;
        }
        else if (pName[0] == '*') {
            *program = MUL;
            program++;
        }
        else if (pName[0] == '/') {
            *program = DIV;
            program++;
        }
        else if (pName[0] == 's') {
            *program = SIN;
            program++;
        }
        else if (pName[0] == 'c') {
            *program = COS;
            program++;
        }
        else if (pName[0] == 'X') {
            string xx = pName.substr(1);
            uint idx = VAR + (uint) stoi(xx);
            *program = idx;
            program++;
        }
        else if (pName == "1") {
            *program = CONST;
            program++;
            *programConstants = 1.;
            programConstants++;
        }
        else if (pName[0] == 'D' && pName[1] == '_') {
            *program = CONST;
            program++;
            double value;
            (*pTree)[i]->primitive_->getValue(&value);
            *programConstants = value;
            programConstants++;
        }
        else {
            cerr << pName << endl;
        }
    }

    // DBG(printSolution(tmp, tmpd);)

    DBG(cerr << "*******************************************************" << endl;)
}

