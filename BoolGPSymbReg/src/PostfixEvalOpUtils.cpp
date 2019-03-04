#include "PostfixEvalOpUtils.h"

#include <stack>

#define DBG(x)

void PostfixEvalOpUtils::ConvertToPostfix(IndividualP individual, char* postfixMem, int& PROG_SIZE)
{
    DBG(std::cerr << "=====================================================" << std::endl;)

    uint nTreeSize, nTree;
    uint nTrees = (uint) individual->size();
    for (nTree = 0; nTree < nTrees; nTree++) {
        TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(nTree));
        nTreeSize = (uint) pTree->size();

        //  prefix print
        DBG(
                for (int i = 0; i < nTreeSize; i++) {
                    string primName = (*pTree)[i]->primitive_->getName();
                    std::cerr << primName << " ";
                }
                std::cerr << endl;)

        //  convert to postfix
        std::stack<std::vector<int>> st;
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
        std::vector<int> result = st.top();


        //  postfix print
        DBG(
                for (int i = 0; i < result.size(); i++) {
                    string pName = (*pTree)[result[i]]->primitive_->getName();
                    std::cerr << pName << " ";
                }
                std::cerr << endl;)


        DBG(std::cerr << "Length:\t" << length << endl;)

        PROG_SIZE = length;
        gp_code_t* program = reinterpret_cast<gp_code_t*>(postfixMem);

        for (int i : result) {
            string pName = (*pTree)[i]->primitive_->getName();
            if (pName == "AND") {
                *program = AND;
                program++;
            }
            else if (pName == "OR") {
                *program = OR;
                program++;
            }
            else if (pName == "NOT") {
                *program = NOT;
                program++;
            }
            else if (pName == "XOR") {
                *program = XOR;
                program++;
            }
            else if (pName == "XNOR") {
                *program = XNOR;
                program++;
            }
            else if (pName == "NAND") {
                *program = NAND;
                program++;
            }
            else if (pName == "NOR") {
                *program = NOR;
                program++;
            }
            else if (pName[0] == 'v') {
                string xx = pName.substr(1);
                gp_code_t idx = VAR + stoi(xx);
                *program = idx;
                program++;
            }
            else {
                std::cerr << "ERROR: Can't convert to postfix, unknown node name " << pName << std::endl;
                return;
            }
        }
    }

    DBG(cerr << "*******************************************************" << endl;)
}
