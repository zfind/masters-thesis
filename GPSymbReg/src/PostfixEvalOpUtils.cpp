#include "PostfixEvalOpUtils.h"

#include <stack>

#define DBG(x)

void PostfixEvalOpUtils::ConvertToPostfix(IndividualP individual, char* programBuffer, int& programSize)
{
    DBG(std::cerr << "=====================================================" << std::endl;)

    DBG(
            uint nTrees = (uint) individual->size();
            if (nTrees != 1) {
                std::cerr << "more than one tree in genotype" << std::endl;
            }
    )

    TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(0));

    programSize = pTree->size();

    //  prefix print
    DBG(
            for (int i = 0; i < programSize; i++) {
                string primName = (*pTree)[i]->primitive_->getName();
                std::cerr << primName << " ";
            }
            std::cerr << endl;
    )

    //  convert to postfix
    std::stack<std::vector<int>> st;
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
    std::vector<int> result = st.top();


    //  postfix print
    DBG(
            for (int i = 0; i < result.size(); i++) {
                string pName = (*pTree)[result[i]]->primitive_->getName();
                std::cerr << pName << " ";
            }
            std::cerr << endl;
    )


    DBG(std::cerr << "Length:\t" << length << endl;)

    gp_code_t* program = reinterpret_cast<gp_code_t*>( programBuffer);

    size_t CONSTANTS_OFFSET =
            (int) ((programSize * sizeof(gp_code_t) + sizeof(gp_val_t) - 1) / sizeof(gp_val_t)) * sizeof(gp_val_t);
    gp_val_t* programConstants = reinterpret_cast<gp_val_t*>(programBuffer + CONSTANTS_OFFSET);

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
            std::string xx = pName.substr(1);
            gp_code_t idx = VAR + stoi(xx);
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
            gp_val_t value;
            (*pTree)[i]->primitive_->getValue(&value);
            *programConstants = value;
            programConstants++;
        }
        else {
            std::cerr << "ERROR: Can't convert to postfix, unknown node name " << pName << std::endl;
            return;
        }
    }


    DBG(std::cerr << "*******************************************************" << std::endl;)
}

