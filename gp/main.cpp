#include <iostream>
#include <vector>
#include <stack>
#include <cstdlib>
#include <ctime>

using namespace std;

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3
#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define ERR 0xFFFFFFFF

typedef unsigned int uint;


uint generateBinaryOperator() {
    uint element;
    int choice = rand() % 4;
    if (choice == 0) {
        return ADD;
    } else if (choice == 1) {
        return SUB;
    } else if (choice == 2) {
        return MUL;
    } else if (choice == 3) {
        return DIV;
    }
    return ERR;
}

uint generateUnaryOperator() {
    uint element;
    int choice = rand() % 2;
    if (choice) {
        return SQR;
    } else {
        return SIN;
    }
}

uint generateOperand() {
    uint element;
    uint choice = rand() % 4;
    return choice;
}

vector<uint> generateIndividual(int chromosomeLength) {
    int count = 0;
    int stackcount = 0;
    stack<uint> st;
    vector<uint> solution;

    uint element = ERR;

    while (count < chromosomeLength) {
        if (stackcount < 1) {
            // generate a random operand, push it to the stack, increment stackcount
            uint operand = generateOperand();
            element = operand;
            st.push(operand);
            stackcount++;
        } else if (stackcount == 1) {
            int x = rand() % 2;
            if (x) {
                uint operand = generateOperand();
                element = operand;
                st.push(operand);
                stackcount++;
            } else {
                uint unary_op = generateUnaryOperator();
                element = unary_op;
                st.pop();
                stackcount--;
                st.push(unary_op);
                stackcount++;
            }
        } else if (stackcount > 1) {
            int y = rand() % 2;
            if (y) {
                uint operand = generateOperand();
                element = operand;
                st.push(operand);
                stackcount++;
            } else {
                int x = rand() % 2;
                if (x) {
                    uint unary_op = generateUnaryOperator();
                    element = unary_op;
                    st.pop();
                    stackcount--;
                    st.push(unary_op);
                    stackcount++;
                } else {
                    uint binary_op = generateBinaryOperator();
                    element = binary_op;
                    st.pop();
                    st.pop();
                    stackcount -= 2;
                    st.push(binary_op);
                    stackcount++;
                }
            }
        }
        solution.push_back(element);
        count++;
    }
    return solution;
}


int main() {
    srand((unsigned) time(nullptr));

    vector<uint> individual = generateIndividual(10);


    for (uint i : individual) {
        switch (i) {
            case ADD:
                cout << "ADD" << endl;
                break;
            case SUB:
                cout << "SUB" << endl;
                break;
            case MUL:
                cout << "MUL" << endl;
                break;
            case DIV:
                cout << "DIV" << endl;
                break;
            case SQR:
                cout << "SQR" << endl;
                break;
            case SIN:
                cout << "SIN" << endl;
                break;
            default:
                cout << i << endl;
                break;
        }
    }

}