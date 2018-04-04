#include <iostream>
#include <vector>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

#define ADD 0xFFFFFFF0
#define SUB 0xFFFFFFF1
#define MUL 0xFFFFFFF2
#define DIV 0xFFFFFFF3
#define SQR 0xFFFFFFF4
#define SIN 0xFFFFFFF5
#define ERR 0xFFFFFFFF

#define OPERAND 0
#define UNARY   1
#define BINARY  2

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

void printSolution(vector<uint> &solution) {
    for (uint i : solution) {
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
                cout << "SQT" << endl;
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

int getArity(uint i) {
    switch (i) {
        case ADD:
        case SUB:
        case MUL:
        case DIV:
            return BINARY;
        case SQR:
        case SIN:
            return UNARY;
        default:
            return OPERAND;
    }
}

int getValidLength(vector<uint> &solution) {
    int length = solution.size();
    int count = 0, validpos = 0;
    int validLength = 0;
    for (int i = 0; i < length; i++) {
        int arity = getArity(solution[i]);
        if (arity == OPERAND) {
            count++;
        } else if (arity == UNARY) {
            count = count - (UNARY - 1);
        } else if (arity == BINARY) {
            count = count - (BINARY - 1);
        }

        if (count == 0) {
            break;
        }
        if (count == 1) {
            validpos = i;
        }
    }
    validLength = validpos + 1;
    return validLength;
}


double evaluate(vector<uint> &solution, vector<double> &input) {
    int validLength = getValidLength(solution);

    double* stack = new double[validLength];
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

                tmp = (abs(o2) > 10E-9) ? o1 / o2 : 1.;

                stack[SP++] = tmp;
                break;
            case SQR:
                o1 = stack[--SP];

                tmp = (o1 >= 0.) ? sqrt(o1) : 1;

                stack[SP++] = tmp;
                break;
            case SIN:
                o1 = stack[--SP];

                tmp = sin(o1);

                stack[SP++] = tmp;
                break;
            case ERR:
                return -1.;
            default:
                tmp = input[solution[i]];

                stack[SP++] = tmp;
                break;
        }
    }

    cerr << "SP:\t" << SP << endl;
    double result = stack[--SP];

    delete[] stack;

    return result;
}


void test1() {
    vector<uint> individual = generateIndividual(10);

    printSolution(individual);

    cout << "valid length:\t" << getValidLength(individual) << endl;
}


void test2() {
    vector<uint> test = {0, 1, MUL, 1, 0, SIN, DIV, ADD};
    vector<double> input = {3., 5.};

    cout << "valid:\t" << getValidLength(test) << endl;
    double eval = evaluate(test, input);
    cout << "eval:\t" << eval << "\ttrue:\t50.4308" << endl;
}


void test3() {

}


int main() {
    srand((unsigned) time(nullptr));

    test2();

    return 0;
}