#include "PostfixEvaluator.h"

#include <chrono>
#include <stack>
#include "Constants.h"
#include "Utils.h"


using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);


// called only once, before the evolution  generates training data
bool PostfixEvaluator::initialize(StateP state) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>("data/input.txt");

    conversionTime = 0L;
    cpuTime = 0L;

    return true;
}


FitnessP PostfixEvaluator::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROGRAM_SIZE;
    Utils::convertToPostfix(individual, programBuffer, PROGRAM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    //  evaluate on CPU
    begin = std::chrono::steady_clock::now();
    vector<double> h_result;
    double h_fitness = h_evaluate(programBuffer, PROGRAM_SIZE, h_result);

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(h_fitness);


    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += postfixConversionTime + diff;

    return fitness;
}

PostfixEvaluator::~PostfixEvaluator() {
    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "CPU time:\t" << cpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
}


double PostfixEvaluator::h_evaluate(char *buffer, uint PROGRAM_SIZE, std::vector<double> &result) {
    result.resize(dataset->size(), 0.);

    double fitness = 0.;
    for (int i = 0; i < dataset->size(); i++) {
        result[i] = h_evaluateIndividual(buffer, PROGRAM_SIZE, dataset->getSampleInput(i));
        fitness += fabs(dataset->getSampleOutput(i) - result[i]);
    }

    return fitness;
}

double PostfixEvaluator::h_evaluateIndividual(char *buffer, uint PROGRAM_SIZE, const std::vector<double> &input) {

    uint *program = reinterpret_cast<uint *>(buffer);

    size_t BUFFER_PROGRAM_SIZE = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    double *programConstants = reinterpret_cast<double *>(buffer + BUFFER_PROGRAM_SIZE);


    double stack[PROGRAM_SIZE];

    int SP = 0;
    double o1, o2, tmp;

    for (int i = 0; i < PROGRAM_SIZE; i++) {

        if (program[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (program[i]) {
                case ADD:
                    tmp = o1 + o2;
                    break;
                case SUB:
                    tmp = o1 - o2;
                    break;
                case MUL:
                    tmp = o1 * o2;
                    break;
                case DIV:
                    tmp = (fabs(o2) > 0.000000001) ? o1 / o2 : 1.;
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (program[i]) {
                case SQR:
                    tmp = (o1 >= 0.) ? sqrt(o1) : 1.;
                    break;
                case SIN:
                    tmp = sin(o1);
                    break;
                case COS:
                    tmp = cos(o1);
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] == CONST) {
            tmp = *programConstants;
            programConstants++;

        } else if (program[i] >= VAR && program[i] < CONST) {
            uint code = program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            CPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    double result = stack[--SP];
    return result;
}
