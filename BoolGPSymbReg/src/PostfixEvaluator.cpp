#include "PostfixEvaluator.h"

#include <chrono>
#include <stack>
#include "Utils.h"


using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);

// called only once, before the evolution  generates training data
bool PostfixEvaluator::initialize(StateP state) {

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));
    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_unique<Dataset>("data/input.txt");

    conversionTime = 0L;
    cpuTime = 0L;


    int NUM_SAMPLES = dataset->size();
    int INPUT_DIMENSION = dataset->dim();

    this->datasetInput.resize(NUM_SAMPLES);
    this->datasetOutput.resize(NUM_SAMPLES);

    for (uint y = 0; y < NUM_SAMPLES; y++) {
        this->datasetInput[y].resize(INPUT_DIMENSION);
        for (uint x = 0; x < INPUT_DIMENSION; x++) {
            this->datasetInput[y][x] = (BOOL_TYPE) dataset->getSampleInput(y)[x];
        }
        this->datasetOutput[y] = (BOOL_TYPE) dataset->getSampleOutput(y);
    }

    return true;
}

PostfixEvaluator::~PostfixEvaluator() {
    delete programBuffer;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "CPU time:\t" << cpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
}

FitnessP PostfixEvaluator::evaluate(IndividualP individual) {

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROG_SIZE, MEM_SIZE;
    Utils::convertToPostfixNew(individual, programBuffer, PROG_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    //  evaluate on CPU
    begin = std::chrono::steady_clock::now();
    vector<BOOL_TYPE> h_result;
    uint h_fitness = h_evaluate(programBuffer, PROG_SIZE, h_result);


    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(h_fitness);


    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += postfixConversionTime + diff;

    return fitness;
}


uint PostfixEvaluator::h_evaluate(char *buffer, uint PROG_SIZE, vector<BOOL_TYPE> &result) {
    result.resize(dataset->size(), 0);

    uint fitness = 0;
    for (int i = 0; i < dataset->size(); i++) {
        result[i] = h_evaluateIndividual(buffer, PROG_SIZE, datasetInput[i]);
        if (result[i] != datasetOutput[i]) {
            fitness++;
        }
    }

    return fitness;
}

BOOL_TYPE
PostfixEvaluator::h_evaluateIndividual(char *postfixMem, uint PROG_SIZE, const std::vector<BOOL_TYPE> &input) {
    uint *program = (uint *) postfixMem;

    BOOL_TYPE stack[PROG_SIZE];

    int SP = 0;
    BOOL_TYPE o1, o2, tmp;

    for (int i = 0; i < PROG_SIZE; i++) {

        if (program[i] >= ARITY_2) {
            o2 = stack[--SP];
            o1 = stack[--SP];

            switch (program[i]) {
                case AND:
                    tmp = o1 && o2;
                    break;
                case OR:
                    tmp = o1 || o2;
                    break;
                case XOR:
                    tmp = (o1 && !o2) || (!o1 && o2);
                    break;
                case XNOR:
                    tmp = (!(o1 && !o2) || (!o1 && o2));
                    break;
                case NAND:
                    tmp = (!o1) || (!o2);
                    break;
                case NOR:
                    tmp = (!o1) && (!o2);
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (program[i]) {
                case NOT:
                    tmp = !o1;
                    break;
                default:
                    CPU_EVALUATE_ERROR
            }

        } else if (program[i] >= VAR && program[i] < CONST) {
            uint code = program[i];
            uint idx = code - VAR;
            tmp = input[idx];

        } else {
            CPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    BOOL_TYPE result = stack[--SP];
    return result;
}