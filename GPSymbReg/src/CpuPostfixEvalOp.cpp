#include "CpuPostfixEvalOp.h"

#include <stack>

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);

void CpuPostfixEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP)(new std::string), ECF::STRING);
}

// called only once, before the evolution  generates training data
bool CpuPostfixEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState] (int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    if (!state->getRegistry()->isModified("dataset.filename"))
        return false;

    voidP pEntry = state->getRegistry()->getEntry("dataset.filename");
    std::string datasetFilename = *(static_cast<std::string*>(pEntry.get()));

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(gp_code_t) + sizeof(gp_val_t) - 1)
            / sizeof(gp_val_t))
            * sizeof(gp_val_t);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(gp_val_t);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_shared<Dataset>(datasetFilename);

    return true;
}

FitnessP CpuPostfixEvalOp::evaluate(IndividualP individual)
{
    cpuTimer.start();

    //  convert to postfix
    conversionTimer.start();
    int programSize;
    PostfixEvalOpUtils::ConvertToPostfix(individual, programBuffer, programSize);
    conversionTimer.pause();

    //  evaluate on CPU
    vector<gp_val_t> h_result; // TODO move to h_evaluate()
    gp_fitness_t h_fitness = h_evaluate(programBuffer, programSize, h_result);

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);
    fitness->setValue(h_fitness);

    cpuTimer.pause();

    return fitness;
}

CpuPostfixEvalOp::~CpuPostfixEvalOp()
{
    delete programBuffer;

    std::stringstream ss;
    ss.precision(7);
    ss << "===== STATS [us] =====" << endl;
    ss << "CPU time:\t" << cpuTimer.get() << endl;
    ss << "Conversion time: " << conversionTimer.get() << endl;
    LOG(1, ss.str());
}

gp_fitness_t CpuPostfixEvalOp::h_evaluate(char* buffer, int programSize, std::vector<gp_val_t>& result)
{
    result.resize(dataset->size(), 0.);

    gp_fitness_t fitness = 0.;
    for (int i = 0; i < dataset->size(); ++i) {
        result[i] = h_evaluateIndividual(buffer, programSize, dataset->getSampleInput(i));
        fitness += fabs(dataset->getSampleOutput(i) - result[i]);
    }

    return fitness;
}

gp_val_t CpuPostfixEvalOp::h_evaluateIndividual(char* buffer, int programSize, const std::vector<gp_val_t>& input)
{

    gp_code_t* program = reinterpret_cast<gp_code_t*>(buffer);

    size_t BUFFER_PROGRAM_SIZE = (int) ((programSize * sizeof(gp_code_t) + sizeof(gp_val_t) - 1)
            / sizeof(gp_val_t))
            * sizeof(gp_val_t);
    gp_val_t* programConstants = reinterpret_cast<gp_val_t*>(buffer + BUFFER_PROGRAM_SIZE);

    gp_val_t stack[programSize];

    int SP = 0;
    gp_val_t o1, o2, tmp;

    for (int i = 0; i < programSize; ++i) {

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

        }
        else if (program[i] >= ARITY_1) {
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

        }
        else if (program[i] == CONST) {
            tmp = *programConstants;
            programConstants++;

        }
        else if (program[i] >= VAR && program[i] < CONST) {
            gp_code_t code = program[i];
            int idx = code - VAR;
            tmp = input[idx];

        }
        else {
            CPU_EVALUATE_ERROR
        }

        stack[SP++] = tmp;
    }

    gp_val_t result = stack[--SP];
    return result;
}
