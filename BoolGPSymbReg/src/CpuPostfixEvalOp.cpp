#include "CpuPostfixEvalOp.h"

#include <stack>

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)
#define CPU_EVALUATE_ERROR do {cerr << "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR" << endl; return NAN; } while(0);

void CpuPostfixEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP) (new std::string), ECF::STRING);
}

bool CpuPostfixEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState](int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    if (!state->getRegistry()->isModified("dataset.filename"))
        return false;

    voidP pEntry = state->getRegistry()->getEntry("dataset.filename");
    std::string datasetFilename = *(static_cast<std::string*>(pEntry.get()));

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));

    programBuffer = new char[BUFFER_SIZE];

    dataset = std::make_unique<Dataset>(datasetFilename);

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

uint CpuPostfixEvalOp::h_evaluate(char* buffer, int programSize, vector<gp_val_t>& result)
{
    result.resize(dataset->size(), 0);

    gp_fitness_t fitness = 0;
    for (int i = 0; i < dataset->size(); ++i) {
        result[i] = h_evaluateIndividual(buffer, programSize, datasetInput[i]);
        if (result[i] != datasetOutput[i]) {
            fitness++;
        }
    }

    return fitness;
}

gp_val_t CpuPostfixEvalOp::h_evaluateIndividual(char* buffer, int programSize, const std::vector<gp_val_t>& input)
{
    gp_code_t* program = reinterpret_cast<gp_code_t*>(buffer);

    gp_val_t stack[programSize];

    int SP = 0;
    gp_val_t o1, o2, tmp;

    for (int i = 0; i < programSize; ++i) {

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

        }
        else if (program[i] >= ARITY_1) {
            o1 = stack[--SP];

            switch (program[i]) {
            case NOT:
                tmp = !o1;
                break;
            default:
                CPU_EVALUATE_ERROR
            }

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