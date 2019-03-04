#include "CpuPostfixEvalOp.h"

#include <stack>

#define CPU_EVALUATE_ERROR do { LOG(1, "ERROR: Unknown operation code"); return NAN; } while(0);

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

    dataset = std::make_unique<Dataset>(datasetFilename);

    size_t BUFFER_SIZE = MAX_PROGRAM_SIZE * sizeof(gp_code_t);
    programBuffer = new char[BUFFER_SIZE];

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

    conversionTimer.start();
    int programSize;
    PostfixEvalOpUtils::ConvertToPostfix(individual, programBuffer, programSize);
    conversionTimer.pause();

    vector<gp_val_t> h_result; // TODO move to h_evaluate()
    gp_fitness_t h_fitness = h_evaluate(programBuffer, programSize, h_result);

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
        result[i] = h_evaluateIndividual(buffer, programSize, dataset->getSampleInput(i));
        if (result[i] != dataset->getSampleOutput(i)) {
            fitness++;
        }
    }

    return fitness;
}

gp_val_t CpuPostfixEvalOp::h_evaluateIndividual(char* buffer, int programSize, const std::vector<gp_val_t>& input)
{
    gp_code_t* program = reinterpret_cast<gp_code_t*>(buffer);

    bool stack[programSize];

    int SP = 0;
    bool o1, o2, tmp;

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

    gp_val_t result = stack[--SP] ? '1' : '0';

    return result;
}