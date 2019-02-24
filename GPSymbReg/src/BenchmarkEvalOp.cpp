#include "BenchmarkEvalOp.h"

#include "PostfixEvalOpUtils.h"

void BenchmarkEvalOp::registerParameters(StateP state)
{
    state->getRegistry()->registerEntry("dataset.filename", (voidP) (new std::string), ECF::STRING);
}

bool BenchmarkEvalOp::initialize(StateP state)
{
    State* pState = state.get();
    LOG = [pState] (int level, std::string msg) {
        ECF_LOG(pState, level, msg);
    };

    symbRegEvalOp = std::make_unique<SymbRegEvalOp>();
    symbRegEvalOp->initialize(state);

    cpuPostfixEvalOp = std::make_unique<CpuPostfixEvalOp>();
    cpuPostfixEvalOp->initialize(state);

    cudaPostfixEvalOp = std::make_unique<CudaPostfixEvalOp>();
    cudaPostfixEvalOp->initialize(state);

    return true;
}

FitnessP BenchmarkEvalOp::evaluate(IndividualP individual)
{
    //  legacy ECF evaluate
    ecfTimer.start();
    FitnessP fitness = symbRegEvalOp->evaluate(individual);
    ecfTimer.pause();

    //  evaluate on CPU
    cpuTimer.start();
    FitnessP h_fitness = cpuPostfixEvalOp->evaluate(individual);
    cpuTimer.pause();

    // evaluate on GPU
    gpuTimer.start();
    FitnessP d_fitness = cudaPostfixEvalOp->evaluate(individual);
    gpuTimer.pause();

    // TODO move to Logger
    //  number of digits in double print
    std::cerr.precision(std::numeric_limits<double>::max_digits10);
    if (fabs(h_fitness->getValue() - d_fitness->getValue()) >
            DOUBLE_EQUALS) {     // std::numeric_limits<double>::epsilon()
        std::cerr << "WARN Host-device difference\t" << "host:\t" << h_fitness->getValue() << "\tdev:\t"
                  << d_fitness->getValue() << "\tdiff:\t"
                  << fabs(h_fitness->getValue() - d_fitness->getValue()) << endl;
    }
    if (fabs(fitness->getValue() - d_fitness->getValue()) > DOUBLE_EQUALS) {
        std::cerr << "WARN ECF-device difference\t" << "ecf:\t" << fitness->getValue() << "host:\t"
                  << h_fitness->getValue()
                  << "\tdev:\t" << d_fitness->getValue()
                  << "\tdiff:\t"
                  << fabs(fitness->getValue() - d_fitness->getValue()) << endl;
    }

    return fitness;
}

BenchmarkEvalOp::~BenchmarkEvalOp()
{
    std::stringstream ss;
    ss.precision(7);

    ss << "===== STATS [us] =====" << endl;

    ss << "ECF time:\t" << ecfTimer.get() << endl;
    ss << "CPU time:\t" << cpuTimer.get() << endl;
    ss << "GPU time:\t" << gpuTimer.get() << endl;

    ss << "CPU vs ECF:\t" << (double) ecfTimer.get() / cpuTimer.get() << endl;
    ss << "GPU vs CPU:\t" << (double) cpuTimer.get() / gpuTimer.get() << endl;
    ss << "GPU vs ECF:\t" << (double) ecfTimer.get() / gpuTimer.get() << endl;

    LOG(1, ss.str());
}
