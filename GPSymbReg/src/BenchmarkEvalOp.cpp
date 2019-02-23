#include "BenchmarkEvalOp.h"

#include "Constants.h"

bool BenchmarkEvalOp::initialize(StateP state)
{
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
    // TODO move to Logger

    cerr.precision(7);

    cerr << "===== STATS [us] =====" << endl;

    cerr << "ECF time:\t" << ecfTimer.get() << endl;
    cerr << "CPU time:\t" << cpuTimer.get() << endl;
    cerr << "GPU time:\t" << gpuTimer.get() << endl;

    cerr << "CPU vs ECF:\t" << (double) ecfTimer.get() / cpuTimer.get() << endl;
    cerr << "GPU vs CPU:\t" << (double) cpuTimer.get() / gpuTimer.get() << endl;
    cerr << "GPU vs ECF:\t" << (double) ecfTimer.get() / gpuTimer.get() << endl;
}
