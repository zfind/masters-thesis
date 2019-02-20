#include "BenchmarkOp.h"

#include <chrono>
#include "Constants.h"


// called only once, before the evolution  generates training data
bool BenchmarkOp::initialize(StateP state) {
    ecfTime = 0L;
    cpuTime = 0L;
    gpuTime = 0L;

    symbRegEvalOp = std::make_unique<SimpleEvaluator>();
    symbRegEvalOp->initialize(state);
    postfixEvalOp = std::make_unique<PostfixEvaluator>();
    postfixEvalOp->initialize(state);
    cuPostfixEvalOp = std::make_unique<CUPostfixEvalOp>();
    cuPostfixEvalOp->initialize(state);

    return true;
}


FitnessP BenchmarkOp::evaluate(IndividualP individual) {
    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  legacy ECF evaluate
    begin = std::chrono::steady_clock::now();
    FitnessP fitness = symbRegEvalOp->evaluate(individual);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    ecfTime += diff;

    //  evaluate on CPU
    begin = std::chrono::steady_clock::now();
    FitnessP h_fitness = postfixEvalOp->evaluate(individual);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += diff;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    FitnessP d_fitness = cuPostfixEvalOp->evaluate(individual);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += diff;


    if (fabs(h_fitness->getValue() - d_fitness->getValue()) >
        DOUBLE_EQUALS) {     // std::numeric_limits<double>::epsilon()
        cerr << "WARN Host-device difference\t" << "host:\t" << h_fitness->getValue() << "\tdev:\t"
             << d_fitness->getValue() << "\tdiff:\t"
             << fabs(h_fitness->getValue() - d_fitness->getValue()) << endl;
    }
    if (fabs(fitness->getValue() - d_fitness->getValue()) > DOUBLE_EQUALS) {
        cerr << "WARN ECF-device difference\t" << "ecf:\t" << fitness->getValue() << "host:\t" << h_fitness->getValue()
             << "\tdev:\t" << d_fitness->getValue()
             << "\tdiff:\t"
             << fabs(fitness->getValue() - d_fitness->getValue()) << endl;
    }

    return fitness;
}

BenchmarkOp::~BenchmarkOp() {
    cerr.precision(7);

    cerr << "===== STATS [us] =====" << endl;

    cerr << "ECF time:\t" << ecfTime << endl;
    cerr << "CPU time:\t" << cpuTime << endl;
    cerr << "GPU time:\t" << gpuTime << endl;

    cerr << "CPU vs ECF:\t" << (double) ecfTime / cpuTime << endl;
    cerr << "GPU vs CPU:\t" << (double) cpuTime / gpuTime << endl;
    cerr << "GPU vs ECF:\t" << (double) ecfTime / gpuTime << endl;
}
