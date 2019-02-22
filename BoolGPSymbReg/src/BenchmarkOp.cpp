#include "BenchmarkOp.h"

#include <chrono>
#include <stack>

using namespace std;

// put "DBG(x) x" to enable debug printout
#define DBG(x)

// called only once, before the evolution  generates training data
bool BenchmarkOp::initialize(StateP state) {

    ecfTime = 0L;
    cpuTime = 0L;
    gpuTime = 0L;

    simpleEvaluator = std::make_unique<SymbRegEvalOp>();
    simpleEvaluator->initialize(state);
    postfixEvalOp = std::make_unique<PostfixEvaluator>();
    postfixEvalOp->initialize(state);
    cudaEvalOp = std::make_unique<CudaEvaluator>();
    cudaEvalOp->initialize(state);

    return true;
}

FitnessP BenchmarkOp::evaluate(IndividualP individual) {

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  legacy ECF evaluate
    begin = std::chrono::steady_clock::now();
    FitnessP fitness = simpleEvaluator->evaluate(individual);
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
    FitnessP d_fitness = cudaEvalOp->evaluate(individual);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += diff;


//    for (uint i = 0; i < h_result.size(); i++) {
//        if (h_result[i] != result[i]) {
//            cerr << "KRIVO\t" << h_result[i] << "\t" << result[i] << endl;
//        }
//    }


    if (fabs(h_fitness->getValue() - d_fitness->getValue()) > DOUBLE_EQUALS) { // std::numeric_limits<double>::epsilon()
        cerr << "FAIL\t"
             << "host:\t" << h_fitness->getValue()
             << "\tdev:\t" << d_fitness->getValue()
             << "\tdiff:\t" << fabs(h_fitness->getValue() - d_fitness->getValue()) << endl;
    }
    if (fabs(fitness->getValue() - d_fitness->getValue()) > DOUBLE_EQUALS) {
        cerr << "FAIL\t"
             << "real:\t" << fitness->getValue()
             << "\thost:\t" << h_fitness->getValue()
             << "\tdev:\t" << d_fitness->getValue()
             << "\tdiff:\t" << fabs(fitness->getValue() - d_fitness->getValue()) << endl;
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
