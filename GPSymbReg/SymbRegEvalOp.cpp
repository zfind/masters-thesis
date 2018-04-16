#include <cmath>
#include <ecf/ECF.h>
#include <stack>
#include <chrono>
#include <limits>

#include "SymbRegEvalOp.h"
#include "Constants.h"


// put "DBG(x) x" to enable debug printout
#define DBG(x)


void SymbRegEvalOp::convertToPostfixNew(IndividualP individual, char *postfixMem, uint &PROG_SIZE, uint &CONST_SIZE) {
    DBG(cerr << "=====================================================" << endl;)

    uint nTreeSize, nTree;
    uint nTrees = (uint) individual->size();
    for (nTree = 0; nTree < nTrees; nTree++) {
        TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(nTree));
        nTreeSize = (uint) pTree->size();

        //  prefix print
        DBG(
                for (int i = 0; i < nTreeSize; i++) {
                    string primName = (*pTree)[i]->primitive_->getName();
                    cerr << primName << " ";
                }
                cerr << endl;)

        //  convert to postfix
        stack<vector<int>> st;
        int length = nTreeSize;
        for (int i = length - 1; i >= 0; i--) {
            int arity = (*pTree)[i]->primitive_->getNumberOfArguments();
            if (arity == 2) {
                vector<int> op1 = st.top();
                st.pop();
                vector<int> op2 = st.top();
                st.pop();
                op1.insert(op1.end(), op2.begin(), op2.end());
                op1.push_back(i);
                st.push(op1);
            } else if (arity == 1) {
                vector<int> op1 = st.top();
                st.pop();
                op1.push_back(i);
                st.push(op1);
            } else {
                vector<int> tmp;
                tmp.push_back(i);
                st.push(tmp);
            }
        }
        vector<int> result = st.top();


        //  postfix ispis
        DBG(
                for (int i = 0; i < result.size(); i++) {
                    string pName = (*pTree)[result[i]]->primitive_->getName();
                    cerr << pName << " ";
                }
                cerr << endl;)


        DBG(cerr << "Velicina:\t" << length << endl;)

        PROG_SIZE = length;
        CONST_SIZE = 0;
        uint *program = (uint *) postfixMem;
        double *programConstants = (double *) &program[PROG_SIZE];

        for (int i : result) {
            string pName = (*pTree)[i]->primitive_->getName();
            if (pName[0] == '+') {
                *program = ADD;
                program++;
            } else if (pName[0] == '-') {
                *program = SUB;
                program++;
            } else if (pName[0] == '*') {
                *program = MUL;
                program++;
            } else if (pName[0] == '/') {
                *program = DIV;
                program++;
            } else if (pName[0] == 's') {
                *program = SIN;
                program++;
            } else if (pName[0] == 'c') {
                *program = COS;
                program++;
            } else if (pName[0] == 'X') {
                string xx = pName.substr(1);
                uint idx = VAR + stoi(xx);
                *program = idx;
                program++;
            } else if (pName == "1") {
                *program = CONST;
                program++;
                *programConstants = 1.;
                programConstants++;
                CONST_SIZE++;
            } else if (pName[0] == 'D' && pName[1] == '_') {
                *program = CONST;
                program++;
                double value;
                (*pTree)[i]->primitive_->getValue(&value);
                *programConstants = value;
                programConstants++;
                CONST_SIZE++;
            } else {
                cerr << pName << endl;
            }
        }

        // DBG(printSolution(tmp, tmpd);)
    }

    DBG(cerr << "*******************************************************" << endl;)
}


// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state) {

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));
    postfixBuffer = new char[BUFFER_SIZE];

    loadFromFile("input.txt", datasetInput, codomain);

    NUM_SAMPLES = datasetInput.size();
    uint INPUT_DIMENSION = datasetInput[0].size();

    evaluator = new CudaEvaluator(NUM_SAMPLES, INPUT_DIMENSION, MAX_PROGRAM_SIZE, datasetInput, codomain);

    conversionTime = 0;
    ecfTime = 0;
    cpuTime = 0;
    gpuTime = 0;

    return true;
}


FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROG_SIZE, MEM_SIZE;
    convertToPostfixNew(individual, postfixBuffer, PROG_SIZE, MEM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    //  evaluate on CPU
    begin = std::chrono::steady_clock::now();
    vector<double> h_result;
    double h_fitness = evaluator->h_evaluate(postfixBuffer, PROG_SIZE, MEM_SIZE, h_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += postfixConversionTime + diff;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<double> d_result;
    double d_fitness = evaluator->d_evaluate(postfixBuffer, PROG_SIZE, MEM_SIZE, d_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;


    //  legacy ECF evaluate

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();
    // (you can also use boost smart pointers:)
    //TreeP tree = boost::static_pointer_cast<Tree::Tree> (individual->getGenotype());

    begin = std::chrono::steady_clock::now();
    double value = 0;
    for (uint i = 0; i < NUM_SAMPLES; i++) {
        // for each test data instance, the x value (domain) must be set
        // tree->setTerminalValue("X", &domain[i]);
        tree->setTerminalValue("X0", &datasetInput[i][0]);
        tree->setTerminalValue("X1", &datasetInput[i][1]);
        tree->setTerminalValue("X2", &datasetInput[i][2]);
        tree->setTerminalValue("X3", &datasetInput[i][3]);
        tree->setTerminalValue("X4", &datasetInput[i][4]);
        // get the y value of the current tree
        double result;
        tree->execute(&result);
        // add the difference
        value += fabs(codomain[i] - result);
    }
    fitness->setValue(value);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    ecfTime += diff;

    if (fabs(h_fitness - d_fitness) > DOUBLE_EQUALS) {     // std::numeric_limits<double>::epsilon()
        cerr << "FAIL\t" << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << "\tdiff:\t"
             << fabs(h_fitness - d_fitness) << endl;
    }
    if (fabs(value - d_fitness) > DOUBLE_EQUALS) {
        cerr << "FAIL\t" << "real:\t" << value << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << "\tdiff:\t"
             << fabs(value - d_fitness) << endl;
    }


    return fitness;
}

SymbRegEvalOp::~SymbRegEvalOp() {
    delete postfixBuffer;

    delete evaluator;

    cerr.precision(7);
    cerr << "===== STATS [us] =====" << endl;
    cerr << "ECF time:\t" << ecfTime << endl;
    cerr << "CPU time:\t" << cpuTime << endl;
    cerr << "GPU time:\t" << gpuTime << endl;
    cerr << "Conversion time: " << conversionTime << endl;
    cerr << "CPU vs ECF:\t" << (double) ecfTime / cpuTime << endl;
    cerr << "GPU vs CPU:\t" << (double) cpuTime / gpuTime << endl;
    cerr << "GPU vs ECF:\t" << (double) ecfTime / gpuTime << endl;
}

void SymbRegEvalOp::loadFromFile(std::string filename,
                                 std::vector<std::vector<double>> &matrix,
                                 std::vector<double> &output) {
    ifstream in(filename);

    if (!in) {
        cerr << "Cannot open file.\n";
        exit(-1);
    }

    int N, DIM;
    in >> N;
    in >> DIM;

    vector<double> initRow;
    initRow.resize(DIM, 0.);
    matrix.resize(N, initRow);

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < DIM; x++) {
            in >> matrix[y][x];
        }
    }

    output.resize(N);
    for (int i = 0; i < N; i++) {
        in >> output[i];
    }

    in.close();
}
