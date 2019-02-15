//
// Created by zac on 01.05.18..
//

#include "SymbRegEvalOp.h"

// put "DBG(x) x" to enable debug printout
#define DBG(x)

FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

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
    vector<BOOL_TYPE> h_result;
    uint h_fitness = evaluator->h_evaluate(postfixBuffer, PROG_SIZE, MEM_SIZE, h_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += postfixConversionTime + diff;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<BOOL_TYPE> d_result;
    uint d_fitness = evaluator->d_evaluate(postfixBuffer, PROG_SIZE, MEM_SIZE, d_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();
    // (you can also use boost smart pointers:)
    //TreeP tree = boost::static_pointer_cast<Tree::Tree> (individual->getGenotype());

    begin = std::chrono::steady_clock::now();
    uint value = 0;

    // get the y value of the current tree
    vector<bool> result;
    result.resize(NUM_SAMPLES, 0);
    tree->execute(&result);

    for (uint i = 0; i < NUM_SAMPLES; i++) {
        // add the difference
        if (codomain[i] != result[i]) {
            value++;
        }
    }

    fitness->setValue(value);

    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    ecfTime += diff;

    for (uint i = 0; i < h_result.size(); i++) {
        if (h_result[i] != result[i]) {
            cerr << "KRIVO\t" << h_result[i] << "\t" << result[i] << endl;
        }
    }


    if (fabs(h_fitness - value) > DOUBLE_EQUALS) {     // std::numeric_limits<double>::epsilon()
        cerr << "FAIL\t"
             << "host:\t" << h_fitness
             << "\tdev:\t" << value
             << "\tdiff:\t" << fabs(h_fitness - value) << endl;
    }
    if (fabs(value - d_fitness) > DOUBLE_EQUALS) {
        cerr << PROG_SIZE << endl;
        cerr << "FAIL\t"
             << "real:\t" << value
             << "\thost:\t" << h_fitness
             << "\tdev:\t" << d_fitness
             << "\tdiff:\t" << fabs(value - d_fitness) << endl;
    }


    return fitness;
}

bool SymbRegEvalOp::initialize(StateP state) {

    uint BUFFER_SIZE = MAX_PROGRAM_SIZE * (sizeof(uint) + sizeof(double));
    postfixBuffer = new char[BUFFER_SIZE];

    loadFromFile("input.txt", datasetInput, codomain);

    Tree::Tree* tree = (Tree::Tree*) state->getGenotypes().at(0).get();
    // zadaj vrijednosti obicnim varijablama (ne mijenjaju se tijekom evolucije!)

    //    for (uint i = 0; i < NUM_SAMPLES; i++) {
    // for each test data instance, the x value (domain) must be set
    // tree->setTerminalValue("X", &domain[i]);
    tree->setTerminalValue("v0", &domain[0]);
    tree->setTerminalValue("v1", &domain[1]);
    tree->setTerminalValue("v2", &domain[2]);
    tree->setTerminalValue("v3", &domain[3]);
    tree->setTerminalValue("v4", &domain[4]);
    tree->setTerminalValue("v5", &domain[5]);
    tree->setTerminalValue("v6", &domain[6]);
    tree->setTerminalValue("v7", &domain[7]);
    tree->setTerminalValue("v8", &domain[8]);
    tree->setTerminalValue("v9", &domain[9]);
    tree->setTerminalValue("v10", &domain[10]);
    tree->setTerminalValue("v11", &domain[11]);
    tree->setTerminalValue("v12", &domain[12]);
    tree->setTerminalValue("v13", &domain[13]);
    tree->setTerminalValue("v14", &domain[14]);
    tree->setTerminalValue("v15", &domain[15]);

    NUM_SAMPLES = datasetInput.size();
    uint INPUT_DIMENSION = datasetInput[0].size();

    evaluator = new CudaEvaluator(NUM_SAMPLES, INPUT_DIMENSION, MAX_PROGRAM_SIZE, datasetInput, codomain);

    conversionTime = 0;
    ecfTime = 0;
    cpuTime = 0;
    gpuTime = 0;

    return true;
}

SymbRegEvalOp::~SymbRegEvalOp() {
    delete[] postfixBuffer;

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
                                 std::vector<std::vector<bool>> &matrix, std::vector<bool> &output) {
    ifstream in(filename);

    if (!in) {
        cerr << "Cannot open file.\n";
        exit(-1);
    }

    int N, DIM;
    in >> N;
    in >> DIM;

    vector<bool> initRow;
    initRow.resize(DIM, 0.);
    matrix.resize(N, initRow);
    output.resize(N);

    uint tmp;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < DIM; x++) {
            in >> tmp;
            matrix[y][x] = static_cast<bool>(tmp);
        }
        in >> tmp;
        output[y] = static_cast<bool>(tmp);
    }


//    vector<BoolV> domain;
    for (int x = 0; x < DIM; x++) {
        vector<bool> v;
        v.resize(N, 0);
        for (int y = 0; y < N; y++) {
            v[y] = matrix[y][x];
        }
        domain.push_back(v);
    }


    in.close();
}

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

        for (int i : result) {
            string pName = (*pTree)[i]->primitive_->getName();
            if (pName == "AND") {
                *program = AND;
                program++;
            } else if (pName == "OR") {
                *program = OR;
                program++;
            } else if (pName == "NOT") {
                *program = NOT;
                program++;
            } else if (pName == "XOR") {
                *program = XOR;
                program++;
            } else if (pName == "XNOR") {
                *program = XNOR;
                program++;
            } else if (pName == "NAND") {
                *program = NAND;
                program++;
            } else if (pName == "NOR") {
                *program = NOR;
                program++;
            } else if (pName[0] == 'v') {
                string xx = pName.substr(1);
                uint idx = VAR + (uint) stoi(xx);
                *program = idx;
                program++;
//            } else if (pName == "1") {
//                *program = CONST;
//                program++;
//                *programConstants = 1.;
//                programConstants++;
//                CONST_SIZE++;
//            } else if (pName[0] == 'D' && pName[1] == '_') {
//                *program = CONST;
//                program++;
//                double value;
//                (*pTree)[i]->primitive_->getValue(&value);
//                *programConstants = value;
//                programConstants++;
//                CONST_SIZE++;
            } else {
                cerr << pName << endl;
            }
        }

        // DBG(printSolution(tmp, tmpd);)
    }

    DBG(cerr << "*******************************************************" << endl;)
}