#include <cmath>
#include <ECF/ECF.h>
#include <stack>
#include <chrono>
#include <limits>

#include "SymbRegEvalOp.h"


// put "DBG(x) x" to enable debug printout
#define DBG(x)


void SymbRegEvalOp::convertToPostfix(IndividualP individual, char *buffer, uint &PROGRAM_SIZE) {
    DBG(cerr << "=====================================================" << endl;)

    DBG(
            uint nTrees = (uint) individual->size();
            if (nTrees != 1) {
                cerr << "more than one tree in genotype" << endl;
            }
    )

    TreeP pTree = boost::dynamic_pointer_cast<Tree::Tree>(individual->getGenotype(0));

    PROGRAM_SIZE = (uint) pTree->size();

    //  prefix print
    DBG(
            for (int i = 0; i < PROGRAM_SIZE; i++) {
                string primName = (*pTree)[i]->primitive_->getName();
                cerr << primName << " ";
            }
            cerr << endl;
    )

    //  convert to postfix
    stack<vector<int>> st;
    for (int i = PROGRAM_SIZE - 1; i >= 0; i--) {
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
            cerr << endl;
    )


    DBG(cerr << "Velicina:\t" << length << endl;)

    uint *program = reinterpret_cast<uint *>( buffer);

    size_t CONSTANTS_OFFSET = (int) ((PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
    double *programConstants = reinterpret_cast<double *>(buffer + CONSTANTS_OFFSET);


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
            uint idx = VAR + (uint) stoi(xx);
            *program = idx;
            program++;
        } else if (pName == "1") {
            *program = CONST;
            program++;
            *programConstants = 1.;
            programConstants++;
        } else if (pName[0] == 'D' && pName[1] == '_') {
            *program = CONST;
            program++;
            double value;
            (*pTree)[i]->primitive_->getValue(&value);
            *programConstants = value;
            programConstants++;
        } else {
            cerr << pName << endl;
        }
    }

    // DBG(printSolution(tmp, tmpd);)

    DBG(cerr << "*******************************************************" << endl;)
}


// called only once, before the evolution  generates training data
bool SymbRegEvalOp::initialize(StateP state) {

    size_t BUFFER_PROGRAM_SIZE = (int) ((MAX_PROGRAM_SIZE * sizeof(uint) + sizeof(double) - 1)
                                        / sizeof(double))
                                 * sizeof(double);
    size_t BUFFER_CONSTANTS_SIZE = MAX_PROGRAM_SIZE * sizeof(double);
    size_t BUFFER_SIZE = BUFFER_PROGRAM_SIZE + BUFFER_CONSTANTS_SIZE;

    programBuffer = new char[BUFFER_SIZE];

    loadFromFile("data/input.txt", datasetInput, codomain);

    N_SAMPLES = datasetInput.size();
    uint SAMPLE_DIMENSION = datasetInput[0].size();

    evaluator = new CudaEvaluator(N_SAMPLES, SAMPLE_DIMENSION, MAX_PROGRAM_SIZE, datasetInput, codomain);

    conversionTime = 0L;
    ecfTime = 0L;
    cpuTime = 0L;
    gpuTime = 0L;

    return true;
}


FitnessP SymbRegEvalOp::evaluate(IndividualP individual) {

    //  number of digits in double print
    cerr.precision(std::numeric_limits<double>::max_digits10);

    std::chrono::steady_clock::time_point begin, end;
    long diff;

    //  convert to postfix
    begin = std::chrono::steady_clock::now();
    uint PROGRAM_SIZE;
    convertToPostfix(individual, programBuffer, PROGRAM_SIZE);
    end = std::chrono::steady_clock::now();

    long postfixConversionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    conversionTime += postfixConversionTime;

    //  evaluate on CPU
    begin = std::chrono::steady_clock::now();
    vector<double> h_result;
    double h_fitness = evaluator->h_evaluate(programBuffer, PROGRAM_SIZE, h_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cpuTime += postfixConversionTime + diff;

    // evaluate on GPU
    begin = std::chrono::steady_clock::now();
    vector<double> d_result;
    double d_fitness = evaluator->d_evaluate(programBuffer, PROGRAM_SIZE, d_result);
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    gpuTime += postfixConversionTime + diff;


    //  legacy ECF evaluate

    // we try to minimize the function value, so we use FitnessMin fitness (for minimization problems)
    FitnessP fitness(new FitnessMin);

    // get the genotype we defined in the configuration file
    Tree::Tree *tree = (Tree::Tree *) individual->getGenotype().get();

    begin = std::chrono::steady_clock::now();
    double value = 0;
    for (uint i = 0; i < N_SAMPLES; i++) {
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
        cerr << "WARN Host-device difference\t" << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << "\tdiff:\t"
             << fabs(h_fitness - d_fitness) << endl;
    }
    if (fabs(value - d_fitness) > DOUBLE_EQUALS) {
        cerr << "WARN ECF-device difference\t" << "ecf:\t" << value << "host:\t" << h_fitness << "\tdev:\t" << d_fitness << "\tdiff:\t"
             << fabs(value - d_fitness) << endl;
    }


    return fitness;
}

SymbRegEvalOp::~SymbRegEvalOp() {
    delete programBuffer;

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

    int N_SAMPLES, SAMPLE_DIMENSION;
    in >> N_SAMPLES;
    in >> SAMPLE_DIMENSION;

    vector<double> initRow;
    initRow.resize(SAMPLE_DIMENSION, 0.);
    matrix.resize(N_SAMPLES, initRow);

    for (int y = 0; y < N_SAMPLES; y++) {
        for (int x = 0; x < SAMPLE_DIMENSION; x++) {
            in >> matrix[y][x];
        }
    }

    output.resize(N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        in >> output[i];
    }

    in.close();
}
