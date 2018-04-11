#ifndef SymbRegEvalOp_h
#define SymbRegEvalOp_h


/**
 * \defgroup symbreg Symbolic regression (GP)
 * \ingroup examples
 * \brief Symbolic regression example - illustration for Tree genotype (see more on this example in help/tutorial.html)
 *
 * \par Description
 * The objective is to find a symbolic form of an unknown function (i.e. find a mapping between the given set 
 * of input and output values). 
 * The fitness is defined with total error for the given set of output values.
 * 
 * \par Tree genotype parameters (in the config file)
 * - set of functions (inner tree nodes)
 * - set of terminals (variables and constants, tree leaves)
 * - min and max tree depth
 *
 */


#include "CudaEvaluator.h"


/**
 * \ingroup symbreg
 * \brief Symbolic regression evaluation operator.
 */
class SymbRegEvalOp : public EvaluateOp {
public:
    FitnessP evaluate(IndividualP individual);

    bool initialize(StateP);

    std::vector<double> domain;
    std::vector<double> codomain;
    uint nSamples;

    void loadFromFile(std::string filename, std::vector<std::vector<double>> &matrix, std::vector<double> &output);
    std::vector<std::vector<double>> datasetInput;
    void printSolution(std::vector<uint> &solution, std::vector<double> &solutionConst);
    void convertToPostfix(IndividualP individual,
                          std::vector<uint> &solution, std::vector<double> &solutionConstants);
    CudaEvaluator *evaluator;

    long convTime, ecfTime, cpuTime, gpuTime;

    ~SymbRegEvalOp();

};

typedef boost::shared_ptr<SymbRegEvalOp> SymbRegEvalOpP;


#endif // SymbRegEvalOp_h
