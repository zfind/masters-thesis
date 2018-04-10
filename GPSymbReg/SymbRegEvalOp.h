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
 
 
/**
 * \ingroup symbreg
 * \brief Symbolic regression evaluation operator.
 */
class SymbRegEvalOp : public EvaluateOp
{
public:
	FitnessP evaluate(IndividualP individual);
	bool initialize(StateP);
	std::vector<double> domain;
	std::vector<std::vector<double>> datasetInput;
	std::vector<double> codomain;
	uint nSamples;
private:
    void convertToPostfix(IndividualP individual, std::vector<uint> &solution, std::vector<double> &solutionConstants);
    void h_evaluateDataset(std::vector<uint> &program, std::vector<double> &programConst,
                           std::vector<vector<double>> &input, std::vector<double> &result);
    double h_evaluatePoint(std::vector<uint> &solution, std::vector<double> &solutionConst, std::vector<double> &input,
                           int validLength);
    void printSolution(std::vector<uint> &solution, std::vector<double> &solutionConst);
};
typedef boost::shared_ptr<SymbRegEvalOp> SymbRegEvalOpP;

#endif // SymbRegEvalOp_h
