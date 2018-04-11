#include <ecf/ECF.h>
#include "SymbRegEvalOp.h"

int main(int argc, char **argv)
{
	StateP state (new State);

    SymbRegEvalOp* symbRegEvalOp = new SymbRegEvalOp;

	// set the evaluation operator
	state->setEvalOp(symbRegEvalOp);

	state->initialize(argc, argv);
	state->run();


    delete symbRegEvalOp;

	return 0;
}
