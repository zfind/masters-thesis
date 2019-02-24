#include "ECF_base.h"
#include "SimpleRandomizer.h"
#include <time.h>


SimpleRandomizer::SimpleRandomizer(unsigned int seed)
{
    if(seed == 0)
		srand((uint) time(NULL));
	else
		srand(seed);
}


int SimpleRandomizer::getRandomInteger(int p, int q) 
{
    return p + (int)( (double)(q-p+1) * (rand() / (RAND_MAX + 1.0)));
}


int SimpleRandomizer::getRandomInteger(int size) 
{
    return (int)( (double)(size) * (rand() / (RAND_MAX + 1.0)));
}


double SimpleRandomizer::getRandomDouble() 
{
	return rand() / (double(RAND_MAX));
}


void SimpleRandomizer::registerParameters(StateP s) 
{
	s->getRegistry()->registerEntry("randomizer.seed", (voidP) new uint(0), ECF::UINT,
		"randomizer seed, 0 uses time(NULL) (default: 0)");
}


bool SimpleRandomizer::initialize(StateP state)
{
	uint seed = 0;
	if(state->getRegistry()->isModified("randomizer.seed")) {
		voidP sptr = state->getRegistry()->getEntry("randomizer.seed");
		seed = *((int*)sptr.get());
	}

	uint *memSeed = new uint;

	if(seed != 0)
		srand(seed);
	else
#ifdef _MPI
		srand((uint) time(NULL) + state->getCommunicator()->getCommRank() + (uint) memSeed);
#else
	{
		srand((uint) time(NULL) + (uint)(size_t) memSeed);
	}
#endif

	delete memSeed;

	return true;
}