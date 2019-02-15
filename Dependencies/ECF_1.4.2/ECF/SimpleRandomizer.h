#ifndef SimpleRandomizer_h
#define SimpleRandomizer_h

#include "Randomizer.h"

/**
 * \ingroup evol
 * \brief A simple randomizer that uses in-built random number generator.
 */
class SimpleRandomizer : public Randomizer 
{
public:
    SimpleRandomizer(unsigned int seed = 0);

	int getRandomInteger(int p, int q);

	int getRandomInteger(int size);

	double getRandomDouble();

	void registerParameters(StateP s);

	/**
	 * Initialize the randomizer with the seed parameter from the Registry.
	 */
	bool initialize(StateP s);
};

typedef boost::shared_ptr<SimpleRandomizer> SimpleRandomizerP;

#endif

