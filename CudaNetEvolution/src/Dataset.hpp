//
// Created by zac on 07.01.18..
//

#ifndef NEURALNET_DATASET_H
#define NEURALNET_DATASET_H

#include <cstring>
#include "Matrix.hpp"

class Dataset {
public:
    const int SIZE = 150;
    const int INPUT_DIM = 4;
    const int OUTPUT_DIM = 3;

    Matrix datasetInput;
    Matrix datasetOutput;

    Dataset();

};


#endif //NEURALNET_DATASET_H
