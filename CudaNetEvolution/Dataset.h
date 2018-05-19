//
// Created by zac on 07.01.18..
//

#ifndef DATASET_H
#define DATASET_H

#include <cstring>
#include "Matrix.h"

class Dataset {
public:
    const int SIZE = 150;
    const int INPUT_DIM = 4;
    const int OUTPUT_DIM = 3;

    Matrix datasetInput;
    Matrix datasetOutput;

    Dataset();
};


#endif //DATASET_H
