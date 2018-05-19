//
// Created by zac on 05.01.18..
//

#ifndef NEURALNET_MATRIX_H
#define NEURALNET_MATRIX_H

#include <cstdio>

using namespace std;


class Matrix {
public:
    const int RR;
    const int CC;
    double *elements;

    Matrix(int RR, int CC);

    ~Matrix();

    void print();
};


#endif //NEURALNET_MATRIX_H
