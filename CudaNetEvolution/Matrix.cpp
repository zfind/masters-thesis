//
// Created by zac on 05.01.18..
//


#include "Matrix.h"


Matrix::Matrix(int RR, int CC) : RR(RR), CC(CC) {
    this->elements = new double[RR * CC];
}

Matrix::~Matrix() {
    delete[](elements);
}

void Matrix::print() {
    printf("----\n");
    for (int i = 0; i < RR; i++) {
        for (int j = 0; j < CC; j++) {
            printf("\t%f, ", elements[i * CC + j]);
        }
        printf("\n");
    }
    printf("----\n");
}
