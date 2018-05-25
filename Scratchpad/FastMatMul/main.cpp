#include <iostream>
#include "Matrix.h"

int main() {

    Matrix matA = loadFromFile("../matA.in");

    Matrix matB = loadFromFile("../matB.in");

    Matrix matC_host = MulHost(matA, matB);

    Matrix matC_check = loadFromFile("../matC.in");

//    Matrix::print(matC_host);

    isEqual(matC_host, matC_check);

    Matrix matC_dev = MulDev(matA, matB);

    isEqual(matC_dev, matC_check);

//    print(matC_dev);

    freeOnHost(matA);
    freeOnHost(matB);
    freeOnHost(matC_host);
    freeOnHost(matC_check);
    freeOnHost(matC_dev);

    return 0;
}