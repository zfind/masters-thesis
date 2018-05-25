//
// Created by zac on 25.05.18..
//

#ifndef CUDAMATMUL_MATRIX_H
#define CUDAMATMUL_MATRIX_H


#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda.h>
//#include <host_defines.h>


#define BLOCK_SIZE 16
#define EPS 1E-7

struct Matrix {

    const int XX;
    const int YY;
    const int STRIDE;

    Matrix(int width, int height);

    Matrix(int width, int height, int stride, double *elements);

//    ~Matrix();

    inline double getElement(int x, int y) const;

    inline void setElement(int x, int y, double val);

    double *elements;

};

Matrix loadFromFile(std::string filename);

void print(const Matrix &A);

void isEqual(const Matrix &matA, const Matrix &matB);

Matrix copyToDevice(const Matrix &A);
Matrix copyToHost(const Matrix& A);

Matrix initOnDevice(int width, int height, int stride);

void freeOnHost(Matrix &matrix);

void freeOnDevice(Matrix &matrix);

Matrix MulHost(const Matrix &A, const Matrix &B);

Matrix MulDev(const Matrix &A, const Matrix &B);

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);

__device__ inline double GetElement(const Matrix &matrix, int x, int y);

__device__ inline void SetElement(Matrix &matrix, int x, int y, double val);

#endif //CUDAMATMUL_MATRIX_H
