//
// Created by zac on 25.05.18..
//

#include "Matrix.h"

Matrix copyToDevice(const Matrix &A) {
    Matrix d_A{A.XX, A.YY, A.STRIDE, nullptr};
    size_t size = A.XX * A.YY * sizeof(double);
    cudaMalloc((void **) &d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    return d_A;
}

Matrix copyToHost(const Matrix& d_A) {
    Matrix h_A{d_A.XX, d_A.YY};
    size_t size = h_A.XX * h_A.YY * sizeof(double);
    cudaMemcpy(h_A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);

    return h_A;
}

Matrix initOnDevice(int width, int height, int stride) {
    Matrix d_A{width, height, stride, nullptr};
    size_t size = width * height * sizeof(double);
    cudaMalloc((void **) &d_A.elements, size);

    return d_A;
}

void freeOnHost(Matrix &matrix) {
    delete[](matrix.elements);
}

void freeOnDevice(Matrix &matrix) {
    cudaFree(matrix.elements);
}

Matrix MulDev(const Matrix &A, const Matrix &B) {
    Matrix d_A = copyToDevice(A);
    Matrix d_B = copyToDevice(B);
    Matrix d_C = initOnDevice(B.XX, A.YY, B.XX);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid;
    dimGrid.x = (B.XX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dimGrid.y = (A.YY + BLOCK_SIZE - 1) / BLOCK_SIZE;

    MatMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

    Matrix h_C = copyToHost(d_C);

    freeOnDevice(d_A);
    freeOnDevice(d_B);
    freeOnDevice(d_C);

    return h_C;
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    double sum = 0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= C.XX || y >= C.YY) return;

    for (int k = 0; k < A.XX; k++) {
        sum += GetElement(A, k, y) * GetElement(B, x, k);
    }
    SetElement(C, x, y, sum);
}

__device__ double GetElement(const Matrix &matrix, int x, int y) {
    return matrix.elements[y * matrix.STRIDE + x];
}

__device__ void SetElement(Matrix &matrix, int x, int y, double val) {
    matrix.elements[y * matrix.STRIDE + x] = val;
}