//
// Created by zac on 04.01.18..
//

#include "Net.h"


Net::Net(vector<int> layers, Dataset &dataset) {
    this->layers = layers;

    int maxCols = 0;
    for (int i = 0; i < layers.size(); i++) {
        if (maxCols < layers[i]) {
            maxCols = layers[i];
        }
    }
    h_output = new double[dataset.SIZE * maxCols];
    h_new_output = new double[dataset.SIZE * maxCols];

    cudaMalloc((void **) &d_datasetInput,
               dataset.datasetInput.RR * dataset.datasetInput.CC * sizeof(double));
    cudaMemcpy(d_datasetInput,
               dataset.datasetInput.elements,
               dataset.datasetInput.RR * dataset.datasetInput.CC * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_datasetOutput,
               dataset.datasetOutput.RR * dataset.datasetOutput.CC * sizeof(double));
    cudaMemcpy(d_datasetOutput,
               dataset.datasetOutput.elements,
               dataset.datasetOutput.RR * dataset.datasetOutput.CC * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_output,
               dataset.SIZE * maxCols * sizeof(double));
    cudaMalloc((void **) &d_new_output,
               dataset.SIZE * maxCols * sizeof(double));

}

Net::~Net() {
    delete[](h_output);
    delete[](h_new_output);

    cudaFree(d_output);
    cudaFree(d_new_output);

    cudaFree(d_datasetInput);
    cudaFree(d_datasetOutput);
}

double Net::evaluate(double weights[], Dataset &dataset) {
    int numberOfLayers = layers.size();
    int breakpoint = 0;
    int h_output_rows;
    int h_output_cols;

    for (int i = 1; i < numberOfLayers; i++) {
        int rows = layers[i - 1] + 1;
        int cols = layers[i];
        int new_breakpoint = breakpoint + rows * cols;


        if (i == 1) {
            mulMatrix(dataset.datasetInput.elements, dataset.SIZE, dataset.INPUT_DIM,
                      &weights[breakpoint], rows, cols,
                      h_output, dataset.SIZE, cols);
            h_output_rows = dataset.SIZE;
            h_output_cols = cols;

        } else {
            int h_new_output_rows = dataset.SIZE;
            int h_new_output_cols = cols;

            mulMatrix(h_output, dataset.SIZE, h_output_cols,
                      &weights[breakpoint], rows, cols,
                      h_new_output, dataset.SIZE, cols);

            std::swap(h_output, h_new_output);
            h_output_rows = h_new_output_rows;
            h_output_cols = h_new_output_cols;
        }

        breakpoint = new_breakpoint;

    }

    double output[dataset.SIZE * dataset.OUTPUT_DIM];
    memcpy(output, h_output, dataset.SIZE * dataset.OUTPUT_DIM * sizeof(double));

    double sum = 0.;
    for (int i = 0; i < dataset.datasetOutput.RR * dataset.datasetOutput.CC; i++) {
        sum += pow(output[i] - dataset.datasetOutput.elements[i], 2);
    }
    sum /= dataset.SIZE;
    return sum;
}

double Net::evaluateGPU(double *weights, Dataset &dataset) {
    int numberOfLayers = layers.size();
    int breakpoint = 0;
    int d_output_rows;
    int d_output_cols;


    for (int i = 1; i < numberOfLayers; i++) {
        int rows = layers[i - 1] + 1;
        int cols = layers[i];
        int new_breakpoint = breakpoint + rows * cols;

        // set up dimensions
        dim3 dimGrid;
        dimGrid.x = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dimGrid.y = (dataset.SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


        if (i == 1) {
            mulMatrixKernel<<<dimGrid, dimBlock>>>(
                    d_datasetInput, dataset.SIZE, dataset.INPUT_DIM,
                    &weights[breakpoint], rows, cols,
                    d_output, dataset.SIZE, cols
            );
            d_output_rows = dataset.SIZE;
            d_output_cols = cols;


        } else {
            int d_new_output_rows = dataset.SIZE;
            int d_new_output_cols = cols;

            mulMatrixKernel<<<dimGrid, dimBlock>>>(
                    d_output, dataset.SIZE, d_output_cols,
                    &weights[breakpoint], rows, cols,
                    d_new_output, dataset.SIZE, cols
            );

            std::swap(d_output, d_new_output);
            d_output_rows = d_new_output_rows;
            d_output_cols = d_new_output_cols;

        }

        breakpoint = new_breakpoint;

    }

    double output[dataset.SIZE * dataset.OUTPUT_DIM];
    cudaMemcpy(&output, d_output, dataset.SIZE * dataset.OUTPUT_DIM * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.;
    for (int i = 0; i < dataset.datasetOutput.RR * dataset.datasetOutput.CC; i++) {
        sum += pow(output[i] - dataset.datasetOutput.elements[i], 2);
    }
    sum /= dataset.SIZE;
    return sum;
}

int Net::getWeightsCount() {
    int count = 0;
    for (int i = 1; i < layers.size(); i++) {
        count += (layers[i - 1] + 1) * layers[i];
    }
    return count;
}

double Net::sigmoid(double x) {
    return 1. / (1. + exp(-1. * x));
}

void Net::mulMatrix(Matrix mA, int rA, int cA, Matrix mB, int rB, int cB, Matrix mC, int rC, int cC) {
    for (int i = 0; i < rA; i++) {
        for (int j = 0; j < cB; j++) {
            double xx = 0.;
            for (int k = 0; k < rB - 1; k++) {
                xx += mA.elements[i * (cA - 1) + k] * mB.elements[k * cB + j];
            }
            xx += mB.elements[(rB - 1) * cB + j];
            mC.elements[i * cC + j] = xx;
        }
    }
}

void Net::mulMatrix(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC) {
    for (int y = 0; y < rA; y++) {
        for (int x = 0; x < cB; x++) {
            double xx = 0.;
            for (int k = 0; k < rB - 1; k++) {
                xx += mA[y * (cA) + k] * mB[k * cB + x];
            }
            xx += mB[(rB - 1) * cB + x];
            xx = 1. / (1. + exp(-1. * xx)); // sigmoid
            mC[y * cC + x] = xx;
        }
    }
}

/*
void Net::mulMatrix(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC) {
    for (int y = 0; y < rA; y++) {
        for (int x = 0; x < cB; x++) {
            double xx = 0.;
            for (int k = 0; k < rB - 1; k++) {
                double valA = GetElement(mA, cA, rA, k, y);
                double valB = GetElement(mB, cB, rB, x, k);
                xx += valA * valB;
            }
            xx += GetElement(mB, cB, rB, x, rB - 1);
            xx = 1. / (1. + exp(-1. * xx)); // sigmoid
            SetElement(mC, cC, rC, x, y, xx);
        }
    }
}
 */

double Net::GetElement(double *matrix, int XX, int YY, int x, int y) {
    return matrix[y * XX + x];
}

void Net::SetElement(double *matrix, int XX, int YY, int x, int y, double val) {
    matrix[y * XX + x] = val;
}


extern "C"
__global__ void mulMatrixKernel(double *mA, int rA, int cA, double *mB, int rB, int cB, double *mC, int rC, int cC) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cC || y >= rC) return;

    double xx = 0.;
    for (int k = 0; k < rB - 1; k++) {
        double valA = GetElement(mA, cA, rA, k, y);
        double valB = GetElement(mB, cB, rB, x, k);
        xx += valA * valB;
    }
    xx += GetElement(mB, cB, rB, x, rB - 1);

    xx = 1. / (1. + exp(-1. * xx)); // sigmoid

    SetElement(mC, cC, rC, x, y, xx);
}

extern "C"
__device__ double GetElement(double *matrix, int XX, int YY, int x, int y) {
    return matrix[y * XX + x];
}

extern "C"
__device__ void SetElement(double *matrix, int XX, int YY, int x, int y, double val) {
    matrix[y * XX + x] = val;
}