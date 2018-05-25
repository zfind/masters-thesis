#include "Matrix.h"


Matrix::Matrix(int width, int height) :
        XX(width), YY(height), STRIDE(width) {
    elements = new double[XX * YY];
}

Matrix::Matrix(int width, int height, int stride, double *elements) :
        XX(width), YY(height), STRIDE(stride) {
    this->elements = elements;
}


Matrix loadFromFile(std::string filename) {
    std::ifstream in(filename);

    if (!in) {
        std::cerr << "Cannot open file.\n";
        exit(-1);
    }

    int ROWS, COLS;
    in >> ROWS;
    in >> COLS;

    Matrix matrix{COLS, ROWS};

    for (int y = 0; y < matrix.YY; y++) {
        for (int x = 0; x < matrix.XX; x++) {
            double val;
            in >> val;
            matrix.setElement(x, y, val);
        }
    }

    in.close();

    return matrix;
}

double Matrix::getElement(int x, int y) const {
    return this->elements[y * this->STRIDE + x];
}

void Matrix::setElement(int x, int y, double val) {
    this->elements[y * this->STRIDE + x] = val;
}

Matrix MulHost(const Matrix &A, const Matrix &B) {
    Matrix C{B.XX, A.YY};
    for (int y = 0; y < C.YY; y++) {
        for (int x = 0; x < C.XX; x++) {
            double sum = 0.;
            for (int k = 0; k < A.XX; k++) {
                double aVal = A.getElement(k, y);
                double bVal = B.getElement(x, k);
                sum += aVal * bVal;
            }
            C.setElement(x, y, sum);
        }
    }
    return C;
}

void isEqual(const Matrix &matA, const Matrix &matB) {
    if (matA.XX != matB.XX) {
        std::cout << "width" << std::endl;
        return;
    }
    if (matA.YY != matB.YY) {
        std::cout << "height" << std::endl;
        return;
    }
    double diff = 0.;
    for (int y = 0; y < matA.YY; y++) {
        for (int x = 0; x < matA.XX; x++) {
            double xx = std::fabs(matA.getElement(x, y) - matB.getElement(x, y));
            if (xx > EPS) {
                diff += xx;
            }
        }
    }
    std::cout << "diff:\t" << diff << std::endl;
}


void print(const Matrix &A) {
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    for (int y = 0; y < A.YY; y++) {
        for (int x = 0; x < A.XX; x++) {
            std::cout << A.getElement(x, y) << "    ";
        }
        std::cout << std::endl;
    }
}


