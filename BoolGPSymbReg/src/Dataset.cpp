#include "Dataset.h"

#include <fstream>
#include <iostream>

using std::vector;


Dataset::Dataset(const std::string &filename) {
    loadFromFile(filename);
}

Dataset::~Dataset() {

}

int Dataset::size() const {
    return N_SIZE;
}

const std::vector<bool> &Dataset::getSampleInput(int i) const {
    return datasetInput[i];
}

bool Dataset::getSampleOutput(int i) const {
    return codomain[i];
}

void Dataset::loadFromFile(const std::string &filename) {
    std::ifstream in(filename);

    if (!in) {
        std::cerr << "Cannot open file.\n";
        exit(-1);
    }

    int N, DIM;
    in >> N;
    in >> DIM;

    vector<bool> initRow;
    initRow.resize(DIM, 0.);
    datasetInput.resize(N, initRow);
    codomain.resize(N);

    uint tmp;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < DIM; x++) {
            in >> tmp;
            datasetInput[y][x] = static_cast<bool>(tmp);
        }
        in >> tmp;
        codomain[y] = static_cast<bool>(tmp);
    }


//    vector<BoolV> domain;
    for (int x = 0; x < DIM; x++) {
        vector<bool> v;
        v.resize(N, 0);
        for (int y = 0; y < N; y++) {
            v[y] = datasetInput[y][x];
        }
        domain.push_back(v);
    }


    in.close();

    N_SIZE = N;
    SAMPLE_DIMENSION = DIM;
}


std::pair<std::vector<bool>, bool> Dataset::getSample(int i) const {
    return std::make_pair(datasetInput[i], codomain[i]);
}

int Dataset::dim() const {
    return SAMPLE_DIMENSION;
}

const std::vector<bool> &Dataset::getOutputVector() const {
    return codomain;
}

std::vector<bool> &Dataset::getSampleInputVector(int i) {
    return domain[i];
}


