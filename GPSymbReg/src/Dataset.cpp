//
// Created by zac on 18.02.19..
//

#include "Dataset.h"
#include <fstream>
#include <iostream>
using std::vector;


Dataset::Dataset(const std::string& filename) {
    loadFromFile(filename);
}

Dataset::~Dataset() {

}

int Dataset::size() const {
    return N_SIZE;
}

const std::vector<double>& Dataset::getSampleInput(int i) const {
    return datasetInput[i];
}

double Dataset::getSampleOutput(int i) const {
    return codomain[i];
}

void Dataset::loadFromFile(const std::string& filename) {
    std::ifstream in(filename);

    if (!in) {
        std::cerr << "Cannot open file.\n";
        exit(-1);
    }

    int N_SAMPLES;
//    int SAMPLE_DIMENSION;
    in >> N_SAMPLES;
    in >> SAMPLE_DIMENSION;

    vector<double> initRow;
    initRow.resize(SAMPLE_DIMENSION, 0.);
    datasetInput.resize(N_SAMPLES, initRow);

    for (int y = 0; y < N_SAMPLES; y++) {
        for (int x = 0; x < SAMPLE_DIMENSION; x++) {
            in >> datasetInput[y][x];
        }
    }

    codomain.resize(N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        in >> codomain[i];
    }

    in.close();

    N_SIZE = N_SAMPLES;
}

std::pair<std::vector<double>, double> Dataset::getSample(int i) const {
    return std::make_pair(datasetInput[i], codomain[i]);
}

int Dataset::dim() const {
    return SAMPLE_DIMENSION;
}

const std::vector<double> &Dataset::getOutputVector() const {
    return codomain;
}

