#include "Dataset.h"

#include <fstream>
#include <iostream>

Dataset::Dataset(const std::string& filename)
{
    if (!loadFromFile(filename)) {
        std::cerr << "Error reading file." << std::endl;
        exit(1);
    }
}

int Dataset::size() const
{
    return SAMPLE_COUNT;
}

const std::vector<gp_val_t>& Dataset::getSampleInput(int i) const
{
    return datasetInput[i];
}

gp_val_t Dataset::getSampleOutput(int i) const
{
    return datasetOutput[i];
}

bool Dataset::loadFromFile(const std::string& filename)
{
    // TODO add error checking

    std::ifstream in(filename);

    if (!in) {
        std::cerr << "Cannot open file." << std::endl;
        exit(-1);
    }

    in >> SAMPLE_COUNT;
    in >> SAMPLE_DIMENSION;

    std::vector<gp_val_t> initRow;
    initRow.resize(SAMPLE_DIMENSION, 0.);
    datasetInput.resize(SAMPLE_COUNT, initRow);

    for (int y = 0; y < SAMPLE_COUNT; ++y) {
        for (int x = 0; x < SAMPLE_DIMENSION; ++x) {
            in >> datasetInput[y][x];
        }
    }

    datasetOutput.resize(SAMPLE_COUNT);
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        in >> datasetOutput[i];
    }

    in.close();

    return true;
}

std::pair<std::vector<gp_val_t>, gp_val_t> Dataset::getSample(int i) const
{
    return std::make_pair(datasetInput[i], datasetOutput[i]);
}

int Dataset::dim() const
{
    return SAMPLE_DIMENSION;
}

const std::vector<gp_val_t>& Dataset::getOutputVector() const
{
    return datasetOutput;
}

