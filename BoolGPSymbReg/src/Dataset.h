#pragma once


#include <string>
#include <vector>


class Dataset {
public:
    explicit Dataset(const std::string &filename);

    ~Dataset();

public:
    int size() const;

    int dim() const;

    const std::vector<bool> &getSampleInput(int i) const;

    bool getSampleOutput(int i) const;

    const std::vector<bool> &getOutputVector() const;

    std::vector<bool> &getSampleInputVector(int i);

    std::pair<std::vector<bool>, bool> getSample(int i) const;


private:
    void loadFromFile(const std::string &filename);

private:
    int N_SIZE;
    int SAMPLE_DIMENSION;
    std::vector<std::vector<bool>> datasetInput;
    std::vector<std::vector<bool>> domain;
    std::vector<bool> codomain;
};
