#pragma once

#include <string>
#include <vector>
#include "PostfixEvalOpUtils.h"

class Dataset {
public:
    explicit Dataset(const std::string& filename);

    ~Dataset() = default;

public:
    int size() const;

    int dim() const;

    const std::vector<gp_val_t>& getSampleInput(int i) const;

    gp_val_t getSampleOutput(int i) const;

    const std::vector<gp_val_t>& getOutputVector() const;

    std::pair<std::vector<gp_val_t>, gp_val_t> getSample(int i) const;

private:
    bool loadFromFile(const std::string& filename);

private:
    int SAMPLE_COUNT;
    int SAMPLE_DIMENSION;
    std::vector<std::vector<gp_val_t>> datasetInput;
    std::vector<gp_val_t> datasetOutput;
};
