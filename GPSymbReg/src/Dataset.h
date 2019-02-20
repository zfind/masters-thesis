//
// Created by zac on 18.02.19..
//

#ifndef GPSYMBREG_DATASET_H
#define GPSYMBREG_DATASET_H

#include <string>
#include <vector>

class Dataset {
public:
    explicit Dataset(const std::string& filename);
    ~Dataset();

public:
    int size() const;
    int dim() const;
    const std::vector<double>& getSampleInput(int i) const;
    double getSampleOutput(int i) const;
    const std::vector<double>& getOutputVector() const;
    std::pair<std::vector<double>, double> getSample(int i) const;


private:
    void loadFromFile(const std::string& filename);

private:
    int N_SIZE;
    int SAMPLE_DIMENSION;
    std::vector<std::vector<double>> datasetInput;
    std::vector<double> codomain;
};


#endif //GPSYMBREG_DATASET_H
