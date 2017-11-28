#include "MNIST_Reader.h"

int main(int argc, char const *argv[]) {
    DataSet *dataset = CreateDataSet();

    dataset->CreateDataPair(TRAIN, 30);

    dataset->GetFeedImage(TRAIN)->PrintShape();
    dataset->GetFeedLabel(TRAIN)->PrintShape();

    dataset->CreateDataPair(TRAIN, 30);

    delete dataset;

    return 0;
}
