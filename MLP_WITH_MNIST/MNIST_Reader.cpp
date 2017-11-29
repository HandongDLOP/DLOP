#include "MNIST_Reader.h"

int main(int argc, char const *argv[]) {
    DataSet *dataset = CreateDataSet();

    for(int i = 0 ; i < 700; i++){
        std::cout << "next" << '\n';
        std::cout << "====================" << '\n';
        dataset->CreateTestDataPair(100);
        std::cout << "\n====================" << '\n';
    }

    delete dataset;

    return 0;
}
