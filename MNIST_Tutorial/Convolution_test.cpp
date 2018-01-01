/*g++ -g -o testing -std=c++11 Convolution_test.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp*/

#include <iostream>
#include <string>

// #include "..//Header//NeuralNetwork.h"
#include "..//Header//DLOP.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             100
#define LOOP_FOR_TRAIN    100
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {

    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();
    dataset->CreateTrainDataPair(BATCH);




    return 0;
}
