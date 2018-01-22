/*g++ -g -o testing -std=c++11 MLP_MSE_With_MNIST_newmodel.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"

#define BATCH             100
#define EPOCH             30
#define LOOP_FOR_TRAIN    (60000 / BATCH)
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

class MLP : public NeuralNetwork<float>{
private:
    // We need to modify kind this interface, after this time any kind of trainable tensor generated in operator(so all tensor will managed in operator)
    // 나중에는 Operator class가 이걸 관리하도록 하여야 한다.
    Operator<float> *w1 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1"));
    Operator<float> *b1 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 15, 1.0), "b1"));

    Operator<float> *w2 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2"));
    Operator<float> *b2 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 1.0), "b2"));

public:
    MLP(Placeholder<float> *x, Placeholder<float> *label) {
        Operator<float> *out = NULL;

        AddPlaceholder(x);
        AddPlaceholder(label);

        // ======================= layer 1======================
        out = AddOperator(new MatMul<float>(x, w1, "matmul1"));
        out = AddOperator(new Add<float>(out, b1, "add1"));
        out = AddOperator(new Sigmoid<float>(out, "relu1"));

        // ======================= layer 2=======================
        out = AddOperator(new MatMul<float>(out, w2, "matmul2"));
        out = AddOperator(new Add<float>(out, b2, "add2"));
        out = AddOperator(new Sigmoid<float>(out, "relu2"));

        // ======================= Error=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetObjectiveFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Optimizer=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetOptimizer(new GradientDescentOptimizer<float>(0.5, MINIMIZE));
    }

    virtual ~MLP() {}
};

int main(int argc, char const *argv[]) {
    // create input, label data placeholder
    Placeholder<float> *x = new Placeholder<float>(1, BATCH, 1, 1, 784, "x");
    Placeholder<float> *label = new Placeholder<float>(1, BATCH, 1, 1, 10, "label");

    // Result of classification
    Operator<float> *result = NULL;

    MLP mlp(x, label);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

    // pytorch check하기
    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Training =======================
        double train_accuracy = 0.f;

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);
            mlp.FeedData(2, dataset->GetTrainFeedImage(), dataset->GetTrainFeedLabel());

            result = mlp.Training();

            train_accuracy += (float)temp::Accuracy(result->GetResult(), label->GetResult(), BATCH);
            printf("\rTraining complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TRAIN, train_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';

        // Caution!
        // Actually, we need to split training set between two set for training set and validation set
        // but in this example we do not above action.
        // ======================= Testing ======================
        double test_accuracy = 0.f;

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            dataset->CreateTestDataPair(BATCH);
            mlp.FeedData(2, dataset->GetTestFeedImage(), dataset->GetTestFeedLabel());

            result = mlp.Testing();

            test_accuracy += (float)temp::Accuracy(result->GetResult(), label->GetResult(), BATCH);
            printf("\rTesting complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TEST, test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';
    }
    // we need to save best weight and bias when occur best acc on test time
    delete dataset;

    return 0;
}
