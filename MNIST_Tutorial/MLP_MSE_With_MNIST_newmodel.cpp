/*g++ -g -o testing -std=c++11 MLP_MSE_With_MNIST_.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

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
    Placeholder<float> *m_x;
    Placeholder<float> *m_label;

    Operator<float> *result;

public:
    MLP(Placeholder<float> *x, Placeholder<float> *label) {
        m_x = this->AddPlaceholder(x);
        m_label = this->AddPlaceholder(label);

        Operator<float> *w1 = this->AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1"));
        Operator<float> *b1 = this->AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 15, 1.0), "b1"));

        Operator<float> *w2 = this->AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2"));
        Operator<float> *b2 = this->AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 1.0), "b2"));

        Operator<float> *out = NULL;
        // ======================= layer 1======================
        out = this->AddOperator(new MatMul<float>(m_x, w1, "matmul1"));
        out = this->AddOperator(new Add<float>(out, b1, "add1"));
        out = this->AddOperator(new Sigmoid<float>(out, "relu1"));

        // ======================= layer 2=======================
        out = this->AddOperator(new MatMul<float>(out, w2, "matmul2"));
        out = this->AddOperator(new Add<float>(out, b2, "add2"));
        result = this->AddOperator(new Sigmoid<float>(out, "relu2"));

        // ======================= Error=======================
        Objective<float> *err = this->SetObjectiveFunction(new MSE<float>(result, m_label, "MSE"));

        // ======================= Optimizer=======================
        this->SetOptimizer(new GradientDescentOptimizer<float>(err, 0.5, MINIMIZE));
    }

    virtual ~MLP() {}

    void SetTensor(Tensor<float> *px, Tensor<float> *plabel) {
        m_x->SetTensor(px);
        m_label->SetTensor(plabel);
    }

    Operator<float>* GetClassificationResult() {
        return result;
    }
};


int main(int argc, char const *argv[]) {
    // create input, label data placeholder
    Placeholder<float> *x = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 784, 1.0), "x");
    Placeholder<float> *label = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.f), "label");

    MLP mlp(x, label);

    Operator<float> *result = mlp.GetClassificationResult();

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Training =======================
        double train_accuracy = 0.f;

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);
            mlp.SetTensor(dataset->GetTrainFeedImage(), dataset->GetTrainFeedLabel());

            mlp.Training();

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
            mlp.SetTensor(dataset->GetTestFeedImage(), dataset->GetTestFeedLabel());

            mlp.Testing();

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
