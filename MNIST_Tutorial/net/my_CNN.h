#include <iostream>
#include <string>

#include "../../Header/NeuralNetwork.h"

class my_CNN : public NeuralNetwork<float>{
private:
public:
    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = NULL;

        int batch_size = x->GetResult()->GetBatchSize();

        out = AddOperator(new Reshape<float>(x, 1, batch_size, 1, 28, 28, "reshape"));

        // ======================= layer 1=======================
        out = AddLayer(new ConvolutionLayer2D<float>(out, 1, 32, 3, 3, 1, 1, VALID, FALSE, "1"));
        out = AddLayer(new BatchNormalizeLayer2D<float>(out, 32, "1"));
        out = AddOperator(new Relu<float>(out, "Relu_1"));
        out = AddOperator(new Maxpooling2D<float>(out, 2, 2, 2, 2, VALID, "MaxPool_1"));


        // ======================= layer 2=======================
        out = AddLayer(new ConvolutionLayer2D<float>(out, 32, 64, 3, 3, 1, 1, VALID, FALSE, "1"));
        out = AddLayer(new BatchNormalizeLayer2D<float>(out, 64, "1"));
        out = AddOperator(new Relu<float>(out, "Relu_2"));
        out = AddOperator(new Maxpooling2D<float>(out, 2, 2, 2, 2, VALID, "MaxPool_2"));

        out = AddOperator(new Reshape<float>(out, 1, batch_size, 1, 1, 5 * 5 * 64, "Flat"));

        // ======================= layer 3=======================
        out = AddLayer(new Linear<float>(out, 5 * 5 * 64, 256, TRUE, "3"));

        out = AddOperator(new Relu<float>(out, "Relu_3"));

        // ======================= layer 4=======================
        out = AddLayer(new Linear<float>(out, 256, 10, TRUE, "4"));

        // ======================= Select Objective Function ===================
        SetObjective(new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE"));
        // SetObjective(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE));
    }

    virtual ~my_CNN() {}
};
