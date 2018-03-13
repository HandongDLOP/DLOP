#include <iostream>

#include "../../Header/NeuralNetwork.h"

enum MODEL_OPTION {
    isSLP,
    isMLP
};

class my_NN : public NeuralNetwork<float>{
private:
public:
    my_NN(Tensorholder<float> *x, Tensorholder<float> *label, MODEL_OPTION pOption) {
        if (pOption == isSLP) SLP(x, label);
        else if (pOption == isMLP) MLP(x, label);
    }

    void SLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = AddLayer(new _Layer::Linear<float>(out, 784, 10, "No", TRUE, "1"));

        // ======================= Select Objective Function ===================
        // Objective<float> *objective = new Objective<float>(out, label,"SCE");
        Objective<float> *objective = new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE");
        // Objective<float> *objective = new MSE<float>(out, label, "MSE");

        SetObjective(objective);

        // ======================= Select Optimizer ===================
        Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

        SetOptimizer(optimizer);
    }

    void MLP(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = x;

        // ======================= layer 1======================
        out = AddLayer(new _Layer::Linear<float>(out, 784, 15, "Sigmoid", TRUE, "1"));

        // ======================= layer 2=======================
        out = AddLayer(new _Layer::Linear<float>(out, 15, 10, "No", TRUE, "2"));

        // ======================= Select Objective Function ===================
        // Objective<float> *objective = new Objective<float>(out, label,"SCE");
        // Objective<float> *objective = new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE");
        Objective<float> *objective = new MSE<float>(out, label, "MSE");

        SetObjective(objective);

        // ======================= Select Optimizer ===================
        Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

        SetOptimizer(optimizer);
    }

    virtual ~my_NN() {}
};
