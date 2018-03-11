#include <iostream>

#include "..//..//Header//NeuralNetwork.h"

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
        Operator<float> *out = NULL;

        // ======================= layer 1======================
        out = AddFullyConnectedLayer(x, 784, 10, FALSE, "1");

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
        Operator<float> *out = NULL;

        // ======================= layer 1======================
        out = AddFullyConnectedLayer(x, 784, 15, TRUE, "1");

        // ======================= layer 2=======================
        out = AddFullyConnectedLayer(out, 15, 10, TRUE, "2");

        // ======================= Select Objective Function ===================
        // Objective<float> *objective = new Objective<float>(out, label,"SCE");
        // Objective<float> *objective = new SoftmaxCrossEntropy<float>(out, label, 0.000001, "SCE");
        Objective<float> *objective = new MSE<float>(out, label, "MSE");

        SetObjective(objective);

        // ======================= Select Optimizer ===================
        Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

        SetOptimizer(optimizer);
    }

    Operator<float>* AddFullyConnectedLayer(Operator<float> *pInput, int pColSize_in, int pColSize_out, int pActivation, std::string pLayernum) {
        Operator<float> *out = NULL;

        Operator<float> *weight = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, pColSize_in, pColSize_out, 0.0, 0.1), "fc_weight" + pLayernum));
        Operator<float> *bias   = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, pColSize_out, 0.1), "fc_bias" + pLayernum));

        out = AddOperator(new BroadcastMatMul<float>(pInput, weight, "fc_matmul" + pLayernum));
        out = AddOperator(new Add<float>(out, bias, "fc_add" + pLayernum));

        if (pActivation) {
            out = AddOperator(new Sigmoid<float>(out, "fc_act" + pLayernum));
        }

        return out;
    }

    virtual ~my_NN() {}
};
