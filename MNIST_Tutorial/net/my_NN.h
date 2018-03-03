#include <iostream>

#include "..//..//Header//NeuralNetwork.h"

enum MODEL_OPTION{
    isSLP,
    isMLP
};

class my_NN : public NeuralNetwork<float>{
private:
public:
    my_NN(Placeholder<float> *x, MODEL_OPTION pOption) {
        if (pOption == isSLP) SLP(x);
        else if (pOption == isMLP) MLP(x);
    }

    void SLP(Placeholder<float> *x) {
        // ======================= layer 1======================
        AddFullyConnectedLayer(x, 784, 10, FALSE, "1");

    }

    void MLP(Placeholder<float> *x) {
        Operator<float> *out = NULL;

        // ======================= layer 1======================
        out = AddFullyConnectedLayer(x, 784, 15, TRUE, "1");

        // ======================= layer 2=======================
        AddFullyConnectedLayer(out, 15, 10, TRUE, "2");

    }

    Operator<float>* AddFullyConnectedLayer(Operator<float> *pInput, int pColSize_in, int pColSize_out, int pActivation, std::string pLayernum) {
        Operator<float> *out = NULL;

        Operator<float> *weight = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, pColSize_in, pColSize_out, 0.0, 0.1), "fc_weight" + pLayernum));
        Operator<float> *bias = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, pColSize_out, 0.1), "fc_bias" + pLayernum));

        out = AddOperator(new MatMul<float>(pInput, weight, "fc_matmul" + pLayernum));
        out = AddOperator(new Add<float>(out, bias, "fc_add" + pLayernum));

        if (pActivation) {
            out = AddOperator(new Sigmoid<float>(out, "fc_act" + pLayernum));
        }

        return out;
    }

    virtual ~my_NN() {}
};
