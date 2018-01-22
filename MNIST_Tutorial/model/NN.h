#include <iostream>
#include <string>

#include "..//..//Header//NeuralNetwork.h"

class NN : public NeuralNetwork<float>{
private:
public:
    NN(Placeholder<float> *x, Placeholder<float> *label) {
        SLP(x, label);
        // MLP(x, label);
    }

    void SLP(Placeholder<float> *x, Placeholder<float> *label) {
        Operator<float> *out = NULL;

        // ======================= layer 1======================
        out = AddFullyConnectedLayer(x, 784, 10, FALSE, "1");

        // ======================= Error=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetObjectiveFunction(new SoftmaxCrossEntropy<float>(out, label, 1e-50, "SCE"));

        // ======================= Optimizer=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetOptimizer(new GradientDescentOptimizer<float>(0.01, MINIMIZE));
    }

    void MLP(Placeholder<float> *x, Placeholder<float> *label) {
        Operator<float> *out = NULL;

        // ======================= layer 1======================
        out = AddFullyConnectedLayer(x, 784, 15, TRUE, "1");

        // ======================= layer 2=======================
        out = AddFullyConnectedLayer(out, 15, 10, TRUE, "2");

        // ======================= Error=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetObjectiveFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Optimizer=======================
        // 추후에는 NN과는 독립적으로 움직이도록 만들기
        SetOptimizer(new GradientDescentOptimizer<float>(0.5, MINIMIZE));
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

    virtual ~NN() {}
};
