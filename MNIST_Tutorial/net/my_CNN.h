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
        out = AddConvLayer(out, 1, 32, "1");

        // ======================= layer 2=======================
        out = AddConvLayer(out, 32, 64, "2");
        out = AddOperator(new Reshape<float>(out, 1, batch_size, 1, 1, 5 * 5 * 64, "flat"));

        // ======================= layer 3=======================
        out = AddLayer(new Linear<float>(out, 5 * 5 * 64, 256, TRUE, "3"));

        out = AddOperator(new Relu<float>(out, "relu"));

        // ======================= layer 4=======================
        out = AddLayer(new Linear<float>(out, 256, 10, TRUE, "4"));


        // ======================= Select Objective Function ===================
        // Objective<float> *objective = new Objective<float>(out, label,"SCE");
        Objective<float> *objective = new SoftmaxCrossEntropy<float>(out, label, 0.0000001, "SCE");
        // Objective<float> *objective = new MSE<float>(out, label, "MSE");

        SetObjective(objective);

        // ======================= Select Optimizer ===================
        Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

        SetOptimizer(optimizer);
    }

    Operator<float>* AddConvLayer(Operator<float> *pInput, int pChannelSize_in, int pChannelSize_out, std::string pLayernum) {
        Operator<float> *out = NULL;

        Operator<float> *weight = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, pChannelSize_out, pChannelSize_in, 3, 3, 0.0, 0.1), "conv_weight" + pLayernum));
        Operator<float> *bias   = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, pChannelSize_out, 1, 1, 0.1), "conv_bias" + pLayernum));

        out = AddOperator(new Convolution2D<float>(pInput, weight, 1, 1, 1, 1, VALID, "conv" + pLayernum));
        out = AddOperator(new Add<float>(out, bias, "conv_add" + pLayernum));
        out = AddOperator(new Relu<float>(out, "conv_relu" + pLayernum));
        out = AddOperator(new Maxpooling2D<float>(out, 2, 2, 2, 2, VALID, "maxpool" + pLayernum));

        return out;
    }


    virtual ~my_CNN() {}
};
