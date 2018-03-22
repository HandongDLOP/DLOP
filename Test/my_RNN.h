#include <iostream>
#include <string>

#include "../Header/NeuralNetwork.h"

#define TIME  2
#define BATCH 2

class my_RNN : public NeuralNetwork<float>{
private:
public:
    my_RNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = NULL;

        // AddPlaceholder(x);
        // AddPlaceholder(label);

        // InputWeight row != HiddenWeight, OutputWeight row
        // bias
        Operator<float> *w1 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 2, 2, 0.0, 0.1), "weight1"));
        Operator<float> *b1 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(TIME, BATCH, 1, 1, 2, 0.1), "bias1"));
        Operator<float> *w2 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 2, 2, 0.0, 0.1), "weight2"));
        Operator<float> *b2 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(TIME, BATCH, 1, 1, 2, 0.1), "bias2"));
        Operator<float> *w3 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 2, 2, 0.0, 0.1), "weight3"));
        Operator<float> *b3 = AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(TIME, BATCH, 1, 1, 2, 0.1), "bias3"));

        std::cout << "w1" << '\n' << w1->GetResult() << '\n';
        std::cout << "w2" << '\n' << w2->GetResult() << '\n';
        std::cout << "w3" << '\n' << w3->GetResult() << '\n';
        std::cout << "b1" << '\n' << b1->GetResult() << '\n';
        std::cout << "b2" << '\n' << b2->GetResult() << '\n';
        std::cout << "b3" << '\n' << b3->GetResult() << '\n';

        out = AddOperator(new Recurrent<float>(x, w1, w2, w3, b1, b2, b3, 3, "rnn"));
        //out->ComputeForwardPropagate();
        //std::cout << out->GetResult() << '\n';

        // ======================= Select Objective Function ===================
        //Objective<float> *objective = new Objective<float>(out, label,"SCE");
        //Objective<float> *objective = new SoftmaxCrossEntropy<float>(out, label, 0.0000001, "SCE");
        Objective<float> *objective = new MSE<float>(out, label, "MSE");

        SetObjective(objective);

        // ======================= Select Optimizer ===================
        Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

        SetOptimizer(optimizer);
    }

    virtual ~my_RNN() {}
};
