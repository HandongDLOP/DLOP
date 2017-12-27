#include "Operator//Tensorholder.h"
// #include "Operator//Variable.h"

#include "Operator//Reshape.h"

// #include "Operator//Threshold.h"
#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"
// #include "Operator//Softmax.h"

#include "Operator//Addfc.h"
#include "Operator//Addconv.h"
#include "Operator//MatMulfc.h"
// #include "Operator//Convolution.h"
// #include "Operator//Maxpooling.h"

#include "Objective//MSE.h"
// #include "Objective//CrossEntropy.h"
#include "Objective//SoftmaxCrossEntropy.h"

// int main(int argc, char const *argv[]) {
// Tensor<float> *_input       = Tensor<float>::Constants(1, 20, 1, 1, 4, 2);
// Tensorholder<float> *input  = new Tensorholder<float>(_input, "tensorholder");
// Tensor<float> *_weight      = Tensor<float>::Constants(1, 1, 1, 4, 3, 2);
// Tensorholder<float> *weight = new Tensorholder<float>(_weight, "tensorholder");
// Tensor<float> *_bias        = Tensor<float>::Constants(1, 1, 1, 1, 3, 1);
// Tensorholder<float> *bias   = new Tensorholder<float>(_bias, "tensorholder");
// Tensor<float> *_label       = Tensor<float>::Constants(1, 20, 1, 1, 3, 0);
// Tensorholder<float> *label  = new Tensorholder<float>(_label, "tensorholder");
//
// Operator<float> *matmul = new MatMul<float>(input, weight, "MatMulfc");
// Operator<float> *add    = new Add<float>(matmul, bias, "addfc");
// Operator<float> *act    = new Relu<float>(add, "act");
//
// Operator<float> *err = new MSE<float>(act, label, "act");
//
//// std::cout << weight->GetResult() << '\n';
//// std::cout << input->GetResult() << '\n';
//
// matmul->ComputeForwardPropagate();
// add->ComputeForwardPropagate();
// act->ComputeForwardPropagate();
// err->ComputeForwardPropagate();
//
// std::cout << input->GetResult() << '\n';
// std::cout << weight->GetResult() << '\n';
// std::cout << matmul->GetResult() << '\n';
// std::cout << bias->GetResult() << '\n';
// std::cout << add->GetResult() << '\n';
// std::cout << act->GetResult() << '\n';
//
// err->ComputeBackPropagate();
// act->ComputeBackPropagate();
// add->ComputeBackPropagate();
// matmul->ComputeBackPropagate();
//
// std::cout << act->GetDelta() << '\n';
// std::cout << add->GetDelta() << '\n';
// std::cout << bias->GetDelta() << '\n';
// std::cout << matmul->GetDelta() << '\n';
// std::cout << weight->GetDelta() << '\n';
//
//
// std::cout << err->GetResult()->GetShape() << '\n';
// std::cout << act->GetResult()->GetShape() << '\n';
// std::cout << add->GetResult()->GetShape() << '\n';
// std::cout << matmul->GetResult()->GetShape() << '\n';
//
//// matmul->ComputeBackPropagate();
////
//// std::cout << input->GetDelta() << ' ';
//// std::cout << weight->GetDelta() << ' ';
//
// delete input;
// delete weight;
// delete matmul;
// delete add;
//// delete act;
//// delete err;
//
//
// return 0;
//
// }

// int main(int argc, char const *argv[]) {
// Tensor<float> *_input      = Tensor<float>::Constants(1, 1, 1, 10, 2, 2);
// Tensorholder<float> *input = new Tensorholder<float>(_input, "tensorholder");
//
// Operator<float> *re = new Reshape<float>(input, 1, 1, 1, 4, 5, "reshape");
//
// re->ComputeForwardPropagate();
//
// int ca             = re->GetDelta()->GetData()->GetCapacity();
// Tensor<float> *del = re->GetDelta();
//
// for (int i = 0; i < ca; i++) {
// (*del)[i] = i;
// }
//
// std::cout << input->GetResult() << '\n';
// std::cout << re->GetResult() << '\n';
//
// re->ComputeBackPropagate();
//
// std::cout << del << '\n';
// std::cout << input->GetDelta() << '\n';
//
// delete input;
// delete re;
//
// return 0;
// }

// int main(int argc, char const *argv[]) {
// Tensor<float> *_input      = Tensor<float>::Constants(1, 5, 3, 2, 2, 2);
// Tensorholder<float> *input = new Tensorholder<float>(_input, "tensorholder");
// Tensor<float> *_bias       = Tensor<float>::Constants(1, 1, 1, 1, 3, 1);
// Tensorholder<float> *bias  = new Tensorholder<float>(_bias, "tensorholder");
//
// Operator<float> *add = new Addconv<float>(input, bias, "addconv");
//
// add->ComputeForwardPropagate();
//
// int ca             = add->GetDelta()->GetData()->GetCapacity();
// Tensor<float> *del = add->GetDelta();
//
// for (int i = 0; i < ca; i++) {
// (*del)[i] = 1;
// }
//
// add->ComputeBackPropagate();
//
// std::cout << bias->GetDelta() << '\n';
// std::cout << input->GetDelta() << '\n';
//
// delete input;
// delete bias;
// delete add;
//
// return 0;
// }

int main(int argc, char const *argv[]) {
    Tensor<float> *_input      = Tensor<float>::Constants(1, 1, 1, 1, 10, 0);
    Tensorholder<float> *input = new Tensorholder<float>(_input, "tensorholder");
    Tensor<float> *_label      = Tensor<float>::Constants(1, 1, 1, 1, 10, 0);
    (*_label)[2] = 1.0;
    Tensorholder<float> *label = new Tensorholder<float>(_label, "tensorholder");

    Operator<float> *err = new SoftmaxCrossEntropy<float>(input, label, "err");

    err->ComputeForwardPropagate();

    std::cout << err->GetResult() << '\n';

    err->ComputeBackPropagate();

    // std::cout << bias->GetDelta() << '\n';
    std::cout << input->GetDelta() << '\n';

    delete input;
    delete label;
    delete err;

    return 0;
}
