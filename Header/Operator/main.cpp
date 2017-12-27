#include "Tensorholder.h"
#include "Relu.h"
#include "Sigmoid.h"
#include "Addfc.h"
#include "MatMulfc.h"

// int main(int argc, char const *argv[]) {
// Tensor<float> *weight = Tensor<float>::Constants(1, 1, 1, 3, 3, 3);
//
// Tensorholder<float> *temp = new Tensorholder<float>(weight, "tensorholder");
// Operator<float>     *act  = new Sigmoid<float>(temp, "act");
//
// temp->ComputeForwardPropagate();
// act->ComputeForwardPropagate();
//
// int capacity = act->GetResult()->GetData()->GetCapacity();
//
// Tensor<float> *result = act->GetResult();
//
// for (int i = 0; i < capacity; i++) {
// std::cout << (*result)[i] << ' ';
// }
//
//// int capacity         = act->GetResult()->GetData()->GetCapacity();
//// Tensor<float> *delta = act->GetDelta();
////
//// for (int i = 0; i < capacity; i++) {
//// (*delta)[i] = 1.0;
//// }
////
//// act->ComputeBackPropagate();
//// temp->ComputeBackPropagate();
////
//// Tensor<float> *grad = temp->GetGradient();
////
//// for (int i = 0; i < capacity; i++) {
//// std::cout << (*grad)[i] << ' ';
//// }
////
//// std::cout << grad->GetShape() << '\n';
//
// delete temp;
// delete act;
//
// return 0;
// }


// int main(int argc, char const *argv[]) {
// Tensor<float> *_input      = Tensor<float>::Constants(1, 10, 1, 1, 10, 3);
// Tensorholder<float> *input = new Tensorholder<float>(_input, "tensorholder");
// Tensor<float> *_bias       = Tensor<float>::Constants(1, 1, 1, 1, 10, 3);
// Tensorholder<float> *bias  = new Tensorholder<float>(_bias, "tensorholder");
//
// Operator<float> *add = new Add<float>(input, bias, "addfc");
//
// add->ComputeForwardPropagate();
//
// int capacity = add->GetResult()->GetData()->GetCapacity();
// Tensor<float> *result = add->GetResult();
//
// for(int i = 0; i < capacity ; i++){
// std::cout << (*result)[i] << ' ';
// }
//
// delete input;
// delete bias;
// delete add;
//
// return 0;
// }


int main(int argc, char const *argv[]) {
    Tensor<float> *_input       = Tensor<float>::Constants(1, 10, 1, 1, 10, 3);
    Tensorholder<float> *input  = new Tensorholder<float>(_input, "tensorholder");
    Tensor<float> *_weight      = Tensor<float>::Constants(1, 1, 1, 10, 5, 3);
    Tensorholder<float> *weight = new Tensorholder<float>(_weight, "tensorholder");

    Operator<float> *matmul = new MatMul<float>(input, weight, "MatMulfc");

    matmul->ComputeForwardPropagate();

    int capacity          = matmul->GetResult()->GetData()->GetCapacity();
    Tensor<float> *result = matmul->GetResult();
    Tensor<float> *delta  = matmul->GetDelta();

    std::cout << result << ' ';

    for (int i = 0; i < capacity; i++) {
        (*delta)[i] = 1;
    }

    matmul->ComputeBackPropagate();

    std::cout << input->GetDelta() << ' ';
    std::cout << weight->GetDelta() << ' ';

    delete input;
    delete weight;
    delete matmul;

    return 0;
}
