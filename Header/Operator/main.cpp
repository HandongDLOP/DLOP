#include "Tensorholder.h"

int main(int argc, char const *argv[]) {
    Tensor<float> *weight = Tensor<float>::Constants(1, 1, 1, 3, 3, 1);

    Tensorholder<float> *temp = new Tensorholder<float>(weight, "tensorholder");

    temp->ComputeForwardPropagate();

    int capacity         = temp->GetResult()->GetData()->GetCapacity();
    Tensor<float> *delta = temp->GetDelta();

    for (int i = 0; i < capacity; i++) {
        (*delta)[i] = 3.0;
    }

    temp->ComputeBackPropagate();

    Tensor<float> *grad = temp->GetGradient();

    for (int i = 0; i < capacity; i++) {
        std::cout << (*grad)[i] << ' ';
    }

    std::cout << grad->GetShape() << '\n';

    delete temp;

    return 0;
}
