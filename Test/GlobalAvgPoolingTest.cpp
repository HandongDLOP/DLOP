#include "../Header/NeuralNetwork.h"

int main(int argc, char const *argv[]) {
    // Tensor<float> *_x = Tensor<float>::Truncated_normal(1, 2, 3, 2, 2, 0.0, 0.1);
    Tensor<float> *_x = Tensor<float>::Constants(1, 2, 3, 2, 2, 1);

    Tensorholder<float> *x = new Tensorholder<float>(_x, "x");

    GlobalAvaragePooling2D<float> *avg = new GlobalAvaragePooling2D<float>(x, "avg");

    avg->ForwardPropagate(int pTime = 0, int pThreadNum = 0);
    avg->BackPropagate(int pTime = 0, int pThreadNum = 0);

    std::cout << _x << '\n';
    std::cout << avg->GetResult() << '\n';
    std::cout << x->GetGradient() << '\n';

    return 0;
}
