#include "Tensorholder.h"

int main(int argc, char const *argv[]) {
    Tensor<float> *weight = Tensor<float>::Constants(1, 100, 1, 28, 28, 1);

    Tensorholder<float> *temp = new Tensorholder<float>(weight, "tensorholder");

    Shape *shape = temp->GetResult()->GetShape();

    std::cout << shape << '\n';

    delete temp;

    return 0;
}
