// #include "DLOP.h"
#include "Operator.h"

int main(int argc, char const *argv[]) {
    Operator<int> *x0 = new Operator<int>("x0");
    Operator<int> *x1 = new Operator<int>("x1");
    Operator<int> *x2 = new Operator<int>(x1, "x2");
    Operator<int> *x3 = new Operator<int>(x1, "x3");
    Operator<int> *x4 = new Operator<int>(x2, x3, "x4");
    Operator<int> *x5 = new Operator<int>(x4, x0, "x5");

    for (int i = 0; i < 20; i++) {
        std::cout << "Forward" << '\n';
        x5->ForwardPropagate();
        std::cout << "Back" << '\n';
        x5->BackPropagate();
    }

    delete x1;
    delete x2;
    delete x3;
    delete x4;
    delete x5;

    return 0;
}
