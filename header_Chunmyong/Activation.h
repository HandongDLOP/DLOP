#ifndef ACTIVATION_H_
#define ACTIVATION_H_    value

#include <string>
#include "Tensor.h"
#include "Layer.h"

// class Activation : public Layer {
// public:
// Activation() {}
//
// virtual ~Activation() {}
//
//// Layer가 가지고 있는 요소들의 주소를 그대로 할당할 수 있도록한다.
//// 상속을 받아서 사용할까 생각했지만 그렇게 사용하기에는, layer하나에 관여하는 요소가 이 것 하나 뿐이 아니다.
// int    Alloc();
// void   Delete();
//
// Tensor Activate();
// Tensor DerActivationFromOutput();
// };

class Relu : public Layer {
public:
    Relu() {
        std::cout << "Relu::Relu() : public Layer" << '\n';
    }

    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }
};

class Sigmoid : public Layer {
public:
    Sigmoid() {
        std::cout << "Sigmoid::Sigmoid() : public Layer" << '\n';
    }

    virtual ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }
};


#endif  // ACTIVATION_H_
