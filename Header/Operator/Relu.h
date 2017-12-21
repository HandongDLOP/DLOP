#ifndef RELU_H_
#define RELU_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Relu : public Operator<DTYPE>{
public:
    Relu(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // RELU_H_
