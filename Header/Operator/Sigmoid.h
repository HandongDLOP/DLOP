#ifndef SIGMOID_H_
#define SIGMOID_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Sigmoid : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Sigmoid::Alloc())
    Sigmoid(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Sigmoid::Sigmoid(Operator *)" << '\n';
        this->Alloc(pInput);
    }

    ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Sigmoid::Alloc(Operator *, Operator *)" << '\n';


        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // SIGMOID_H_
