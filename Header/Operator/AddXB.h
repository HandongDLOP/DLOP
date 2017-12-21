#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Add : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Add::Alloc())
    Add(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "Add::Add(Operator<DTYPE> *, MetaParameter *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    Add(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "Add::Add(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // ADD_H_
