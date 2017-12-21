#ifndef RESHAPE_H_
#define RESHAPE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Reshape : public Operator<DTYPE>{
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Reshape::Alloc())
    Reshape(Operator<DTYPE> *pInput, int pChannel, int pRow, int pCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        this->Alloc(pInput, pChannel, pRow, pCol);
    }

    ~Reshape() {
        std::cout << "Reshape::~Reshape()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput, int pChannel, int pRow, int pCol) {
        std::cout << "Reshape::Alloc(Operator *, Operator *)" << '\n';
        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // RESHAPE_H_
