#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Maxpooling : public Operator<DTYPE>{
public:
    Maxpooling(Operator<DTYPE> *pInput, int strideRow, int strideCol) : Operator<DTYPE>(pInput) {
        std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, int, int)" << '\n';
        this->Alloc(pInput, strideRow, strideCol);
    }

    Maxpooling(Operator<DTYPE> *pInput, int strideRow, int strideCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(pInput, strideRow, strideCol);
    }

    ~Maxpooling() {
        std::cout << "Maxpooling::~Maxpooling()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol) {
        std::cout << "Maxpooling::Alloc(Operator<DTYPE> *, int, int)" << '\n';

        return 1;
    }

    //
    virtual int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';
        return 1;
    }

    //
    virtual int ComputeBackPropagate() {
        return 1;
    }
};
//
#endif  // MAXPOOLING_H_
