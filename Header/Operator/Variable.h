#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Variable : public Operator<DTYPE>{
public:
    Variable(std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor<DTYPE> *pTensor, std::string pName, int pTrainable = 1) : Operator<DTYPE>(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor<DTYPE> *, std::string)" << '\n';

        this->Alloc(pTensor, pTrainable);
    }

    ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual int Alloc(Tensor<DTYPE> *pTensor, int pTrainable) {
        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // VARIABLE_H_
