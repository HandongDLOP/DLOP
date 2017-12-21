#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MatMul : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    MatMul(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    MatMul(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // MATMUL_H_
