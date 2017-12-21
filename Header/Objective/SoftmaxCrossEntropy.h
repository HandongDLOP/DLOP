#ifndef SOFTMAXCROSSENTROPY_H_
#define SOFTMAXCROSSENTROPY_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class SoftmaxCrossEntropy : public Operator<DTYPE>{
private:
    DTYPE m_epsilon = 0.0;  // for backprop

public:
    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon = 1e-20) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        Alloc(pInput0, pInput1, epsilon);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        Alloc(pInput0, pInput1);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        Alloc(pInput0, pInput1, epsilon);
    }

    ~SoftmaxCrossEntropy() {
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon = 1e-20) {
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // SOFTMAXCROSSENTROPY_H_
