#ifndef MSE_H_
#define MSE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MSE : public Operator<DTYPE>{
public:
    MSE(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    MSE(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';


        return 1;
    }

    virtual int ComputeForwardPropagate() {
        return 1;
    }

    virtual int ComputeBackPropagate() {
        return 1;
    }
};

#endif  // MSE_H_
