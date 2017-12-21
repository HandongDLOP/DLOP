#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Convolution : public Operator<DTYPE>{
public:
    Convolution(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int stride1, int stride2, int stride3, int stride4, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        Alloc(pInput0, pInput1, stride1, stride2, stride3, stride4);
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int stride1, int stride2, int stride3, int stride4) {
        return 1;
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }

    virtual int ComputeForwardPropagate() {
        return true;
    }

    virtual int ComputeBackPropagate() {
        std::cout << this->GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};

#endif  // CONVOLUTION_H_
