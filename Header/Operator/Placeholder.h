#ifndef PLACEHOLDER_H_
#define PLACEHOLDER_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Placeholder : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    Placeholder(std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "Placeholder::Placeholder(std::string)" << '\n';
    }

    Placeholder(Tensor<DTYPE> *pTensor, std::string pName) : Operator<DTYPE>(pTensor, pName) {
        std::cout << "Placeholder::Placeholder(Tensor *, std::string)" << '\n';

        this->Alloc(pTensor);
    }

    ~Placeholder() {
        std::cout << "Placeholder::~Placeholder()" << '\n';
    }

    virtual int Alloc(Tensor<DTYPE> *pTensor) {
        this->SetOutput(pTensor);

        Tensor<DTYPE> *delta = new Tensor<DTYPE>(pTensor->GetShape());
        this->SetDelta(delta);

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return 1;
    }

    virtual int ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return 1;
    }
};


#endif  // PLACEHOLDER_H_
