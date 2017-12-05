#ifndef PLACEHOLDER_H_
#define PLACEHOLDER_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Placeholder : public Operator<DTYPE>{
private:
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

    bool Alloc(Tensor<DTYPE> *pTensor) {
        this->SetOutput(pTensor);

        // no meaning
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(pTensor->GetShape());
        this->SetDelta(delta);

        return true;
    }

    bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return true;
    }

    bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};


#endif  // PLACEHOLDER_H_
