#ifndef PLACEHOLDER_H_
#define PLACEHOLDER_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Placeholder : public Operator<DTYPE>{
private:

public:
    Placeholder(Tensor<DTYPE> *pTensor, std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "Placeholder<DTYPE>::Placeholder(Tensor *, std::string)" << '\n';

        this->Alloc(pTensor);
    }

    ~Placeholder() {
        std::cout << "Placeholder<DTYPE>::~Placeholder()" << '\n';
    }

    int Alloc(Tensor<DTYPE> *pTensor) {
        std::cout << "Placeholder<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        Shape *shapeOfDelta = new Shape(pTensor->GetShape());
        this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return TRUE;
    }

    int ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return TRUE;
    }

    void SetTensor(Tensor<DTYPE> * pTensor){
        this->SetResult(pTensor);
    }

};


#endif  // PLACEHOLDER_H_
