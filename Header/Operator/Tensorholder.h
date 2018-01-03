#ifndef TENSORHOLDER_H_
#define TENSORHOLDER_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Tensorholder : public Operator<DTYPE>{
private:
    int m_isTrainable;

public:
    Tensorholder(Tensor<DTYPE> *pTensor, std::string pName, int pTrainable = 1) : Operator<DTYPE>(pName) {
        std::cout << "Tensorholder<DTYPE>::Tensorholder(Tensor<DTYPE> *, std::string)" << '\n';
        this->Alloc(pTensor, pTrainable);
    }

    ~Tensorholder() {
        std::cout << "Tensorholder<DTYPE>::~Tensorholder()" << '\n';
    }

    int Alloc(Tensor<DTYPE> *pTensor, int pTrainable) {
        std::cout << "Tensorholder<DTYPE>::Alloc(Tensor<DTYPE> *, std::string)" << '\n';

        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        // Shape *shapeOfDelta = new Shape(pTensor->GetShape());
        // this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        Shape *shapeOfGradient = new Shape(pTensor->GetShape());
        this->SetGradient(new Tensor<DTYPE>(shapeOfGradient));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        return TRUE;
    }

    int ComputeBackPropagate() {
// int capacity = this->GetResult()->GetData()->GetCapacity();
//
// Tensor<DTYPE> *delta    = this->GetDelta();
// Tensor<DTYPE> *gradient = this->GetGradient();
//
// for(int i = 0; i < capacity; i++){
// (*gradient)[i] = (*delta)[i];
// }

        return TRUE;
    }
};

#endif  // TENSORHOLDER_H_
