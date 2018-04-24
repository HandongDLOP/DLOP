#ifndef ONEHOT_H_
#define ONEHOT_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Onehot : public Operator<DTYPE>{
private:
    int m_capacity;

public:
    Onehot(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Onehot::Onehot(Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput);
    }

    ~Onehot() {
        std::cout << "Onehot::~Onehot()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pLeftInput) {
        std::cout << "Onehot::Alloc(Operator<DTYPE> *)" << '\n';

        Shape *pInputTenShape = pLeftInput->GetResult()->GetShape();

        int timesize    = (*pInputTenShape)[0];

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_input  = (*input_contatiner)[0]->GetResult();

        return TRUE;
    }

    int ComputeBackPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_input_grad  = (*input_contatiner)[0]->GetGradient();

        return TRUE;
    }
};
