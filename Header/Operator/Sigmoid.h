#ifndef SIGMOID_H_
#define SIGMOID_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Sigmoid : public Operator<DTYPE>{
public:
    Sigmoid(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Sigmoid::Sigmoid(Operator *)" << '\n';
        this->Alloc(pInput);
    }

    ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Sigmoid::Alloc(Operator *, Operator *)" << '\n';

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        int capacity          = input->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*result)[i] = this->SIGMOID((*input)[i]);
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        int capacity               = result->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] += (*result)[i] * (1 - (*result)[i]) * (*this_delta)[i];
        }
        return TRUE;
    }

    inline DTYPE SIGMOID(DTYPE data) {
        return 1.F / (1.F + (DTYPE)exp(-data));
    }
};

#endif  // SIGMOID_H_
