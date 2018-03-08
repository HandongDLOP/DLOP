#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Add : public Operator<DTYPE>{
public:
    Add(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) : Operator<DTYPE>(pInput, pBias, pName) {
        std::cout << "Add::Add(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput, pBias);
    }

    ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        std::cout << "Add::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Shape *shapeOfResult = new Shape(pInput->GetResult()->GetShape());
        this->SetResult(new Tensor<DTYPE>(shapeOfResult));

        Shape *shapeOfDelta = new Shape(pInput->GetResult()->GetShape());
        this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *bias   = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int count       = timesize * batchsize * channelsize * rowsize;

        int bias_capacity = bias->GetCapacity();

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < bias_capacity; j++) {
                index = i * bias_capacity + j;

                (*result)[index] = (*input)[index] + (*bias)[j];
            }
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();
        Tensor<DTYPE> *bias_gradient = this->GetInput()[1]->GetGradient();
        bias_gradient->Reset();

        int timesize    = input_delta->GetTimeSize();
        int batchsize   = input_delta->GetBatchSize();
        int channelsize = input_delta->GetChannelSize();
        int rowsize     = input_delta->GetRowSize();
        int count       = timesize * batchsize * channelsize * rowsize;

        int bias_capacity = bias_gradient->GetCapacity();

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < bias_capacity; j++) {
                index = i * bias_capacity + j;

                (*input_delta)[index] = (*this_delta)[index];
                (*bias_gradient)[j]  += (*this_delta)[index];
            }
        }

        return TRUE;
    }
};

#endif  // ADD_H_
