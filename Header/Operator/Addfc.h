#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Add : public Operator<DTYPE>{
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

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int count       = timesize * batchsize * channelsize * rowsize;

        int bias_capacity = bias->GetData()->GetCapacity();

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < bias_capacity; j++) {
                index = i * bias_capacity + j;

                (*result)[index] = (*input)[index] + (*bias)[j];
                std::cout << index << '\n';
                std::cout << j << '\n';
            }
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        return TRUE;
    }
};

#endif  // ADD_H_
