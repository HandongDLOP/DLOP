#ifndef MSE_H_
#define MSE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MSE : public Operator<DTYPE>{
public:
    MSE(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, std::string pName) : Operator<DTYPE>(pInput, pLabel, pName) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput, pLabel);
    }

    ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel) {
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int timesize  = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *label  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        int timesize  = input->GetTimeSize();
        int batchsize = input->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < capacity; j++) {
                index = i * capacity + j;

                (*result)[i] += Error((*input)[index], (*label)[index], capacity);
            }
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *label       = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();

        int capacity = input_delta->GetData()->GetCapacity();

        int channelsize    = input->GetChannelSize();
        int rowsize        = input->GetRowSize();
        int colsize        = input->GetColSize();
        int numOfOutputDim = channelsize * rowsize * colsize;

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] += ((*input)[i] - (*label)[i]) / numOfOutputDim;
        }

        return TRUE;
    }

    inline DTYPE Error(DTYPE pred, DTYPE ans, int numOfOutputDim) {
        return (pred - ans) * (pred - ans) / numOfOutputDim * 0.5;
    }
};

#endif  // MSE_H_
