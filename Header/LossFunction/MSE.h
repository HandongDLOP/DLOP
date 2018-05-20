#ifndef MSE_H_
#define MSE_H_    value

#include "..//LossFunction.h"

template<typename DTYPE>
class MSE : public LossFunction<DTYPE>{
public:
    MSE(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #if __DEBUG__
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    virtual ~MSE() {
        #if __DEBUG__
        std::cout << "MSE::~MSE()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pOperator) {
        #if __DEBUG__
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(
            new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1)
            );

        this->SetGradient(
            new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize)
            );

        return TRUE;
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *input    = this->GetTensor();
        Tensor<DTYPE> *label    = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result   = this->GetResult();
        Tensor<DTYPE> *gradient = this->GetGradient();

        int timesize  = input->GetTimeSize();
        int batchsize = input->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int index = 0;

        int i = 0;

        int ti          = 0;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            i = ti * batchsize + ba;

            for (int j = 0; j < capacity; j++) {
                index              = i * capacity + j;
                (*result)[i]      += Error((*input)[index], (*label)[index]);
                (*gradient)[index] = ((*input)[index] - (*label)[index]);
            }
        }

        return result;
    }

    Tensor<DTYPE>* BackPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *gradient    = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int timesize  = gradient->GetTimeSize();
        int batchsize = gradient->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = gradient->GetChannelSize();
        int rowsize     = gradient->GetRowSize();
        int colsize     = gradient->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int index = 0;
        int i     = 0;

        int ti          = 0;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            i = ti * batchsize + ba;

            for (int j = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[index] += (*gradient)[index] / batchsize;
            }
        }

        return NULL;
    }

#if __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;
    }

    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif  // __CUDNN__


    inline DTYPE Error(DTYPE pred, DTYPE ans) {
        return (pred - ans) * (pred - ans) / 2;
    }
};

#endif  // MSE_H_
