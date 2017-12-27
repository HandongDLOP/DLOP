#ifndef SOFTMAXCROSSENTROPY_H_
#define SOFTMAXCROSSENTROPY_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class SoftmaxCrossEntropy : public Operator<DTYPE>{
private:
    Tensor<DTYPE> *m_aSoftmaxResult;
    DTYPE m_epsilon;  // for backprop

public:
    SoftmaxCrossEntropy(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, DTYPE epsilon = 1e-2) : Operator<DTYPE>(pInput, pLabel) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        Alloc(pInput, pLabel, epsilon);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, std::string pName) : Operator<DTYPE>(pInput, pLabel, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        Alloc(pInput, pLabel);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pInput, pLabel, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        Alloc(pInput, pLabel, epsilon);
    }

    ~SoftmaxCrossEntropy() {
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
        delete m_aSoftmaxResult;
    }

    virtual int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, DTYPE epsilon = 1e-2) {
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';

        int timesize  = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        Shape *shapeOfSoftmaxResult = new Shape(pInput->GetResult()->GetShape());
        m_aSoftmaxResult = new Tensor<DTYPE>(shapeOfSoftmaxResult);

        m_epsilon = epsilon;

        return TRUE;
    }

    virtual int ComputeForwardPropagate() {
        Tensor<DTYPE> *input         = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *label         = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();
        result->Reset();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        DTYPE sum[timesize][batchsize] = { 0.f, };
        DTYPE max[timesize][batchsize] = { 0.f, };
        int   numOfOutputDim           = 0;

        int count    = timesize * batchsize;
        int capacity = colsize;


        int start = 0;
        int end   = 0;

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                start = (ti * batchsize + ba) * capacity;
                end   = start + capacity;

                max[ti][ba] = Max(input, start, end);
            }
        }

        DTYPE temp = 0.f;

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                start = (ti * batchsize + ba) * capacity;
                end   = start + capacity;

                for (int i = start; i < end; i++) {
                    temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
                }
                sum[ti][ba] = temp;
                temp        = 0.f;
            }
        }

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                start = (ti * batchsize + ba) * capacity;
                end   = start + capacity;

                for (int i = start; i < end; i++) {
                    (*softmaxresult)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
                    std::cout << -(*label)[i] * log((*softmaxresult)[i] + m_epsilon) / capacity << '\n';
                    (*result)[start] += -(*label)[i] * log((*softmaxresult)[i] + m_epsilon) / capacity;
                }
            }
        }

        return TRUE;
    }

    virtual int ComputeBackPropagate() {
        Tensor<DTYPE> *label         = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;

        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();

        int capacity = input_delta->GetData()->GetCapacity();

        int numOfOutputDim = label->GetColSize();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] = ((*softmaxresult)[i] - (*label)[i]) / numOfOutputDim;
        }

        return TRUE;
    }

    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }
};

#endif  // SOFTMAXCROSSENTROPY_H_
