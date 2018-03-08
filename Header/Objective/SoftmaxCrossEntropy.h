#ifndef SOFTMAXCROSSENTROPY_H_
#define SOFTMAXCROSSENTROPY_H_    value

#include "..//Objective.h"

template<typename DTYPE>
class SoftmaxCrossEntropy : public Objective<DTYPE>{
private:
    Tensor<DTYPE> *m_aSoftmaxResult;
    DTYPE m_epsilon;  // for backprop

public:
    SoftmaxCrossEntropy(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE epsilon = 1e-2, std::string pName = "NO NAME") : Objective<DTYPE>(pOperator, pLabel, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        Alloc(pOperator, epsilon);
    }

    virtual ~SoftmaxCrossEntropy() {
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
        Delete();
    }

    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon) {
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';

        Operator<DTYPE> *pInput = pOperator;

        int timesize  = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        Shape *shapeOfSoftmaxResult = new Shape(pInput->GetResult()->GetShape());

        m_aSoftmaxResult = new Tensor<DTYPE>(shapeOfSoftmaxResult);

        Shape *shapeOfGradient = new Shape(pInput->GetResult()->GetShape());

        this->SetGradient(new Tensor<DTYPE>(shapeOfGradient));

        m_epsilon = epsilon;

        return TRUE;
    }

    virtual void Delete() {
        if (m_aSoftmaxResult) {
            delete m_aSoftmaxResult;
            m_aSoftmaxResult = NULL;
        }
    }

    virtual Tensor<DTYPE>* ForwardPropagate() {
        // 추가로  backprop을 계속해서 구성해나가게 되면, 진행하는 것이 가능하다. label 값을 따로 저장하는 작업이 필요가 없어진다.

        Tensor<DTYPE> *input         = this->GetTensor();
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();
        Tensor<DTYPE> *gradient      = this->GetGradient();
        result->Reset();
        gradient->Reset();

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

                    (*result)[ti * batchsize + ba] += -(*label)[i] * log((*softmaxresult)[i] + m_epsilon);

                    (*gradient)[i] = (*softmaxresult)[i] - (*label)[i];
                }
            }
        }


        return result;
    }

    virtual Tensor<DTYPE>* BackPropagate() {
        Tensor<DTYPE> *gradient = this->GetGradient();

        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;

        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();
        input_delta->Reset();

        int capacity = input_delta->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] = (*gradient)[i];
        }


        return NULL;
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
