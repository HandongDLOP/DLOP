#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MatMul : public Operator<DTYPE>{
public:
    MatMul(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput, pWeight);
    }

    ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight) {
        std::cout << "Add::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int count       = timesize * batchsize * channelsize;

        int rowsize = result->GetRowSize();
        int colsize = result->GetColSize();

        int hiddensize = input->GetColSize();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        DTYPE temp = 0.f;

        for (int i = 0; i < count; i++) {
            for (int ro = 0; ro < rowsize; ro++) {
                for (int co = 0; co < colsize; co++) {
                    for (int hid = 0; hid < hiddensize; hid++) {
                        input_index  = (i * rowsize + ro) * hiddensize + hid;
                        weight_index = hid * colsize + co;

                        temp += (*input)[input_index] * (*weight)[weight_index];
                    }
                    result_index            = (i * rowsize + ro) * colsize + co;
                    (*result)[result_index] = temp;
                    temp                    = 0.f;
                }
            }
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();
        Tensor<DTYPE> *weight_delta = this->GetInput()[1]->GetDelta();
        weight_delta->Reset();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int count       = timesize * batchsize * channelsize;

        int rowsize    = this_delta->GetRowSize();
        int colsize    = this_delta->GetColSize();
        int hiddensize = input_delta->GetColSize();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int i = 0; i < count; i++) {
            for (int ro = 0; ro < rowsize; ro++) {
                for (int co = 0; co < colsize; co++) {
                    for (int hid = 0; hid < hiddensize; hid++) {
                        input_index  = (i * rowsize + ro) * hiddensize + hid;
                        weight_index = hid * colsize + co;
                        result_index = (i * rowsize + ro) * colsize + co;

                        (*input_delta)[input_index]   += (*weight)[weight_index] * (*this_delta)[result_index];
                        (*weight_delta)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                    }
                }
            }
        }

        return TRUE;
    }
};

#endif  // MATMUL_H_
