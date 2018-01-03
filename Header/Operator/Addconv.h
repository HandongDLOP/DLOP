#ifndef AddconvCONV_H_
#define AddconvCONV_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Addconv : public Operator<DTYPE>{
public:
    Addconv(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) : Operator<DTYPE>(pInput, pBias, pName) {
        std::cout << "Addconv::Addconv(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput, pBias);
    }

    ~Addconv() {
        std::cout << "Addconv::~Addconv()" << '\n';
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

        int timesize  = input->GetTimeSize();
        int batchsize = input->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = input->GetChannelSize();

        int rowsize   = input->GetRowSize();
        int colsize   = input->GetColSize();
        int planesize = rowsize * colsize;

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int j = 0; j < planesize; j++) {
                    index = i * channelsize * planesize + ch * planesize + j;

                    (*result)[index] = (*input)[index] + (*bias)[ch];
                }
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

        int timesize  = input_delta->GetTimeSize();
        int batchsize = input_delta->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = input_delta->GetChannelSize();

        int rowsize   = input_delta->GetRowSize();
        int colsize   = input_delta->GetColSize();
        int planesize = rowsize * colsize;

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int j = 0; j < planesize; j++) {
                    index = i * channelsize * planesize + ch * planesize + j;

                    (*input_delta)[index] = (*this_delta)[index];
                    (*bias_gradient)[ch]    += (*this_delta)[index];
                }
            }
        }

        return TRUE;
    }
};

#endif  // AddconvCONV_H_
