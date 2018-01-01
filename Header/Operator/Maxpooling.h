#ifndef Maxpooling4D_H_
#define Maxpooling4D_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Maxpooling4D : public Operator<DTYPE>{
private:
    int stride[2] = { 0, };
    int mask[2]   = { 0, };

    Tensor<int> *indexOfMaxInput;

public:
    Maxpooling4D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol) : Operator<DTYPE>(pInput) {
        std::cout << "Maxpooling4D::Maxpooling4D(Operator<DTYPE> *, int, int)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    Maxpooling4D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling4D::Maxpooling4D(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    ~Maxpooling4D() {
        std::cout << "Maxpooling4D::~Maxpooling4D()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol) {
        std::cout << "Maxpooling4D::Alloc(Operator<DTYPE> *, int, int)" << '\n';

        Shape *shapeOfInput = pInput->GetResult()->GetShape();

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }  // 4D

        int rowsize = 0;
        int colsize = 0;

        if ((*shapeOfInput)[3] % strideRow > 0) {
            rowsize = (*shapeOfInput)[3] / strideRow + 1;
        } else {
            rowsize = (*shapeOfInput)[3] / strideRow;
        }

        if ((*shapeOfInput)[4] % strideCol > 0) {
            colsize = (*shapeOfInput)[4] / strideCol + 1;
        } else {
            colsize = (*shapeOfInput)[4] / strideCol;
        }

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));

        stride[0] = strideRow;
        stride[1] = strideCol;

        mask[0] = maskRow;
        mask[1] = maskCol;

        indexOfMaxInput = new Tensor<int>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize);

        return TRUE;
    }

    //
    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input   = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput    = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();
        result->Reset();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int rowsizeOfMask = mask[0];
        int colsizeOfMask = mask[1];

        DTYPE max = 0.f;

        int indexOfResult = 0;
        int indexOfInput  = 0;

        int temprow = 0;
        int tempcol = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int mro = 0; mro < rowsizeOfMask; mro++) {
                            for (int mco = 0; mco < colsizeOfMask; mco++) {
                                temprow = stride[0] * ro + mro;
                                tempcol = stride[1] * co + mco;

                                indexOfResult = Index4D(shapeOfResult, ba, ch, ro, co);
                                indexOfInput  = Index4D(shapeOfInput, ba, ch, temprow, tempcol);

                                if ((mro == 0) && (mco == 0)) {
                                    max                               = (*input)[indexOfInput];
                                    (*result)[indexOfResult]          = max;
                                    (*indexOfMaxInput)[indexOfResult] = indexOfInput;
                                } else {
                                    if(temprow < rowsizeOfInput && tempcol < colsizeOfInput){
                                        if (max < (*input)[indexOfInput]) {
                                            max                               = (*input)[indexOfInput];
                                            (*result)[indexOfResult]          = max;
                                            (*indexOfMaxInput)[indexOfResult] = indexOfInput;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // std::cout << indexOfMaxInput << '\n';

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfDelta  = this_delta->GetShape();

        int batchsize   = (*shapeOfDelta)[1];
        int channelsize = (*shapeOfDelta)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfDelta)[3];
        int colsize     = (*shapeOfDelta)[4];

        int indexOfDelta = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        indexOfDelta = Index4D(shapeOfDelta, ba, ch, ro, co);
                        (*input_delta)[(*indexOfMaxInput)[indexOfDelta]] += (*this_delta)[indexOfDelta];
                    }
                }
            }
        }

        return TRUE;
    }
};
//
#endif  // Maxpooling4D_H_
