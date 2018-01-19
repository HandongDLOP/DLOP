#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Convolution2D : public Operator<DTYPE>{
private:
    int stride[4] = { 0, };

public:
    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride0, stride1, stride2, stride3);
    }

    virtual ~Convolution2D() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3) {
        Shape *shapeOfInput = pInput->GetResult()->GetShape();
        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        int width = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + 1) / stride1;
        int height = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + 1) / stride2;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], height, width));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], height, width));

        stride[0] = stride0;
        stride[1] = stride1;
        stride[2] = stride2;
        stride[3] = stride3;

        return TRUE;
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult = result->GetShape();
        result->Reset();

        int batchsize = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize = (*shapeOfResult)[3];
        int colsize = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight = (*shapeOfWeight)[3];
        int colsizeOfWeight = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    (*result)[Index4D(shapeOfResult, ba, ch, ro, co)]
                                        += ((*input)[Index4D(shapeOfInput, ba, wch, stride[1] * ro + wro, stride[2] * co + wco)]
                                            * (*weight)[Index4D(shapeOfWeight, ch, wch, wro, wco)]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput = input->GetShape();
        input_delta->Reset();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight = weight->GetShape();
        weight_gradient->Reset();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult = this_delta->GetShape();

        int batchsize = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize = (*shapeOfResult)[3];
        int colsize = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight = (*shapeOfWeight)[3];
        int colsizeOfWeight = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int input_index = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    input_index = Index4D(shapeOfInput, ba, wch, stride[1] * ro + wro, stride[2] * co + wco);
                                    weight_index = Index4D(shapeOfWeight, ch, wch, wro, wco);
                                    result_index = Index4D(shapeOfResult, ba, ch, ro, co);

                                    (*input_delta)[input_index]
                                        += ((*weight)[weight_index]
                                            * (*this_delta)[result_index]);
                                    (*weight_gradient)[weight_index]
                                        += ((*input)[input_index]
                                            * (*this_delta)[result_index]);

                                    // (*result)[Index4D(shapeOfResult, ba, ch, ro, co)]
                                    // += ((*input)[Index4D(shapeOfInput, ba, wch, ro + wro, co + wco)]
                                    // * (*weight)[Index4D(shapeOfWeight, ch, wch, wro, wco)]);
                                }
                            }
                        }
                    }
                }
            }
        }


        return TRUE;
    }
};

template<typename DTYPE> class Convolution3D : public Operator<DTYPE>{
private:
public:
    Convolution3D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride0, stride1, stride2, stride3);
    }

    virtual ~Convolution3D() {
        std::cout << "Convolution3D::~Convolution3D()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3) {
        return TRUE;
    }

    int ComputeForwardPropagate() {
        return TRUE;
    }

    int ComputeBackPropagate() {
        return TRUE;
    }
};

#endif  // CONVOLUTION_H_
