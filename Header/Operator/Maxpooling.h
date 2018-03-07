#ifndef Maxpooling4D_H_
#define Maxpooling4D_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Maxpooling4D : public Operator<DTYPE>{
private:
#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    float *pDevInput;
    float *pDevOutput;
    float *pDevFilter;
    float *pDevInputDelta;
    float *pDevDelta;
#endif  // __CUDNN__
    int m_stride[2] = { 0, };
    int m_mask[2]    = { 0, };
    int m_padding[2] = { 0, };

    Tensor<int> *indexOfMaxInput;

public:
    Maxpooling4D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol) : Operator<DTYPE>(pInput) {
        std::cout << "Maxpooling4D::Maxpooling4D(Operator<DTYPE> *, int, int)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol);
    }

    Maxpooling4D(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling4D::Maxpooling4D(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(pInput, strideRow, strideCol, maskRow, maskCol, padding);
    }

    ~Maxpooling4D() {
        std::cout << "Maxpooling4D::~Maxpooling4D()" << '\n';
        #if __CUDNN__
        destroyHandles();
        #endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol, int maskRow, int maskCol, int padding) {
        std::cout << "Maxpooling4D::Alloc(Operator<DTYPE> *, int, int)" << '\n';
        #if __CUDNN__
        createHandles();
        #endif  // if __CUDNN__

        Shape *shapeOfInput = pInput->GetResult()->GetShape();

        m_stride[0] = strideRow;
        m_stride[1] = strideCol;

        m_mask[0] = maskRow;
        m_mask[1] = maskCol;

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }  // 4D

        int rowsize = 0;
        int colsize = 0;

        if (padding == SAME) {
            m_padding[0] = (((*shapeOfInput)[3] - 1) * m_stride[0] - (*shapeOfInput)[3] + maskCol) / 2;
            m_padding[1] = (((*shapeOfInput)[4] - 1) * m_stride[1] - (*shapeOfInput)[4] + maskRow) / 2;
        }

        rowsize = ((*shapeOfInput)[3] - maskRow + (2 * m_padding[0])) / strideRow + 1;
        colsize = ((*shapeOfInput)[4] - maskCol + (2 * m_padding[1])) / strideCol + 1;

        // rowsize = (*shapeOfInput)[3] / strideRow;
        // colsize = (*shapeOfInput)[4] / strideCol;

        // if ((*shapeOfInput)[3] % strideRow > 0) {
        // rowsize = (*shapeOfInput)[3] / strideRow + 1;
        // } else {
        // rowsize = (*shapeOfInput)[3] / strideRow;
        // }
        //
        // if ((*shapeOfInput)[4] % strideCol > 0) {
        // colsize = (*shapeOfInput)[4] / strideCol + 1;
        // } else {
        // colsize = (*shapeOfInput)[4] / strideCol;
        // }

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize));


        indexOfMaxInput = new Tensor<int>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], rowsize, colsize);

        #if __CUDNN__
        pDevInput      = NULL;
        pDevFilter     = NULL;
        pDevOutput     = NULL;
        pDevInputDelta = NULL;
        pDevDelta      = NULL;
        #endif  // if __CUDNN__

        return TRUE;
    }

#if __CUDNN__
    void createHandles() {
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));
    }

    void destroyHandles() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDeltaDesc));
    }

#endif  // if __CUDNN__


    //
    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();
        result->Reset();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int rowsizeOfMask = m_mask[0];
        int colsizeOfMask = m_mask[1];

        DTYPE max = 0.f;

        int indexOfResult = 0;
        int indexOfInput  = 0;

        int temprow = 0;
        int tempcol = 0;

#if 0
        int   padding_h = 0; int padding_w = 0;
        int   n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int   c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];
        float alpha = 1; float beta = 0;
        int   inputCapacity  = input->GetData()->GetCapacity();
        int   outputCapacity = result->GetData()->GetCapacity();

        float *hostInput  = new float[inputCapacity];
        float *hostOutput = new float[outputCapacity];

        if ((hostInput == NULL) || (hostOutput == NULL)) {
            printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        input->ConvertTo1D(hostInput);

        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevOutput, (outputCapacity * sizeof(float))));
        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));


        checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                               m_mask[0], m_mask[1], m_padding[0], m_padding[1], m_stride[0], m_stride[1]));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));

        int outputDim[4] = { 0, };
        checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, 4, outputDim));

        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              outputDim[0], outputDim[1], outputDim[2], outputDim[3]));


        checkCUDNN(cudnnPoolingForward(this->GetCudnnHandle(), poolingDesc, &alpha, inputTensorDesc, pDevInput,
                                       &beta, outputTensorDesc, pDevOutput));
        checkCudaErrors(cudaMemcpy(hostOutput, pDevOutput, (outputCapacity * sizeof(float)), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());

        for (int i = 0; i < inputCapacity; i++) {
            // if(i % w == 0) printf("\n");
            // printf("%f ", (*input)[i]);
        }
        // printf("input shape : %d,%d,%d,%d\n",n,c,h,w);

        for (int i = 0; i < outputCapacity; i++) {
            (*result)[i] = hostOutput[i];
            // if(i % outputDim[3] == 0) printf("\n");
            // printf("%f ", (*result)[i]);
        }
        // printf("output shape : %d,%d,%d,%d\n",outputDim[0],outputDim[1],outputDim[2],outputDim[3]);

        delete[] hostInput;
        delete[] hostOutput;
        // checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(pDevInput));
        checkCudaErrors(cudaFree(pDevOutput));
        pDevInput  = NULL;
        pDevOutput = NULL;
#else  // if 0

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int mro = 0; mro < rowsizeOfMask; mro++) {
                            for (int mco = 0; mco < colsizeOfMask; mco++) {
                                temprow = m_stride[0] * ro + mro;
                                tempcol = m_stride[1] * co + mco;

                                indexOfResult = Index4D(shapeOfResult, ba, ch, ro, co);
                                indexOfInput  = Index4D(shapeOfInput, ba, ch, temprow, tempcol);

                                if ((mro == 0) && (mco == 0)) {
                                    max                               = (*input)[indexOfInput];
                                    (*result)[indexOfResult]          = max;
                                    (*indexOfMaxInput)[indexOfResult] = indexOfInput;
                                } else {
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

        // for(int i = 0; i < inputCapacity; i++){
        // if(i % w == 0) printf("\n");
        // printf("%f ", (*input)[i]);
        // }
        // printf("input shape : %d,%d,%d,%d\n",n,c,h,w);

        // for(int i = 0; i < outputCapacity; i++){
        // (*result)[i] = hostOutput[i];
        // if(i % outputDim[3] == 0) printf("\n");
        // printf("%f ", (*result)[i]);
        // }
        // printf("output shape : %d,%d,%d,%d\n",outputDim[0],outputDim[1],outputDim[2],outputDim[3]);

#endif  // if 0
        // std::cout << indexOfMaxInput << '\n';

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        input_delta->Reset();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfDelta       = this_delta->GetShape();

        int batchsize   = (*shapeOfDelta)[1];
        int channelsize = (*shapeOfDelta)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfDelta)[3];
        int colsize     = (*shapeOfDelta)[4];

        int indexOfDelta = 0;

#if 0
        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];

        int out_ch = (*shapeOfResult)[2];
        int out_r  = (*shapeOfResult)[3];
        int out_c  = (*shapeOfResult)[4];

        float alpha = 1; float beta = 0;
        int   inputCapacity      = input->GetData()->GetCapacity();
        int   outputCapacity     = result->GetData()->GetCapacity();
        int   deltaCapacity      = this_delta->GetData()->GetCapacity();
        int   inputDeltaCapacity = inputCapacity;

        float *hostInput      = new float[inputCapacity];
        float *hostOutput     = new float[outputCapacity];
        float *hostInputDelta = new float[inputDeltaCapacity];
        float *hostDelta      = new float[deltaCapacity];

        if ((hostInput == NULL) || (hostInputDelta == NULL) || (hostDelta == NULL) || (hostOutput == NULL)) {
            printf("Failed to allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        this_delta->ConvertTo1D(hostDelta);
        input->ConvertTo1D(hostInput);
        result->ConvertTo1D(hostOutput);

        checkCudaErrors(cudaMalloc(&pDevInputDelta, (inputDeltaCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevDelta, (deltaCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevOutput, (outputCapacity * sizeof(float))));

        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pDevOutput, hostOutput, (outputCapacity * sizeof(float)), cudaMemcpyHostToDevice));

        checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                               m_mask[0], m_mask[1], m_padding[0], m_padding[1], m_stride[0], m_stride[1]));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, out_ch, out_r, out_c));
        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, out_ch, out_r, out_c));


        checkCUDNN(cudnnPoolingBackward(this->GetCudnnHandle(), poolingDesc, &alpha, outputTensorDesc, pDevOutput,
                                        deltaDesc, pDevDelta, inputTensorDesc, pDevInput, &beta, inputDeltaDesc, pDevInputDelta));

        checkCudaErrors(cudaMemcpy(hostInputDelta, pDevInputDelta, (inputDeltaCapacity * sizeof(float)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] = hostInputDelta[i];
        }


        delete[] hostInput;
        delete[] hostOutput;
        delete[] hostInputDelta;
        delete[] hostDelta;

        checkCudaErrors(cudaFree(pDevInput));
        checkCudaErrors(cudaFree(pDevOutput));
        checkCudaErrors(cudaFree(pDevDelta));
        checkCudaErrors(cudaFree(pDevInputDelta));
#else  // if 0

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        indexOfDelta                                      = Index4D(shapeOfDelta, ba, ch, ro, co);
                        (*input_delta)[(*indexOfMaxInput)[indexOfDelta]] += (*this_delta)[indexOfDelta];
                    }
                }
            }
        }
#endif  // if 0

        return TRUE;
    }
};
//
#endif  // Maxpooling4D_H_
