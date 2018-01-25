#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Convolution2D : public Operator<DTYPE>{
private:
#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionFwdAlgo_t algo;
    float *devInput;
    float *devOutput;
    float *devFilter;
#endif //__CUDNN__
    int m_stride[4] = { 0, };

public:
    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
#if __CUDNN__
        createHandles();
#endif
        Alloc(pInput, pWeight, stride0, stride1, stride2, stride3);
    }

    virtual ~Convolution2D() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
#if __CUDNN__
        destroyHandles();
#endif
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

        m_stride[0] = stride0;
        m_stride[1] = stride1;
        m_stride[2] = stride2;
        m_stride[3] = stride3;

#if __CUDNN__
        devInput = NULL;
        devFilter = NULL;
        devOutput = NULL;

        checkCudaErrors(cudaMalloc(&devInput, (*shapeOfInput)[3] * (*shapeOfInput)[4] * sizeof(float)));
        checkCudaErrors(cudaMalloc(&devOutput, height * width * sizeof(float)));
        checkCudaErrors(cudaMalloc(&devFilter, (*shapeOfWeight)[3] * (*shapeOfWeight)[4] * sizeof(float)));
        if(devInput == NULL || devOutput == NULL || devFilter == NULL){
          printf("Failed to allcate DEVICE memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
          return FALSE;
        }
        checkCudaErrors(cudaMemcpy(devInput, pInput->GetResult()->GetData(), (*shapeOfInput)[3] * (*shapeOfInput)[4] * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devFilter, pWeight->GetResult()->GetData(), height * width * sizeof(float), cudaMemcpyHostToDevice));
#endif

        return TRUE;
    }

#if __CUDNN__
    void createHandles(){
      checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
      checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    }

    void destroyHandles(){
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
      }


#endif


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

#if __CUDNN__
        /* n : # of images(batch size),       c : # of feature maps per image
           h : height of each feature map,    w : width of each feature map*/
        int n = (*shapeOfInput)[1];     int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2];     int w = (*shapeOfInput)[4];
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            n, c, h, w));

        // cudnnSetFilter4dDescriptor(filterDesc, DATA_TYPE, TENSOR_FORMAT, k, c, h, w)
        /* k : # of output feature map,       c : # of input feature map,
           h : height of each filter,         w : width of each filter */
        int k = (*shapeOfWeight)[1];
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            k, c, (*shapeOfWeight)[3], (*shapeOfWeight)[4]));

        int padding_h = 0; int padding_w = 0;
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, m_stride[1], m_stride[2], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
           BUT, THESE ALREADY EXIST IN OUR MODEL*/
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
               n, channelsize, rowsize, colsize));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(*(this->m_pCudnnHandle), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));


        size_t sizeInBytes = 0;
        void* devWorkSpace = NULL;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(*(this->m_pCudnnHandle), inputTensorDesc, filterDesc, convDesc, outputTensorDesc, algo, &sizeInBytes));

        if (sizeInBytes != 0){
          checkCudaErrors(cudaMalloc(&devWorkSpace, sizeInBytes));
          if (devWorkSpace == NULL){
            printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
          }
        }

        float alpha = 1;
        float beta = 0;
        checkCUDNN(cudnnConvolutionForward(*(this->m_pCudnnHandle), &alpha, inputTensorDesc, devInput, filterDesc, devFilter, convDesc,
            algo, devWorkSpace, sizeInBytes, &beta, outputTensorDesc, devOutput))

        //float *hostOutput = (float*)calloc (rowsize * colsize, sizeof(float));
        float *hostOutput = &((*(result->GetData()))[0]);
        checkCudaErrors(cudaMemcpy(hostOutput, devOutput, rowsize * colsize * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        //result->GetData() = hostOutput;

        if (sizeInBytes != 0){
          checkCudaErrors( cudaFree(devWorkSpace) );
        }
        checkCudaErrors(cudaFree(devInput));
        checkCudaErrors(cudaFree(devFilter));
        checkCudaErrors(cudaFree(devOutput));

#else
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    (*result)[Index4D(shapeOfResult, ba, ch, ro, co)]
                                        += ((*input)[Index4D(shapeOfInput, ba, wch, m_stride[1] * ro + wro, m_stride[2] * co + wco)]
                                            * (*weight)[Index4D(shapeOfWeight, ch, wch, wro, wco)]);
                                }
                            }
                        }
                    }
                }
            }
        }
#endif //__CUDNN__
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
                                    input_index = Index4D(shapeOfInput, ba, wch, m_stride[1] * ro + wro, m_stride[2] * co + wco);
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
