#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"
#include <cstdio>

template<typename DTYPE> class Convolution2D : public Operator<DTYPE>{
private:
    int m_stride[2]  = { 0, };
    int m_padding[2] = { 0, };

#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta;
    DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostFilter, *m_aHostInputDelta, *m_aHostDelta;
#endif  // __CUDNN__

    Device m_Device;

public:
    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, 0, 0);
    }

    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding, padding);
    }

    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2, std::string pName = "NO NAME") : Operator<DTYPE>(pInput, pWeight, pName) {
        Alloc(pInput, pWeight, stride1, stride2, padding1, padding2);
    }

    virtual ~Convolution2D() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, int padding1, int padding2) {
        int outputWidth  = 0;
        int outputHeight = 0;

        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        m_Device = Device::CPU;

        m_stride[0] = stride1;
        m_stride[1] = stride2;

        m_padding[0] = padding1;
        m_padding[1] = padding2;

        outputHeight = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + (2 * m_padding[0])) / m_stride[0] + 1;
        outputWidth  = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + (2 * m_padding[1])) / m_stride[1] + 1;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));

#if __CUDNN__
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));
        // for cudnn
        m_pDevInput      = NULL;
        m_pDevFilter     = NULL;
        m_pDevOutput     = NULL;
        m_pDevInputDelta = NULL;
        m_pDevDelta      = NULL;

        m_aHostInput      = new DTYPE[pInput->GetResult()->GetCapacity()];
        m_aHostOutput     = new DTYPE[this->GetResult()->GetCapacity()];
        m_aHostFilter     = new DTYPE[pWeight->GetResult()->GetCapacity()];
        m_aHostInputDelta = new DTYPE[pInput->GetResult()->GetCapacity()];
        m_aHostDelta      = new DTYPE[this->GetResult()->GetCapacity()];

#endif  // if __CUDNN__

        return TRUE;
    }

    void Delete() {
#if __CUDNN__
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDeltaDesc));

        delete[] m_aHostInput;
        delete[] m_aHostOutput;
        delete[] m_aHostFilter;
        delete[] m_aHostInputDelta;
        delete[] m_aHostDelta;
#endif  // if __CUDNN__
    }

    int ComputeForwardPropagate() {
        if (m_Device == Device::CPU) ComputeForwardPropagateOnCPU();
#ifdef __CUDNN__
        else if (m_Device == Device::GPU) ComputeForwardPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int ComputeBackPropagate() {
        if (m_Device == Device::CPU) ComputeBackPropagateOnCPU();
#ifdef __CUDNN__
        else if (m_Device == Device::GPU) ComputeBackPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    void SetDeviceCPU() {
        m_Device = Device::CPU;
    }

#ifdef __CUDNN__
    void SetDeviceGPU() {
        m_Device = Device::GPU;
    }

#endif  // if __CUDNN__

    int ComputeForwardPropagateOnCPU() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

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
                                        += ((*input)[Index4D(shapeOfInput, ba, wch, m_stride[0] * ro + wro, m_stride[1] * co + wco)]
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

    int ComputeBackPropagateOnCPU() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    input_index  = Index4D(shapeOfInput, ba, wch, m_stride[0] * ro + wro, m_stride[1] * co + wco);
                                    weight_index = Index4D(shapeOfWeight, ch, wch, wro, wco);
                                    result_index = Index4D(shapeOfResult, ba, ch, ro, co);

                                    (*input_delta)[input_index]
                                        += ((*weight)[weight_index] * (*this_delta)[result_index]);

                                    (*weight_gradient)[weight_index]
                                        += ((*input)[input_index] * (*this_delta)[result_index]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ComputeForwardPropagateOnGPU() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        size_t freeMem  = 0;
        size_t totalMem = 0;
        size_t allocMem = 0;

        int i_n = (*shapeOfInput)[1]; int i_h = (*shapeOfInput)[3];
        int i_c = (*shapeOfInput)[2]; int i_w = (*shapeOfInput)[4];
        int f_n = (*shapeOfWeight)[1];
        int f_h = (*shapeOfWeight)[3]; int f_w = (*shapeOfWeight)[4];

        int inputCapacity  = input->GetCapacity();
        int filterCapacity = weight->GetCapacity();

        input->ConvertTo1D(m_aHostInput);
        weight->ConvertTo1D(m_aHostFilter);

        checkCudaErrors(cudaMalloc(&m_pDevInput, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc(&m_pDevFilter, (filterCapacity * sizeof(DTYPE))));

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));


        cudnnConvolutionFwdAlgo_t algo;

        /* n : # of images(batch size),       c : # of feature maps per image
         * h : height of each feature map,    w : width of each feature map*/
        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));


        /* k : # of output feature map,       c : # of input feature map,
         * h : height of each filter,         w : width of each filter */
        int k = (*shapeOfWeight)[1];
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              k, c, (*shapeOfWeight)[3], (*shapeOfWeight)[4]));


        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[0], m_stride[1], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        // cudnnGetConvolutionNdForwardOutputDim( ... )
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, channelsize, rowsize, colsize));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &algo));


        size_t sizeInBytes  = 0;
        void  *devWorkSpace = NULL;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc, algo, &sizeInBytes));

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&devWorkSpace, sizeInBytes));

            if (devWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        DTYPE alpha          = 1;
        DTYPE beta           = 0;
        int   outputCapacity = result->GetCapacity();

        checkCudaErrors(cudaMalloc(&m_pDevOutput, (outputCapacity * sizeof(DTYPE))));

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, filterDesc, m_pDevFilter, convDesc,
                                           algo, devWorkSpace, sizeInBytes, &beta, outputTensorDesc, m_pDevOutput));

        // &m_pDevOutput
        checkCudaErrors(cudaMemcpy(m_aHostOutput, m_pDevOutput, (outputCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());

        for (int i = 0; i < outputCapacity; i++) {
            (*result)[i] = m_aHostOutput[i];
        }

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaFree(devWorkSpace));
        }

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(m_pDevInput));
        checkCudaErrors(cudaFree(m_pDevFilter));
        checkCudaErrors(cudaFree(m_pDevOutput));

        m_pDevInput  = NULL;
        m_pDevFilter = NULL;
        m_pDevOutput = NULL;

        return TRUE;
    }

    int ComputeBackPropagateOnGPU() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        cudnnConvolutionBwdFilterAlgo_t filterAlgo;
        cudnnConvolutionBwdDataAlgo_t   dataAlgo;
        DTYPE alpha = 1;
        DTYPE beta  = 0;

        int inputCapacity  = input->GetCapacity();
        int filterCapacity = weight->GetCapacity();


        input->ConvertTo1D(m_aHostInput);
        weight->ConvertTo1D(m_aHostFilter);

        // printf("\n***** TEMPORARY VARIABLES ARE COPIED *****\n");
        checkCudaErrors(cudaMalloc(&m_pDevInput, (inputCapacity * sizeof(DTYPE))));
        // checkCudaErrors(cudaMalloc(&m_pDevOutput, (outputCapacity * sizeof(DTYPE)) ));
        checkCudaErrors(cudaMalloc(&m_pDevFilter, (filterCapacity * sizeof(DTYPE))));

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));

        // printf("\n***** DEVICE VARIABLES ARE COPIED FROM HOST TO DEVICE *****\n");

        /* n : # of images(batch size),       c : # of feature maps per image
         * h : height of each feature map,    w : width of each feature map*/
        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int ch = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, ch, h, w));

        // cudnnSetFilter4dDescriptor(filterDesc, DATA_TYPE, TENSOR_FORMAT, k, c, h, w)

        /* k : # of output feature map,       c : # of input feature map,
         * h : height of each filter,         w : width of each filter */
        int k = (*shapeOfWeight)[1];
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              k, ch, (*shapeOfWeight)[3], (*shapeOfWeight)[4]));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[0], m_stride[1], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfInput)[4]));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDeltaDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              (*shapeOfWeight)[1], (*shapeOfWeight)[2], (*shapeOfWeight)[3], (*shapeOfWeight)[4]));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        // CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        // CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc,
                                                            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &dataAlgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDeltaDesc,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &filterAlgo));

        // printf("\n***** CUDA INITIALIZATION SUCCESS *****\n");

        size_t dataSizeInBytes = 0; size_t filterSizeInBytes = 0;
        void  *dataDevWorkSpace = NULL; void *filterDevWorkSpace = NULL;

        // *(this->m_pCudnnHandle)
        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(this->GetCudnnHandle(), filterDesc, deltaDesc, convDesc, inputDeltaDesc, dataAlgo, &dataSizeInBytes));
        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, deltaDesc, convDesc, filterDesc, filterAlgo, &filterSizeInBytes));

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&dataDevWorkSpace, dataSizeInBytes));

            if (dataDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&filterDevWorkSpace, filterSizeInBytes));

            if (filterDevWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        int inputDeltaCapacity = input->GetCapacity();
        int deltaCapacity      = this_delta->GetCapacity();

        checkCudaErrors(cudaMalloc(&m_pDevInputDelta, (inputDeltaCapacity) * sizeof(DTYPE)));
        checkCudaErrors(cudaMalloc(&m_pDevDelta, (deltaCapacity) * sizeof(DTYPE)));

        this_delta->ConvertTo1D(m_aHostDelta);

        checkCudaErrors(cudaMemcpy(m_pDevDelta, m_aHostDelta, (deltaCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));


        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &alpha, filterDesc, m_pDevFilter, deltaDesc, m_pDevDelta, convDesc,
                                                dataAlgo, dataDevWorkSpace, dataSizeInBytes, &beta, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, deltaDesc, m_pDevDelta, convDesc,
                                                  filterAlgo, filterDevWorkSpace, filterSizeInBytes, &beta, filterDesc, m_pDevFilter));

        checkCudaErrors(cudaMemcpy(m_aHostInputDelta, m_pDevInputDelta, (inputDeltaCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_aHostFilter, m_pDevFilter, (filterCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] += m_aHostInputDelta[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            (*weight_gradient)[i] += m_aHostFilter[i];
        }

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaFree(dataDevWorkSpace));
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaFree(filterDevWorkSpace));
        }

        checkCudaErrors(cudaFree(m_pDevInput));
        checkCudaErrors(cudaFree(m_pDevFilter));
        checkCudaErrors(cudaFree(m_pDevInputDelta));
        checkCudaErrors(cudaFree(m_pDevDelta));

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // CONVOLUTION_H_
