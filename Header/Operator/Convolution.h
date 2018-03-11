#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"
#include <cstdio>

template<typename DTYPE> class Convolution2D : public Operator<DTYPE>{
private:
#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    float *pDevInput;
    float *pDevOutput;
    float *pDevFilter;
    float *pDevInputDelta;
    float *pDevDelta;
#endif  // __CUDNN__
    int m_stride[4] = { 0, };
    int m_padding[2] = { 0, };

public:
    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
    #if __CUDNN__
        createHandles();
    #endif  // if __CUDNN__
        Alloc(pInput, pWeight, stride0, stride1, stride2, stride3);
    }

    Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, int padding, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
    #if __CUDNN__
        createHandles();
    #endif  // if __CUDNN__
        Alloc(pInput, pWeight, stride0, stride1, stride2, stride3, padding);
    }

    virtual ~Convolution2D() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
    #if __CUDNN__
        destroyHandles();
    #endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3, int padding) {
        if ((padding != SAME) && (padding != VALID)) {
            printf("Receive invalid padding in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();

        m_stride[0] = stride0;
        m_stride[1] = stride1;
        m_stride[2] = stride2;
        m_stride[3] = stride3;

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }
        int outputWidth  = 0;
        int outputHeight = 0;

        if (padding == SAME) {
            m_padding[0] = (((*shapeOfInput)[3] - 1) * m_stride[0] - (*shapeOfInput)[3] + (*shapeOfWeight)[3]) / 2;
            m_padding[1] = (((*shapeOfInput)[4] - 1) * m_stride[1] - (*shapeOfInput)[4] + (*shapeOfWeight)[4]) / 2;
        }

        outputHeight = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + (2 * m_padding[0])) / m_stride[1] + 1;
        outputWidth  = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + (2 * m_padding[0])) / m_stride[2] + 1;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], outputHeight, outputWidth));


    #if __CUDNN__
        pDevInput      = NULL;
        pDevFilter     = NULL;
        pDevOutput     = NULL;
        pDevInputDelta = NULL;
        pDevDelta      = NULL;
    #endif  // if __CUDNN__

        return TRUE;
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride0, int stride1, int stride2, int stride3) {
        Shape *shapeOfInput  = pInput->GetResult()->GetShape();
        Shape *shapeOfWeight = pWeight->GetResult()->GetShape();

        if ((*shapeOfInput)[0] != 1) {
            printf("Receive invalid timesize value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        int width  = ((*shapeOfInput)[4] - (*shapeOfWeight)[4] + 1) / stride1;
        int height = ((*shapeOfInput)[3] - (*shapeOfWeight)[3] + 1) / stride2;

        this->SetResult(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], height, width));
        this->SetDelta(new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfWeight)[1], height, width));

        m_stride[0] = stride0;
        m_stride[1] = stride1;
        m_stride[2] = stride2;
        m_stride[3] = stride3;

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
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));
    }

    void destroyHandles() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDeltaDesc));
    }

# define mexPrintf    printf
    inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

            if (abort) exit(code);
        }
    }

# define gpuErrchk(ans)    { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuMemReport(size_t *avail, size_t *total,
                             const char *title = 0, const size_t *free = 0, const bool sense = true) {
        char tstring[32] = { '\0' };

        gpuErrchk(cudaMemGetInfo(avail, total));

        if (free) {
            if (title) {
                strncpy(tstring, title, 31);
            }
            mexPrintf("%s Memory avaliable: Free: %zu, Total: %zu, %s: %zu\n",
                      tstring, *avail, *total, (sense) ? "Allocated\0" : "Freed\0",
                      (sense) ? (*free - *avail) : (*avail - *free));
        } else {
            mexPrintf("Memory avaliable: Free: %zu, Total: %zu\n", *avail, *total);
        }
    }

#endif  // if __CUDNN__

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();
        // result->Reset();

        int batchsize   = (*shapeOfResult)[1];
        int channelsize = (*shapeOfResult)[2];  // == shapeOfWeight[1]
        int rowsize     = (*shapeOfResult)[3];
        int colsize     = (*shapeOfResult)[4];

        int channelsizeOfWeight = (*shapeOfWeight)[2];
        int rowsizeOfWeight     = (*shapeOfWeight)[3];
        int colsizeOfWeight     = (*shapeOfWeight)[4];

        int rowsizeOfInput = (*shapeOfInput)[3];
        int colsizeOfInput = (*shapeOfInput)[4];


#if __CUDNN__
        // printf("\n***** START CUDNN CONVOLUTION FORWARD FUNCTION *****\n");
        size_t freeMem  = 0;
        size_t totalMem = 0;
        size_t allocMem = 0;

        int i_n = (*shapeOfInput)[1]; int i_h = (*shapeOfInput)[3];
        int i_c = (*shapeOfInput)[2]; int i_w = (*shapeOfInput)[4];
        int f_n = (*shapeOfWeight)[1];
        int f_h = (*shapeOfWeight)[3]; int f_w = (*shapeOfWeight)[4];

        int inputCapacity  = input->GetCapacity();
        int filterCapacity = weight->GetCapacity();

        float *hostInput  = new float[inputCapacity];
        float *hostFilter = new float[filterCapacity];

        if ((hostInput == NULL) || (hostFilter == NULL)) {
            printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        input->ConvertTo1D(hostInput);
        weight->ConvertTo1D(hostFilter);

        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        // checkCudaErrors(cudaMalloc(&pDevOutput, (outputCapacity * sizeof(float)) ));
        checkCudaErrors(cudaMalloc(&pDevFilter, (filterCapacity * sizeof(float))));

        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pDevFilter, hostFilter, (filterCapacity * sizeof(float)), cudaMemcpyHostToDevice));

        delete[] hostInput;
        delete[] hostFilter;


        cudnnConvolutionFwdAlgo_t algo;

        /* n : # of images(batch size),       c : # of feature maps per image
         * h : height of each feature map,    w : width of each feature map*/
        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));

        // cudnnSetFilter4dDescriptor(filterDesc, DATA_TYPE, TENSOR_FORMAT, k, c, h, w)

        /* k : # of output feature map,       c : # of input feature map,
         * h : height of each filter,         w : width of each filter */
        int k = (*shapeOfWeight)[1];
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              k, c, (*shapeOfWeight)[3], (*shapeOfWeight)[4]));


        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[1], m_stride[2], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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

        float  alpha          = 1;
        float  beta           = 0;
        int    outputCapacity = result->GetCapacity();
        float *hostOutput     = new float[outputCapacity];

        if (hostOutput == NULL) {
            printf("Failed to allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }
        checkCudaErrors(cudaMalloc(&pDevOutput, (outputCapacity * sizeof(float))));

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &alpha, inputTensorDesc, pDevInput, filterDesc, pDevFilter, convDesc,
                                           algo, devWorkSpace, sizeInBytes, &beta, outputTensorDesc, pDevOutput));

        // &pDevOutput
        checkCudaErrors(cudaMemcpy(hostOutput, pDevOutput, (outputCapacity * sizeof(float)), cudaMemcpyDeviceToHost));
        // checkCudaErrors(cudaDeviceSynchronize());

        for (int i = 0; i < outputCapacity; i++) {
            (*result)[i] = hostOutput[i];
        }
        delete[] hostOutput;

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaFree(devWorkSpace));
        }

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(pDevInput));
        checkCudaErrors(cudaFree(pDevFilter));
        checkCudaErrors(cudaFree(pDevOutput));

        pDevInput  = NULL;
        pDevFilter = NULL;
        pDevOutput = NULL;


#else  // if __CUDNN__ is undefined

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
#endif  // __CUDNN__

        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();
        // input_delta->Reset();

        Tensor<DTYPE> *weight          = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();
        // weight_gradient->Reset();

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

#if __CUDNN__
        cudnnConvolutionBwdFilterAlgo_t filterAlgo;
        cudnnConvolutionBwdDataAlgo_t   dataAlgo;
        float alpha = 1;
        float beta  = 0;

        int inputCapacity  = input->GetCapacity();
        int filterCapacity = weight->GetCapacity();

        float *hostInput  = new float[inputCapacity];
        float *hostFilter = new float[filterCapacity];

        if ((hostInput == NULL) || (hostFilter == NULL)) {
            printf("Failed to allocate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        input->ConvertTo1D(hostInput);
        weight->ConvertTo1D(hostFilter);

        // printf("\n***** TEMPORARY VARIABLES ARE COPIED *****\n");
        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        // checkCudaErrors(cudaMalloc(&pDevOutput, (outputCapacity * sizeof(float)) ));
        checkCudaErrors(cudaMalloc(&pDevFilter, (filterCapacity * sizeof(float))));

        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity) * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pDevFilter, hostFilter, (filterCapacity) * sizeof(float), cudaMemcpyHostToDevice));

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

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, m_padding[0], m_padding[1], m_stride[1], m_stride[2], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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

        float *hostInputDelta = new float[inputDeltaCapacity];
        float *hostDelta      = new float[deltaCapacity];

        if ((hostInputDelta == NULL) || (hostDelta == NULL) || (hostFilter == NULL)) {
            printf("Failed to allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        checkCudaErrors(cudaMalloc(&pDevInputDelta, (inputDeltaCapacity) * sizeof(float)));
        checkCudaErrors(cudaMalloc(&pDevDelta, (deltaCapacity) * sizeof(float)));

        this_delta->ConvertTo1D(hostDelta);

        checkCudaErrors(cudaMemcpy(pDevDelta, hostDelta, (deltaCapacity) * sizeof(float), cudaMemcpyHostToDevice));


        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &alpha, filterDesc, pDevFilter, deltaDesc, pDevDelta, convDesc,
                                                dataAlgo, dataDevWorkSpace, dataSizeInBytes, &beta, inputDeltaDesc, pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &alpha, inputTensorDesc, pDevInput, deltaDesc, pDevDelta, convDesc,
                                                  filterAlgo, filterDevWorkSpace, filterSizeInBytes, &beta, filterDesc, pDevFilter));

        checkCudaErrors(cudaMemcpy(hostInputDelta, pDevInputDelta, (inputDeltaCapacity) * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostFilter, pDevFilter, (filterCapacity) * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] = hostInputDelta[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            (*weight_gradient)[i] = hostFilter[i];
        }

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaFree(dataDevWorkSpace));
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaFree(filterDevWorkSpace));
        }

        checkCudaErrors(cudaFree(pDevInput));
        checkCudaErrors(cudaFree(pDevFilter));
        checkCudaErrors(cudaFree(pDevInputDelta));
        checkCudaErrors(cudaFree(pDevDelta));
        delete[] hostInput;
        delete[] hostFilter;
        delete[] hostInputDelta;
        delete[] hostDelta;

#else  // if __CUDNN__

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {  // Batchsize of weight kernel
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int wch = 0; wch < channelsizeOfWeight; wch++) {  // == (*shapeOfInput)[2];
                            for (int wro = 0; wro < rowsizeOfWeight; wro++) {
                                for (int wco = 0; wco < colsizeOfWeight; wco++) {
                                    input_index  = Index4D(shapeOfInput, ba, wch, m_stride[1] * ro + wro, m_stride[2] * co + wco);
                                    weight_index = Index4D(shapeOfWeight, ch, wch, wro, wco);
                                    result_index = Index4D(shapeOfResult, ba, ch, ro, co);

                                    (*input_delta)[input_index]
                                        += ((*weight)[weight_index] * (*this_delta)[result_index]);

                                    (*weight_gradient)[weight_index]
                                        += ((*input)[input_index] * (*this_delta)[result_index]);

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

#endif  // if __CUDNN__

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
