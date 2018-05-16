#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"
#include <cstdio>

template<typename DTYPE> class MatMul : public Operator<DTYPE>{
private:
#if __CUDNN__
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc, filterDeltaDesc;
    DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevFilter, *m_pDevInputDelta, *m_pDevDelta, *m_pDevFilterDelta;
    DTYPE *m_aHostInput, *m_aHostOutput, *m_aHostFilter, *m_aHostInputDelta, *m_aHostDelta, *m_aHostFilterDelta;
#endif  // __CUDNN__

public:
    MatMul(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pWeight, pInput, pName) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pWeight, pInput);
    }

    virtual ~MatMul() {
        std::cout << "Convolution2D::~Convolution2D()" << '\n';
        Delete();
    }

    int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput) {
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetRowSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

#if __CUDNN__
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDeltaDesc));

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();
        int filterCapacity = pWeight->GetResult()->GetCapacity();

        checkCudaErrors(cudaMalloc((void **)&m_pDevInput, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevOutput, (outputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevFilter, (filterCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevInputDelta, (inputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevDelta, (outputCapacity * sizeof(DTYPE))));
        checkCudaErrors(cudaMalloc((void **)&m_pDevFilterDelta, (filterCapacity * sizeof(DTYPE))));

        m_aHostInput       = new DTYPE[inputCapacity];
        m_aHostOutput      = new DTYPE[outputCapacity];
        m_aHostFilter      = new DTYPE[filterCapacity];
        m_aHostInputDelta  = new DTYPE[inputCapacity];
        m_aHostDelta       = new DTYPE[outputCapacity];
        m_aHostFilterDelta = new DTYPE[filterCapacity];

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

        checkCudaErrors(cudaFree(m_pDevInput));
        checkCudaErrors(cudaFree(m_pDevOutput));
        checkCudaErrors(cudaFree(m_pDevFilter));
        checkCudaErrors(cudaFree(m_pDevInputDelta));
        checkCudaErrors(cudaFree(m_pDevDelta));
        checkCudaErrors(cudaFree(m_pDevFilterDelta));

        delete[] m_aHostInput;
        delete[] m_aHostOutput;
        delete[] m_aHostFilter;
        delete[] m_aHostInputDelta;
        delete[] m_aHostDelta;
        delete[] m_aHostFilterDelta;
#endif  // if __CUDNN__
    }

    int ForwardPropagate() {
        if (this->GetDevice() == CPU) ComputeForwardPropagateOnCPU();
        // if (this->GetDevice() == CPU) ComputeForwardPropagateOnCPU_MT();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeForwardPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int BackPropagate() {
        if (this->GetDevice() == CPU) ComputeBackPropagateOnCPU();
        // if (this->GetDevice() == CPU) ComputeBackPropagateOnCPU_MT();
#ifdef __CUDNN__
        else if (this->GetDevice() == GPU) ComputeBackPropagateOnGPU();
#endif  // if __CUDNN__
        else return FALSE;
        return TRUE;
    }

    int ComputeForwardPropagateOnCPU() {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                                (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                    += (*weight)[Index5D(weightTenShape, 0, 0, 0, co, hid)]
                                       * (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)];
                                // (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                // += (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)]
                                // * (*weight)[Index5D(weightTenShape, 0, 0, 0, hid, co)];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int ComputeBackPropagateOnCPU() {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();
        int hiddensize  = input_delta->GetColSize();

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                                weight_index = Index5D(weightTenShape, 0, 0, 0, co, hid);
                                input_index  = Index5D(inputTenShape, ti, ba, ch, ro, hid);
                                result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                                (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
                                (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int ForwardPropagate(int pTime, int pThreadNum) {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti          = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                += (*weight)[Index5D(weightTenShape, 0, 0, 0, co, hid)]
                                   * (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime, int pThreadNum) {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();
        int hiddensize  = input_delta->GetColSize();

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int weight_index = 0;
        int input_index  = 0;
        int result_index = 0;

        int ti          = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                            weight_index = Index5D(weightTenShape, 0, 0, 0, co, hid);
                            input_index  = Index5D(inputTenShape, ti, ba, ch, ro, hid);
                            result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                            (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
                            (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ComputeForwardPropagateOnGPU() {
        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Shape *shapeOfWeight  = weight->GetShape();

        Tensor<DTYPE> *input = this->GetInput()[1]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        Tensor<DTYPE> *result = this->GetResult();
        Shape *shapeOfResult  = result->GetShape();

        int batchsize = (*shapeOfResult)[1];
        int colsize   = (*shapeOfResult)[4];

        int rowsizeOfWeight = (*shapeOfWeight)[3];
        int colsizeOfWeight = (*shapeOfWeight)[4];

        int batchsizeOfInput = (*shapeOfInput)[1];
        int colsizeOfInput   = (*shapeOfInput)[4];

        cudnnConvolutionFwdAlgo_t algo;
        DTYPE alpha = 1;
        DTYPE beta  = 0;

        int inputCapacity  = input->GetCapacity();
        int outputCapacity = result->GetCapacity();
        int filterCapacity = weight->GetCapacity();

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            m_aHostFilter[i] = (*weight)[i];
        }

        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, 1, 1, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              rowsizeOfWeight, 1, 1, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1,
                                                   1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        // cudnnGetConvolutionNdForwardOutputDim( ... )
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, colsize, 1, 1));

        /* FIND THE BEST ALGORITHM ACCORDING TO PREFERENCE */
        // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &algo));

        size_t sizeInBytes  = 0;
        void  *devWorkSpace = NULL;

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->GetCudnnHandle(), inputTensorDesc, filterDesc, convDesc,
                                                           outputTensorDesc, algo, &sizeInBytes));

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaMalloc(&devWorkSpace, sizeInBytes));

            if (devWorkSpace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                return FALSE;
            }
        }

        checkCUDNN(cudnnConvolutionForward(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, filterDesc, m_pDevFilter, convDesc,
                                           algo, devWorkSpace, sizeInBytes, &beta, outputTensorDesc, m_pDevOutput));

        checkCudaErrors(cudaMemcpy(m_aHostOutput, m_pDevOutput, (outputCapacity * sizeof(DTYPE)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < outputCapacity; i++) {
            (*result)[i] = m_aHostOutput[i];
        }

        if (sizeInBytes != 0) {
            checkCudaErrors(cudaFree(devWorkSpace));
        }

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

    int ComputeBackPropagateOnGPU() {
        Tensor<DTYPE> *weight          = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        Shape *shapeOfWeight           = weight->GetShape();

        Tensor<DTYPE> *input       = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[1]->GetDelta();
        Shape *shapeOfInput        = input->GetShape();

        Tensor<DTYPE> *this_delta = this->GetDelta();
        Shape *shapeOfResult      = this_delta->GetShape();

        int batchsize = (*shapeOfResult)[1];
        int colsize   = (*shapeOfResult)[4];

        int rowsizeOfWeight = (*shapeOfWeight)[3];
        int colsizeOfWeight = (*shapeOfWeight)[4];

        int batchsizeOfInput = (*shapeOfInput)[1];
        int colsizeOfInput   = (*shapeOfInput)[4];

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        cudnnConvolutionBwdFilterAlgo_t filterAlgo;
        cudnnConvolutionBwdDataAlgo_t   dataAlgo;
        DTYPE alpha = 1;
        DTYPE beta  = 0;

        int inputCapacity       = input->GetCapacity();
        int filterCapacity      = weight->GetCapacity();
        int inputDeltaCapacity  = input->GetCapacity();
        int deltaCapacity       = this_delta->GetCapacity();
        int filterDeltaCapacity = weight_gradient->GetCapacity();

        for (int i = 0; i < inputCapacity; i++) {
            m_aHostInput[i] = (*input)[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            m_aHostFilter[i] = (*weight)[i];
        }

        for (int i = 0; i < deltaCapacity; i++) {
            m_aHostDelta[i] = (*this_delta)[i];
        }


        checkCudaErrors(cudaMemcpy(m_pDevInput, m_aHostInput, (inputCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevFilter, m_aHostFilter, (filterCapacity * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_pDevDelta, m_aHostDelta, (deltaCapacity) * sizeof(DTYPE), cudaMemcpyHostToDevice));

        // printf("\n***** DEVICE VARIABLES ARE COPIED FROM HOST TO DEVICE *****\n");

        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, 1, 1, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              rowsizeOfWeight, 1, 1, colsizeOfWeight));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1,
                                                   1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        /* WE CAN OBTAIN THE OUTPUT DIMENSION FROM cudnnGetConvolutionNdForwardOutputDim() FUNCTION
         * BUT, THESE ALREADY EXIST IN OUR MODEL*/
        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, colsize, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsizeOfInput, 1, 1, colsizeOfInput));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDeltaDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              rowsizeOfWeight, 1, 1, colsizeOfWeight));

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

        checkCUDNN(cudnnConvolutionBackwardData(this->GetCudnnHandle(), &alpha, filterDesc, m_pDevFilter, deltaDesc, m_pDevDelta, convDesc,
                                                dataAlgo, dataDevWorkSpace, dataSizeInBytes, &beta, inputDeltaDesc, m_pDevInputDelta));

        checkCUDNN(cudnnConvolutionBackwardFilter(this->GetCudnnHandle(), &alpha, inputTensorDesc, m_pDevInput, deltaDesc, m_pDevDelta, convDesc,
                                                  filterAlgo, filterDevWorkSpace, filterSizeInBytes, &beta, filterDesc, m_pDevFilterDelta));

        checkCudaErrors(cudaMemcpy(m_aHostInputDelta, m_pDevInputDelta, (inputDeltaCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_aHostFilterDelta, m_pDevFilterDelta, (filterDeltaCapacity) * sizeof(DTYPE), cudaMemcpyDeviceToHost));

        // Device to mem
        for (int i = 0; i < inputDeltaCapacity; i++) {
            (*input_delta)[i] += m_aHostInputDelta[i];
        }

        for (int i = 0; i < filterCapacity; i++) {
            (*weight_gradient)[i] += m_aHostFilterDelta[i];
        }

        if (dataSizeInBytes != 0) {
            checkCudaErrors(cudaFree(dataDevWorkSpace));
        }

        if (filterSizeInBytes != 0) {
            checkCudaErrors(cudaFree(filterDevWorkSpace));
        }

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // MATMUL_H_
