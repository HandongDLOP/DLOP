#ifndef RELU_H_
#define RELU_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Relu : public Operator<DTYPE>{
#if __CUDNN__

private:
    cudnnActivationDescriptor_t actDesc;
    cudnnTensorDescriptor_t inputTensorDesc, outputTensorDesc, deltaDesc, inputDeltaDesc;
    float *pDevInput;
    float *pDevOutput;
    float *pDevDelta;
    float *pDevInputDelta;
#endif  // if __CUDNN__

public:
    Relu(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Relu::Relu(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
#if __CUDNN__
        destroyHandles();
#endif  // if __CUDNN__
    }

    int Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Relu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
#if __CUDNN__
        createHandles();
        pDevInput      = NULL;
        pDevOutput     = NULL;
        pDevDelta      = NULL;
        pDevInputDelta = NULL;
#endif  // if __CUDNN__
        Shape *shapeOfResult = new Shape(pInput->GetResult()->GetShape());
        this->SetResult(new Tensor<DTYPE>(shapeOfResult));

        Shape *shapeOfDelta = new Shape(pInput->GetResult()->GetShape());
        this->SetDelta(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    #if __CUDNN__
    void createHandles() {
        checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&deltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
    }

    void destroyHandles() {
        checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(deltaDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inputDeltaDesc));
        checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
    }

    #endif  // if __CUDNN__

      #define mexPrintf    printf

    inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

            if (abort) exit(code);
        }
    }

     #define gpuErrchk(ans)    { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuMemReport(size_t *avail, size_t *total,
                             const char *title = 0, const size_t *free = 0, const bool sense = true) {
        char tstring[32] = { '\0' };

        gpuErrchk(cudaMemGetInfo(avail, total));

        if (free) {
            if (title) {
                strncpy(tstring, title, 31);
            }
            mexPrintf("%s Memory available: Free: %zu, Total: %zu, %s: %zu\n",
                      tstring, *avail, *total, (sense) ? "Allocated\0" : "Freed\0",
                      (sense) ? (*free - *avail) : (*avail - *free));
        } else {
            mexPrintf("Memory available: Free: %zu, Total: %zu\n", *avail, *total);
        }
    }

    int ComputeForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        int inputCapacity     = input->GetData()->GetCapacity();

#if __CUDNN__
        float  alpha = 1;
        float  beta = 0;
        Shape *shapeOfInput = input->GetShape();
        int    n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int    c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];

        float *hostInput = new float[inputCapacity];

        if (hostInput == NULL) {
            printf("Failed to allocate memory in %s (%s %d)", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        input->ConvertTo1D(hostInput);

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));

        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));

        // devInput and devOutput pointers may be equal but, identical Descriptor.
        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc, &alpha, inputTensorDesc, pDevInput, &beta,
                                          outputTensorDesc, pDevInput));

        checkCudaErrors(cudaMemcpy(hostInput, pDevInput, (inputCapacity * sizeof(float)), cudaMemcpyDeviceToHost));

        for (int i = 0; i < inputCapacity; i++) {
            (*result)[i] = hostInput[i];
        }

        delete[] hostInput;
        // checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(pDevInput));
        pDevInput = NULL;

#else  // if __CUDNN__

        for (int i = 0; i < inputCapacity; i++) {
            (*result)[i] = this->MAX((*input)[i], 0.f);
        }
#endif  // if __CUDNN__
        return TRUE;
    }

    int ComputeBackPropagate() {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        int capacity               = result->GetData()->GetCapacity();


#if __CUDNN__
        Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
        Shape *shapeOfInput  = input->GetShape();

        int   inputCapacity      = input->GetData()->GetCapacity();
        int   outputCapacity     = result->GetData()->GetCapacity();
        int   deltaCapacity      = this_delta->GetData()->GetCapacity();
        int   inputDeltaCapacity = input_delta->GetData()->GetCapacity();
        float alpha              = 1;
        float beta               = 0;

        float *hostInput      = new float[inputCapacity];
        float *hostOutput     = new float[outputCapacity];
        float *hostDelta      = new float[deltaCapacity];
        float *hostInputDelta = new float[inputDeltaCapacity];

        int n = (*shapeOfInput)[1]; int h = (*shapeOfInput)[3];
        int c = (*shapeOfInput)[2]; int w = (*shapeOfInput)[4];

        if ((hostInput == NULL) || (hostOutput == NULL) || (hostDelta == NULL) || (hostInputDelta == NULL)) {
            printf("Failed to allocate memory in %s (%s %d)", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        input->ConvertTo1D(hostInput);
        this_delta->ConvertTo1D(hostDelta);
        result->ConvertTo1D(hostOutput);

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(deltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));
        checkCUDNN(cudnnSetTensor4dDescriptor(inputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              n, c, h, w));

        checkCudaErrors(cudaMalloc(&pDevInput, (inputCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevOutput, (inputCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevDelta, (deltaCapacity * sizeof(float))));
        checkCudaErrors(cudaMalloc(&pDevInputDelta, (inputDeltaCapacity * sizeof(float))));

        checkCudaErrors(cudaMemcpy(pDevInput, hostInput, (inputCapacity * sizeof(float)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pDevDelta, hostDelta, (deltaCapacity * sizeof(float)), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pDevOutput, hostOutput, (outputCapacity * sizeof(float)), cudaMemcpyHostToDevice));


        checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), actDesc, &alpha, outputTensorDesc, pDevOutput,
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

#else  // if __CUDNN__

        for (int i = 0; i < capacity; i++) {
            if ((*result)[i] > 0.0) (*input_delta)[i] = (*this_delta)[i];
            else (*input_delta)[i] = 0;
        }
#endif  // if __CUDNN__
        return TRUE;
    }

    inline DTYPE MAX(DTYPE data1, DTYPE data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }
};

#endif  // RELU_H_
